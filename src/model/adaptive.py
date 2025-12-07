"""
HIERARCHICAL ADAPTIVE TRANSFORMER WRAPPER

Integrates exit gates and MoE routers for two-dimensional adaptive computation:
- Vertical (depth): Early exit via exit gates
- Horizontal (width): Selective FFN via routers

Optimized for speedup with research-backed improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any
import random

from .gate import ExitGate
from .router import MoERouter
from .tokenstate import TokenState


class HierarchicalTransformerWrapper(nn.Module):
    """
    Hierarchical wrapper that adds adaptive computation to a frozen base model.
    
    Features:
    - Exit gates at strategic layers for early termination
    - MoE routers at all layers for selective FFN computation
    - Progressive layer dropout during training
    - Capacity scheduling support
    """

    def __init__(
        self,
        base_model: nn.Module,
        exit_layers: List[int] = None,
        capacity: float = 0.55,
        disable_exit_gates: bool = False,
        use_layer_dropout: bool = False,
        layer_dropout_max: float = 0.2,
    ):
        """
        Initialize the hierarchical wrapper.
        
        Args:
            base_model: Frozen base transformer model
            exit_layers: Layer indices with exit gates (default: [4, 8, 12, 16, 19])
            capacity: Initial router capacity (fraction of tokens using FFN)
            disable_exit_gates: If True, disable exit gates (for router-only training)
            use_layer_dropout: If True, apply progressive layer dropout during training
            layer_dropout_max: Maximum layer dropout rate for final layer
        """
        super().__init__()
        
        self.base_model = base_model
        self.hidden_dim = base_model.config.hidden_size
        self.num_layers = base_model.config.num_hidden_layers
        self.capacity = capacity
        self.disable_exit_gates = disable_exit_gates
        self.use_layer_dropout = use_layer_dropout
        self.layer_dropout_max = layer_dropout_max
        
        # Default exit layers optimized for 22-layer model
        if exit_layers is None:
            exit_layers = [4, 8, 12, 16, 19]
        self.exit_layers = exit_layers

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        # Create exit gates (only at strategic layers)
        self.exit_gates = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            if layer_idx in exit_layers:
                gate = ExitGate(
                    self.hidden_dim,
                    temperature=0.5,
                    use_learnable_temp=True,
                )
                self.exit_gates.append(gate)
            else:
                self.exit_gates.append(None)

        # Create routers for all layers
        self.skip_routers = nn.ModuleList([
            MoERouter(
                self.hidden_dim,
                capacity=capacity,
                initial_temp=1.0,
                min_temp=0.1,
            )
            for _ in range(self.num_layers)
        ])
        
        # Training state
        self.current_progress = 0.0

    def count_trainable_params(self) -> int:
        """Count total trainable parameters in gates and routers"""
        total = 0
        
        for gate in self.exit_gates:
            if gate is not None:
                total += sum(p.numel() for p in gate.parameters() if p.requires_grad)
        
        for router in self.skip_routers:
            total += sum(p.numel() for p in router.parameters() if p.requires_grad)
        
        return total

    def set_progress(self, progress: float) -> None:
        """Set training progress for capacity/temperature scheduling"""
        self.current_progress = max(0.0, min(1.0, progress))

    def set_capacity(self, new_capacity: float) -> None:
        """Update capacity for all routers"""
        self.capacity = max(0.1, min(0.9, new_capacity))
        for router in self.skip_routers:
            router.adjust_capacity(self.capacity)

    def _should_drop_layer(self, layer_idx: int) -> bool:
        """Determine if layer should be dropped (training only)"""
        if not self.use_layer_dropout or not self.training:
            return False
        
        # Progressive dropout: higher rate for later layers
        position_ratio = layer_idx / (self.num_layers - 1)
        base_rate = self.layer_dropout_max * position_ratio
        
        # Ramp up during first half of training
        scale = min(self.current_progress * 2, 1.0)
        dropout_rate = base_rate * scale
        
        return random.random() < dropout_rate

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
        progress: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with adaptive computation.
        
        Args:
            input_ids: [batch, seq_len] input token IDs
            attention_mask: [batch, seq_len] attention mask
            training: Whether in training mode
            progress: Training progress 0.0-1.0 (for scheduling)
        
        Returns:
            Dict containing:
                - logits: Output logits
                - hidden_states: Final hidden states
                - efficiency_metrics: Compute fraction, speedup, etc.
                - token_states: Exit layers, skip counts, etc.
                - aux_losses: List of auxiliary losses from routers
        """
        if progress is not None:
            self.current_progress = progress
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.base_model.model.embed_tokens.weight.dtype

        # Get embeddings from base model
        hidden = self.base_model.model.embed_tokens(input_ids)
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Create proper 4D causal attention mask for LLaMA
        # Use large negative value instead of -inf to avoid NaN in float16
        # Shape: (batch, 1, seq_len, seq_len)
        min_dtype = torch.finfo(dtype).min
        
        if attention_mask is not None:
            # Convert 2D mask to 4D causal mask
            # First create causal mask with finite large negative value
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), min_dtype, device=device, dtype=dtype),
                diagonal=1
            )
            # Expand for batch (use clone to avoid memory issues)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).clone()
            
            # Apply padding mask - only mask where attention_mask is 0
            padding_mask = attention_mask[:, None, None, :].to(dtype)
            # Where padding_mask is 0, we want to mask out (large negative)
            # Where padding_mask is 1, we want to keep the existing mask
            causal_mask = causal_mask.masked_fill(padding_mask == 0, min_dtype)
            
            attention_mask_4d = causal_mask
        else:
            attention_mask_4d = None
        
        # Initialize token state tracker
        tracker = TokenState(batch_size, seq_len, device)
        
        # Collect auxiliary losses
        aux_losses = []
        all_ffn_masks = []

        # Process each layer
        for layer_idx in range(self.num_layers):
            layer = self.base_model.model.layers[layer_idx]

            # Check if any tokens are still active
            if not tracker.active.any():
                break
            
            # Progressive layer dropout (training only)
            if self._should_drop_layer(layer_idx):
                continue

            # === Exit Gate Decision ===
            if (self.exit_gates[layer_idx] is not None and 
                not self.disable_exit_gates):
                
                exit_probs = self.exit_gates[layer_idx](hidden, training=training)
                exit_mask = (exit_probs > 0.5) & tracker.active
                
                if exit_mask.any():
                    tracker.update_exit(exit_mask, layer_idx)
                    
                    # Early termination if all tokens exited
                    if not tracker.active.any():
                        break

            # === Self-Attention (always computed for active tokens) ===
            attn_input = layer.input_layernorm(hidden)
            
            # Try new transformers API first, fall back to simpler API
            try:
                # New transformers API: requires position_embeddings
                cos, sin = layer.self_attn.rotary_emb(attn_input, position_ids)
                position_embeddings = (cos, sin)
                
                attn_output = layer.self_attn(
                    attn_input,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask_4d,
                )[0]
            except (TypeError, AttributeError):
                # Fallback for mock models or different API
                try:
                    attn_result = layer.self_attn(attn_input, attention_mask=attention_mask_4d)
                    attn_output = attn_result[0] if isinstance(attn_result, tuple) else attn_result
                except TypeError:
                    # Simplest fallback: just hidden states
                    attn_result = layer.self_attn(attn_input)
                    attn_output = attn_result[0] if isinstance(attn_result, tuple) else attn_result
            
            hidden = hidden + attn_output

            # === Router Decision for FFN ===
            if tracker.active.any():
                ffn_mask, aux_loss = self.skip_routers[layer_idx](
                    hidden,
                    tracker.active,
                    training=training,
                    progress=self.current_progress,
                )
                
                if aux_loss is not None:
                    aux_losses.append(aux_loss)
                
                all_ffn_masks.append(ffn_mask)
                
                # Update skip count (tokens that are active but not selected for FFN)
                skip_mask = tracker.active & ~ffn_mask
                tracker.update_skip(skip_mask, layer_idx)

                # === Selective FFN Computation ===
                if ffn_mask.any():
                    ffn_input = layer.post_attention_layernorm(hidden)
                    
                    # Get indices of tokens that need FFN
                    compute_indices = torch.where(ffn_mask)
                    compute_tokens = ffn_input[compute_indices]

                    if compute_tokens.numel() > 0:
                        # Compute FFN for selected tokens
                        ffn_output = layer.mlp(compute_tokens)
                        
                        # Update hidden states (need clone to avoid in-place issues)
                        hidden = hidden.clone()
                        hidden[compute_indices] = hidden[compute_indices] + ffn_output

        # Final normalization and LM head
        hidden = self.base_model.model.norm(hidden)
        logits = self.base_model.lm_head(hidden)

        # Get efficiency metrics
        metrics = tracker.get_efficiency_metrics(total_layers=self.num_layers)

        return {
            'logits': logits,
            'hidden_states': hidden,
            'efficiency_metrics': metrics,
            'token_states': {
                'active': tracker.active,
                'exit_layer': tracker.exit_layer,
                'skip_count': tracker.skip_count,
                'ffn_masks': all_ffn_masks,
                'layer_stats': tracker.layer_stats,
            },
            'aux_losses': aux_losses,
        }

    def get_exit_gate_params(self):
        """Get parameters for exit gates only"""
        params = []
        for gate in self.exit_gates:
            if gate is not None:
                params.extend(gate.parameters())
        return params

    def get_router_params(self):
        """Get parameters for routers only"""
        params = []
        for router in self.skip_routers:
            params.extend(router.parameters())
        return params

    def freeze_routers(self):
        """Freeze router parameters"""
        for router in self.skip_routers:
            for param in router.parameters():
                param.requires_grad = False

    def freeze_exit_gates(self):
        """Freeze exit gate parameters"""
        for gate in self.exit_gates:
            if gate is not None:
                for param in gate.parameters():
                    param.requires_grad = False

    def unfreeze_routers(self):
        """Unfreeze router parameters"""
        for router in self.skip_routers:
            for param in router.parameters():
                param.requires_grad = True

    def unfreeze_exit_gates(self):
        """Unfreeze exit gate parameters"""
        for gate in self.exit_gates:
            if gate is not None:
                for param in gate.parameters():
                    param.requires_grad = True

    def get_routing_analysis(self) -> Dict[str, Any]:
        """Get routing statistics for analysis"""
        router_temps = [r.temperature.item() for r in self.skip_routers]
        router_compliance = [r.get_capacity_compliance() for r in self.skip_routers]
        
        gate_temps = []
        for gate in self.exit_gates:
            if gate is not None:
                gate_temps.append(gate.temp_forward.item())
        
        return {
            'router_temperatures': router_temps,
            'router_capacity_compliance': router_compliance,
            'gate_temperatures': gate_temps,
            'current_capacity': self.capacity,
        }


def create_hierarchical_model(
    base_model: nn.Module,
    config: Dict[str, Any] = None,
) -> HierarchicalTransformerWrapper:
    """
    Factory function to create hierarchical model with optimal config.
    
    Args:
        base_model: Base transformer model
        config: Optional configuration dict
    
    Returns:
        Configured HierarchicalTransformerWrapper
    """
    if config is None:
        from ..config import get_optimal_config
        config = get_optimal_config()
    
    return HierarchicalTransformerWrapper(
        base_model=base_model,
        exit_layers=config.get('exit_layers', [4, 8, 12, 16, 19]),
        capacity=config.get('capacity', 0.55),
        use_layer_dropout=config.get('training', {}).get('use_layer_dropout', False),
        layer_dropout_max=config.get('training', {}).get('layer_dropout_max', 0.2),
    )