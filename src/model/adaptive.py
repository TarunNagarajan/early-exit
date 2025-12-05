import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from .gate import ExitGate
from .router import MoERouter
from ..token_state_tracker import TokenStateTracker

class HierarchicalTransformerWrapper(nn.Module):
    """
    - wrapper for exit gates + routers
    """

    def __init__(self,
                 base_model: nn.Module,
                 exit_layers: List[int] = [5, 10, 15, 18],
                 capacity: float = 0.7,
                 disable_exit_gates: bool = False):
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = base_model.config.hidden_size
        self.num_layers = base_model.config.num_hidden_layers
        self.capacity = capacity
        self.disable_exit_gates = disable_exit_gates
        self.exit_layers = exit_layers

        for param in base_model.parameters():
            param.requires_grad = False

        self.exit_gates = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            if layer_idx in exit_layers and not disable_exit_gates:
                self.exit_gates.append(ExitGate(self.hidden_dim))
            else:
                self.exit_gates.append(None)

        self.skip_routers = nn.ModuleList([
            MoERouter(self.hidden_dim, capacity=capacity)
            for _ in range(self.num_layers)
        ])

    def count_trainable_params(self):
        total = 0
        for gate in self.exit_gates:
            if gate is not None:
                total += sum(p.numel() for p in gate.parameters())
        for router in self.skip_routers:
            total += sum(p.numel() for p in router.parameters())
        return total

    def forward(self, input_ids, attention_mask=None, training=True, global_step=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        hidden = self.base_model.model.embed_tokens(input_ids)
        tracker = TokenStateTracker(batch_size, seq_len, device)
        
        aux_losses = []
        all_ffn_masks = []

        for layer_idx in range(self.num_layers):
            layer = self.base_model.model.layers[layer_idx]

            if not tracker.active.any():
                break

            if self.exit_gates[layer_idx] is not None and not self.disable_exit_gates:
                exit_probs = self.exit_gates[layer_idx](hidden, training=training)
                exit_mask = (exit_probs > 0.5) & tracker.active
                if exit_mask.any():
                    tracker.update_exit(exit_mask, layer_idx)

            attn_input = layer.input_layernorm(hidden)
            attn_output = layer.self_attn(attn_input, attention_mask=attention_mask)[0]
            hidden = hidden + attn_output

            if tracker.active.any():
                ffn_mask, aux_loss = self.skip_routers[layer_idx](hidden, tracker.active, training=training, global_step=global_step)
                if aux_loss is not None:
                    aux_losses.append(aux_loss)
                all_ffn_masks.append(ffn_mask)
                tracker.update_skip(~ffn_mask, layer_idx)

                if ffn_mask.any():
                    ffn_input = layer.post_attention_layernorm(hidden)
                    compute_indices = torch.where(ffn_mask)
                    compute_tokens = ffn_input[compute_indices]

                    if compute_tokens.numel() > 0:
                        ffn_output = layer.mlp(compute_tokens)
                        hidden = hidden.clone()
                        hidden[compute_indices] = hidden[compute_indices] + ffn_output

        hidden = self.base_model.model.norm(hidden)
        logits = self.base_model.lm_head(hidden)

        metrics = tracker.get_efficiency_metrics(total_layers=self.num_layers)

        return {
            'logits': logits,
            'hidden_states': hidden,
            'efficiency_metrics': metrics,
            'token_states': {
                'active': tracker.active,
                'exit_layer': tracker.exit_layer,
                'skip_count': tracker.skip_count,
                'ffn_mask': all_ffn_masks
            },
            'aux_losses': aux_losses
        }

    def set_capacity(self, new_capacity):
        self.capacity = new_capacity
        for router in self.skip_routers:
            router.adjust_capacity(new_capacity)