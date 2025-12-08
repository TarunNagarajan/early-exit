"""
OPTIMIZED MOE ROUTER WITH RESEARCH-BACKED IMPROVEMENTS

Features:
- Router z-loss (ST-MoE) for training stability
- Entropy regularization for exploration
- Cosine temperature annealing
- Improved load balancing loss formulation
- Full capacity range [0.1, 0.9]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MoERouter(nn.Module):
    """
    Optimized MoE-style router for selective FFN computation.
    
    Research-backed improvements:
    1. Router z-loss (ST-MoE): Prevents extreme logits for stability
    2. Entropy regularization: Encourages exploration
    3. Cosine temperature annealing: Smoother than exponential
    4. Improved load balancing: Better formulation
    """

    def __init__(
        self,
        hidden_dim: int,
        capacity: float = 0.55,
        initial_temp: float = 1.0,
        min_temp: float = 0.1,
        lb_weight: float = 0.01,
        z_loss_weight: float = 1e-4,
        entropy_weight: float = 0.001,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.capacity = capacity
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        
        # Loss weights
        self.lb_weight = lb_weight
        self.z_loss_weight = z_loss_weight
        self.entropy_weight = entropy_weight
        
        # Single linear router (minimalist design)
        self.router = nn.Linear(hidden_dim, 1, bias=True)
        
        # Temperature buffer
        self.register_buffer('temperature', torch.tensor(initial_temp))
        self.register_buffer('current_step', torch.tensor(0))
        
        # Statistics tracking
        self.register_buffer('usage_history', torch.zeros(100))
        self.history_idx = 0
        
        # Better initialization (small weights for stability)
        nn.init.xavier_uniform_(self.router.weight, gain=0.01)
        nn.init.zeros_(self.router.bias)

    def update_temperature(self, progress: float) -> float:
        """
        Cosine annealing schedule (smoother than exponential).
        
        Args:
            progress: Training progress from 0.0 to 1.0
        
        Returns:
            Current temperature value
        """
        # Cosine decay from initial to min
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        new_temp = self.min_temp + (self.initial_temp - self.min_temp) * cosine_decay
        self.temperature = torch.tensor(new_temp, device=self.router.weight.device)
        return new_temp

    def compute_load_balance_loss(
        self,
        router_probs: torch.Tensor,
        selected_mask: torch.Tensor,
        num_active: int
    ) -> torch.Tensor:
        """
        Differentiable load balancing loss.
        
        Uses soft router_probs (sigmoid of scores) to maintain gradient flow,
        rather than the hard selected_mask which has no gradients.
        """
        if num_active == 0 or router_probs.numel() == 0:
            # Return zero that still has gradient capability
            return router_probs.new_zeros((), requires_grad=True)
        
        # Use mean of router probs as differentiable proxy for actual fraction
        # This maintains gradient flow through the router parameters
        soft_fraction = router_probs.mean()
        target_fraction = torch.tensor(self.capacity, device=router_probs.device, dtype=router_probs.dtype)
        
        # MSE loss for capacity compliance (now differentiable!)
        lb_loss = F.mse_loss(soft_fraction, target_fraction)
        
        return lb_loss

    def compute_router_z_loss(
        self,
        scores: torch.Tensor,
        active_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Router z-loss (ST-MoE) for training stability.
        
        Penalizes large logits to prevent extreme routing decisions.
        """
        active_scores = scores[active_mask]
        if active_scores.numel() == 0:
            # Return zero that still has gradient capability
            return scores.new_zeros((), requires_grad=True)
        
        # Penalize squared logits
        z_loss = (active_scores ** 2).mean()
        return z_loss

    def compute_entropy_loss(
        self,
        scores: torch.Tensor,
        active_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Entropy regularization to encourage exploration.
        
        Higher entropy = more exploration (less deterministic routing).
        """
        active_scores = scores[active_mask]
        if active_scores.numel() == 0:
            # Return zero that still has gradient capability
            return scores.new_zeros((), requires_grad=True)
        
        # Convert to probabilities and clamp for numerical stability
        probs = torch.sigmoid(active_scores).clamp(1e-4, 1 - 1e-4)
        
        # Binary entropy - use larger epsilon for float16 (1e-10 underflows to 0)
        entropy = -(
            probs * torch.log(probs) +
            (1 - probs) * torch.log(1 - probs)
        ).mean()
        
        # Negative because we maximize entropy (minimize negative entropy)
        return -entropy

    def forward(
        self,
        hidden_states: torch.Tensor,
        active_mask: torch.Tensor,
        training: bool = True,
        progress: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute FFN selection mask.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            active_mask: [batch, seq_len] - tokens still being processed
            training: Whether in training mode
            progress: Training progress 0.0-1.0 for temperature annealing
        
        Returns:
            ffn_mask: [batch, seq_len] - boolean mask for FFN computation
            aux_loss: Combined auxiliary loss (or None in inference)
        """
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        original_dtype = hidden_states.dtype
        
        # CRITICAL: Convert to float32 for ALL computations to prevent NaN
        # Float16 has limited range and can easily overflow in gradient computation
        hidden_states_fp32 = hidden_states.float()
        
        # Update temperature during training
        if training and progress is not None:
            self.update_temperature(progress)
        
        # Compute routing scores in float32
        # Cast router weights to float32 for this computation
        with torch.amp.autocast(device_type='cuda', enabled=False):
            scores = self.router(hidden_states_fp32).squeeze(-1)  # [batch, seq_len], float32
        
        # Mask inactive tokens with large negative value
        scores = torch.where(
            active_mask,
            scores,
            torch.tensor(-1e9, device=device, dtype=scores.dtype)
        )

        num_active = active_mask.sum().item()
        
        # Handle edge case: no active tokens
        if num_active == 0:
            return torch.zeros_like(active_mask, dtype=torch.bool), None

        # Calculate k (number of tokens to select for FFN)
        k = max(1, min(int(self.capacity * num_active + 0.5), num_active))

        aux_loss = None

        if training:
            # Add Gumbel noise for differentiable sampling
            # Clamp uniform to avoid extreme values that cause NaN in float16
            uniform = torch.rand_like(scores).clamp(1e-6, 1 - 1e-6)
            # Compute Gumbel noise with clamping to prevent float16 overflow
            gumbel = -torch.log(-torch.log(uniform) + 1e-6)
            gumbel = gumbel.clamp(-10, 10)  # Prevent extreme values
            noisy_scores = scores + gumbel * self.temperature

            # Top-k selection
            flat_scores = noisy_scores.view(-1)
            _, indices = torch.topk(flat_scores, k=k)

            ffn_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
            ffn_mask[indices] = True
            ffn_mask = ffn_mask.view(batch_size, seq_len) & active_mask

            # Compute auxiliary losses - scores is already float32
            router_probs = torch.sigmoid(scores[active_mask])  # Already float32
            
            lb_loss = self.compute_load_balance_loss(router_probs, ffn_mask, num_active)
            z_loss = self.compute_router_z_loss(scores, active_mask)
            entropy_loss = self.compute_entropy_loss(scores, active_mask)

            # Debug: Check if any losses are NaN/Inf BEFORE nan_to_num
            lb_nan = torch.isnan(lb_loss) or torch.isinf(lb_loss)
            z_nan = torch.isnan(z_loss) or torch.isinf(z_loss)
            ent_nan = torch.isnan(entropy_loss) or torch.isinf(entropy_loss)
            
            if lb_nan or z_nan or ent_nan:
                # Increment counter (stored as buffer for debugging)
                if not hasattr(self, '_nan_count'):
                    self._nan_count = 0
                self._nan_count += 1
                if self._nan_count <= 3:  # Only print first 3
                    print(f"ROUTER NaN DETECTED: lb={lb_loss.item() if not lb_nan else 'NaN'}, "
                          f"z={z_loss.item() if not z_nan else 'NaN'}, "
                          f"ent={entropy_loss.item() if not ent_nan else 'NaN'}", flush=True)

            # Guard against NaN in any component - use nan_to_num to preserve gradients
            lb_loss = torch.nan_to_num(lb_loss.float(), nan=0.0, posinf=0.0, neginf=0.0)
            z_loss = torch.nan_to_num(z_loss.float(), nan=0.0, posinf=0.0, neginf=0.0)
            entropy_loss = torch.nan_to_num(entropy_loss.float(), nan=0.0, posinf=0.0, neginf=0.0)

            # Combined auxiliary loss
            aux_loss = (
                self.lb_weight * lb_loss +
                self.z_loss_weight * z_loss +
                self.entropy_weight * entropy_loss
            )

            # Track usage statistics
            actual_usage = ffn_mask.float().sum().item() / max(num_active, 1)
            self.usage_history[self.history_idx % 100] = actual_usage
            self.history_idx += 1

        else:
            # Inference: deterministic top-k
            flat_scores = scores.view(-1)
            _, indices = torch.topk(flat_scores, k=k)

            ffn_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
            ffn_mask[indices] = True
            ffn_mask = ffn_mask.view(batch_size, seq_len) & active_mask

        return ffn_mask, aux_loss

    def get_capacity_compliance(self) -> float:
        """Check how close actual capacity is to target"""
        if self.history_idx == 0:
            return 0.0
        
        valid_count = min(self.history_idx, 100)
        avg_usage = self.usage_history[:valid_count].mean().item()
        return abs(avg_usage - self.capacity)

    def adjust_capacity(self, new_capacity: float) -> None:
        """Adjust capacity with full range support"""
        self.capacity = max(0.1, min(0.9, new_capacity))

    def reset_stats(self) -> None:
        """Reset usage statistics"""
        self.usage_history.zero_()
        self.history_idx = 0

    def get_routing_entropy(self, hidden_states: torch.Tensor) -> float:
        """Compute routing entropy for analysis"""
        with torch.no_grad():
            scores = self.router(hidden_states).squeeze(-1)
            probs = torch.sigmoid(scores).clamp(1e-4, 1 - 1e-4)
            entropy = -(
                probs * torch.log(probs) +
                (1 - probs) * torch.log(1 - probs)
            ).mean()
            return entropy.item()


class MultiHeadRouter(nn.Module):
    """
    Multi-head router for more expressive routing decisions.
    Each head independently routes, and decisions are aggregated.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        capacity: float = 0.55,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            MoERouter(hidden_dim, capacity=capacity, **kwargs)
            for _ in range(num_heads)
        ])
        
        # Learnable head weights
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
    
    def forward(self, hidden_states, active_mask, training=True, progress=None):
        # Get scores from all heads
        all_scores = []
        all_aux = []
        
        for head in self.heads:
            scores = head.router(hidden_states).squeeze(-1)
            all_scores.append(scores)
        
        # Weighted average of scores
        weights = F.softmax(self.head_weights, dim=0)
        combined_scores = sum(w * s for w, s in zip(weights, all_scores))
        
        # Use first head's logic for mask computation
        # (could be optimized but keeps interface simple)
        temp_router = self.heads[0]
        temp_router_scores = combined_scores
        
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        
        temp_router_scores = torch.where(
            active_mask, temp_router_scores,
            torch.tensor(-1e9, device=device, dtype=temp_router_scores.dtype)
        )
        
        num_active = active_mask.sum().item()
        if num_active == 0:
            return torch.zeros_like(active_mask, dtype=torch.bool), None
        
        k = max(1, min(int(temp_router.capacity * num_active + 0.5), num_active))
        
        if training:
            uniform = torch.rand_like(temp_router_scores).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform) + 1e-6).clamp(-10, 10)
            noisy = temp_router_scores + gumbel * temp_router.temperature
            
            _, indices = torch.topk(noisy.view(-1), k=k)
            ffn_mask = torch.zeros(batch_size * seq_len, dtype=torch.bool, device=device)
            ffn_mask[indices] = True
            ffn_mask = ffn_mask.view(batch_size, seq_len) & active_mask
            
            # Compute combined aux loss
            aux_loss = sum(
                head.compute_load_balance_loss(
                    torch.sigmoid(combined_scores[active_mask]),
                    ffn_mask, num_active
                ) for head in self.heads
            ) / self.num_heads
            
            return ffn_mask, aux_loss * self.heads[0].lb_weight
        else:
            _, indices = torch.topk(temp_router_scores.view(-1), k=k)
            ffn_mask = torch.zeros(batch_size * seq_len, dtype=torch.bool, device=device)
            ffn_mask[indices] = True
            return ffn_mask.view(batch_size, seq_len) & active_mask, None
    
    def adjust_capacity(self, new_capacity):
        for head in self.heads:
            head.adjust_capacity(new_capacity)