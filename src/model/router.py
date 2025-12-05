import torch
import torch.nn as nn
import torch.nn.functional as F 

class MoERouter(nn.Module):
    """
    SLM-OPTIMIZED Router for TinyLlama 1.1B
    - Lightweight design
    - Temperature annealing
    - Load balancing loss
    - Higher default capacity (0.7)
    """

    def __init__(self, hidden_dim, capacity=0.7, initial_temp=0.5, min_temp=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.capacity = capacity
        self.router = nn.Linear(hidden_dim, 1)
        
        # Temperature annealing (starts lower for SLMs)
        self.register_buffer('temperature', torch.tensor(initial_temp))
        self.register_buffer('step_count', torch.tensor(0))
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.anneal_rate = 5e-5  # Faster annealing for fewer steps
        
        # Load balancing
        self.lb_weight = 0.005  # Lower weight for SLMs
        self.z_loss_weight = 5e-6
        
        # Statistics
        self.register_buffer('usage_stats', torch.zeros(100))
        self.current_step = 0
        
        # Initialization
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router.bias)

    def update_temperature(self, global_step):
        """Exponential temperature annealing"""
        annealed = self.initial_temp * torch.exp(-self.anneal_rate * global_step)
        self.temperature = torch.maximum(
            torch.tensor(self.min_temp, device=self.temperature.device),
            annealed
        )
        self.step_count += 1
        return self.temperature.item()

    def compute_load_balance_loss(self, router_probs, selected_mask, num_active):
        """Load balancing loss to encourage uniform routing"""
        if num_active == 0:
            return torch.tensor(0.0, device=router_probs.device)
        
        # Fraction of active tokens that were selected
        f = selected_mask.float().sum() / num_active
        
        # Average router probability for active tokens
        p = router_probs.mean()
        
        # Penalize deviation from target capacity
        target = torch.tensor(self.capacity, device=router_probs.device)
        lb_loss = (f - target).pow(2) + (p - target).pow(2)
        
        return lb_loss

    def compute_router_z_loss(self, logits, active_mask):
        """Router z-loss for training stability"""
        # Only compute on active tokens
        active_logits = logits[active_mask]
        if active_logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        z_loss = active_logits.pow(2).mean()
        return z_loss

    def forward(self, hidden_states, active_mask, training=True, global_step=None):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Update temperature during training
        if training and global_step is not None:
            self.update_temperature(global_step)
        
        # Compute routing scores
        scores = self.router(hidden_states).squeeze(-1)
        scores = torch.where(
            active_mask, 
            scores, 
            torch.tensor(-float('inf'), device=scores.device)
        )

        num_active = active_mask.sum().item()
        if num_active == 0:
            return torch.zeros_like(active_mask, dtype=torch.bool), None

        # Calculate k based on capacity
        k = max(1, min(int(self.capacity * num_active), num_active))

        aux_loss = None
        
        if training:
            # Gumbel-Softmax with temperature
            gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
            noisy = (scores + gumbel) / self.temperature
            
            # Top-k selection
            flat_scores = noisy.view(-1)
            topk_vals, indices = torch.topk(flat_scores, k=k)
            
            ffn_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
            ffn_mask[indices] = True
            ffn_mask = ffn_mask.view(batch_size, seq_len) & active_mask
            
            # Compute auxiliary losses
            router_probs = torch.sigmoid(scores[active_mask])
            lb_loss = self.compute_load_balance_loss(router_probs, ffn_mask, num_active)
            z_loss = self.compute_router_z_loss(scores, active_mask)
            
            # Combined auxiliary loss
            aux_loss = self.lb_weight * lb_loss + self.z_loss_weight * z_loss
            
            # Track usage statistics
            if self.current_step < 100:
                actual_capacity = ffn_mask.float().sum().item() / num_active
                self.usage_stats[self.current_step % 100] = actual_capacity
                self.current_step += 1
        
        else:
            # Inference: deterministic top-k
            flat_scores = scores.view(-1)
            topk_vals, indices = torch.topk(flat_scores, k=k)
            
            ffn_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
            ffn_mask[indices] = True
            ffn_mask = ffn_mask.view(batch_size, seq_len) & active_mask

        return ffn_mask, aux_loss
    
    def get_capacity_compliance(self):
        """Check how close actual capacity is to target"""
        if self.current_step == 0:
            return 0.0
        valid = min(self.current_step, 100)
        return abs(self.usage_stats[:valid].mean().item() - self.capacity)
    
    def adjust_capacity(self, new_capacity):
        """Adjust capacity during inference"""
        self.capacity = max(0.5, min(0.9, new_capacity))
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.usage_stats.zero_()
        self.current_step = 0