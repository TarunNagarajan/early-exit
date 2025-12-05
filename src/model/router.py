import torch
import torch.nn as nn
import torch.nn.functional as F 

class MoERouter(nn.Module):
    """
    - MoE-style router for FFN skipping
    - top-k selection with exact capacity control
    """

    def __init__(self, hidden_dim, capacity = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.capacity = capacity
        self.router = nn.Linear(hidden_dim, 1)

        nn.init.normal_(self.router.weight, mean = 0.0, std = 0.02)
        nn.init.zeros_(self.router.bias)

        self.current_step = 0 
        self.register_buffer('usage_stats', torch.zeros(100))

    def forward(self, hidden_states, active_mask, training = True):
        batch_size, seq_len = hidden_states.shape[:2]
        
        scores = self.router(hidden_states).squeeze(-1)
        scores = torch.where(active_mask, scores, torch.tensor(-float('inf'), device = scores.device))

        num_active = active_mask.sum().item()
        if num_active == 0:
            return torch.zeros_like(active_mask, dtype = torch.bool)

        k = int(self.capacity * num_active)
        k = max(1, min(k, num_active))

        if training:
            # FIX: add temperature to gumbel noise
            temperature = 0.5
            gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
            noisy = scores + temperature * gumbel

            flat_scores = noisy.view(-1)
            topy, indices = torch.topk(flat_scores, k = k)
            
            ffn_mask = torch.zeros_like(flat_scores, dtype = torch.bool)
            ffn_mask[indices] = True
            ffn_mask = ffn_mask.view(batch_size, seq_len)
        
        else:
            flat_scores = scores.view(-1)
            topy, indices = torch.topk(flat_scores, k = k)

            ffn_mask = torch.zeros_like(flat_scores, dtype = torch.bool)
            ffn_mask[indices] = True
            ffn_mask = ffn_mask.view(batch_size, seq_len)

        ffn_mask = ffn_mask & active_mask

        if training and self.current_step < 100:
            actual_capacity = ffn_mask.float().sum().item() / num_active if num_active > 0 else 0 
            self.usage_stats[self.current_step % 100] = actual_capacity
            self.current_step += 1

        return ffn_mask
            
    # []: get_capacity_compliance(self), adjust_capacity(self, updated_capacity)

    def get_capacity_compliance(self):
        if self.current_step == 0:
            return 0.0
        
        valid = min(self.current_step, 100)
        recent_stats = self.usage_stats[:valid]
        avg_capacity = recent_stats.mean().item()

        return abs(avg_capacity - self.capacity)

    def adjust_capacity(self, updated_capacity):
        self.capacity = max(0.1, min(0.9, updated_capacity))

    def reset_stats(self):
        self.usage_stats.zero_()
        self.current_step = 0


