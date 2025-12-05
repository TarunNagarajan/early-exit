import torch
import torch.nn as nn
import torch.nn.functional as F 

class TokenState:
    """
    - tracks token lifecycle
    - active, exit layer, skip counts
    """
    def __init__(self, batch_size, seq_len, device):
        self.active = torch.ones(batch_size, seq_len, dtype = torch.bool, device = device)
        self.exits = torch.full((batch_size, seq_len), -1, dtype = torch.long, device = device)
        self.skips = torch.zeros(batch_size, seq_len, dtype = torch.long, device = device)
        self.layer_stats = []

    def update_exit(self, exit_mask, layer_index):
        exited = exit_mask & self.active
        if exited.any():
            self.exits[exited] = layer_index
            self.active[exited] = False

        return exited

    def update_skip(self, skip_mask, layer_index):
        skipped = skip_mask & self.active
        if skipped.any():
            self.skips += 1

        self.layer_stats.append({
            'layer': layer_index,
            'active_fract': self.active.float().mean().item(),
            'ffn_fract': (~skip_mask).float().mean().item() if skip_mask.numel() > 0 else 0.0
        })

        return skipped

    def get_efficiency_metrics(self, total_layers = 32):
        batch_size, seq_len = self.active.shape
        exit_mask = self.exits >= 0 
        layers_used_exited = 0 
        
        if exit_mask.any():
            layers_used_exited = (self.exits[exit_mask] + 1).sum().item()

        not_exited_mask = ~exited_mask
        layers_used_not_exited = 0

        if not_exited_mask.any():
            layers_used_not_exited = (total_layers - self.skip_count[not_exited_mask]).sum().item()

        total_layers_used = layers_used_exited + layers_used_not_exited
        thoretical_max_layers = total_layers_used * batch_size * seq_len

        compute_fract = total_layers_used / thoretical_max_layers if thoretical_max_layers > 0 else 0 
        speedup = 1.0 / compute_fract if compute_fract > 0 else 1.0 

        return {
            'compute_fract': compute_fract,
            'speedup': speedup,
            'exit_rate': exit_mask.float().mean().item(),
            'avg_skip_rate': self.skips.float().mean().item() / total_layers if total_layers > 0 else 0.0
        }
        
