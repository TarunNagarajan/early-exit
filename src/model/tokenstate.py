import torch
import torch.nn as nn

class TokenState:
    """Track token lifecycle: active, exit layer, skip counts"""
    
    def __init__(self, batch_size, seq_len, device):
        self.active = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        self.exit_layer = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)
        self.skip_count = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        self.layer_stats = []
    
    def update_exit(self, exit_mask, layer_idx):
        newly_exited = exit_mask & self.active
        if newly_exited.any():
            self.exit_layer[newly_exited] = layer_idx
            self.active[newly_exited] = False
        return newly_exited
    
    def update_skip(self, skip_mask, layer_idx):
        newly_skipped = skip_mask & self.active
        if newly_skipped.any():
            self.skip_count[newly_skipped] += 1
        
        self.layer_stats.append({
            'layer': layer_idx,
            'active_fraction': self.active.float().mean().item(),
            'ffn_fraction': (~skip_mask).float().mean().item() if skip_mask.numel() > 0 else 0.0
        })
        
        return newly_skipped
    
    def get_efficiency_metrics(self, total_layers=22):
        batch_size, seq_len = self.active.shape
        
        exited_mask = self.exit_layer >= 0
        layers_used_exited = 0
        if exited_mask.any():
            layers_used_exited = (self.exit_layer[exited_mask] + 1).sum().item()
        
        not_exited_mask = ~exited_mask
        layers_used_not_exited = 0
        if not_exited_mask.any():
            layers_used_not_exited = (total_layers - self.skip_count[not_exited_mask]).sum().item()
        
        total_layers_used = layers_used_exited + layers_used_not_exited
        total_possible_layers = batch_size * seq_len * total_layers
        
        compute_fraction = total_layers_used / total_possible_layers if total_possible_layers > 0 else 0
        speedup = 1.0 / compute_fraction if compute_fraction > 0 else 1.0
        
        return {
            'compute_fraction': compute_fraction,
            'speedup': speedup,
            'exit_rate': exited_mask.float().mean().item(),
            'avg_skip_rate': self.skip_count.float().mean().item() / total_layers if total_layers > 0 else 0.0
        }
    
    def get_token_trajectories(self):
        return {
            'active_final': self.active.cpu(),
            'exit_layer': self.exit_layer.cpu(),
            'skip_count': self.skip_count.cpu(),
            'layer_stats': self.layer_stats
        }