"""
OPTIMIZED TOKEN STATE TRACKER

Fixed efficiency calculation that properly accounts for:
- FFN skips before early exit
- Accurate compute fraction estimation
- Additional metrics for analysis
"""

import torch
from typing import Dict, List, Any


class TokenState:
    """
    Track token lifecycle through the hierarchical transformer.
    
    Tracks:
    - active: Which tokens are still being processed
    - exit_layer: Layer where each token exited (-1 = didn't exit)
    - skip_count: Number of FFN computations skipped
    - layer_stats: Per-layer statistics for analysis
    """
    
    def __init__(self, batch_size: int, seq_len: int, device: torch.device):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        
        # Token state tensors
        self.active = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        self.exit_layer = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)
        self.skip_count = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Per-layer statistics
        self.layer_stats: List[Dict[str, Any]] = []
        
        # Track FFN decisions for each token
        self.ffn_decisions: List[torch.Tensor] = []
    
    def update_exit(self, exit_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Update state when tokens exit.
        
        Args:
            exit_mask: [batch, seq_len] - tokens that want to exit
            layer_idx: Current layer index (0-indexed)
        
        Returns:
            newly_exited: Tokens that actually exited (were active and chose to exit)
        """
        # Only active tokens can exit
        newly_exited = exit_mask & self.active
        
        if newly_exited.any():
            self.exit_layer[newly_exited] = layer_idx
            self.active[newly_exited] = False
        
        return newly_exited
    
    def update_skip(self, skip_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Update state when tokens skip FFN.
        
        Args:
            skip_mask: [batch, seq_len] - tokens that skipped FFN
            layer_idx: Current layer index
        
        Returns:
            actually_skipped: Tokens that were active and skipped
        """
        # Only count skips for active tokens
        actually_skipped = skip_mask & self.active
        
        if actually_skipped.any():
            self.skip_count[actually_skipped] += 1
        
        # Track layer statistics
        num_active = self.active.sum().item()
        num_skipped = actually_skipped.sum().item()
        
        self.layer_stats.append({
            'layer': layer_idx,
            'active_count': num_active,
            'active_fraction': num_active / (self.batch_size * self.seq_len),
            'skip_count': num_skipped,
            'skip_fraction': num_skipped / max(num_active, 1),
            'ffn_count': num_active - num_skipped,
        })
        
        return actually_skipped
    
    def get_efficiency_metrics(self, total_layers: int = 22) -> Dict[str, float]:
        """
        Compute efficiency metrics with FIXED calculation.
        
        The compute fraction accounts for:
        - Attention is always computed for active tokens
        - FFN is only computed when not skipped
        - For exited tokens, only count layers up to exit point
        
        Assuming attention ≈ FFN in compute cost:
        - Full layer = attention + FFN = 2 units
        - Skipped layer = attention only = 1 unit
        - Compute fraction = actual_units / max_possible_units
        """
        total_tokens = self.batch_size * self.seq_len
        
        # Separate exited and non-exited tokens
        exited_mask = self.exit_layer >= 0
        
        # For exited tokens
        total_compute_exited = 0.0
        if exited_mask.any():
            exit_depths = self.exit_layer[exited_mask].float() + 1  # 1-indexed
            skips = self.skip_count[exited_mask].float()
            
            # Each layer has attention (always) + FFN (if not skipped)
            # Compute units = exit_depth * 2 - skips (each skip saves 1 unit)
            compute_units = exit_depths * 2 - skips
            total_compute_exited = compute_units.sum().item()
        
        # For non-exited tokens (went through all layers)
        total_compute_not_exited = 0.0
        not_exited_mask = ~exited_mask
        if not_exited_mask.any():
            skips = self.skip_count[not_exited_mask].float()
            
            # Full path: total_layers * 2 - skips
            compute_units = total_layers * 2 - skips
            total_compute_not_exited = compute_units.sum().item()
        
        # Total compute
        total_compute = total_compute_exited + total_compute_not_exited
        max_compute = total_tokens * total_layers * 2  # All tokens, all layers, attn+ffn
        
        compute_fraction = total_compute / max_compute if max_compute > 0 else 0
        speedup = 1.0 / compute_fraction if compute_fraction > 0 else 1.0
        
        # Additional metrics
        avg_exit_depth = -1.0
        if exited_mask.any():
            avg_exit_depth = self.exit_layer[exited_mask].float().mean().item()
        
        return {
            'compute_fraction': compute_fraction,
            'speedup': speedup,
            'exit_rate': exited_mask.float().mean().item(),
            'avg_exit_depth': avg_exit_depth,
            'avg_skip_count': self.skip_count.float().mean().item(),
            'avg_skip_rate': self.skip_count.float().mean().item() / total_layers,
            'tokens_active_final': self.active.sum().item(),
            'tokens_exited': exited_mask.sum().item(),
        }
    
    def get_token_trajectories(self) -> Dict[str, Any]:
        """Get detailed token trajectories for analysis"""
        return {
            'active_final': self.active.cpu(),
            'exit_layer': self.exit_layer.cpu(),
            'skip_count': self.skip_count.cpu(),
            'layer_stats': self.layer_stats,
        }
    
    def get_layer_summary(self) -> Dict[str, List[float]]:
        """Get per-layer summary statistics"""
        if not self.layer_stats:
            return {}
        
        return {
            'active_fraction': [s['active_fraction'] for s in self.layer_stats],
            'skip_fraction': [s['skip_fraction'] for s in self.layer_stats],
            'ffn_count': [s['ffn_count'] for s in self.layer_stats],
        }
    
    def get_exit_distribution(self, total_layers: int = 22) -> Dict[int, int]:
        """Get distribution of exit layers"""
        distribution = {}
        
        for layer_idx in range(-1, total_layers):
            count = (self.exit_layer == layer_idx).sum().item()
            if count > 0:
                distribution[layer_idx] = count
        
        return distribution
    
    def __repr__(self) -> str:
        metrics = self.get_efficiency_metrics()
        return (
            f"TokenState(batch={self.batch_size}, seq={self.seq_len}, "
            f"exit_rate={metrics['exit_rate']:.2%}, "
            f"speedup={metrics['speedup']:.2f}x)"
        )


class TokenStatePool:
    """
    Pool of TokenState objects for memory efficiency.
    Reuses TokenState objects across forward passes.
    """
    
    def __init__(self, max_pool_size: int = 10):
        self.pool: List[TokenState] = []
        self.max_pool_size = max_pool_size
    
    def get(self, batch_size: int, seq_len: int, device: torch.device) -> TokenState:
        """Get a TokenState from pool or create new one"""
        # Try to find matching size in pool
        for i, state in enumerate(self.pool):
            if state.batch_size == batch_size and state.seq_len == seq_len:
                return self.pool.pop(i)
        
        # Create new if not found
        return TokenState(batch_size, seq_len, device)
    
    def release(self, state: TokenState) -> None:
        """Return TokenState to pool"""
        if len(self.pool) < self.max_pool_size:
            # Reset state for reuse
            state.active.fill_(True)
            state.exit_layer.fill_(-1)
            state.skip_count.zero_()
            state.layer_stats.clear()
            self.pool.append(state)