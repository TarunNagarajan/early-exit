"""
OPTIMIZED CONFIGURATION FOR HIERARCHICAL ADAPTIVE TRANSFORMER
Research-backed settings prioritizing speedup with acceptable quality trade-off

Based on:
- LayerSkip (2024): Progressive layer dropout, optimal exit placement
- ST-MoE: Router z-loss for stability
- Decoupled ST-GS: Separate forward/backward temperatures
"""

import math

# ============================================================================
# TINYLLAMA ARCHITECTURE (VERIFIED)
# ============================================================================

TINYLLAMA_NUM_LAYERS = 22
TINYLLAMA_HIDDEN_DIM = 2048
TINYLLAMA_ATTENTION_HEADS = 32
TINYLLAMA_KV_HEADS = 4  # Grouped-Query Attention

# ============================================================================
# EXIT GATE CONFIGURATION (Speedup-Focused)
# ============================================================================

# 5 exit points for finer control, earlier first exit for common tokens
OPTIMAL_EXIT_LAYERS = [4, 8, 12, 16, 19]

# Exit gate architecture
EXIT_GATE_CONFIG = {
    'hidden_sizes': [64, 32],      # Deeper network for better decisions
    'dropout': 0.1,
    'initial_temp_forward': 0.5,   # Decoupled temperatures
    'initial_temp_backward': 1.0,
    'min_temp': 0.1,
    'use_learnable_temp': True,
}

# ============================================================================
# ROUTER CONFIGURATION (Aggressive for Speedup)
# ============================================================================

# Lower capacity = more skipping = higher speedup
OPTIMAL_CAPACITY = 0.55  # 55% FFN usage (aggressive)

ROUTER_CONFIG = {
    'capacity': OPTIMAL_CAPACITY,
    'initial_temp': 1.0,
    'min_temp': 0.1,
    'temp_schedule': 'cosine',     # Smoother than exponential
    
    # Auxiliary losses (research-calibrated)
    'lb_loss_weight': 0.01,        # Load balancing
    'z_loss_weight': 1e-4,         # Router z-loss (ST-MoE)
    'entropy_weight': 0.001,       # Entropy regularization
    
    # Capacity range
    'min_capacity': 0.1,
    'max_capacity': 0.9,
}

# ============================================================================
# TRAINING CONFIGURATION (Optimized for Speed + Quality)
# ============================================================================

TRAINING_CONFIG = {
    # Phase 1: Router training (critical for learning skip patterns)
    'router_epochs': 3,
    'router_lr': 1e-3,
    'router_warmup_ratio': 0.1,
    'router_weight_decay': 0.01,
    
    # Phase 2: Exit gate training
    'exit_epochs': 2,
    'exit_lr': 5e-4,
    'exit_warmup_ratio': 0.05,
    'exit_weight_decay': 0.01,
    
    # Batch configuration
    'batch_size': 4,
    'gradient_accumulation': 4,   # Effective batch = 16
    'max_seq_length': 512,
    
    # Regularization
    'gradient_clip': 1.0,
    'label_smoothing': 0.1,
    
    # Exit timing loss weights
    'exit_early_weight': 0.03,     # Encourage early exit
    'exit_diversity_weight': 0.01, # Prevent clustering
    'exit_monotonicity_weight': 0.005,
    
    # Progressive training
    'use_layer_dropout': True,
    'layer_dropout_max': 0.2,      # Max 20% dropout for later layers
    
    # Capacity scheduling
    'use_capacity_schedule': True,
    'capacity_start': 0.9,
    'capacity_end': 0.55,
    'capacity_warmup_ratio': 0.3,
}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    'name': 'wikitext',
    'version': 'wikitext-2-raw-v1',
    'train_samples': None,  # Use all
    'eval_samples': 1000,
    'test_samples': 1000,
}

# ============================================================================
# MEMORY OPTIMIZATION
# ============================================================================

MEMORY_CONFIG = {
    'use_4bit_quantization': True,
    'use_gradient_checkpointing': True,
    'empty_cache_frequency': 10,
    'use_flash_attention': True,
}

# ============================================================================
# EXPECTED PERFORMANCE (Speedup-Focused)
# ============================================================================

EXPECTED_PERFORMANCE = {
    'target_speedup': 1.7,           # Ambitious target
    'min_speedup': 1.5,              # Minimum acceptable
    'max_perplexity_increase': 0.12, # 12% max degradation
    'target_exit_rate': 0.40,        # 40% tokens exit early
    'target_skip_rate': 0.45,        # 45% FFN skipped
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_optimal_config():
    """Returns complete optimal configuration"""
    return {
        'num_layers': TINYLLAMA_NUM_LAYERS,
        'hidden_dim': TINYLLAMA_HIDDEN_DIM,
        'exit_layers': OPTIMAL_EXIT_LAYERS,
        'capacity': OPTIMAL_CAPACITY,
        'exit_gate': EXIT_GATE_CONFIG,
        'router': ROUTER_CONFIG,
        'training': TRAINING_CONFIG,
        'dataset': DATASET_CONFIG,
        'memory': MEMORY_CONFIG,
        'expected': EXPECTED_PERFORMANCE,
    }


def get_capacity_at_step(step, total_steps, config=None):
    """Calculate capacity based on training progress"""
    if config is None:
        config = TRAINING_CONFIG
    
    if not config.get('use_capacity_schedule', False):
        return OPTIMAL_CAPACITY
    
    progress = step / total_steps if total_steps > 0 else 0
    warmup = config['capacity_warmup_ratio']
    
    if progress < warmup:
        return config['capacity_start']
    
    # Linear decay after warmup
    decay_progress = (progress - warmup) / (1 - warmup)
    capacity = config['capacity_start'] - (
        config['capacity_start'] - config['capacity_end']
    ) * decay_progress
    
    return max(config['capacity_end'], capacity)


def get_layer_dropout_rate(layer_idx, progress, config=None):
    """Calculate layer dropout rate (higher for later layers)"""
    if config is None:
        config = TRAINING_CONFIG
    
    if not config.get('use_layer_dropout', False):
        return 0.0
    
    # Position ratio (0 = first layer, 1 = last layer)
    position_ratio = layer_idx / (TINYLLAMA_NUM_LAYERS - 1)
    
    # Dropout increases with depth
    base_rate = config['layer_dropout_max'] * position_ratio
    
    # Ramp up during first half of training
    scale = min(progress * 2, 1.0)
    
    return base_rate * scale


if __name__ == '__main__':
    config = get_optimal_config()
    
    print("=" * 80)
    print("OPTIMIZED HIERARCHICAL TRANSFORMER CONFIGURATION")
    print("=" * 80)
    
    print(f"\n📐 Architecture:")
    print(f"   Layers: {config['num_layers']}")
    print(f"   Hidden dim: {config['hidden_dim']}")
    print(f"   Exit points: {config['exit_layers']}")
    print(f"   Base capacity: {config['capacity']*100:.0f}%")
    
    print(f"\n🎯 Expected Performance (Speedup-Focused):")
    print(f"   Target speedup: {config['expected']['target_speedup']:.1f}x")
    print(f"   Max perplexity increase: {config['expected']['max_perplexity_increase']*100:.0f}%")
    print(f"   Target exit rate: {config['expected']['target_exit_rate']*100:.0f}%")
    
    print(f"\n🏋️ Training:")
    print(f"   Router epochs: {config['training']['router_epochs']}")
    print(f"   Exit epochs: {config['training']['exit_epochs']}")
    print(f"   Layer dropout: {'✅' if config['training']['use_layer_dropout'] else '❌'}")
    print(f"   Capacity schedule: {'✅' if config['training']['use_capacity_schedule'] else '❌'}")
    
    print(f"\n🔧 Router Settings:")
    print(f"   Z-loss weight: {config['router']['z_loss_weight']}")
    print(f"   Entropy weight: {config['router']['entropy_weight']}")
    print(f"   Capacity range: [{config['router']['min_capacity']}, {config['router']['max_capacity']}]")
