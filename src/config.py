"""
OPTIMAL CONFIGURATION FOR TINYLLAMA 1.1B
Research-backed settings for Small Language Models
"""

# ============================================================================
# TinyLlama Architecture (22 layers, 2048 hidden_dim, 32 heads)
# ============================================================================

# CRITICAL: TinyLlama has 22 layers, NOT 32!
TINYLLAMA_NUM_LAYERS = 22  
TINYLLAMA_HIDDEN_DIM = 2048
TINYLLAMA_ATTENTION_HEADS = 32

# ============================================================================
# EXIT GATE CONFIGURATION (Conservative for SLMs)
# ============================================================================

# For SLMs, use FEWER exit points to preserve model capacity
# Rule of thumb: exit_points = num_layers / 4 (instead of / 3 for larger models)
OPTIMAL_EXIT_LAYERS = [5, 10, 15, 18]  # 4 exit points for 22-layer model

# Why this works for SLMs:
# - Early exit at layer 5 (~23% depth) for simple tokens
# - Mid exits at 10 and 15 for moderate complexity
# - Late exit at 18 (82% depth) before final layers
# - Preserves last 4 layers for complex reasoning

# ============================================================================
# ROUTER CAPACITY CONFIGURATION (Aggressive for SLMs)
# ============================================================================

# SLMs benefit from HIGHER capacity to preserve learned features
OPTIMAL_CAPACITY = 0.7  # 70% FFN computation (vs 50% for larger models)

# Why higher capacity for SLMs:
# - Less redundancy in small models - every layer matters more
# - Research shows 1B models need 60-80% capacity for competitive performance
# - Below 60% causes significant quality degradation

# ============================================================================
# TRAINING HYPERPARAMETERS (SLM-Specific)
# ============================================================================

TRAINING_CONFIG = {
    # Phase 1: Router training
    'router_epochs': 2,  # Fewer epochs for small models (vs 3-5 for large)
    'router_lr': 5e-4,   # Higher LR for faster convergence with less data
    'router_warmup_steps': 100,  # Short warmup
    
    # Phase 2: Exit gate training  
    'exit_epochs': 1,    # Just 1 epoch to avoid overfitting
    'exit_lr': 3e-4,     # Slightly lower than router
    'exit_warmup_steps': 50,
    
    # Batch configuration
    'batch_size': 4,     # Larger batches work better with 4-bit quantization
    'gradient_accumulation': 2,  # Effective batch = 8
    'max_seq_length': 512,  # Keep moderate for memory efficiency
    
    # Regularization (critical for SLMs)
    'weight_decay': 0.01,
    'dropout': 0.1,
    'gradient_clip': 0.5,  # Lower clip for stability
    
    # Loss weights
    'lb_loss_weight': 0.005,  # Lower for SLMs (vs 0.01)
    'exit_timing_weight': 0.05,  # Lower to avoid aggressive early exits
}

# ============================================================================
# GATE & ROUTER ARCHITECTURE (Lightweight for SLMs)
# ============================================================================

GATE_CONFIG = {
    'hidden_size': 32,  # Keep small - don't overparameterize
    'initial_temp': 0.5,  # Lower initial temp for SLMs
    'min_temp': 0.05,
    'anneal_rate': 5e-5,  # Faster annealing for fewer training steps
    'use_position_bias': False,  # Skip for simplicity in SLMs
}

ROUTER_CONFIG = {
    'initial_temp': 0.5,  # Lower starting temp
    'min_temp': 0.05,
    'anneal_rate': 5e-5,
    'lb_loss_weight': 0.005,
    'z_loss_weight': 5e-6,
}

# ============================================================================
# DATASET CONFIGURATION (Small model = smaller dataset)
# ============================================================================

DATASET_CONFIG = {
    'name': 'wikitext',
    'version': 'wikitext-2-raw-v1',  # Smaller version
    'train_samples': 10000,  # Limit training samples
    'eval_samples': 1000,
    'test_samples': 1000,
}

# ============================================================================
# MEMORY OPTIMIZATION (Critical for Colab)
# ============================================================================

MEMORY_CONFIG = {
    'use_4bit_quantization': True,  # Keep base model in 4-bit
    'use_gradient_checkpointing': True,  # Essential for limited memory
    'empty_cache_frequency': 10,  # Clear cache every N steps
}

# ============================================================================
# EVALUATION METRICS (Track degradation)
# ============================================================================

ACCEPTABLE_DEGRADATION = {
    'max_perplexity_increase': 0.15,  # Accept 15% increase (vs 10% for large)
    'min_speedup': 1.3,  # Require 1.3x speedup minimum
    'target_compute_fraction': 0.7,  # Target 70% computation
}

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def get_optimal_config():
    """Returns optimal configuration for TinyLlama 1.1B"""
    return {
        'num_layers': TINYLLAMA_NUM_LAYERS,
        'hidden_dim': TINYLLAMA_HIDDEN_DIM,
        'exit_layers': OPTIMAL_EXIT_LAYERS,
        'capacity': OPTIMAL_CAPACITY,
        'training': TRAINING_CONFIG,
        'gate': GATE_CONFIG,
        'router': ROUTER_CONFIG,
        'dataset': DATASET_CONFIG,
        'memory': MEMORY_CONFIG,
        'acceptance': ACCEPTABLE_DEGRADATION,
    }

if __name__ == '__main__':
    config = get_optimal_config()
    print("=" * 80)
    print("OPTIMAL TINYLLAMA 1.1B CONFIGURATION")
    print("=" * 80)
    print(f"\nArchitecture:")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Exit points: {config['exit_layers']}")
    print(f"  Capacity: {config['capacity']*100:.0f}% ")
    
    print(f"\nExpected Performance:")
    print(f"  Target speedup: ~1.4-1.5x")
    print(f"  Compute fraction: ~70%")
    print(f"  Max perplexity increase: <15%")
    
    print(f"\nTraining:")
    print(f"  Router epochs: {config['training']['router_epochs']}")
    print(f"  Exit epochs: {config['training']['exit_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    
