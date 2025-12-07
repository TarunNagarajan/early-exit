# Hierarchical Adaptive Transformer

A **two-dimensional adaptive computation** framework for efficient transformer inference.

![Speedup](https://img.shields.io/badge/Speedup-1.5--1.8x-green)
![Quality](https://img.shields.io/badge/Perplexity%20Increase-%3C12%25-blue)
![Parameters](https://img.shields.io/badge/Overhead-~100K%20params-orange)

## Key Innovation

This project implements a hierarchical approach to adaptive computation with **no explicit difficulty predictor**:

1. **Vertical Adaptation (Exit Gates)**: Tokens exit at different depths (layers 4, 8, 12, 16, 19)
2. **Horizontal Adaptation (Skip Gates)**: Active tokens skip FFN computation at variable rates using MoE-style routing

The routing decisions learn difficulty implicitly—no auxiliary classification task required.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Tokens                          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Embedding Layer                         │
└─────────────────────────────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Layer 1     │  │   Layer 4     │  │   Layer 19    │
│               │  │ + Exit Gate   │  │ + Exit Gate   │
│ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │
│ │ Attention │ │  │ │ Attention │ │  │ │ Attention │ │
│ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │
│      │        │  │      │        │  │      │        │
│      ▼        │  │      ▼        │  │      ▼        │
│ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │
│ │  Router   │◄┼──┼─┤  Router   │◄┼──┼─┤  Router   │ │
│ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │
│      │        │  │      │        │  │      │        │
│   55% FFN     │  │   55% FFN     │  │   55% FFN     │
└───────────────┘  └───────────────┘  └───────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               Output Logits (Speedup: 1.5-1.8x)         │
└─────────────────────────────────────────────────────────┘
```

## Features

- **Base Model**: TinyLlama-1.1B (22 layers, 2048 hidden, 32 heads) - FROZEN
- **Exit Gates**: 5 strategic layers with Gumbel-Softmax + decoupled temperatures
- **MoE Routers**: All 22 layers with z-loss + entropy regularization
- **Two-Phase Training**: Routers first, exit gates second
- **Capacity Range**: 10-90% (default 55%)

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/early-exit.git
cd early-exit

# Install dependencies
pip install -r requirements.txt

# Run environment setup (first time only)
python environment.py
```

### Training

```bash
# Full training (routers + exit gates)
python src/training/train.py --phase full --save checkpoints/model.pth

# Train routers only
python src/training/train.py --phase routers

# Train exit gates only (requires pretrained routers)
python src/training/train.py --phase exit --resume checkpoints/routers.pth
```

### Inference

```bash
# Generate with fixed capacity
python src/inference/inference.py --prompt "The future of AI is" --capacity 0.5

# Generate with target speedup (binary search)
python src/inference/inference.py --prompt "The future of AI is" --target-speedup 1.6

# Benchmark different capacities
python src/inference/inference.py --benchmark
```

### Evaluation

```bash
python src/inference/evaluate.py --checkpoint checkpoints/model.pth
```

## Expected Performance

| Capacity | Speedup | Perplexity Increase |
|----------|---------|---------------------|
| 90% | 1.1x | <1% |
| 70% | 1.3x | ~3% |
| 55% | 1.6x | ~8% |
| 40% | 1.8x | ~12% |

## Configuration

Key hyperparameters in `src/config.py`:

```python
# Exit layers (5 strategic points for 22-layer model)
OPTIMAL_EXIT_LAYERS = [4, 8, 12, 16, 19]

# Router capacity (55% = aggressive speedup focus)
OPTIMAL_CAPACITY = 0.55

# Training
TRAINING_CONFIG = {
    'router_epochs': 3,
    'exit_epochs': 2,
    'router_lr': 1e-3,
    'exit_lr': 5e-4,
    'gradient_clip': 1.0,
    'use_layer_dropout': True,
    'use_capacity_schedule': True,
}
```

## Research-Backed Optimizations

This implementation incorporates techniques from:

1. **LayerSkip (2024)**: Progressive layer dropout, optimal exit placement
2. **ST-MoE**: Router z-loss for training stability
3. **Decoupled ST-GS**: Separate forward/backward temperatures
4. **Mixture-of-Depths**: Dynamic compute allocation

## Project Structure

```
early-exit/
├── src/
│   ├── config.py                 # Optimal configuration
│   ├── model/
│   │   ├── adaptive.py           # HierarchicalTransformerWrapper
│   │   ├── gate.py               # ExitGate with decoupled temps
│   │   ├── router.py             # MoERouter with z-loss
│   │   ├── tokenstate.py         # Token lifecycle tracking
│   │   └── load.py               # Model loading utilities
│   ├── training/
│   │   └── train.py              # Two-phase training
│   └── inference/
│       ├── inference.py          # AdaptiveInference
│       ├── evaluate.py           # Perplexity evaluation
│       └── analyze.py            # Routing analysis
├── tests/
│   └── test_components.py        # Comprehensive test suite
├── checkpoints/                  # Saved models
├── environment.py                # Colab setup
└── requirements.txt
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_components.py::TestMoERouter -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Colab Usage

```python
# 1. Clone and setup
!git clone https://github.com/your-username/early-exit.git
%cd early-exit
!pip install -r requirements.txt

# 2. Train model
!python src/training/train.py --phase full --save checkpoints/model.pth

# 3. Evaluate
!python src/inference/evaluate.py --checkpoint checkpoints/model.pth
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hierarchical-adaptive-transformer,
  title={Hierarchical Adaptive Transformer: Two-Dimensional Adaptive Computation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/early-exit}
}
```

## License

MIT License - see LICENSE for details.

## Acknowledgments

- TinyLlama team for the base model
- LayerSkip, ST-MoE, and related papers for research insights