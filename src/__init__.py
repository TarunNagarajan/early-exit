"""
Hierarchical Adaptive Transformer

Two-dimensional adaptive computation framework:
- Vertical (depth): Early exit via exit gates
- Horizontal (width): Selective FFN via MoE routers
"""

from .config import get_optimal_config

__version__ = "1.0.0"
__all__ = ["get_optimal_config"]