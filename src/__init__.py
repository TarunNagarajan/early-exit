"""
Hierarchical Adaptive Transformer Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Token-Level Adaptive Computation with Exit Gates and MoE-Style FFN Skipping"

from .token_state_tracker import TokenStateTracker
from .model.gate import ExitGate
from .model.router import MoERouter
from .model.adaptive import HierarchicalTransformerWrapper

__all__ = [
    "TokenStateTracker",
    "ExitGate", 
    "MoERouter",
    "HierarchicalTransformerWrapper"
]