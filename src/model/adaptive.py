import torch
import torch.nn as nn 
import torch.nn.functional as F 
from typing import List, Dict, Optional
from .gate import ExitGate
from .router import Router 
from .tokenstate import TokenState 

class Adaptive(nn.Module):
    """
    - wrapper for exit gates + routers
    """
    def __init__(self,
                 base_model: nn.Module,
                 exit_layers: List[int] = None,
                 capacity: float = 0.5, 
                 disable_exit_gates: bool = False):
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = base_model.config.hidden_size
        self.num_layers = base_model.config.num_hidden_layers
        self.capacity = capacity
        self.disable_exit_gates = disable_exit_gates

        if exit_layers is None:
            exit_layers = [4, 8, 12, 16, 20, 24, 28]

        self.exit_layers = exit_layers

        for param in base_model.parameters():
            param.requires_grad = False

        self.exit_gates = nn.ModuleList()
        for layer_index in range(self.num_layers):
            if layer_index in exit_layers and not disable_exit_gates:
                self.exit_gates.append(ExitGate(self.hidden_dim))
            else:
                self.exit_gates.append(None)

        self.skip_routers = nn.ModuleList([
            Router(self.hidden_dim, capacity = capacity)
            for _ in range(self.num_layers)
        ])

