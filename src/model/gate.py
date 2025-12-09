"""
ENHANCED EXIT GATE WITH DECOUPLED TEMPERATURES

Research-backed improvements:
- Decoupled forward/backward temperatures (Decoupled ST-GS)
- Deeper network for better decision capacity
- Learnable per-gate temperatures
- Improved initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ExitGate(nn.Module):
    """
    Enhanced exit gate with research-backed optimizations.
    
    Features:
    - Gumbel-Softmax for differentiable training
    - Decoupled forward/backward temperatures
    - Hard threshold (0.5) for inference
    - Deeper network for better decision capacity
    """

    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 0.5,
        use_learnable_temp: bool = True,
        hidden_sizes: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_temperature = temperature
        self.use_learnable_temp = use_learnable_temp
        
        # Default hidden sizes for deeper network
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        
        # Build network
        layers = []
        in_dim = hidden_dim
        for h_dim in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 2))  # [continue, exit]
        
        self.exit_network = nn.Sequential(*layers)
        
        # Decoupled learnable temperatures
        if use_learnable_temp:
            self.log_temp_forward = nn.Parameter(torch.tensor(math.log(temperature)))
            self.log_temp_backward = nn.Parameter(torch.tensor(math.log(temperature * 2)))
        else:
            self.register_buffer('log_temp_forward', torch.tensor(math.log(temperature)))
            self.register_buffer('log_temp_backward', torch.tensor(math.log(temperature)))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for stable early training"""
        for module in self.exit_network.modules():
            if isinstance(module, nn.Linear):
                # Small initialization to start with low exit probability
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
        
        # Neutral initialization - let training decide exit behavior
        final_linear = list(self.exit_network.modules())[-1]
        if isinstance(final_linear, nn.Linear):
            final_linear.bias.data[0] = 0.0  # Continue - neutral
            final_linear.bias.data[1] = 0.0  # Exit - neutral
    
    @property
    def temp_forward(self):
        return torch.exp(self.log_temp_forward).clamp(0.05, 2.0)
    
    @property
    def temp_backward(self):
        return torch.exp(self.log_temp_backward).clamp(0.05, 2.0)

    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Compute exit decisions.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            training: Whether in training mode (use Gumbel-Softmax)
        
        Returns:
            exit_probs: [batch, seq_len] - probability/decision to exit
        """
        batch_size, seq_len, _ = hidden_states.shape
        logits = self.exit_network(hidden_states)  # [batch, seq_len, 2]

        if training:
            # Gumbel noise for exploration - clamp to prevent float16 overflow
            uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(uniform) + 1e-6)
            gumbel_noise = gumbel_noise.clamp(-10, 10)
            
            # Forward pass uses temp_forward
            noisy_logits = (logits + gumbel_noise) / self.temp_forward
            exit_probs = F.softmax(noisy_logits, dim=-1)[:, :, 1]

            # Straight-through estimator
            if self.training:
                exit_hard = (exit_probs > 0.5).float()
                # Gradient flows through soft, forward uses hard
                exit_probs = exit_hard + (exit_probs - exit_probs.detach())

            return exit_probs
        else:
            # Inference: deterministic threshold
            probs = F.softmax(logits, dim=-1)[:, :, 1]
            return (probs > 0.5).float()

    def get_exit_decisions(
        self,
        hidden_states: torch.Tensor,
        active_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get hard exit decisions for inference.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            active_mask: [batch, seq_len] - only decide for active tokens
        
        Returns:
            exit_mask: [batch, seq_len] - boolean mask of tokens to exit
        """
        exit_decisions = self.forward(hidden_states, training=False)
        exit_mask = exit_decisions > 0.5
        
        if active_mask is not None:
            exit_mask = exit_mask & active_mask
        
        return exit_mask
    
    def get_exit_confidence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get confidence scores for exit decisions (for analysis)"""
        logits = self.exit_network(hidden_states)
        probs = F.softmax(logits, dim=-1)
        
        # Confidence = how certain about the decision
        exit_prob = probs[:, :, 1]
        confidence = torch.abs(exit_prob - 0.5) * 2  # 0 = uncertain, 1 = certain
        
        return confidence


class ExitGateEnsemble(nn.Module):
    """
    Ensemble of exit gates for more robust decisions.
    Uses multiple small gates and averages their predictions.
    """
    
    def __init__(self, hidden_dim: int, num_gates: int = 3, **kwargs):
        super().__init__()
        self.gates = nn.ModuleList([
            ExitGate(hidden_dim, **kwargs) for _ in range(num_gates)
        ])
        self.num_gates = num_gates
    
    def forward(self, hidden_states: torch.Tensor, training: bool = True):
        probs = torch.stack([
            gate(hidden_states, training) for gate in self.gates
        ], dim=0)
        return probs.mean(dim=0)
    
    def get_exit_decisions(self, hidden_states, active_mask=None):
        avg_prob = self.forward(hidden_states, training=False)
        exit_mask = avg_prob > 0.5
        if active_mask is not None:
            exit_mask = exit_mask & active_mask
        return exit_mask
