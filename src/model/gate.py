import torch
import torch.nn as nn
import torch.nn.functional as F

class ExitGate(nn.Module):
    """
    - exit decision at 'strategic' layers
    - gumbel-softmax for differentiable training
    """

    def __init__(self, hidden_dim, temperature=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.exit_network = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),  # [continue, exit]
        )

        for layer in self.exit_network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, hidden_states, training=True):
        batch_size, seq_len, _ = hidden_states.shape
        logits = self.exit_network(hidden_states)

        if training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            noisy_logits = (logits + gumbel_noise) / self.temperature
            exit_probs = F.softmax(noisy_logits, dim=-1)[:, :, 1]

            if self.training:
                exit_decision = (exit_probs > 0.5).float()
                exit_probs = exit_decision + (exit_probs - exit_probs.detach())

            return exit_probs
        else:
            exit_probs = F.softmax(logits, dim=-1)[:, :, 1]
            return (exit_probs > 0.5).float()

    def get_exit_decisions(self, hidden_states, active_mask=None):
        exit_probs = self.forward(hidden_states, training=False)
        if active_mask is not None:
            exit_probs = exit_probs * active_mask.float()
        return (exit_probs > 0.5)

