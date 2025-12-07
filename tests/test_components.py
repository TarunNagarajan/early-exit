"""
COMPREHENSIVE TEST SUITE FOR HIERARCHICAL ADAPTIVE TRANSFORMER

Tests:
1. Component tests (ExitGate, MoERouter, TokenState)
2. Integration tests (HierarchicalTransformerWrapper)
3. Gradient flow verification
4. Edge case handling
5. Efficiency metric validation
"""

import pytest
import torch
import torch.nn as nn
import sys
import math

sys.path.append('.')

from src.model.gate import ExitGate
from src.model.router import MoERouter
from src.model.tokenstate import TokenState


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def hidden_dim():
    return 2048


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def sample_hidden_states(batch_size, seq_len, hidden_dim, device):
    return torch.randn(batch_size, seq_len, hidden_dim, device=device)


@pytest.fixture
def active_mask(batch_size, seq_len, device):
    return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)


# ============================================================================
# EXIT GATE TESTS
# ============================================================================

class TestExitGate:
    """Tests for ExitGate component"""
    
    def test_init(self, hidden_dim):
        """Test initialization"""
        gate = ExitGate(hidden_dim)
        assert gate.hidden_dim == hidden_dim
        assert hasattr(gate, 'exit_network')
        assert hasattr(gate, 'log_temp_forward')
        assert hasattr(gate, 'log_temp_backward')
    
    def test_forward_training(self, hidden_dim, sample_hidden_states, device):
        """Test forward pass in training mode"""
        gate = ExitGate(hidden_dim).to(device)
        gate.train()
        
        output = gate(sample_hidden_states, training=True)
        
        batch_size, seq_len = sample_hidden_states.shape[:2]
        assert output.shape == (batch_size, seq_len)
        assert output.dtype == torch.float32
        # Should be probabilities between 0 and 1
        assert (output >= 0).all() and (output <= 1).all()
    
    def test_forward_inference(self, hidden_dim, sample_hidden_states, device):
        """Test forward pass in inference mode"""
        gate = ExitGate(hidden_dim).to(device)
        gate.eval()
        
        output = gate(sample_hidden_states, training=False)
        
        batch_size, seq_len = sample_hidden_states.shape[:2]
        assert output.shape == (batch_size, seq_len)
        # Should be binary (0 or 1)
        assert torch.all((output == 0) | (output == 1))
    
    def test_gradient_flow(self, hidden_dim, sample_hidden_states, device):
        """Test that gradients flow through exit gate"""
        gate = ExitGate(hidden_dim).to(device)
        gate.train()
        
        sample_hidden_states.requires_grad = True
        output = gate(sample_hidden_states, training=True)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert sample_hidden_states.grad is not None
        for param in gate.parameters():
            assert param.grad is not None
    
    def test_get_exit_decisions(self, hidden_dim, sample_hidden_states, active_mask, device):
        """Test get_exit_decisions method"""
        gate = ExitGate(hidden_dim).to(device)
        gate.eval()
        
        decisions = gate.get_exit_decisions(sample_hidden_states, active_mask)
        
        assert decisions.dtype == torch.bool
        assert decisions.shape == active_mask.shape
    
    def test_temperature_clamping(self, hidden_dim):
        """Test temperature values are properly clamped"""
        gate = ExitGate(hidden_dim)
        
        # Set extreme log temperatures
        gate.log_temp_forward.data = torch.tensor(10.0)  # Very high
        gate.log_temp_backward.data = torch.tensor(-10.0)  # Very low
        
        assert gate.temp_forward <= 2.0
        assert gate.temp_backward >= 0.05


# ============================================================================
# MOE ROUTER TESTS
# ============================================================================

class TestMoERouter:
    """Tests for MoERouter component"""
    
    def test_init(self, hidden_dim):
        """Test initialization"""
        router = MoERouter(hidden_dim, capacity=0.5)
        assert router.hidden_dim == hidden_dim
        assert router.capacity == 0.5
        assert hasattr(router, 'router')
        
        # Check parameter count
        param_count = sum(p.numel() for p in router.parameters())
        assert param_count == hidden_dim + 1  # Weight + bias
    
    def test_forward_training(self, hidden_dim, sample_hidden_states, active_mask, device):
        """Test forward pass in training mode"""
        router = MoERouter(hidden_dim).to(device)
        router.train()
        
        ffn_mask, aux_loss = router(sample_hidden_states, active_mask, training=True)
        
        assert ffn_mask.dtype == torch.bool
        assert ffn_mask.shape == active_mask.shape
        assert aux_loss is not None
        assert aux_loss.requires_grad
    
    def test_forward_inference(self, hidden_dim, sample_hidden_states, active_mask, device):
        """Test forward pass in inference mode"""
        router = MoERouter(hidden_dim).to(device)
        router.eval()
        
        ffn_mask, aux_loss = router(sample_hidden_states, active_mask, training=False)
        
        assert ffn_mask.dtype == torch.bool
        assert aux_loss is None
    
    def test_capacity_compliance(self, hidden_dim, sample_hidden_states, active_mask, device):
        """Test that capacity constraint is respected"""
        capacity = 0.5
        router = MoERouter(hidden_dim, capacity=capacity).to(device)
        
        ffn_mask, _ = router(sample_hidden_states, active_mask, training=False)
        
        num_active = active_mask.sum().item()
        num_selected = ffn_mask.sum().item()
        
        # Should select approximately capacity * num_active tokens
        expected = int(capacity * num_active + 0.5)
        assert abs(num_selected - expected) <= 1  # Allow off-by-one
    
    def test_inactive_tokens_excluded(self, hidden_dim, sample_hidden_states, device):
        """Test that inactive tokens are never selected"""
        router = MoERouter(hidden_dim).to(device)
        
        # Create mask with some inactive tokens
        batch_size, seq_len = sample_hidden_states.shape[:2]
        active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        active_mask[:, seq_len//2:] = False  # Second half inactive
        
        ffn_mask, _ = router(sample_hidden_states, active_mask, training=False)
        
        # Inactive tokens should not be selected
        inactive_selected = ffn_mask[:, seq_len//2:].sum().item()
        assert inactive_selected == 0
    
    def test_no_active_tokens(self, hidden_dim, sample_hidden_states, device):
        """Test edge case: no active tokens"""
        router = MoERouter(hidden_dim).to(device)
        
        batch_size, seq_len = sample_hidden_states.shape[:2]
        active_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        ffn_mask, aux_loss = router(sample_hidden_states, active_mask, training=True)
        
        assert ffn_mask.sum().item() == 0
        assert aux_loss is None
    
    def test_capacity_adjustment(self, hidden_dim):
        """Test capacity adjustment with clamping"""
        router = MoERouter(hidden_dim)
        
        # Test lower bound
        router.adjust_capacity(0.05)
        assert router.capacity >= 0.1
        
        # Test upper bound
        router.adjust_capacity(0.95)
        assert router.capacity <= 0.9
        
        # Test valid value
        router.adjust_capacity(0.6)
        assert router.capacity == 0.6
    
    def test_temperature_annealing(self, hidden_dim):
        """Test temperature annealing"""
        router = MoERouter(hidden_dim, initial_temp=1.0, min_temp=0.1)
        
        # At start (progress=0)
        temp_start = router.update_temperature(0.0)
        assert abs(temp_start - 1.0) < 0.01
        
        # At end (progress=1)
        temp_end = router.update_temperature(1.0)
        assert abs(temp_end - 0.1) < 0.01
        
        # At middle (progress=0.5) - cosine decay
        temp_mid = router.update_temperature(0.5)
        assert temp_mid > 0.1 and temp_mid < 1.0
    
    def test_z_loss_computation(self, hidden_dim, sample_hidden_states, active_mask, device):
        """Test z-loss computation"""
        router = MoERouter(hidden_dim, z_loss_weight=0.1).to(device)
        
        scores = router.router(sample_hidden_states).squeeze(-1)
        z_loss = router.compute_router_z_loss(scores, active_mask)
        
        assert z_loss.item() >= 0  # z-loss should be non-negative


# ============================================================================
# TOKEN STATE TESTS
# ============================================================================

class TestTokenState:
    """Tests for TokenState component"""
    
    def test_init(self, batch_size, seq_len, device):
        """Test initialization"""
        state = TokenState(batch_size, seq_len, device)
        
        assert state.active.shape == (batch_size, seq_len)
        assert state.exit_layer.shape == (batch_size, seq_len)
        assert state.skip_count.shape == (batch_size, seq_len)
        assert state.active.all()  # All tokens start active
        assert (state.exit_layer == -1).all()  # No exits yet
        assert (state.skip_count == 0).all()  # No skips yet
    
    def test_update_exit(self, batch_size, seq_len, device):
        """Test exit updates"""
        state = TokenState(batch_size, seq_len, device)
        
        # Exit some tokens
        exit_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        exit_mask[0, :4] = True  # First 4 tokens of first batch
        
        newly_exited = state.update_exit(exit_mask, layer_idx=5)
        
        assert newly_exited.sum().item() == 4
        assert state.active[0, :4].sum().item() == 0  # Exited tokens inactive
        assert (state.exit_layer[0, :4] == 5).all()  # Exit layer recorded
    
    def test_update_skip(self, batch_size, seq_len, device):
        """Test skip updates"""
        state = TokenState(batch_size, seq_len, device)
        
        # Skip FFN for some tokens
        skip_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        skip_mask[0, :8] = True
        
        state.update_skip(skip_mask, layer_idx=3)
        
        assert (state.skip_count[0, :8] == 1).all()
        assert len(state.layer_stats) == 1
    
    def test_exited_tokens_cannot_skip(self, batch_size, seq_len, device):
        """Test that exited tokens don't accumulate skips"""
        state = TokenState(batch_size, seq_len, device)
        
        # Exit first token
        exit_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        exit_mask[0, 0] = True
        state.update_exit(exit_mask, layer_idx=2)
        
        # Try to skip all tokens
        skip_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        state.update_skip(skip_mask, layer_idx=3)
        
        # Exited token should not have skip counted
        assert state.skip_count[0, 0].item() == 0
    
    def test_efficiency_metrics_no_exits(self, batch_size, seq_len, device):
        """Test efficiency metrics when no tokens exit"""
        state = TokenState(batch_size, seq_len, device)
        
        # Skip some FFNs but no exits
        skip_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        for layer in range(5):  # Simulate 5 layers
            state.update_skip(skip_mask, layer)
        
        metrics = state.get_efficiency_metrics(total_layers=22)
        
        assert metrics['exit_rate'] == 0.0
        assert metrics['speedup'] > 1.0  # Should show speedup from skips
        assert 0 < metrics['compute_fraction'] < 1.0
    
    def test_efficiency_metrics_with_exits(self, batch_size, seq_len, device):
        """Test efficiency metrics with early exits"""
        state = TokenState(batch_size, seq_len, device)
        
        # Exit half the tokens at layer 5
        exit_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        exit_mask[:, :seq_len//2] = True
        state.update_exit(exit_mask, layer_idx=5)
        
        metrics = state.get_efficiency_metrics(total_layers=22)
        
        assert metrics['exit_rate'] == 0.5
        assert metrics['speedup'] > 1.0
        assert metrics['avg_exit_depth'] == 5.0
    
    def test_compute_fraction_accuracy(self, device):
        """Test that compute fraction is calculated correctly"""
        # Simple case: 1 token, exits at layer 10 of 22
        state = TokenState(1, 1, device)
        
        exit_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
        state.update_exit(exit_mask, layer_idx=10)
        
        metrics = state.get_efficiency_metrics(total_layers=22)
        
        # Token used 11 layers (0-10), each with attn+ffn
        # Compute units = 11 * 2 = 22
        # Max compute = 1 * 22 * 2 = 44
        # Fraction = 22/44 = 0.5
        assert abs(metrics['compute_fraction'] - 0.5) < 0.01


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for the full hierarchical model"""
    
    @pytest.fixture
    def mock_base_model(self, hidden_dim, device):
        """Create a mock base model for testing"""
        class MockConfig:
            hidden_size = hidden_dim
            num_hidden_layers = 6  # Small for testing
        
        class MockLayer(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.input_layernorm = nn.LayerNorm(hidden_dim)
                self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
                self.self_attn = nn.Identity()
                self.mlp = nn.Linear(hidden_dim, hidden_dim)
            
            def forward(self, x):
                return x
        
        class MockSelfAttn(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.proj = nn.Linear(hidden_dim, hidden_dim)
            
            def forward(self, x, attention_mask=None):
                return (self.proj(x),)
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(1000, hidden_dim)
                self.model.layers = nn.ModuleList([
                    MockLayer(hidden_dim) for _ in range(6)
                ])
                for layer in self.model.layers:
                    layer.self_attn = MockSelfAttn(hidden_dim)
                self.model.norm = nn.LayerNorm(hidden_dim)
                self.lm_head = nn.Linear(hidden_dim, 1000)
        
        return MockModel().to(device)
    
    def test_wrapper_creation(self, mock_base_model, hidden_dim):
        """Test wrapper initialization"""
        from src.model.adaptive import HierarchicalTransformerWrapper
        
        wrapper = HierarchicalTransformerWrapper(
            mock_base_model,
            exit_layers=[2, 4],
            capacity=0.5,
        )
        
        assert wrapper.num_layers == 6
        assert len(wrapper.exit_gates) == 6
        assert len(wrapper.skip_routers) == 6
        
        # Only specified layers have exit gates
        assert wrapper.exit_gates[2] is not None
        assert wrapper.exit_gates[4] is not None
        assert wrapper.exit_gates[0] is None
    
    def test_forward_pass(self, mock_base_model, device):
        """Test forward pass"""
        from src.model.adaptive import HierarchicalTransformerWrapper
        
        wrapper = HierarchicalTransformerWrapper(
            mock_base_model,
            exit_layers=[2, 4],
            capacity=0.5,
        ).to(device)
        
        input_ids = torch.randint(0, 1000, (2, 16), device=device)
        
        outputs = wrapper(input_ids, training=True)
        
        assert 'logits' in outputs
        assert 'efficiency_metrics' in outputs
        assert 'token_states' in outputs
        assert 'aux_losses' in outputs
        
        assert outputs['logits'].shape == (2, 16, 1000)
    
    def test_gradient_flow_through_wrapper(self, mock_base_model, device):
        """Test that gradients flow through the wrapper"""
        from src.model.adaptive import HierarchicalTransformerWrapper
        
        wrapper = HierarchicalTransformerWrapper(
            mock_base_model,
            exit_layers=[2, 4],
            capacity=0.5,
        ).to(device)
        
        input_ids = torch.randint(0, 1000, (2, 16), device=device)
        
        outputs = wrapper(input_ids, training=True)
        loss = outputs['logits'].sum()
        loss.backward()
        
        # Check gradients for trainable components
        for router in wrapper.skip_routers:
            for param in router.parameters():
                if param.requires_grad:
                    assert param.grad is not None
    
    def test_trainable_param_count(self, mock_base_model):
        """Test trainable parameter counting"""
        from src.model.adaptive import HierarchicalTransformerWrapper
        
        hidden_dim = mock_base_model.config.hidden_size
        
        wrapper = HierarchicalTransformerWrapper(
            mock_base_model,
            exit_layers=[2, 4],
            capacity=0.5,
        )
        
        trainable = wrapper.count_trainable_params()
        
        # Should only count gates and routers, not base model
        # 2 gates + 6 routers
        assert trainable > 0
        
        # Base model should be frozen
        for param in mock_base_model.parameters():
            assert not param.requires_grad


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
