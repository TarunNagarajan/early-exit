
import torch
import sys
import math
from typing import Dict, Any

sys.path.append('.')

from src.model.gate import ExitGate
from src.model.router import MoERouter
from src.model.tokenstate import TokenState
from src.model.adaptive import HierarchicalTransformerWrapper

def test_token_state_efficiency():
    print("Testing TokenState efficiency metrics...")
    device = torch.device('cpu')
    batch_size = 2
    seq_len = 5
    total_layers = 10
    
    state = TokenState(batch_size, seq_len, device)
    
    # Simulate execution of 5 layers
    # Layer 0-4 executed
    
    # Simulate some skips
    # Token (0,0) skips layer 0
    skip_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    skip_mask[0, 0] = True
    state.update_skip(skip_mask, layer_idx=0)
    
    # Simulate execution stats for 5 layers
    # For simplicity, let's say we ran 5 layers
    # We need to manually populate layer_stats because update_skip mostly handles it
    # But usually the loop does it.
    
    # Let's use the loop pattern to be accurate
    for i in range(5):
        # some skips
        skip_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        if i % 2 == 0:
            skip_mask[:, :] = True # Skip all on even layers
        
        state.update_skip(skip_mask, layer_idx=i)
        
    metrics = state.get_efficiency_metrics(total_layers=total_layers)
    
    print(f"  Executed layers: {metrics['executed_layers']} (Expected: 5)")
    assert metrics['executed_layers'] == 5
    
    # Check compute fraction
    # Total tokens = 10
    # Attention cost = 10 * 5 = 50
    # FFN cost:
    # 5 layers. Even layers (0,2,4) skipped -> 0 FFN cost.
    # Odd layers (1,3) not skipped -> 10 * 2 = 20 FFN cost.
    # Total compute = 50 + 20 = 70.
    
    # Max compute = 10 * 10 * 2 = 200.
    # Expected fraction = 70 / 200 = 0.35
    
    print(f"  Compute fraction: {metrics['compute_fraction']:.4f} (Expected: 0.35)")
    assert abs(metrics['compute_fraction'] - 0.35) < 0.001
    print("  ✅ TokenState efficiency metrics verified.")


def test_router_gradients():
    print("\nTesting MoERouter gradients...")
    device = torch.device('cpu')
    hidden_dim = 32
    router = MoERouter(hidden_dim, capacity=0.5).to(device)
    
    # Allow gradients on input
    inputs = torch.randn(2, 5, hidden_dim, requires_grad=True, device=device)
    active_mask = torch.ones(2, 5, dtype=torch.bool, device=device)
    
    # Forward pass
    ffn_mask, router_probs, aux_loss = router(inputs, active_mask, training=True)
    
    print(f"  Router output shapes: mask={ffn_mask.shape}, probs={router_probs.shape}")
    
    # Simulate usage in Adaptive Wrapper (Applying STE)
    # hidden = hidden + ffn_output * (1 + probs - probs.detach())
    # but strictly speaking, we just need to check if gradients flow to 'inputs' from 'router_probs'
    
    # Let's define a loss on router_probs
    loss = router_probs.sum() + aux_loss
    loss.backward()
    
    # Check gradients
    print(f"  Input gradients: {'Present' if inputs.grad is not None else 'Missing'}")
    if inputs.grad is not None:
        print(f"  Input gradient norm: {inputs.grad.norm().item():.4f}")
        
    print(f"  Router weight gradients: {'Present' if router.router.weight.grad is not None else 'Missing'}")
    
    assert inputs.grad is not None
    assert inputs.grad.norm().item() > 0
    assert router.router.weight.grad is not None
    print("  ✅ Router gradients verified.")

def test_integration_flow():
    print("\nTesting Integration (Wrapper) Flow...")
    device = torch.device('cpu')
    
    # Mock base model components
    class MockConfig:
        hidden_size = 32
        num_hidden_layers = 2
    
    class MockLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(32)
            self.post_attention_layernorm = torch.nn.LayerNorm(32)
            self.self_attn = lambda x, **kwargs: (x,)
            self.mlp = torch.nn.Linear(32, 32)
            
    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(100, 32)
            self.model.norm = torch.nn.Identity()
            self.model.layers = torch.nn.ModuleList([MockLayer() for _ in range(2)])
            self.lm_head = torch.nn.Linear(32, 100)
            
    base = MockBaseModel()
    wrapper = HierarchicalTransformerWrapper(base, exit_layers=[], capacity=0.5)
    wrapper.unfreeze_routers()
    
    # Forward pass
    input_ids = torch.randint(0, 100, (2, 5))
    outputs = wrapper(input_ids, training=True)
    
    # Check outputs
    logits = outputs['logits']
    aux_losses = outputs['aux_losses']
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Aux losses count: {len(aux_losses)}")
    
    # Compute generic loss
    loss = logits.sum() + sum(aux_losses)
    loss.backward()
    
    # Check gradients in routers
    grad_found = False
    for param in wrapper.skip_routers[0].parameters():
        if param.grad is not None:
            grad_found = True
            break
            
    print(f"  Gradients found in wrapper routers: {grad_found}")
    assert grad_found
    print("  ✅ Integration flow verified.")

if __name__ == "__main__":
    try:
        test_token_state_efficiency()
        test_router_gradients()
        test_integration_flow()
        print("\nAll verification checks passed!")
    except AssertionError as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        sys.exit(1)
