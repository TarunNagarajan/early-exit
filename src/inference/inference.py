#!/usr/bin/env python3
"""
ADAPTIVE INFERENCE WITH BUDGET CONTROL

Features:
- Binary search for target speedup
- Dynamic capacity adjustment
- Batch processing with efficiency tracking
"""

import torch
import torch.nn.functional as F
import argparse
import sys
import math
from typing import Tuple, Dict, Any, Optional

sys.path.append('.')
from src.model.load import load_model_and_tokenizer


class AdaptiveInference:
    """
    Budget-aware inference that achieves target speedup via binary search.
    
    Adjusts router capacity to meet speedup targets while maintaining quality.
    """
    
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()
    
    def infer_with_target_speedup(
        self,
        input_ids: torch.Tensor,
        target_speedup: float = 1.5,
        tolerance: float = 0.05,
        max_iterations: int = 10,
        min_capacity: float = 0.1,
        max_capacity: float = 0.9,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Binary search for optimal capacity to achieve target speedup.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            target_speedup: Desired speedup factor
            tolerance: Acceptable deviation from target
            max_iterations: Maximum binary search iterations
            min_capacity: Minimum capacity to try
            max_capacity: Maximum capacity to try
        
        Returns:
            Tuple of (model outputs, optimal capacity)
        """
        low, high = min_capacity, max_capacity
        best_capacity = 0.5
        best_diff = float('inf')
        best_outputs = None
        
        for iteration in range(max_iterations):
            mid_capacity = (low + high) / 2
            self.model.set_capacity(mid_capacity)
            
            with torch.no_grad():
                outputs = self.model(input_ids.to(self.device), training=False)
            
            actual_speedup = outputs['efficiency_metrics']['speedup']
            diff = abs(actual_speedup - target_speedup)
            
            # Track best result
            if diff < best_diff:
                best_diff = diff
                best_capacity = mid_capacity
                best_outputs = outputs
            
            # Check convergence
            if diff < tolerance:
                break
            
            # Adjust search range
            if actual_speedup < target_speedup:
                # Need more speedup = lower capacity
                high = mid_capacity
            else:
                # Too much speedup = higher capacity
                low = mid_capacity
        
        # Set model to best capacity found
        self.model.set_capacity(best_capacity)
        
        return best_outputs, best_capacity
    
    def generate_with_budget(
        self,
        prompt: str,
        max_length: int = 100,
        target_speedup: Optional[float] = None,
        capacity: Optional[float] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text with compute budget constraint.
        
        Args:
            prompt: Input prompt string
            max_length: Maximum tokens to generate
            target_speedup: If set, use binary search for capacity
            capacity: If set, use fixed capacity
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
        
        Returns:
            Tuple of (generated text, efficiency metrics)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Set capacity
        if target_speedup is not None:
            # Find optimal capacity for target speedup
            _, optimal_capacity = self.infer_with_target_speedup(
                input_ids, target_speedup=target_speedup
            )
            self.model.set_capacity(optimal_capacity)
        elif capacity is not None:
            self.model.set_capacity(capacity)
        
        # Generate tokens
        generated = input_ids
        all_metrics = []
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(generated, training=False)
                all_metrics.append(outputs['efficiency_metrics'])
                
                next_token_logits = outputs['logits'][:, -1, :]
                
                if do_sample and temperature > 0:
                    # Sample with temperature
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Stop at EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode output
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = sum(values) / len(values)
        
        return text, avg_metrics
    
    def benchmark_capacities(
        self,
        input_ids: torch.Tensor,
        capacities: list = None,
    ) -> Dict[float, Dict[str, float]]:
        """
        Benchmark different capacity settings.
        
        Args:
            input_ids: Input token IDs
            capacities: List of capacities to test
        
        Returns:
            Dict mapping capacity to efficiency metrics
        """
        if capacities is None:
            capacities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = {}
        
        for cap in capacities:
            self.model.set_capacity(cap)
            
            with torch.no_grad():
                outputs = self.model(input_ids.to(self.device), training=False)
            
            results[cap] = outputs['efficiency_metrics']
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Adaptive Inference with Budget Control")
    parser.add_argument("--prompt", type=str, 
                       default="The quantum computer calculated the solution to",
                       help="Input prompt for generation")
    parser.add_argument("--length", type=int, default=50,
                       help="Maximum tokens to generate")
    parser.add_argument("--target-speedup", type=float, default=None,
                       help="Target speedup (uses binary search)")
    parser.add_argument("--capacity", type=float, default=0.55,
                       help="Fixed capacity (if --target-speedup not set)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run capacity benchmark instead of generation")
    args = parser.parse_args()

    print("Loading model...")
    base_model, tokenizer = load_model_and_tokenizer()

    # Load checkpoint or create new model
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        from src.model.adaptive import HierarchicalTransformerWrapper
        hierarchical_model = HierarchicalTransformerWrapper(
            base_model=base_model,
            exit_layers=checkpoint['config']['exit_layers'],
            capacity=checkpoint['config']['capacity'],
        )
        hierarchical_model.load_state_dict(checkpoint['wrapper_state'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        from src.model.adaptive import HierarchicalTransformerWrapper
        hierarchical_model = HierarchicalTransformerWrapper(base_model=base_model)
        print("Using untrained model (for testing)")

    hierarchical_model = hierarchical_model.to(base_model.device)
    
    # Create adaptive inference engine
    inference = AdaptiveInference(hierarchical_model, tokenizer)
    
    if args.benchmark:
        # Run capacity benchmark
        print("\nRunning capacity benchmark...")
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
        results = inference.benchmark_capacities(input_ids)
        
        print("\n" + "=" * 60)
        print("CAPACITY BENCHMARK RESULTS")
        print("=" * 60)
        print(f"{'Capacity':>10} {'Speedup':>10} {'Exit Rate':>12} {'Skip Rate':>12}")
        print("-" * 60)
        
        for cap in sorted(results.keys()):
            metrics = results[cap]
            print(f"{cap*100:>9.0f}% {metrics['speedup']:>9.2f}x "
                  f"{metrics['exit_rate']*100:>11.1f}% "
                  f"{metrics['avg_skip_rate']*100:>11.1f}%")
    else:
        # Generate text
        print(f"\nPrompt: {args.prompt}")
        
        if args.target_speedup:
            print(f"Target speedup: {args.target_speedup}x (using binary search)")
            text, metrics = inference.generate_with_budget(
                args.prompt,
                max_length=args.length,
                target_speedup=args.target_speedup,
            )
        else:
            print(f"Fixed capacity: {args.capacity*100:.0f}%")
            text, metrics = inference.generate_with_budget(
                args.prompt,
                max_length=args.length,
                capacity=args.capacity,
            )
        
        print("\n" + "=" * 60)
        print("GENERATION RESULTS")
        print("=" * 60)
        print(f"\nGenerated text:\n  {text}")
        print(f"\nEfficiency metrics:")
        print(f"  Speedup: {metrics['speedup']:.2f}x")
        print(f"  Compute fraction: {metrics['compute_fraction']*100:.1f}%")
        print(f"  Exit rate: {metrics['exit_rate']*100:.1f}%")
        print(f"  Skip rate: {metrics['avg_skip_rate']*100:.1f}%")


if __name__ == "__main__":
    main()