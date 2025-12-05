#!/usr/bin/env python3
"""
ADAPTIVE INFERENCE WITH BUDGET CONTROL
"""

import torch
import argparse
import sys
sys.path.append('.')
from src.model.load import load_model_and_tokenizer

def generate_with_budget(model, tokenizer, prompt, max_length=100, budget=0.5):
    """Generate text with compute budget constraint"""
    print("[STARTING GENERATION WITH BUDGET]")
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    model.set_capacity(budget)

    generated = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated, training=False)

            next_token_logits = outputs['logits'][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    efficiency = outputs['efficiency_metrics']

    return text, efficiency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The quantum computer calculated")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--budget", type=float, default=0.5, help="Target compute fraction (0-1)")
    parser.add_argument("--checkpoint", type=str, help="Optional checkpoint")

    args = parser.parse_args()

    print("[LOADING MODEL]")
    base_model, tokenizer_obj = load_model_and_tokenizer()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        from src.model.adaptive import HierarchicalTransformerWrapper
        hierarchical_model = HierarchicalTransformerWrapper(
            base_model=base_model,
            exit_layers=checkpoint['config']['exit_layers'],
            capacity=checkpoint['config']['capacity']
        )
        hierarchical_model.load_state_dict(checkpoint['wrapper_state'])
    else:
        from src.model.adaptive import HierarchicalTransformerWrapper
        hierarchical_model = HierarchicalTransformerWrapper(base_model=base_model)

    hierarchical_model = hierarchical_model.to(base_model.device)

    print(f"[GENERATING WITH {args.budget*100:.0f}% COMPUTE BUDGET...]")
    print(f"Prompt: {args.prompt}")

    text, efficiency = generate_with_budget(
        hierarchical_model, tokenizer_obj, args.prompt, args.length, args.budget
    )

    print("\n" + "=" * 80)
    print("[GENERATION RESULTS]")
    print("=" * 80)
    print(f"\n[GENERATED TEXT:]")
    print(f"   {text}")

    print(f"\n[EFFICIENCY:]")
    print(f"   Target budget: {args.budget*100:.0f}%")
    print(f"   Actual compute: {efficiency['compute_fraction']*100:.1f}%")
    print(f"   Speedup: {efficiency['speedup']:.2f}x")
    print(f"   Exit rate: {efficiency['exit_rate']:.2%}")

    return text, efficiency

if __name__ == "__main__":
    main()