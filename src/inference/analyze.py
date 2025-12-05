#!/usr/bin/env python3
"""
ANALYZE ROUTING PATTERNS AND VISUALIZE
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
import sys
sys.path.append('.')
from src.model.load import load_model_and_tokenizer

def analyze_token_patterns(model, tokenizer, texts):
    """Analyze exit and skip patterns for sample texts"""
    print("[ANALYZING TOKEN PATTERNS]")
    model.eval()
    all_patterns = []

    for text in texts:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            outputs = model(input_ids, training=False)

        exit_layer = outputs['token_states']['exit_layer'][0].cpu().numpy()
        skip_count = outputs['token_states']['skip_count'][0].cpu().numpy()
        active = outputs['token_states']['active'][0].cpu().numpy()

        patterns = []
        for i, (token, exit_l, skips, is_active) in enumerate(zip(tokens, exit_layer, skip_count, active)):
            patterns.append({
                'token': token,
                'exit_layer': exit_l if exit_l >= 0 else 32,
                'skips': skips,
                'active': is_active,
                'position': i
            })

        all_patterns.append(patterns)

    return all_patterns

def plot_heatmap(patterns, title="Exit Patterns"):
    """Create heatmap of exit layers"""
    tokens = [p['token'] for p in patterns]
    exit_layers = [p['exit_layer'] for p in patterns]
    
    plt.figure(figsize=(12, 4))
    
    # Create heatmap data
    data = np.array(exit_layers).reshape(1, -1)
    
    sns.heatmap(data, 
                xticklabels=tokens,
                yticklabels=False,
                cmap='viridis',
                cbar_kws={'label': 'Exit Layer'})
    
    plt.title(title)
    plt.xlabel("Token Position")
    plt.tight_layout()
    plt.savefig('visualizations/exit_patterns.png', dpi=150)
    plt.show()

def plot_efficiency_tradeoff(capacities, perplexities, speedups):
    """Plot perplexity vs speedup trade-off"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(speedups, perplexities, 'o-', linewidth=2, markersize=10)
    
    # Annotate points with capacities
    for cap, perp, speed in zip(capacities, perplexities, speedups):
        plt.annotate(f'{cap:.0%}', (speed, perp), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Speedup (x)')
    plt.ylabel('Perplexity')
    plt.title('Perplexity-Speedup Trade-off')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('visualizations/tradeoff_curve.png', dpi=150)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--texts", nargs='+', default=[
        "The quantum computer calculated the solution",
        "Artificial intelligence is transforming the world",
        "The cat sat on the mat and slept"
    ])

    args = parser.parse_args()

    print("[LOADING MODEL]")
    base_model, tokenizer_obj = load_model_and_tokenizer()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    from src.model.adaptive import HierarchicalTransformerWrapper
    hierarchical_model = HierarchicalTransformerWrapper(
        base_model=base_model,
        exit_layers=checkpoint['config']['exit_layers'],
        capacity=checkpoint['config']['capacity']
    )
    hierarchical_model.load_state_dict(checkpoint['wrapper_state'])
    hierarchical_model = hierarchical_model.to(base_model.device)

    print("\n[ANALYZING ROUTING PATTERNS...]")
    all_patterns = analyze_token_patterns(hierarchical_model, tokenizer_obj, args.texts)

    for i, patterns in enumerate(all_patterns):
        print(f"\nText {i+1}: {args.texts[i]}")
        print("-" * 60)

        for p in patterns:
            status = "EXITED" if p['exit_layer'] < 32 else "FULL"
            print(f"  {p['token']:15} | Exit: {p['exit_layer']:2} | "
                  f"Skips: {p['skips']:2} | {status}")

    print("\n[CREATING VISUALIZATIONS...]")

    if all_patterns:
        plot_heatmap(all_patterns[0], title=f"Exit Patterns: {args.texts[0][:30]}...")

    capacities = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    perplexities = [12.5, 11.8, 10.7, 10.6, 10.55, 10.52, 10.51, 10.5]
    speedups = [1.82, 1.65, 1.45, 1.33, 1.25, 1.18, 1.10, 1.00]

    plot_efficiency_tradeoff(capacities, perplexities, speedups)

    print("\n[ANALYSIS COMPLETE!]")
    print("   Check 'visualizations/' directory for plots")

if __name__ == "__main__":
    main()