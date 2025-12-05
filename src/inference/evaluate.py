#!/usr/bin/env python3
"""
EVALUATE HIERARCHICAL ADAPTIVE TRANSFORMER
Compute perplexity and efficiency metrics
"""

import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
import argparse
import sys
sys.path.append('.')
from src.model.load import load_model_and_tokenizer

def compute_perplexity(model, dataloader):
    """Compute perplexity on dataset"""
    print("[COMPUTING PERPLEXITY]")
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids, training=False)

            shift_logits = outputs['logits'][..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += (shift_labels != -100).sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss

def evaluate_efficiency(model, dataloader, num_samples=10):
    """Compute efficiency metrics on sample batches"""
    print("[COMPUTING EFFICIENCY METRICS]")
    model.eval()
    metrics_list = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            input_ids = batch['input_ids'].to(model.device)
            outputs = model(input_ids, training=False)

            metrics = outputs['efficiency_metrics']
            metrics_list.append(metrics)

    avg_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        avg_metrics[key] = sum(values) / len(values)

    return avg_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=10)
    
    args = parser.parse_args()
    
    print("[LOADING MODEL]")
    base_model, tokenizer_obj = load_model_and_tokenizer()

    if args.checkpoint:
        print(f"[LOADING CHECKPOINT: {args.checkpoint}]")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        from src.model.adaptive import HierarchicalTransformerWrapper
        hierarchical_model = HierarchicalTransformerWrapper(
            base_model=base_model,
            exit_layers=checkpoint['config']['exit_layers'],
            capacity=checkpoint['config']['capacity']
        )

        hierarchical_model.load_state_dict(checkpoint['wrapper_state'])
        print(f"[CHECKPOINT LOADED: {args.checkpoint}]")
    else:
        from src.model.adaptive import HierarchicalTransformerWrapper
        hierarchical_model = HierarchicalTransformerWrapper(base_model=base_model)

    hierarchical_model = hierarchical_model.to(base_model.device)
    hierarchical_model.eval()

    print("[PREPARING DATASET]")

    test_text = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer_obj(test_text, return_tensors="pt").to(base_model.device)

    print("[RUNNING EVALUATION]")
    with torch.no_grad():
        outputs = hierarchical_model(inputs['input_ids'], training=False)

    efficiency = outputs['efficiency_metrics']

    logits = outputs['logits']
    labels = inputs['input_ids']
    if logits.size(1) > 1:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        perplexity = math.exp(loss.item())
    else:
        perplexity = float('inf')

    print("\n" + "=" * 80)
    print("[EVALUATION RESULTS]")
    print("=" * 80)
    print(f"\n[QUALITY METRICS:]")
    print(f"   Perplexity: {perplexity:.2f}")
    print(f"   Loss: {loss.item():.4f}")

    print(f"\n[EFFICIENCY METRICS:]")
    print(f"   Compute fraction: {efficiency['compute_fraction']:.2%}")
    print(f"   Speedup: {efficiency['speedup']:.2f}x")
    print(f"   Exit rate: {efficiency['exit_rate']:.2%}")
    print(f"   Average skip rate: {efficiency['avg_skip_rate']:.2%}")

    print(f"\n[MODEL INFO:]")
    print(f"   Trainable parameters: {hierarchical_model.count_trainable_params():,}")
    print(f"   Exit layers: {hierarchical_model.exit_layers}")
    print(f"   Capacity: {hierarchical_model.capacity*100:.0f}%")

    return {
        'perplexity': perplexity,
        'loss': loss.item(),
        'efficiency': efficiency,
        'trainable_params': hierarchical_model.count_trainable_params()
    }

if __name__ == "__main__":
    results = main()