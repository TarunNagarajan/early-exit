#!/usr/bin/env python3
"""
TRAINING SCRIPT FOR HIERARCHICAL ADAPTIVE TRANSFORMER
Two phases: Routers → Exit gates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import sys
sys.path.append('.')
from src.model.adaptive import Adaptive
from src.model.load import model, tokenizer

def prepare_wikitext_dataset(tokenizer, seq_len=512):
    """Prepare Wikitext-2 dataset"""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=seq_len)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized = tokenized.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    return tokenized

def create_dataloader(dataset, batch_size=2, split='train'):
    """Create DataLoader with padding"""
    
    def collate_fn(batch):
        max_len = max(len(item['input_ids']) for item in batch)
        
        input_ids = []
        attention_masks = []
        labels = []
        
        for item in batch:
            pad_len = max_len - len(item['input_ids'])
            input_ids.append(item['input_ids'] + [tokenizer.pad_token_id] * pad_len)
            attention_masks.append([1] * len(item['input_ids']) + [0] * pad_len)
            labels.append(item.get('labels', item['input_ids']) + [-100] * pad_len)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_masks),
            'labels': torch.tensor(labels)
        }
    
    return DataLoader(
        dataset[split],
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=collate_fn
    )

def train_phase_routers(model, dataloader, epochs=3, lr=1e-3):
    """Phase 1: Train routers only (exit gates disabled)"""

    for name, param in model.named_parameters():
        if 'skip_router' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.disable_exit_gates = True

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_tokens = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids, training=True)

            loss = F.cross_entropy(
                outputs['logits'].view(-1, outputs['logits'].size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += input_ids.numel()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'compute': f"{outputs['efficiency_metrics']['compute_fraction']:.2%}"
            })

        avg_loss = total_loss / len(dataloader)

    return model

def train_phase_exit(model, dataloader, epochs=2, lr=1e-3):
    """Phase 2: Train exit gates only (routers frozen)"""

    for name, param in model.named_parameters():
        if 'exit_gate' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.disable_exit_gates = False

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_tokens = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids, training=True)

            loss = F.cross_entropy(
                outputs['logits'].view(-1, outputs['logits'].size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            # Add exit timing regularization
            exit_layer = outputs['token_states']['exit_layer']
            active_mask = outputs['token_states']['active']

            exit_loss = compute_exit_timing_loss(exit_layer, input_ids, active_mask)

            total_loss_combined = loss + 0.1 * exit_loss

            optimizer.zero_grad()
            total_loss_combined.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += input_ids.numel()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'exit': f"{outputs['efficiency_metrics']['exit_rate']:.2%}"
            })

        avg_loss = total_loss / len(dataloader)

    return model

def compute_exit_timing_loss(exit_layer, input_ids, active_mask):
    """Encourage early exit for common tokens"""
    common_mask = input_ids < 1000
    exited_mask = exit_layer >= 0

    if not exited_mask.any():
        return torch.tensor(0.0, device=exit_layer.device)

    common_exited = common_mask & exited_mask
    if common_exited.any():
        exit_layers_common = exit_layer[common_exited].float()
        target_layers = torch.full_like(exit_layers_common, 4.0)
        loss_common = F.mse_loss(exit_layers_common, target_layers)
    else:
        loss_common = torch.tensor(0.0, device=exit_layer.device)

    return loss_common

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["routers", "exit", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--capacity", type=float, default=0.5)
    parser.add_argument("--save", type=str, default="checkpoints/model.pth")

    args = parser.parse_args()

from src.model.load import load_model_and_tokenizer

    # Load model and tokenizer
    base_model, tokenizer_obj = load_model_and_tokenizer()

from src.model.adaptive import HierarchicalTransformerWrapper

    # Create hierarchical wrapper
    hierarchical_model = HierarchicalTransformerWrapper(
        base_model=base_model,
        capacity=args.capacity,
        disable_exit_gates=(args.phase == "routers")
    )

    # Prepare dataset
    dataset = prepare_wikitext_dataset(tokenizer_obj, seq_len=args.seq_len)
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, split='train')

    # Training
    if args.phase == "routers" or args.phase == "full":
        hierarchical_model = train_phase_routers(hierarchical_model, dataloader, epochs=args.epochs, lr=args.lr)

    if args.phase == "exit" or args.phase == "full":
        hierarchical_model.disable_exit_gates = False
        hierarchical_model = train_phase_exit(hierarchical_model, dataloader, epochs=min(2, args.epochs), lr=args.lr)

    # Save checkpoint
    torch.save({
        'wrapper_state': hierarchical_model.state_dict(),
        'config': {
            'exit_layers': hierarchical_model.exit_layers,
            'capacity': hierarchical_model.capacity,
            'phase': args.phase,
            'epochs': args.epochs
        }
    }, args.save)

if __name__ == "__main__":
    main()