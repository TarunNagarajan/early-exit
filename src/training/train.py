#!/usr/bin/env python3
"""
TRAINING SCRIPT FOR HIERARCHICAL ADAPTIVE TRANSFORMER
Optimized for TinyLlama 1.1B with multi-GPU support via Accelerate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
import sys
import os
from accelerate import Accelerator

sys.path.append('.')
from src.model.load import load_model_and_tokenizer
from src.model.adaptive import HierarchicalTransformerWrapper
from src.config import get_optimal_config

def prepare_wikitext_dataset(tokenizer, config):
    """Prepare Wikitext-2 dataset"""
    dataset = load_dataset(config['dataset']['name'], config['dataset']['version'])

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=config['training']['max_seq_length'])

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized = tokenized.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    return tokenized

def create_dataloader(dataset, tokenizer, batch_size, split='train'):
    """Create DataLoader with padding"""
    
    def collate_fn(batch):
        max_len = max(len(item['input_ids']) for item in batch)
        
        input_ids, attention_masks, labels = [], [], []
        
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

def train_phase_routers(model, dataloader, config, accelerator):
    """Phase 1: Train routers only"""
    training_config = config['training']
    
    for name, param in model.named_parameters():
        param.requires_grad = 'skip_router' in name

    model.disable_exit_gates = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_config['router_lr'],
        weight_decay=training_config['weight_decay']
    )
    
    num_training_steps = len(dataloader) * training_config['router_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config['router_warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    model.train()
    global_step = 0

    for epoch in range(training_config['router_epochs']):
        pbar = tqdm(dataloader, desc=f"Router Epoch {epoch+1}/{training_config['router_epochs']}", disable=not accelerator.is_main_process)
        for i, batch in enumerate(pbar):
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], training=True, global_step=global_step)
            
            logits = outputs['logits']
            labels = batch['labels']
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            total_aux_loss = sum(l for l in outputs['aux_losses'] if l is not None)
            total_loss = loss + total_aux_loss
            
            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(model.parameters(), training_config['gradient_clip'])
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if accelerator.is_main_process:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'aux': f"{total_aux_loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.1e}"
                })
            global_step += 1

    return accelerator.unwrap_model(model)

def train_phase_exit(model, dataloader, config, accelerator):
    """Phase 2: Train exit gates only"""
    training_config = config['training']

    for name, param in model.named_parameters():
        param.requires_grad = 'exit_gate' in name

    model.disable_exit_gates = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_config['exit_lr'],
        weight_decay=training_config['weight_decay']
    )
    
    num_training_steps = len(dataloader) * training_config['exit_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config['exit_warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    model.train()

    for epoch in range(training_config['exit_epochs']):
        pbar = tqdm(dataloader, desc=f"Exit Epoch {epoch+1}/{training_config['exit_epochs']}", disable=not accelerator.is_main_process)
        for batch in pbar:
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], training=True)
            
            loss = F.cross_entropy(
                outputs['logits'].view(-1, outputs['logits'].size(-1)),
                batch['labels'].view(-1),
                ignore_index=-100
            )

            exit_layer = outputs['token_states']['exit_layer']
            exit_loss = compute_exit_timing_loss(exit_layer, batch['input_ids'])
            
            total_loss = loss + training_config['exit_timing_weight'] * exit_loss
            
            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(model.parameters(), training_config['gradient_clip'])
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if accelerator.is_main_process:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'exit_loss': f"{exit_loss.item():.4f}",
                    'exit_rate': f"{outputs['efficiency_metrics']['exit_rate']:.2%}"
                })
            
    return accelerator.unwrap_model(model)

def compute_exit_timing_loss(exit_layer, input_ids):
    """Encourage early exit for common tokens"""
    common_mask = input_ids < 1000
    exited_mask = exit_layer >= 0
    if not exited_mask.any():
        return torch.tensor(0.0, device=exit_layer.device)

    common_exited = common_mask & exited_mask
    if common_exited.any():
        exit_layers_common = exit_layer[common_exited].float()
        return F.mse_loss(exit_layers_common, torch.zeros_like(exit_layers_common))
    return torch.tensor(0.0, device=exit_layer.device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["routers", "exit", "full"], default="full")
    parser.add_argument("--save", type=str, default="checkpoints/tinyllama_full.pth")
    args = parser.parse_args()

    accelerator = Accelerator()
    config = get_optimal_config()
    
    base_model, tokenizer = load_model_and_tokenizer()

    hierarchical_model = HierarchicalTransformerWrapper(
        base_model=base_model,
        exit_layers=config['exit_layers'],
        capacity=config['capacity']
    )

    dataset = prepare_wikitext_dataset(tokenizer, config)
    dataloader = create_dataloader(dataset, tokenizer, config['training']['batch_size'])

    if args.phase == "routers" or args.phase == "full":
        if accelerator.is_main_process:
            print("--- PHASE 1: TRAINING ROUTERS ---")
        hierarchical_model = train_phase_routers(hierarchical_model, dataloader, config, accelerator)

    if args.phase == "exit" or args.phase == "full":
        if accelerator.is_main_process:
            print("\n--- PHASE 2: TRAINING EXIT GATES ---")
        hierarchical_model = train_phase_exit(hierarchical_model, dataloader, config, accelerator)

    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        torch.save({
            'wrapper_state': hierarchical_model.state_dict(),
            'config': {
                'exit_layers': hierarchical_model.exit_layers,
                'capacity': hierarchical_model.capacity,
            }
        }, args.save)
        print(f"\n✅ Model saved to {args.save}")

if __name__ == "__main__":
    main()