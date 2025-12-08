#!/usr/bin/env python3
"""
OPTIMIZED TRAINING SCRIPT FOR HIERARCHICAL ADAPTIVE TRANSFORMER

Features:
- Two-phase training (routers → exit gates)
- Progressive layer dropout
- Capacity scheduling
- Improved exit timing loss
- Multi-GPU support via Accelerate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import argparse
import sys
import os
import math
from accelerate import Accelerator

sys.path.append('.')
from src.model.load import load_model_and_tokenizer
from src.model.adaptive import HierarchicalTransformerWrapper
from src.config import get_optimal_config, get_capacity_at_step


def get_lr(step, warmup_steps, total_steps, base_lr):
    """
    SIMPLE CONSTANT LR - no warmup, no decay.
    Just works.
    """
    return base_lr


def set_lr(optimizer, lr):
    """Set learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def prepare_wikitext_dataset(tokenizer, config):
    """Prepare Wikitext-2 dataset"""
    dataset = load_dataset(
        config['dataset']['name'],
        config['dataset']['version']
    )

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=config['training']['max_seq_length'],
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        num_proc=4,
    )
    
    # Filter out very short sequences
    tokenized = tokenized.filter(lambda x: len(x['input_ids']) > 10)
    
    # Add labels
    tokenized = tokenized.map(
        lambda examples: {'labels': examples['input_ids']},
        batched=True,
    )
    
    return tokenized


def create_dataloader(dataset, tokenizer, batch_size, split='train'):
    """Create DataLoader with dynamic padding"""
    
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
            'labels': torch.tensor(labels),
        }
    
    return DataLoader(
        dataset[split],
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )


def compute_exit_timing_loss(
    exit_layer: torch.Tensor,
    input_ids: torch.Tensor,
    num_layers: int = 22,
    config: dict = None,
) -> torch.Tensor:
    """
    Improved exit timing loss with three components:
    1. Early exit encouragement for common tokens
    2. Diversity (prevent all tokens exiting at same layer)
    3. Position monotonicity (later positions exit later - soft)
    """
    device = exit_layer.device
    exited_mask = exit_layer >= 0
    
    if not exited_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    if config is None:
        config = {
            'exit_early_weight': 0.03,
            'exit_diversity_weight': 0.01,
            'exit_monotonicity_weight': 0.005,
        }
    
    exit_layers_float = exit_layer[exited_mask].float()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # 1. Early exit encouragement for common tokens
    common_mask = input_ids < 1000  # Common tokens have low IDs
    common_exited = common_mask & exited_mask
    
    if common_exited.any():
        common_exits = exit_layer[common_exited].float()
        # Target: common tokens should exit at ~30% depth
        target_depth = 0.3 * num_layers
        early_loss = F.smooth_l1_loss(
            common_exits,
            torch.full_like(common_exits, target_depth)
        )
        total_loss = total_loss + config['exit_early_weight'] * early_loss
    
    # 2. Diversity loss (prevent clustering at single layer)
    if exit_layers_float.numel() > 1:
        exit_std = exit_layers_float.std()
        # Encourage std of at least 2 layers
        diversity_loss = F.relu(2.0 - exit_std)
        total_loss = total_loss + config['exit_diversity_weight'] * diversity_loss
    
    # 3. Position monotonicity (weak preference for later positions to exit later)
    batch_size, seq_len = exit_layer.shape
    position_ids = torch.arange(seq_len, device=device).expand(batch_size, -1)
    
    valid_exits = exit_layer[exited_mask].float()
    valid_positions = position_ids[exited_mask].float()
    
    if valid_exits.numel() > 1 and valid_positions.std() > 0:
        # Normalize and compute correlation
        exit_norm = (valid_exits - valid_exits.mean()) / (valid_exits.std() + 1e-6)
        pos_norm = (valid_positions - valid_positions.mean()) / (valid_positions.std() + 1e-6)
        correlation = (exit_norm * pos_norm).mean()
        
        # Penalize negative correlation
        monotonicity_loss = F.relu(-correlation)
        total_loss = total_loss + config['exit_monotonicity_weight'] * monotonicity_loss
    
    return total_loss


def train_phase_routers(model, dataloader, config, accelerator):
    """
    Phase 1: Train routers only (exit gates disabled).
    """
    training_config = config['training']
    
    # Freeze everything except routers
    for param in model.parameters():
        param.requires_grad = False
    
    for router in model.skip_routers:
        for param in router.parameters():
            param.requires_grad = True
    
    # Disable exit gates
    model.disable_exit_gates = True
    model.use_layer_dropout = training_config.get('use_layer_dropout', False)

    # Prepare model and dataloader with accelerator FIRST
    # This is important because accelerator.prepare() may change dataloader length
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # Now calculate training steps with the correct dataloader length
    num_epochs = training_config['router_epochs']
    num_training_steps = len(dataloader) * num_epochs
    warmup_steps = int(num_training_steps * training_config['router_warmup_ratio'])
    
    if accelerator.is_main_process:
        print(f"  Training steps: {num_training_steps}, warmup: {warmup_steps}")
    
    # Optimizer (we use manual LR scheduling, so set initial LR to 0)
    base_lr = training_config['router_lr']
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,  # Will be overwritten by manual scheduling
        weight_decay=training_config['router_weight_decay'],
    )
    
    # Prepare optimizer only
    optimizer = accelerator.prepare(optimizer)
    
    # Debug info
    if accelerator.is_main_process:
        print(f"  ✅ Using MANUAL LR scheduling (no HuggingFace scheduler)")
        print(f"  Base LR: {base_lr:.2e}, warmup steps: {warmup_steps}")
        # Test the LR function
        test_lrs = [get_lr(s, warmup_steps, num_training_steps, base_lr) for s in [0, 100, warmup_steps, num_training_steps//2]]
        print(f"  LR at step 0: {test_lrs[0]:.2e}, step 100: {test_lrs[1]:.2e}, step {warmup_steps}: {test_lrs[2]:.2e}")

    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_aux_loss = 0.0
        
        pbar = tqdm(
            dataloader,
            desc=f"Router Training Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_main_process
        )
        
        for batch in pbar:
            # === MANUAL LR UPDATE ===
            current_lr = get_lr(global_step, warmup_steps, num_training_steps, base_lr)
            set_lr(optimizer, current_lr)
            
            # Calculate progress for scheduling
            progress = global_step / num_training_steps
            
            # Update capacity based on schedule
            if training_config.get('use_capacity_schedule', False):
                current_capacity = get_capacity_at_step(global_step, num_training_steps, training_config)
                accelerator.unwrap_model(model).set_capacity(current_capacity)
            
            # Forward pass
            outputs = model(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                training=True,
                progress=progress,
            )
            
            # Language modeling loss
            logits = outputs['logits']
            labels = batch['labels']
            
            # Cast logits to float32 for stable loss computation
            logits_fp32 = logits.float()
            
            lm_loss = F.cross_entropy(
                logits_fp32.view(-1, logits_fp32.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                label_smoothing=training_config.get('label_smoothing', 0.0),
            )
            
            # Auxiliary losses from routers
            aux_losses = outputs['aux_losses']
            total_aux_loss = sum(l for l in aux_losses if l is not None)
            
            # Check if aux_loss is valid (has gradients and not NaN)
            aux_valid = (
                isinstance(total_aux_loss, torch.Tensor) and 
                not torch.isnan(total_aux_loss) and 
                not torch.isinf(total_aux_loss)
            )
            
            # Total loss - only add aux if valid to preserve gradient chain
            if aux_valid:
                total_loss = lm_loss + total_aux_loss
            else:
                total_loss = lm_loss
                total_aux_loss = torch.tensor(0.0)  # For logging only
            
            # Skip batch if NaN detected - with detailed debugging
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                if accelerator.is_main_process and global_step < 10:
                    lm_nan = torch.isnan(lm_loss) or torch.isinf(lm_loss)
                    aux_nan = isinstance(total_aux_loss, torch.Tensor) and (torch.isnan(total_aux_loss) or torch.isinf(total_aux_loss))
                    tqdm.write(f"⚠️ NaN at step {global_step}: lm_loss={'NaN' if lm_nan else f'{lm_loss.item():.2f}'}, aux={'NaN' if aux_nan else 'OK'}")
                optimizer.zero_grad()
                global_step += 1  # Still increment step even when skipping
                continue
            
            # Backward pass
            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(
                model.parameters(),
                training_config['gradient_clip']
            )
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging
            epoch_loss += lm_loss.item()
            if isinstance(total_aux_loss, torch.Tensor):
                epoch_aux_loss += total_aux_loss.item()
            
            if accelerator.is_main_process:
                metrics = outputs['efficiency_metrics']
                pbar.set_postfix({
                    'loss': f"{lm_loss.item():.4f}",
                    'aux': f"{total_aux_loss.item() if isinstance(total_aux_loss, torch.Tensor) else 0:.4f}",
                    'speed': f"{metrics['speedup']:.2f}x",
                    'skip': f"{metrics['avg_skip_rate']:.0%}",
                    'lr': f"{current_lr:.1e}",
                })
                
                # Print loss every 50 steps using print() for Kaggle
                if global_step % 50 == 0:
                    print(f"Step {global_step:5d} | Loss: {lm_loss.item():.4f} | LR: {current_lr:.2e}", flush=True)
            
            global_step += 1
        
        if accelerator.is_main_process:
            avg_loss = epoch_loss / len(dataloader)
            avg_aux = epoch_aux_loss / len(dataloader)
            # Recalculate current_lr with the final global_step value
            final_lr = get_lr(global_step - 1, warmup_steps, num_training_steps, base_lr)
            # Use scientific notation for small values
            loss_str = f"{avg_loss:.4f}" if avg_loss >= 0.0001 else f"{avg_loss:.2e}"
            aux_str = f"{avg_aux:.4f}" if avg_aux >= 0.0001 else f"{avg_aux:.2e}"
            print(f"  Epoch {epoch+1} [step {global_step}] avg loss: {loss_str}, avg aux: {aux_str}, lr: {final_lr:.2e}")

    return accelerator.unwrap_model(model)


def train_phase_exit(model, dataloader, config, accelerator):
    """
    Phase 2: Train exit gates only (routers frozen).
    """
    training_config = config['training']

    # Freeze everything except exit gates
    for param in model.parameters():
        param.requires_grad = False
    
    for gate in model.exit_gates:
        if gate is not None:
            for param in gate.parameters():
                param.requires_grad = True
    
    # Enable exit gates
    model.disable_exit_gates = False

    # Prepare model and dataloader FIRST (important for correct step count)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    num_epochs = training_config['exit_epochs']
    num_training_steps = len(dataloader) * num_epochs
    warmup_steps = int(num_training_steps * training_config['exit_warmup_ratio'])
    
    if accelerator.is_main_process:
        print(f"  Training steps: {num_training_steps}, warmup: {warmup_steps}")

    # Optimizer with manual LR scheduling
    base_lr = training_config['exit_lr']
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=training_config['exit_weight_decay'],
    )
    
    optimizer = accelerator.prepare(optimizer)
    
    if accelerator.is_main_process:
        print(f"  ✅ Using MANUAL LR scheduling, base LR: {base_lr:.2e}")
    
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        pbar = tqdm(
            dataloader,
            desc=f"Exit Gate Training Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_main_process
        )
        
        for batch in pbar:
            # === MANUAL LR UPDATE ===
            current_lr = get_lr(global_step, warmup_steps, num_training_steps, base_lr)
            set_lr(optimizer, current_lr)
            
            progress = global_step / num_training_steps
            
            outputs = model(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                training=True,
                progress=progress,
            )
            
            # Language modeling loss (cast to float32 for stability)
            logits_fp32 = outputs['logits'].float()
            lm_loss = F.cross_entropy(
                logits_fp32.view(-1, logits_fp32.size(-1)),
                batch['labels'].view(-1),
                ignore_index=-100,
                label_smoothing=training_config.get('label_smoothing', 0.0),
            )

            # Exit timing loss
            exit_layer = outputs['token_states']['exit_layer']
            exit_loss = compute_exit_timing_loss(
                exit_layer,
                batch['input_ids'],
                num_layers=config['num_layers'],
                config=training_config,
            )
            
            # Combined loss
            total_loss = lm_loss + exit_loss
            
            # Skip batch if NaN detected
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                optimizer.zero_grad()
                continue
            
            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(
                model.parameters(),
                training_config['gradient_clip']
            )
            
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.is_main_process:
                metrics = outputs['efficiency_metrics']
                pbar.set_postfix({
                    'loss': f"{lm_loss.item():.4f}",
                    'exit': f"{exit_loss.item():.4f}",
                    'rate': f"{metrics['exit_rate']:.0%}",
                    'speed': f"{metrics['speedup']:.2f}x",
                    'lr': f"{current_lr:.1e}",
                })
            
            global_step += 1

    return accelerator.unwrap_model(model)


def evaluate_model(model, dataloader, accelerator):
    """Quick evaluation to get perplexity and efficiency"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            outputs = model(
                batch['input_ids'].to(accelerator.device),
                attention_mask=batch['attention_mask'].to(accelerator.device),
                training=False,
            )
            
            # Compute loss
            shift_logits = outputs['logits'][..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].to(accelerator.device).contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='sum',
            )
            
            total_loss += loss.item()
            total_tokens += (shift_labels != -100).sum().item()
            all_metrics.append(outputs['efficiency_metrics'])
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    # Average efficiency metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return perplexity, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Hierarchical Adaptive Transformer")
    parser.add_argument("--phase", choices=["routers", "exit", "full"], default="full",
                       help="Training phase: routers only, exit gates only, or full")
    parser.add_argument("--save", type=str, default="checkpoints/hierarchical_optimized.pth",
                       help="Path to save checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation (requires --resume)")
    args = parser.parse_args()

    accelerator = Accelerator()
    config = get_optimal_config()
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("HIERARCHICAL ADAPTIVE TRANSFORMER - OPTIMIZED TRAINING")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Exit layers: {config['exit_layers']}")
        print(f"  Target capacity: {config['capacity']*100:.0f}%")
        print(f"  Router epochs: {config['training']['router_epochs']}")
        print(f"  Exit epochs: {config['training']['exit_epochs']}")
        print(f"  Target speedup: {config['expected']['target_speedup']:.1f}x")
        print()
    
    # Load base model
    base_model, tokenizer = load_model_and_tokenizer()
    
    # Create or load hierarchical model
    if args.resume:
        if accelerator.is_main_process:
            print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        hierarchical_model = HierarchicalTransformerWrapper(
            base_model=base_model,
            exit_layers=checkpoint['config']['exit_layers'],
            capacity=checkpoint['config']['capacity'],
            use_layer_dropout=config['training'].get('use_layer_dropout', False),
        )
        hierarchical_model.load_state_dict(checkpoint['wrapper_state'])
    else:
        hierarchical_model = HierarchicalTransformerWrapper(
            base_model=base_model,
            exit_layers=config['exit_layers'],
            capacity=config['capacity'],
            use_layer_dropout=config['training'].get('use_layer_dropout', False),
        )
    
    if accelerator.is_main_process:
        print(f"Trainable parameters: {hierarchical_model.count_trainable_params():,}")
    
    # Prepare dataset
    dataset = prepare_wikitext_dataset(tokenizer, config)
    train_dataloader = create_dataloader(
        dataset, tokenizer,
        config['training']['batch_size'],
        split='train'
    )
    val_dataloader = create_dataloader(
        dataset, tokenizer,
        config['training']['batch_size'],
        split='validation'
    )
    
    # Evaluation only mode
    if args.eval_only:
        if not args.resume:
            print("Error: --eval-only requires --resume")
            return
        
        hierarchical_model = hierarchical_model.to(accelerator.device)
        perplexity, metrics = evaluate_model(hierarchical_model, val_dataloader, accelerator)
        
        if accelerator.is_main_process:
            print(f"\nEvaluation Results:")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Speedup: {metrics['speedup']:.2f}x")
            print(f"  Exit rate: {metrics['exit_rate']:.1%}")
            print(f"  Skip rate: {metrics['avg_skip_rate']:.1%}")
        return

    # Training
    if args.phase == "routers" or args.phase == "full":
        if accelerator.is_main_process:
            print("\n" + "=" * 40)
            print("PHASE 1: TRAINING ROUTERS")
            print("=" * 40)
        hierarchical_model = train_phase_routers(
            hierarchical_model, train_dataloader, config, accelerator
        )

    if args.phase == "exit" or args.phase == "full":
        if accelerator.is_main_process:
            print("\n" + "=" * 40)
            print("PHASE 2: TRAINING EXIT GATES")
            print("=" * 40)
        hierarchical_model = train_phase_exit(
            hierarchical_model, train_dataloader, config, accelerator
        )

    # Final evaluation
    if accelerator.is_main_process:
        print("\n" + "=" * 40)
        print("FINAL EVALUATION")
        print("=" * 40)
    
    hierarchical_model = hierarchical_model.to(accelerator.device)
    perplexity, metrics = evaluate_model(hierarchical_model, val_dataloader, accelerator)
    
    if accelerator.is_main_process:
        print(f"\nResults:")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Speedup: {metrics['speedup']:.2f}x")
        print(f"  Compute fraction: {metrics['compute_fraction']:.1%}")
        print(f"  Exit rate: {metrics['exit_rate']:.1%}")
        print(f"  Avg skip rate: {metrics['avg_skip_rate']:.1%}")
        
        # Check against targets
        expected = config['expected']
        print(f"\nTarget Comparison:")
        speedup_ok = metrics['speedup'] >= expected['min_speedup']
        print(f"  Speedup: {metrics['speedup']:.2f}x {'✅' if speedup_ok else '❌'} (target: ≥{expected['min_speedup']}x)")

    # Save checkpoint
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        torch.save({
            'wrapper_state': hierarchical_model.state_dict(),
            'config': {
                'exit_layers': hierarchical_model.exit_layers,
                'capacity': hierarchical_model.capacity,
                'num_layers': hierarchical_model.num_layers,
            },
            'metrics': {
                'perplexity': perplexity,
                'efficiency': metrics,
            },
        }, args.save)
        print(f"\n✅ Model saved to {args.save}")


if __name__ == "__main__":
    main()