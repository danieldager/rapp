"""Debug script to explore variance reduction strategies for LSTM convergence.

Tests different initialization schemes and learning rate schedules to make 
convergence more consistent across random seeds.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.train.datasets import TokenDataset, collate_fn, VOCAB_SIZE
from scripts.train.models import LSTM, LSTMConfig


def custom_init_lstm(lstm, init_scale=1.0, forget_bias=1.0):
    """Custom LSTM initialization with adjustable scale."""
    for name, param in lstm.named_parameters():
        if "weight_ih" in name:
            # Input-hidden: scaled Xavier
            nn.init.xavier_uniform_(param.data, gain=init_scale)
        elif "weight_hh" in name:
            # Hidden-hidden: scaled orthogonal
            nn.init.orthogonal_(param.data, gain=init_scale)
        elif "bias" in name:
            param.data.zero_()
            n = param.size(0)
            param.data[n // 4 : n // 2].fill_(forget_bias)


def get_lr_schedule(optimizer, warmup_steps, max_lr):
    """Create linear warmup schedule."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_training_experiment(config_name, model_fn, optimizer_fn, num_seeds=5):
    """Run training with multiple seeds and return statistics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dir = "/scratch2/ddager/rapp/tokens/chunks30_spidr_base/"
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*70}")
    
    for seed in range(num_seeds):
        torch.manual_seed(42 + seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + seed)
        
        # Create model and optimizer
        model, optimizer, scheduler = model_fn(), None, None
        model = model.to(device)
        optimizer, scheduler = optimizer_fn(model)
        
        # CRITICAL: Create fresh data loader for each seed to ensure same batches!
        train_dataset = TokenDataset(train_dir)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, collate_fn=collate_fn
        )
        
        losses = []
        print(f"  Seed {seed+1}: ", end="", flush=True)
        
        for i, batch in enumerate(train_loader):
            if i >= 100:
                break
            
            input_ids = batch["input_ids"].to(device)
            
            optimizer.zero_grad()
            output = model(input_ids=input_ids, labels=input_ids)
            loss = output["loss"]
            loss.backward()
            
            # Always clip to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            losses.append(loss.item())
            
            if i in [0, 20, 40, 60, 80]:
                print(f"{loss.item():.3f} ", end="", flush=True)
        
        print(f"→ {losses[-1]:.3f}")
        results.append({"losses": losses, "step_40": losses[40], "final": losses[-1]})
    
    # Calculate statistics
    step_40_losses = [r["step_40"] for r in results]
    final_losses = [r["final"] for r in results]
    
    mean_40 = sum(step_40_losses) / num_seeds
    std_40 = (sum((x - mean_40)**2 for x in step_40_losses) / num_seeds) ** 0.5
    mean_final = sum(final_losses) / num_seeds
    std_final = (sum((x - mean_final)**2 for x in final_losses) / num_seeds) ** 0.5
    
    # Check for "stuck" seeds (loss > 5.0 at step 40)
    stuck_seeds = sum(1 for loss in step_40_losses if loss > 5.0)
    
    print(f"\n  📊 RESULTS ({num_seeds} seeds):")
    print(f"    Step 40:  {mean_40:.4f} ± {std_40:.4f}  (stuck: {stuck_seeds}/{num_seeds})")
    print(f"    Step 100: {mean_final:.4f} ± {std_final:.4f}")
    print(f"    Variance reduction: {std_40:.4f} (lower is better)")
    
    return {
        "name": config_name,
        "mean_40": mean_40,
        "std_40": std_40,
        "mean_final": mean_final,
        "std_final": std_final,
        "stuck_seeds": stuck_seeds,
    }


def main():
    print("\n" + "=" * 70)
    print("VARIANCE REDUCTION EXPERIMENTS")
    print("=" * 70)
    print("\nTesting strategies to reduce convergence variance:\n")
    
    lstm_config = LSTMConfig(
        vocab_size=VOCAB_SIZE,
        embedding_dim=256,
        hidden_size=256,
        num_layers=2,
        dropout=0.0,
    )
    
    experiments = []
    
    # ============================================================
    # Experiment 1: Baseline (current setup)
    # ============================================================
    def baseline_model():
        return LSTM(lstm_config)
    
    def baseline_opt(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.99))
        return opt, None
    
    result = run_training_experiment(
        "1. BASELINE: Adam lr=1e-2, beta2=0.99, default init",
        baseline_model,
        baseline_opt,
        num_seeds=5
    )
    experiments.append(result)
    
    # ============================================================
    # Experiment 2: Learning Rate Warmup
    # ============================================================
    def warmup_opt(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.99))
        scheduler = get_lr_schedule(opt, warmup_steps=10, max_lr=1e-2)
        return opt, scheduler
    
    result = run_training_experiment(
        "2. LR WARMUP: 10-step warmup from 0 to 1e-2",
        baseline_model,
        warmup_opt,
        num_seeds=5
    )
    experiments.append(result)
    
    # ============================================================
    # Experiment 3: Smaller initialization scale
    # ============================================================
    def small_init_model():
        model = LSTM(lstm_config)
        custom_init_lstm(model.lstm, init_scale=0.5, forget_bias=1.0)
        return model
    
    result = run_training_experiment(
        "3. SMALL INIT: 0.5x weight scale, forget_bias=1.0",
        small_init_model,
        baseline_opt,
        num_seeds=5
    )
    experiments.append(result)
    
    # ============================================================
    # Experiment 4: Higher forget gate bias
    # ============================================================
    def high_forget_model():
        model = LSTM(lstm_config)
        custom_init_lstm(model.lstm, init_scale=1.0, forget_bias=2.0)
        return model
    
    result = run_training_experiment(
        "4. HIGH FORGET: forget_bias=2.0 (remember more)",
        high_forget_model,
        baseline_opt,
        num_seeds=5
    )
    experiments.append(result)
    
    # ============================================================
    # Experiment 5: Lower LR without warmup
    # ============================================================
    def lower_lr_opt(model):
        opt = torch.optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.99))
        return opt, None
    
    result = run_training_experiment(
        "5. LOWER LR: lr=5e-3 (half of baseline)",
        baseline_model,
        lower_lr_opt,
        num_seeds=5
    )
    experiments.append(result)
    
    # ============================================================
    # Experiment 6: Warmup + Small Init (combo)
    # ============================================================
    result = run_training_experiment(
        "6. COMBO: Warmup + 0.5x init scale",
        small_init_model,
        warmup_opt,
        num_seeds=5
    )
    experiments.append(result)
    
    # ============================================================
    # Experiment 7: Warmup + High Forget Bias (combo)
    # ============================================================
    result = run_training_experiment(
        "7. COMBO: Warmup + forget_bias=2.0",
        high_forget_model,
        warmup_opt,
        num_seeds=5
    )
    experiments.append(result)
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL RANKING: Sorted by Variance (Std@40)")
    print(f"{'='*70}\n")
    
    experiments.sort(key=lambda x: x["std_40"])
    
    print(f"{'Rank':<6}{'Std@40':<10}{'Mean@40':<10}{'Stuck':<8}{'Config'}")
    print("-" * 70)
    for i, exp in enumerate(experiments, 1):
        marker = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{marker} {i:<3} {exp['std_40']:>6.4f}    {exp['mean_40']:>6.4f}    {exp['stuck_seeds']}/5     {exp['name']}")
    
    print(f"\n{'='*70}")
    print("INSIGHTS:")
    print(f"{'='*70}")
    best = experiments[0]
    baseline = [e for e in experiments if "BASELINE" in e["name"]][0]
    
    improvement = ((baseline["std_40"] - best["std_40"]) / baseline["std_40"]) * 100
    print(f"Best strategy: {best['name']}")
    print(f"Variance reduction: {improvement:.1f}% vs baseline")
    print(f"Stuck seeds: {best['stuck_seeds']}/5 (vs {baseline['stuck_seeds']}/5 baseline)")
    print(f"Mean loss @40: {best['mean_40']:.4f} (vs {baseline['mean_40']:.4f} baseline)")


if __name__ == "__main__":
    main()
