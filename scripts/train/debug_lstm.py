"""Debug script to systematically identify LSTM plateau issue.

Run tests incrementally to isolate the problem:
1. Data sanity checks
2. Loss calculation verification
3. Gradient flow analysis
4. Training dynamics comparison
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add repo root to path for debugging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.train.datasets import (
    TokenDataset,
    EvalDataset,
    collate_fn,
    MAX_TOKENS,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    VOCAB_SIZE,
)
from scripts.train.models import LSTM, LSTMConfig
from transformers import GPT2Config, GPT2LMHeadModel


def test_1_data_statistics():
    """Test 1: Verify token distribution and check for data issues."""
    print("\n" + "=" * 70)
    print("TEST 1: DATA STATISTICS")
    print("=" * 70)

    train_dir = "/scratch2/ddager/rapp/tokens/chunks30_spidr_base/"
    eval_dir = "/scratch2/ddager/rapp/tokens/chunks-eval_spidr_base/"

    print(f"\nLoading datasets...")
    train_dataset = TokenDataset(train_dir)
    eval_dataset = EvalDataset(eval_dir)

    # Sample some batches
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)

    print(f"\nAnalyzing token distributions from first 100 batches...")
    all_tokens = []
    for i, batch in enumerate(train_loader):
        if i >= 100:
            break
        all_tokens.extend(batch["input_ids"].flatten().tolist())

    all_tokens = torch.tensor(all_tokens)

    print(f"Total tokens sampled: {len(all_tokens)}")
    print(f"Unique tokens: {torch.unique(all_tokens).numel()} (expected: {VOCAB_SIZE})")
    print(f"Token range: [{all_tokens.min()}, {all_tokens.max()}]")
    print(f"BOS occurrences: {(all_tokens == BOS_TOKEN_ID).sum()}")
    print(f"EOS occurrences: {(all_tokens == EOS_TOKEN_ID).sum()}")

    # Check token frequency distribution
    token_counts = torch.bincount(all_tokens, minlength=VOCAB_SIZE)
    top_tokens = torch.topk(token_counts, k=10)

    print(f"\nTop 10 most frequent tokens:")
    for idx, (count, token_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
        print(
            f"  {idx+1}. Token {token_id.item()}: {count.item()} times ({100*count.item()/len(all_tokens):.2f}%)"
        )

    # Check for any tokens that never appear
    missing_tokens = (token_counts == 0).sum()
    print(f"\nTokens that never appear: {missing_tokens.item()}/{VOCAB_SIZE}")

    # Calculate theoretical minimum loss (entropy)
    probs = token_counts.float() / token_counts.sum()
    probs = probs[probs > 0]  # Remove zeros for log
    entropy = -(probs * probs.log()).sum()
    print(f"\nTheoretical minimum loss (empirical entropy): {entropy:.4f}")
    print(
        f"Random baseline loss (uniform distribution): {torch.log(torch.tensor(VOCAB_SIZE)):.4f}"
    )


def test_2_loss_calculation():
    """Test 2: Verify loss calculation is correct for both models."""
    print("\n" + "=" * 70)
    print("TEST 2: LOSS CALCULATION VERIFICATION")
    print("=" * 70)

    # Create small synthetic batch
    batch_size = 4
    seq_len = 128

    # Create input with known distribution
    input_ids = torch.randint(0, 256, (batch_size, seq_len))

    print(f"\nTesting with synthetic batch: {input_ids.shape}")

    # Test LSTM (using paper's original parameters)
    lstm_config = LSTMConfig(
        vocab_size=VOCAB_SIZE,
        embedding_dim=200,  # Paper default
        hidden_size=1024,  # Paper default
        num_layers=3,  # Paper default
        dropout=0.1,  # Paper default
    )
    lstm_model = LSTM(lstm_config)

    lstm_output = lstm_model(input_ids=input_ids, labels=input_ids)
    print(f"\nLSTM Loss: {lstm_output['loss'].item():.4f}")
    print(f"LSTM Logits shape: {lstm_output['logits'].shape}")

    # Test GPT2
    gpt2_config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=seq_len,
        n_ctx=seq_len,
        n_embd=256,
        n_layer=2,
        n_head=4,
    )
    gpt2_model = GPT2LMHeadModel(gpt2_config)

    gpt2_output = gpt2_model(input_ids=input_ids, labels=input_ids)
    print(f"\nGPT2 Loss: {gpt2_output.loss.item():.4f}")
    print(f"GPT2 Logits shape: {gpt2_output.logits.shape}")

    # Manual loss calculation to verify
    lstm_logits_shifted = lstm_output["logits"][:, :-1, :].contiguous()
    labels_shifted = input_ids[:, 1:].contiguous()

    manual_loss = F.cross_entropy(
        lstm_logits_shifted.view(-1, VOCAB_SIZE), labels_shifted.view(-1)
    )
    print(f"\nManual LSTM loss calculation: {manual_loss.item():.4f}")
    print(
        f"Matches model output: {torch.allclose(lstm_output['loss'], manual_loss, atol=1e-4)}"
    )

    # Check if initialized logits are reasonable (should be roughly uniform)
    with torch.no_grad():
        lstm_probs = F.softmax(lstm_output["logits"][0, 0], dim=-1)
        gpt2_probs = F.softmax(gpt2_output.logits[0, 0], dim=-1)

        print(f"\nInitialized logit distributions (should be relatively flat):")
        print(
            f"LSTM: max prob = {lstm_probs.max():.4f}, entropy = {-(lstm_probs * lstm_probs.log()).sum():.4f}"
        )
        print(
            f"GPT2: max prob = {gpt2_probs.max():.4f}, entropy = {-(gpt2_probs * gpt2_probs.log()).sum():.4f}"
        )


def test_3_gradient_flow():
    """Test 3: Check if gradients are flowing properly through LSTM."""
    print("\n" + "=" * 70)
    print("TEST 3: GRADIENT FLOW ANALYSIS")
    print("=" * 70)

    # Create small model and data
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 256, (batch_size, seq_len))

    # Using paper's original parameters
    lstm_config = LSTMConfig(
        vocab_size=VOCAB_SIZE,
        embedding_dim=200,  # Paper default
        hidden_size=1024,  # Paper default
        num_layers=3,  # Paper default
        dropout=0.1,  # Paper default
    )
    lstm_model = LSTM(lstm_config)

    # Forward + backward pass
    output = lstm_model(input_ids=input_ids, labels=input_ids)
    loss = output["loss"]
    loss.backward()

    # Check gradient statistics for each layer
    print(f"\nGradient statistics by layer:")
    print(f"{'Layer':<30} {'Mean':>12} {'Std':>12} {'Max':>12} {'% Zero':>12}")
    print("-" * 70)

    for name, param in lstm_model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            mean = grad.abs().mean().item()
            std = grad.std().item()
            max_val = grad.abs().max().item()
            pct_zero = (grad.abs() < 1e-8).float().mean().item() * 100
            print(
                f"{name:<30} {mean:>12.6f} {std:>12.6f} {max_val:>12.6f} {pct_zero:>11.2f}%"
            )
        else:
            print(f"{name:<30} {'NO GRADIENT'}")

    # Check for vanishing/exploding gradients
    all_grads = []
    for param in lstm_model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.flatten())

    if all_grads:
        all_grads = torch.cat(all_grads)
        grad_norm = all_grads.norm().item()
        print(f"\nOverall gradient norm: {grad_norm:.6f}")

        if grad_norm < 1e-4:
            print("WARNING: Gradients are very small (vanishing gradient problem)")
        elif grad_norm > 1e3:
            print("WARNING: Gradients are very large (exploding gradient problem)")
        else:
            print("✓ Gradients appear to be in a reasonable range")


def test_4_training_dynamics():
    """Test 4: Mini grid search with multiple random seeds."""
    print("\n" + "=" * 70)
    print("TEST 4: MINI GRID SEARCH (3 seeds per config)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load real data
    train_dir = "/scratch2/ddager/rapp/tokens/chunks30_spidr_base/"
    train_dataset = TokenDataset(train_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, collate_fn=collate_fn
    )

    # Create LSTM config
    lstm_config = LSTMConfig(
        vocab_size=VOCAB_SIZE,
        embedding_dim=256,
        hidden_size=256,
        num_layers=2,
        dropout=0.0,
    )
    n_params = sum(
        p.numel() for p in torch.nn.ModuleList([LSTM(lstm_config)]).parameters()
    )

    print(
        f"\nModel config: {lstm_config.embedding_dim}/{lstm_config.hidden_size}/{lstm_config.num_layers}"
    )
    print(f"Parameters: {n_params/1e6:.1f}M")
    print(f"Running 3 random seeds per configuration for robustness\n")

    # Grid search configs - focus on 7e-3 to 1.2e-2 range (the sweet spot)
    configs = [
        # Previous winners
        {"name": "Adam lr=7e-3, clip=5", "lr": 7e-3, "clip": 5.0},
        {"name": "Adam lr=1e-2, clip=5", "lr": 1e-2, "clip": 5.0},
        # Explore around 7e-3 to 1e-2
        {"name": "Adam lr=8e-3, clip=5", "lr": 8e-3, "clip": 5.0},
        {"name": "Adam lr=9e-3, clip=5", "lr": 9e-3, "clip": 5.0},
        # Explore higher LRs
        {"name": "Adam lr=1.2e-2, clip=5", "lr": 1.2e-2, "clip": 5.0},
        {"name": "Adam lr=1.5e-2, clip=5", "lr": 1.5e-2, "clip": 5.0},
        # Try different clipping with best LR
        {"name": "Adam lr=1e-2, clip=3", "lr": 1e-2, "clip": 3.0},
        {"name": "Adam lr=1e-2, clip=7", "lr": 1e-2, "clip": 7.0},
        # Try different Adam betas (default is beta2=0.999)
        {
            "name": "Adam lr=1e-2, beta2=0.99, clip=5",
            "lr": 1e-2,
            "beta2": 0.99,
            "clip": 5.0,
        },
        {
            "name": "Adam lr=1e-2, beta2=0.98, clip=5",
            "lr": 1e-2,
            "beta2": 0.98,
            "clip": 5.0,
        },
    ]

    # Store all results for summary
    all_results = []

    for config in configs:
        name = config["name"]
        lr = config["lr"]
        clip = config["clip"]
        beta2 = config.get("beta2", 0.999)  # Default Adam beta2

        print(f"\n{'='*70}")
        print(f"CONFIG: {name}")
        print(f"{'='*70}")

        seed_results = []

        for seed in range(3):
            print(f"\n  [Seed {seed+1}/3]")

            # Set seed for reproducibility
            torch.manual_seed(42 + seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42 + seed)

            # Create fresh model for this seed
            lstm_model = LSTM(lstm_config).to(device)
            optimizer = torch.optim.Adam(
                lstm_model.parameters(), lr=lr, betas=(0.9, beta2)
            )

            losses = []
            for i, batch in enumerate(train_loader):
                if i >= 100:
                    break

                input_ids = batch["input_ids"].to(device)

                optimizer.zero_grad()
                output = lstm_model(input_ids=input_ids, labels=input_ids)
                loss = output["loss"]
                loss.backward()

                # Gradient clipping
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), clip)

                optimizer.step()
                losses.append(loss.item())

                if i % 20 == 0:
                    print(f"    Step {i:3d}: {loss.item():.4f}", end="")
                    if i == 40:
                        print(f"  ← @40: {loss.item():.4f}")
                    else:
                        print()

            seed_results.append(
                {
                    "initial": losses[0],
                    "step_40": losses[40],
                    "step_100": losses[-1],
                    "improvement_40": losses[0] - losses[40],
                    "improvement_100": losses[0] - losses[-1],
                }
            )

        # Calculate statistics across seeds
        step_40_losses = [r["step_40"] for r in seed_results]
        step_100_losses = [r["step_100"] for r in seed_results]
        imp_40 = [r["improvement_40"] for r in seed_results]
        imp_100 = [r["improvement_100"] for r in seed_results]

        mean_40 = sum(step_40_losses) / 3
        mean_100 = sum(step_100_losses) / 3
        std_40 = (sum((x - mean_40) ** 2 for x in step_40_losses) / 3) ** 0.5
        std_100 = (sum((x - mean_100) ** 2 for x in step_100_losses) / 3) ** 0.5

        result = {
            "name": name,
            "lr": lr,
            "clip": clip,
            "beta2": beta2,
            "mean_40": mean_40,
            "std_40": std_40,
            "mean_100": mean_100,
            "std_100": std_100,
            "best_40": min(step_40_losses),
            "best_100": min(step_100_losses),
        }
        all_results.append(result)

        print(f"\n  📊 RESULTS ACROSS 3 SEEDS:")
        print(
            f"    Step 40:  {mean_40:.4f} ± {std_40:.4f}  (best: {min(step_40_losses):.4f})"
        )
        print(
            f"    Step 100: {mean_100:.4f} ± {std_100:.4f}  (best: {min(step_100_losses):.4f})"
        )

        if mean_40 <= 1.80:
            print(f"    🎯 EXCELLENT: Mean@40 ≤ 1.80!")
        elif mean_100 <= 1.80:
            print(f"    ✓ GOOD: Mean@100 ≤ 1.80")
        elif mean_100 <= 2.00:
            print(f"    ⚠️  OK: Learning but can improve")
        else:
            print(f"    ❌ SLOW: Needs faster convergence")

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY: All Configurations Ranked")
    print(f"{'='*70}\n")

    # Sort by mean loss at step 40
    all_results.sort(key=lambda x: x["mean_40"])

    print(f"{'Rank':<6}{'Mean@40':<12}{'Std@40':<10}{'Mean@100':<12}{'Config'}")
    print("-" * 70)
    for i, r in enumerate(all_results, 1):
        marker = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(
            f"{marker} {i:<3} {r['mean_40']:>6.4f}     ±{r['std_40']:.4f}    {r['mean_100']:>6.4f}      {r['name']}"
        )

    print(f"\n{'='*70}")
    print(f"RECOMMENDATION:")
    print(f"{'='*70}")
    best = all_results[0]
    print(f"Best config: {best['name']}")
    print(f"  Step 40 loss: {best['mean_40']:.4f} ± {best['std_40']:.4f}")
    print(f"  Step 100 loss: {best['mean_100']:.4f} ± {best['std_100']:.4f}")
    print(f"  LR: {best['lr']}, Clip: {best['clip']}, Beta2: {best['beta2']}")


def test_5_precision_issues():
    """Test 5: Check if BF16 is causing numerical issues with LSTM."""
    print("\n" + "=" * 70)
    print("TEST 5: PRECISION ANALYSIS (BF16 vs FP32)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create synthetic batch
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 256, (batch_size, seq_len)).to(device)

    # Using paper's original parameters
    lstm_config = LSTMConfig(
        vocab_size=VOCAB_SIZE,
        embedding_dim=200,  # Paper default
        hidden_size=1024,  # Paper default
        num_layers=3,  # Paper default
        dropout=0.1,  # Paper default
    )

    # Test FP32
    lstm_fp32 = LSTM(lstm_config).to(device)
    output_fp32 = lstm_fp32(input_ids=input_ids, labels=input_ids)
    loss_fp32 = output_fp32["loss"]
    loss_fp32.backward()

    grad_norm_fp32 = (
        torch.cat(
            [p.grad.flatten() for p in lstm_fp32.parameters() if p.grad is not None]
        )
        .norm()
        .item()
    )

    print(f"\nFP32:")
    print(f"  Loss: {loss_fp32.item():.6f}")
    print(f"  Gradient norm: {grad_norm_fp32:.6f}")

    # Test BF16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        lstm_bf16 = LSTM(lstm_config).to(device).to(torch.bfloat16)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_bf16 = lstm_bf16(input_ids=input_ids, labels=input_ids)
            loss_bf16 = output_bf16["loss"]

        loss_bf16.backward()

        grad_norm_bf16 = (
            torch.cat(
                [p.grad.flatten() for p in lstm_bf16.parameters() if p.grad is not None]
            )
            .norm()
            .item()
        )

        print(f"\nBF16:")
        print(f"  Loss: {loss_bf16.item():.6f}")
        print(f"  Gradient norm: {grad_norm_bf16:.6f}")

        print(f"\nDifference:")
        print(f"  Loss difference: {abs(loss_fp32.item() - loss_bf16.item()):.6f}")
        print(f"  Grad norm ratio: {grad_norm_bf16 / grad_norm_fp32:.4f}")

        if abs(loss_fp32.item() - loss_bf16.item()) > 0.1:
            print("  WARNING: Significant difference between FP32 and BF16")
    else:
        print("\nWARNING: BF16 not supported on this device, skipping BF16 test")


def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="LSTM Plateau Debugging Suite")
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="Test to run: 0=all, 1=data stats, 2=loss calc, 3=gradients, 4=training, 5=precision",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("LSTM PLATEAU DEBUGGING SUITE")
    print("=" * 70)
    print("\nThis script will run diagnostic tests to identify why LSTM is plateauing:")
    print("  1. Data statistics (token distribution, entropy)")
    print("  2. Loss calculation verification")
    print("  3. Gradient flow analysis")
    print("  4. Training dynamics with different optimizers")
    print("  5. Precision issues (FP32 vs BF16)")

    tests = [
        ("Data Statistics", test_1_data_statistics),
        ("Loss Calculation", test_2_loss_calculation),
        ("Gradient Flow", test_3_gradient_flow),
        ("Training Dynamics", test_4_training_dynamics),
        ("Precision Issues", test_5_precision_issues),
    ]

    choice = args.test

    if choice == 0:
        print("\nRunning ALL tests...\n")
        for name, test_fn in tests:
            try:
                test_fn()
            except Exception as e:
                print(f"\nTest '{name}' failed with error: {e}")
                import traceback

                traceback.print_exc()
    elif 1 <= choice <= len(tests):
        name, test_fn = tests[choice - 1]
        print(f"\nRunning test: {name}\n")
        test_fn()
    else:
        print("Invalid choice")

    print("\n" + "=" * 70)
    print("DEBUGGING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
