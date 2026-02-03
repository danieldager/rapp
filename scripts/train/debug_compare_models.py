"""Compare LSTM vs GPT2 training dynamics to understand why GPT2 converges reliably.

This will help us identify what architectural or training differences make GPT2 robust.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.train.datasets import TokenDataset, collate_fn, VOCAB_SIZE
from scripts.train.models import LSTM, LSTMConfig
from transformers import GPT2Config, GPT2LMHeadModel


def analyze_model_training(
    model_name, model, optimizer, train_loader, device, num_steps=100, num_seeds=5
):
    """Run multiple seeds and analyze training dynamics."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {model_name}")
    print(f"{'='*70}\n")

    all_seed_results = []

    for seed in range(num_seeds):
        # Set seed
        torch.manual_seed(42 + seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + seed)

        # Reset model
        if hasattr(model, "apply"):
            model.apply(
                lambda m: (
                    m.reset_parameters() if hasattr(m, "reset_parameters") else None
                )
            )

        # Recreate optimizer
        if "gpt2" in model_name.lower():
            opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.99))

        # Recreate data loader to ensure same batches
        loader = torch.utils.data.DataLoader(
            TokenDataset("/scratch2/ddager/rapp/tokens/chunks30_spidr_base/"),
            batch_size=32,
            collate_fn=collate_fn,
        )

        losses = []
        grad_norms = []

        print(f"  Seed {seed+1}/{num_seeds}: ", end="", flush=True)

        for i, batch in enumerate(loader):
            if i >= num_steps:
                break

            input_ids = batch["input_ids"].to(device)

            opt.zero_grad()

            if "gpt2" in model_name.lower():
                output = model(input_ids=input_ids, labels=input_ids)
                loss = output.loss
            else:
                output = model(input_ids=input_ids, labels=input_ids)
                loss = output["loss"]

            loss.backward()

            # Measure gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            opt.step()
            losses.append(loss.item())

            if i in [0, 20, 40, 60, 80]:
                print(f"{loss.item():.3f} ", end="", flush=True)

        print(f"→ {losses[-1]:.3f}")

        all_seed_results.append(
            {
                "losses": losses,
                "grad_norms": grad_norms,
                "step_40": losses[40] if len(losses) > 40 else losses[-1],
                "final": losses[-1],
                "plateaued": losses[40] > 5.0 if len(losses) > 40 else losses[-1] > 5.0,
            }
        )

    # Analyze results
    step_40_losses = [r["step_40"] for r in all_seed_results]
    final_losses = [r["final"] for r in all_seed_results]
    plateaued_count = sum(1 for r in all_seed_results if r["plateaued"])

    mean_40 = sum(step_40_losses) / num_seeds
    std_40 = (sum((x - mean_40) ** 2 for x in step_40_losses) / num_seeds) ** 0.5
    mean_final = sum(final_losses) / num_seeds
    std_final = (sum((x - mean_final) ** 2 for x in final_losses) / num_seeds) ** 0.5

    # Gradient analysis
    avg_grad_norms_early = [sum(r["grad_norms"][:10]) / 10 for r in all_seed_results]
    avg_grad_norms_late = [sum(r["grad_norms"][-10:]) / 10 for r in all_seed_results]

    print(f"\n  📊 STATISTICS:")
    print(f"    Step 40:  {mean_40:.4f} ± {std_40:.4f}")
    print(f"    Step 100: {mean_final:.4f} ± {std_final:.4f}")
    print(
        f"    Plateaued seeds: {plateaued_count}/{num_seeds} ({100*plateaued_count/num_seeds:.0f}%)"
    )
    print(
        f"    Grad norm (early): {sum(avg_grad_norms_early)/num_seeds:.3f} ± {(sum((x-sum(avg_grad_norms_early)/num_seeds)**2 for x in avg_grad_norms_early)/num_seeds)**0.5:.3f}"
    )
    print(
        f"    Grad norm (late):  {sum(avg_grad_norms_late)/num_seeds:.3f} ± {(sum((x-sum(avg_grad_norms_late)/num_seeds)**2 for x in avg_grad_norms_late)/num_seeds)**0.5:.3f}"
    )

    return {
        "model_name": model_name,
        "mean_40": mean_40,
        "std_40": std_40,
        "mean_final": mean_final,
        "std_final": std_final,
        "plateaued_pct": 100 * plateaued_count / num_seeds,
    }


def main():
    print("\n" + "=" * 70)
    print("LSTM vs GPT2: Why Does GPT2 Converge Reliably?")
    print("=" * 70)
    print("\nComparing architectural and training differences...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Create data loader
    train_dir = "/scratch2/ddager/rapp/tokens/chunks30_spidr_base/"
    train_dataset = TokenDataset(train_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, collate_fn=collate_fn
    )

    results = []

    # ============================================================
    # Test 1: LSTM (current best config)
    # ============================================================
    lstm_config = LSTMConfig(
        vocab_size=VOCAB_SIZE,
        embedding_dim=256,
        hidden_size=256,
        num_layers=2,
        dropout=0.0,
    )
    lstm_model = LSTM(lstm_config).to(device)
    lstm_opt = torch.optim.Adam(lstm_model.parameters(), lr=1e-2, betas=(0.9, 0.99))

    result = analyze_model_training(
        "LSTM 256/256/2 (Adam lr=1e-2)",
        lstm_model,
        lstm_opt,
        train_loader,
        device,
        num_seeds=5,
    )
    results.append(result)

    # ============================================================
    # Test 2: GPT2 (similar size to LSTM)
    # ============================================================
    gpt2_config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=2048,
        n_ctx=2048,
        n_embd=256,
        n_layer=2,
        n_head=4,
    )
    gpt2_model = GPT2LMHeadModel(gpt2_config).to(device)
    gpt2_opt = torch.optim.AdamW(gpt2_model.parameters(), lr=5e-4, weight_decay=0.01)

    result = analyze_model_training(
        "GPT2 256/2/4 (AdamW lr=5e-4)",
        gpt2_model,
        gpt2_opt,
        train_loader,
        device,
        num_seeds=5,
    )
    results.append(result)

    # ============================================================
    # Test 3: LSTM with GPT2's learning rate
    # ============================================================
    lstm_model_low_lr = LSTM(lstm_config).to(device)
    lstm_opt_low = torch.optim.AdamW(
        lstm_model_low_lr.parameters(), lr=5e-4, weight_decay=0.01
    )

    result = analyze_model_training(
        "LSTM 256/256/2 (AdamW lr=5e-4, like GPT2)",
        lstm_model_low_lr,
        lstm_opt_low,
        train_loader,
        device,
        num_seeds=5,
    )
    results.append(result)

    # ============================================================
    # Test 4: LSTM with LayerNorm (architectural change)
    # ============================================================
    # We'll test if adding layer normalization helps
    class LSTMWithLayerNorm(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
            self.lstm = nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=0.0,
                batch_first=True,
            )
            self.layer_norm = nn.LayerNorm(config.hidden_size)
            self.output = nn.Linear(config.hidden_size, config.vocab_size)

            # Initialize
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.zero_()
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

        def forward(self, input_ids, labels=None, **kwargs):
            embeddings = self.embedding(input_ids)
            lstm_output, _ = self.lstm(embeddings)
            lstm_output = self.layer_norm(lstm_output)  # Add LayerNorm!
            logits = self.output(lstm_output)

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
                )
            return {"loss": loss, "logits": logits}

    lstm_ln_model = LSTMWithLayerNorm(lstm_config).to(device)
    lstm_ln_opt = torch.optim.Adam(
        lstm_ln_model.parameters(), lr=1e-2, betas=(0.9, 0.99)
    )

    result = analyze_model_training(
        "LSTM+LayerNorm 256/256/2 (Adam lr=1e-2)",
        lstm_ln_model,
        lstm_ln_opt,
        train_loader,
        device,
        num_seeds=5,
    )
    results.append(result)

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}\n")

    print(f"{'Model':<40} {'Mean@40':>10} {'Std@40':>10} {'Plateau%':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['model_name']:<40} {r['mean_40']:>10.4f} {r['std_40']:>10.4f} {r['plateaued_pct']:>9.0f}%"
        )

    print(f"\n{'='*70}")
    print("KEY INSIGHTS:")
    print(f"{'='*70}")

    gpt2_result = [r for r in results if "GPT2" in r["model_name"]][0]
    lstm_result = [r for r in results if r["model_name"].startswith("LSTM 256")][0]

    print(f"\n1. Variance comparison:")
    print(f"   GPT2 std: {gpt2_result['std_40']:.4f}")
    print(f"   LSTM std: {lstm_result['std_40']:.4f}")
    print(
        f"   → LSTM is {lstm_result['std_40']/gpt2_result['std_40']:.1f}x more variable"
    )

    print(f"\n2. Plateau rate:")
    print(f"   GPT2: {gpt2_result['plateaued_pct']:.0f}% plateau")
    print(f"   LSTM: {lstm_result['plateaued_pct']:.0f}% plateau")

    print(f"\n3. Best performing variant:")
    best = min(results, key=lambda x: x["std_40"])
    print(f"   {best['model_name']}")
    print(
        f"   Std@40: {best['std_40']:.4f} (vs LSTM baseline {lstm_result['std_40']:.4f})"
    )


if __name__ == "__main__":
    main()
