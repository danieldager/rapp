"""Evaluate language models on sWuggy word-nonword classification task.

The sWuggy score measures how well a model distinguishes between real words
and phonetically matched nonwords based on sequence probabilities.
"""

import sys
from pathlib import Path
import time

import polars as pl
import torch
import webdataset as wds

from scripts.train.datasets import BOS_TOKEN_ID, EOS_TOKEN_ID
from scripts.swuggy.utils import load_checkpoint


def print_evaluation_summary(config, n_samples, model_info):
    """Print evaluation configuration summary."""
    print(f"\n{'='*60}")
    print("SWUGGY EVALUATION")
    print(f"{'='*60}")
    print(f"Dataset:    {n_samples} samples")
    print(f"Model:      {model_info['type'].upper()} ({model_info['n_params']/1e6:.1f}M params)")
    print(f"Checkpoint: {Path(config['checkpoint_path']).name}")
    print(f"Device:     {config['device']}")
    print(f"{'='*60}\n")


def calculate_sequence_log_probability(model, tokens, device):
    """
    Calculate log P(sequence) using autoregressive factorization.
    Returns (log_prob, log_prob_normalized, num_tokens).
    """
    tokens = tokens.unsqueeze(0).to(device)  # [1, seq_len]

    with torch.no_grad():
        outputs = model(input_ids=tokens, labels=tokens)

        # Get logits [1, seq_len, vocab_size]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        # Shift for next-token prediction: predict tokens[1:] given tokens[:-1]
        shift_logits = logits[:, :-1, :].contiguous()  # [1, seq_len-1, vocab_size]
        shift_labels = tokens[:, 1:].contiguous()  # [1, seq_len-1]

        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        target_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(
            -1
        )  # [1, seq_len-1]

        # Sum to get sequence log probability
        # This computes: log P(token_1|BOS) + log P(token_2|BOS,token_1) + ... + log P(EOS|full_context)
        sequence_log_prob = target_log_probs.sum().item()

        # Length-normalized version (average log probability per token)
        num_tokens = target_log_probs.shape[1]
        normalized_log_prob = sequence_log_prob / num_tokens if num_tokens > 0 else 0.0

    return sequence_log_prob, normalized_log_prob, num_tokens


def load_tokens_from_tar(tokens_dir: str) -> dict:
    """
    Load all token samples from WebDataset tar files.

    Returns:
        Dict mapping file_id to token tensor
    """
    tokens_dir_path = Path(tokens_dir)
    urls = sorted([str(p) for p in tokens_dir_path.glob("*.tar")])

    if not urls:
        raise ValueError(f"No tar files found in {tokens_dir}")

    print(f"\nLoading tokens from {len(urls)} tar files...")

    tokens_dict = {}
    dataset = wds.WebDataset(urls, shardshuffle=False).decode()  # type: ignore

    for sample in dataset:
        key = sample["__key__"]  # This is the file_id
        tokens = sample.get("tokens.npy")

        if tokens is None:
            print(f"Warning: No tokens found for {key}")
            continue

        # Add BOS/EOS and convert to tensor
        token_list = [BOS_TOKEN_ID] + tokens.tolist() + [EOS_TOKEN_ID]
        token_tensor = torch.tensor(token_list, dtype=torch.long)
        tokens_dict[key] = token_tensor

    print(f"Loaded {len(tokens_dict)} token sequences")
    return tokens_dict


def evaluate_and_add_probabilities(
    model, metadata_df: pl.DataFrame, tokens_dict: dict, device: str
) -> pl.DataFrame:
    """Calculate log probabilities for all samples."""
    print("\nCalculating log probabilities...")
    
    total = len(metadata_df)
    log_probs, log_probs_norm, seq_lengths = [], [], []
    missing = 0
    start_time = time.time()
    log_interval = max(1, total // 20)

    for idx, row in enumerate(metadata_df.iter_rows(named=True), 1):
        file_id = row["file_id"]

        # Get tokens for this file_id
        tokens = tokens_dict.get(file_id)

        if tokens is None:
            print(f"Warning: No tokens found for {file_id}")
            log_probs.append(None)
            log_probs_norm.append(None)
            seq_lengths.append(None)
            missing += 1
            continue

        # Calculate log probability (both unnormalized and normalized)
        log_prob, log_prob_norm, num_tokens = calculate_sequence_log_probability(
            model, tokens, device
        )
        log_probs.append(log_prob)
        log_probs_norm.append(log_prob_norm)
        seq_lengths.append(num_tokens)

        # Progress logging
        if idx % log_interval == 0 or idx == total:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (total - idx) / rate if rate > 0 else 0
            pct = 100 * idx / total

            print(
                f"  {idx}/{total} ({pct:.1f}%) | "
                f"Rate: {rate:.1f} s/s | "
                f"Elapsed: {elapsed/60:.1f}m | "
                f"ETA: {remaining/60:.1f}m"
            )

    if missing > 0:
        print(f"Warning: {missing} samples missing tokens")

    # Add log_prob columns to dataframe
    df_with_probs = metadata_df.with_columns(
        [
            pl.Series("log_prob", log_probs),
            pl.Series("log_prob_norm", log_probs_norm),
            pl.Series("num_tokens", seq_lengths),
        ]
    )

    total_time = time.time() - start_time
    print(
        f"✓ Calculated {len(log_probs) - missing} log probabilities in {total_time/60:.1f} minutes"
    )
    return df_with_probs


if __name__ == "__main__":
    config = {
        "metadata_path": "/scratch2/ddager/rapp/metadata/swuggy.parquet",
        "tokens_dir": "/scratch2/ddager/rapp/tokens/swuggy_spidr_base",
        "checkpoint_path": "/scratch2/ddager/rapp/weights/gpt2_e768_l12_h12/checkpoint-8200",
        "output_path": "/scratch2/ddager/rapp/metadata/swuggy_evaluated.parquet",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Load data and model
    print("Loading metadata...")
    df = pl.read_parquet(config["metadata_path"])
    
    print("Loading model...")
    model, model_config = load_checkpoint(config["checkpoint_path"], config["device"])
    
    model_info = {
        "type": model_config.model_type,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    print_evaluation_summary(config, len(df), model_info)

    # Evaluate
    tokens_dict = load_tokens_from_tar(config["tokens_dir"])
    df_with_probs = evaluate_and_add_probabilities(model, df, tokens_dict, config["device"])

    # Save results
    print(f"\nSaving to {config['output_path']}...")
    df_with_probs.write_parquet(config["output_path"])
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nNext: python scripts/swuggy/analysis.py\n")
