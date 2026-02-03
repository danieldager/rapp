"""Utilities for loading trained models and preparing sWuggy dataset."""

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import polars as pl
import torch
import torchaudio
import torch.nn.functional as F
from safetensors.torch import load_file
import webdataset as wds
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.train.models import LSTM, LSTMConfig
from spidr.models import spidr_base


def load_checkpoint(checkpoint_path: str, device: str = "cuda") -> Tuple[Any, Any]:
    """
    Load a trained model from a checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory containing config.json and model.safetensors
        device: Device to load model onto ('cuda' or 'cpu')

    Returns:
        Tuple of (model, config)
    """
    checkpoint_dir = Path(checkpoint_path)

    # Load config
    with open(checkpoint_dir / "config.json", "r") as f:
        config_dict = json.load(f)

    model_type = config_dict.get("model_type", "gpt2")

    # Create model
    if model_type == "lstm":
        config = LSTMConfig(**config_dict)
        model = LSTM(config)
    else:
        config = GPT2Config(**config_dict)
        model = GPT2LMHeadModel(config)

    # Load weights
    state_dict = load_file(str(checkpoint_dir / "model.safetensors"))

    # Handle GPT2 weight tying: lm_head.weight is tied to transformer.wte.weight
    if model_type == "gpt2" and "lm_head.weight" not in state_dict:
        if "transformer.wte.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    model.load_state_dict(state_dict)

    # Move to device and eval
    model.to(device)  # type: ignore
    model.eval()

    print(f"✓ Loaded {model_type} model from {checkpoint_path}")
    return model, config


# =============================================================================
# Prepare and Encode sWuggy Dataset
# =============================================================================

SAMPLE_RATE = 16000
FRAME_RATE = 50
MAX_SHARD_SIZE = 1 * 1024**3
MAX_SHARD_COUNT = 10000


def preprocess_audio_bytes(audio_bytes: bytes) -> torch.Tensor:
    """Load audio from bytes, convert to mono, and normalize."""
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

    # Convert to mono (take first channel)
    if waveform.shape[0] > 1:
        waveform = waveform[0:1, :]

    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE
        )

    # Normalize
    waveform = F.layer_norm(waveform, waveform.shape)

    return waveform


def encode_audio(encoder, waveform: torch.Tensor, device: str) -> np.ndarray:
    """Encode audio waveform to discrete tokens using SPIDR model."""
    if device == "cuda":
        waveform = waveform.cuda()

    with torch.no_grad():
        # SPIDR returns codebooks for layers > 4, we use layer 6
        codebooks = encoder.get_codebooks(waveform, onehot=True)
        tokens = codebooks[5].argmax(dim=-1)

        # Deduplicate consecutive tokens for language modeling
        tokens = torch.unique_consecutive(tokens)

    return tokens.cpu().numpy().astype(np.int16)


def prepare_and_encode_swuggy(
    raw_parquet_pattern: str = "/store/projects/lexical-benchmark/swuggy/data/*.parquet",
    output_tokens_dir: str = "/scratch2/ddager/rapp/tokens/swuggy_spidr_base",
    output_metadata_path: str = "/scratch2/ddager/rapp/metadata/swuggy.parquet",
    device: str = "cuda",
) -> None:
    """
    Load raw sWuggy dataset, encode audio to tokens, and create clean metadata.

    Steps:
        1. Load raw parquet files with embedded audio bytes
        2. Unpivot: separate positive/negative into individual rows
        3. Encode audio bytes to tokens using SPIDR
        4. Write tokens to WebDataset tar files
        5. Save clean metadata parquet (without audio bytes)

    Args:
        raw_parquet_pattern: Path pattern to raw sWuggy parquet files
        output_tokens_dir: Directory to write token tar files
        output_metadata_path: Path to save clean metadata parquet
        device: Device for encoding ('cuda' or 'cpu')
    """
    print("=" * 60)
    print("SWUGGY PREPARE & ENCODE PIPELINE")
    print("=" * 60)

    # Load raw data
    print(f"\nLoading raw data from {raw_parquet_pattern}...")
    df_raw = pl.read_parquet(raw_parquet_pattern)
    print(f"Loaded {len(df_raw)} word pairs")

    # Unpivot: create separate rows for positive and negative samples
    print("\nUnpivoting dataset...")

    # Extract positive samples
    df_positive = df_raw.select(
        [
            pl.col("id").alias("word_id"),
            pl.col("positive").struct.field("bytes").alias("audio_bytes"),
            pl.col("positive_word").alias("word"),
            pl.col("positive_phones").alias("phones"),
            pl.col("voice"),
            pl.lit(True).alias("positive"),
        ]
    )

    # Extract negative samples
    df_negative = df_raw.select(
        [
            pl.col("id").alias("word_id"),
            pl.col("negative").struct.field("bytes").alias("audio_bytes"),
            pl.col("negative_word").alias("word"),
            pl.col("negative_phones").alias("phones"),
            pl.col("voice"),
            pl.lit(False).alias("positive"),
        ]
    )

    # Concatenate
    df = pl.concat([df_positive, df_negative])

    # Create file_id
    df = df.with_columns(
        pl.concat_str(
            [
                pl.col("word_id").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("voice"),
                pl.lit("_"),
                pl.when(pl.col("positive"))
                .then(pl.lit("pos"))
                .otherwise(pl.lit("neg")),
            ]
        ).alias("file_id")
    )

    print(
        f"Created {len(df)} samples ({len(df_positive)} positive + {len(df_negative)} negative)"
    )

    # Setup encoder
    print(f"\nLoading SPIDR encoder on {device}...")
    encoder = spidr_base().to(device)
    encoder.eval()

    # Setup WebDataset writer
    output_dir = Path(output_tokens_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = str(output_dir / "shard-%03d.tar")

    print(f"\nEncoding audio and writing to {output_dir}...")

    processed = 0
    skipped = 0
    start_time = time.time()

    with wds.ShardWriter(shard_pattern, maxsize=MAX_SHARD_SIZE, maxcount=MAX_SHARD_COUNT) as sink:  # type: ignore
        for row in df.iter_rows(named=True):
            file_id = row["file_id"]
            audio_bytes = row["audio_bytes"]

            try:
                # Preprocess audio
                waveform = preprocess_audio_bytes(audio_bytes)

                # Skip very short segments
                duration = waveform.shape[-1] / SAMPLE_RATE
                if duration < 0.5:
                    skipped += 1
                    print(f"  Skipped {file_id}: too short ({duration:.2f}s)")
                    continue

                # Encode to tokens
                tokens = encode_audio(encoder, waveform, device)

                if len(tokens) == 0:
                    skipped += 1
                    print(f"  Skipped {file_id}: zero tokens")
                    continue

                # Write to tar
                sample = {
                    "__key__": file_id,
                    "tokens.npy": tokens,
                    "json": {
                        "file_id": file_id,
                        "word_id": row["word_id"],
                        "word": row["word"],
                        "voice": row["voice"],
                        "positive": row["positive"],
                        "token_count": len(tokens),
                        "duration_sec": duration,
                    },
                }
                sink.write(sample)
                processed += 1

                # Progress update
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    print(
                        f"  Processed: {processed}/{len(df)} | Rate: {rate:.1f} samples/sec"
                    )

            except Exception as e:
                skipped += 1
                print(f"  Error processing {file_id}: {str(e)[:100]}")

    # Save clean metadata (without audio bytes)
    print(f"\nSaving metadata to {output_metadata_path}...")
    df_clean = df.select(
        [
            "word_id",
            "file_id",
            "word",
            "phones",
            "voice",
            "positive",
        ]
    )

    Path(output_metadata_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.write_parquet(output_metadata_path)

    # Summary
    elapsed_min = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print("ENCODING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed:  {processed} samples")
    print(f"Skipped:    {skipped} samples")
    print(f"Time:       {elapsed_min:.1f} minutes")
    print(f"Tokens:     {output_tokens_dir}")
    print(f"Metadata:   {output_metadata_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prepare_and_encode_swuggy(
        raw_parquet_pattern="/store/projects/lexical-benchmark/swuggy/data/*.parquet",
        output_tokens_dir="/scratch2/ddager/rapp/tokens/swuggy_spidr_base",
        output_metadata_path="/scratch2/ddager/rapp/metadata/swuggy.parquet",
        device=device,
    )
