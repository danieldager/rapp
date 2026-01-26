#!/usr/bin/env python3
"""Tokenize audio files using mHuBERT + k-means with optional splitting."""
# NOTE - do we really want encode to rely on the VAD metadata? (i.e. duration column)
# in any case, we need to refactor the manifest reading function (to account for shards)
# and the output path (always write to top level tokens folder, correct folder name)
# ALSO, tasks should get consecutive files! (not strided slicing, df needs to be sorted)

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, cast

import numpy as np
import polars as pl
import webdataset as wds

import torch
import torchaudio
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from spidr.models import spidr_base

# Monkey-patching
# sys.path.insert(0, str(Path(__file__).parent))
# from sts import *
# from textless.data.speech_encoder import SpeechEncoder  # type: ignore

# Constants
FRAME_RATE = 50  # mHuBERT: 50 Hz = 20ms hop
VAD_HOP_SIZE = 256
SAMPLE_RATE = 16000
TIMING_WINDOW = 1000
MAX_SHARD_SIZE = 1 * 1024**3  # 1GB per shard
MAX_SHARD_COUNT = 10000  # Max samples per shard


@dataclass
class Config:
    """Configuration for tokenization."""

    manifest: str  # Path to manifest CSV
    model: str  # e.g., "mhubert-base-vp_mls_cv_8lang/kmeans/2000"
    model_name: str  # e.g., "mhubert"
    task_id: int = 0
    num_tasks: int = 1
    overwrite: bool = False
    device: str = "cuda"


class ProgressTracker:
    """Track timing and summary statistics."""

    def __init__(self, window_size: int = TIMING_WINDOW):
        self.window_size = window_size
        self.timings = {"load": [], "encode": [], "write": []}
        self.processed = 0
        self.short_segments = 0  # < 3s
        self.long_segments = 0  # > 30s
        self.start_time = time.time()

    def add_timing(self, stage: str, duration: float) -> None:
        """Record timing for a stage."""
        self.timings[stage].append(duration)
        if len(self.timings[stage]) > self.window_size:
            self.timings[stage] = self.timings[stage][-self.window_size :]

    def get_avg_timing(self, stage: str) -> float:
        """Get average timing for a stage (last window)."""
        times = self.timings[stage]
        if not times:
            return 0.0
        return sum(times[-self.window_size :]) / min(self.window_size, len(times))

    def get_throughput(self) -> float:
        """Get samples per second."""
        elapsed = time.time() - self.start_time
        return self.processed / elapsed if elapsed > 0 else 0

    def get_elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        return (time.time() - self.start_time) / 60


def logProgress(counter: int, total_files: int, tracker: ProgressTracker) -> None:
    # Progress update every 1000 files
    if counter % 1000 == 0 or counter == 100:
        rate = tracker.get_throughput()
        avg_load = tracker.get_avg_timing("load")
        avg_encode = tracker.get_avg_timing("encode")
        avg_write = tracker.get_avg_timing("write")

        # Calculate ETA
        remaining = total_files - counter
        if rate > 0:
            eta_sec = remaining / rate if tracker.processed > 0 else 0
            eta_h = int(eta_sec // 3600)
            eta_m = int((eta_sec % 3600) // 60)
            eta_str = f"{eta_h}h{eta_m}m"
        else:
            eta_str = "--h--m"

        print(
            f"  [{counter:6d}/{total_files}] | "
            f"{tracker.processed} samples | "
            f"{rate:.1f} s/sec | "
            # f"load={avg_load:.3f}s encode={avg_encode:.3f}s write={avg_write:.3f}s | "
            f"ETA: {eta_str}",
            flush=True,
        )


def logSummary(config: Config, tracker: ProgressTracker) -> None:
    print(f"\n{'='*60}")
    print(f"Completed: {tracker.processed} samples processed")
    print(
        f"Short segments (<3s): {tracker.short_segments}, "
        f"Long segments (>30s): {tracker.long_segments}"
    )
    print(f"Task {config.task_id} finished: {datetime.now().strftime('%H:%M:%S')}")
    print(
        f"Total time: {tracker.get_elapsed_minutes():.1f} min "
        f"({tracker.get_throughput():.1f} samples/sec)"
    )


# =============================================================================
# Manifest Processing
# =============================================================================


def parse_splits(splits_val: Optional[object]) -> Optional[List[int]]:
    """Parse splits column value (handle NaN, strings, and lists)."""
    if splits_val is None:
        return None

    if isinstance(splits_val, float):
        return None if splits_val != splits_val else None  # NaN check

    if isinstance(splits_val, str) and splits_val.strip():
        try:
            return [int(x) for x in eval(splits_val)]
        except Exception:
            return None

    if isinstance(splits_val, list):
        return [int(x) for x in splits_val]

    return None


# NOTE - we want to store split indices in the original sample rate (16kHz),
# Thus this function will be soon depreciated.
def create_segments_polars(row: dict) -> List[Tuple[int, int]]:
    """Convert VAD split points into segment boundaries (in samples)."""
    total_samples = int(row["duration"] * SAMPLE_RATE)
    splits = row["splits"]

    if splits is None or not isinstance(splits, list) or len(splits) == 0:
        return [(0, total_samples)]

    # Convert VAD frame indices to audio sample indices
    boundaries = [0] + [int(s * VAD_HOP_SIZE) for s in splits] + [total_samples]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def load_manifest(manifest_path: str) -> pl.DataFrame:
    """Load and validate manifest CSV or Parquet file."""

    if manifest_path.endswith(".parquet"):
        df = pl.read_parquet(manifest_path)
    else:
        df = pl.read_csv(manifest_path)

    # if path column exists, rename to audio_filepath
    if "path" in df.columns:
        df = df.rename({"path": "audio_filepath"})

    # Validate required columns
    # required_cols = {"file_id", "audio_filepath", "duration"}
    required_cols = {"file_id", "audio_filepath"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # # Parse splits if present
    # if "splits" in df.columns:
    #     df = df.with_columns(
    #         pl.col("splits")
    #         .map_batches(lambda s: s.map_elements(parse_splits))
    #         .alias("splits")
    #     )
    # else:
    #     df = df.with_columns(pl.lit(None).alias("splits"))

    # # Create segment boundaries
    # df = df.with_columns(
    #     pl.struct(["duration", "splits"])
    #     .map_elements(create_segments_polars)
    #     .alias("segments")
    # )

    return df


# =============================================================================
# Audio Processing
# =============================================================================


def preprocess_audio(audio_path: str) -> Tuple[torch.Tensor, float]:
    """Load audio file, ensure mono, and resample to target SR."""
    t0 = time.time()
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.cuda()

    # Convert to mono (take first channel)
    if waveform.shape[0] > 1:
        waveform = waveform[0, :]

    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE
        )

    # Normalize
    waveform = F.layer_norm(waveform, waveform.shape)

    return waveform, time.time() - t0


# =============================================================================
# Encoding & Writing
# =============================================================================


def encode_segment(encoder, segment: torch.Tensor, device: str) -> np.ndarray:
    """Encode audio segment to tokens."""
    if device == "cuda":
        segment = segment.cuda()

    with torch.no_grad():
        tokens = encoder(segment)["units"].cpu().numpy().astype(np.int32)

    return tokens


def write_sample_to_sink(
    sink: wds.ShardWriter,  # type: ignore
    file_id: str,
    segment_id: int,
    tokens: torch.Tensor,
    audio_filepath: str,
) -> None:
    """Write tokenized sample to WebDataset shard."""
    file_stem = Path(file_id).stem
    key = f"{file_stem}_s{segment_id:03d}"

    # convert tokens to numpy int16
    tokens_npy = tokens.cpu().numpy().astype(np.int16)

    sample = {
        "__key__": key,
        "tokens.npy": tokens_npy,
        "json": {
            "file_id": file_stem,
            "segment_id": segment_id,
            "token_count": len(tokens_npy),
            "audio_filepath": str(audio_filepath),
        },
    }

    sink.write(sample)


# =============================================================================
# Setup Helpers
# =============================================================================


# def load_encoder(config: Config) -> SpeechEncoder:
#     """Load and initialize mHuBERT encoder."""
#     print("Loading encoder...")
#     dense_model, quantizer, vocab_size = config.model.split("/")
#     encoder = SpeechEncoder.by_name(
#         dense_model_name=dense_model,
#         quantizer_model_name=quantizer,
#         vocab_size=int(vocab_size),
#         deduplicate=True,
#         need_f0=False,
#     )
#     if config.device == "cuda" and torch.cuda.is_available():
#         encoder = encoder.cuda()
#     print(f"Encoder loaded on {config.device}\n")
#     return encoder


def setup_writer(manifest: str, model_name: str, task_id: int) -> wds.ShardWriter:  # type: ignore
    """Create output directory and WebDataset writer."""
    manifest_path_obj = Path(manifest)
    root_path = manifest_path_obj.parent.parent.parent
    dataset_name = manifest_path_obj.stem.split("_")[0]
    output_dir = root_path / "tokens" / f"{dataset_name}_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")

    shard_pattern = str(output_dir / f"task{task_id:03d}-shard%03d.tar")
    return wds.ShardWriter(shard_pattern, maxsize=MAX_SHARD_SIZE, maxcount=MAX_SHARD_COUNT)  # type: ignore


def log_error(file_id: str, audio_filepath: str, message: str) -> None:
    """Log error to stderr."""
    print(f"  ERROR [{audio_filepath}]: {message}", flush=True, file=sys.stderr)


# =============================================================================
# Main Processing
# =============================================================================


# NOTE - segments will soon take the form of (start_idx, duration) in 16kHz sample rate
# Until then, we will just give 0 for start and the full waveform length as the end index.
def process_audio_file(
    model,
    file_id: str,
    audio_filepath: str,
    # segments: List[Tuple[int, int]],
    tracker: ProgressTracker,
    sink: wds.ShardWriter,  # type: ignore
) -> None:
    """Process a single audio file and encode all segments."""

    # Load audio
    waveform, load_time = preprocess_audio(audio_filepath)
    tracker.add_timing("load", load_time)

    if waveform.shape[-1] == 0:
        log_error(file_id, audio_filepath, "Invalid audio: zero-length waveform")
        return

    # # Fix last segment boundary to match actual waveform length
    # if segments:
    #     segments = list(segments)
    #     segments[-1] = (segments[-1][0], waveform.shape[-1])

    # NOTE - Hardcode single segment for now
    segments = [(0, waveform.shape[-1])]

    # Process each segment
    for segment_id, (start, end) in enumerate(segments):
        t_encode_start = time.time()
        segment = waveform[:, start:end]
        duration = segment.shape[-1] / SAMPLE_RATE

        # Validate segment length
        # NOTE - is this an appropriate length to skip?
        if duration < 3:
            tracker.short_segments += 1
            log_error(file_id, audio_filepath, f"{segment_id} under 3 seconds")
            continue
        if duration > 30:
            tracker.long_segments += 1

        # Encode
        # NOTE - spidr returns a codebook for every layer L > 4
        # We extract layer 6 and argmax to get discrete tokens
        codebooks = model.get_codebooks(segment, onehot=True)
        tokens = codebooks[5].argmax(dim=-1)

        # NOTE - deduplicate for language modeling
        tokens = torch.unique_consecutive(tokens)

        tracker.add_timing("encode", time.time() - t_encode_start)

        if len(tokens) == 0:
            log_error(file_id, audio_filepath, f"{segment_id} 0 tokens")
            continue

        # Write
        t_write_start = time.time()
        write_sample_to_sink(sink, file_id, segment_id, tokens, audio_filepath)
        tracker.add_timing("write", time.time() - t_write_start)
        tracker.processed += 1


def tokenize_manifest(config: Config, model) -> None:
    """Main tokenization pipeline."""

    # Load and prepare data
    # NOTE - we also want to accept .parquet manifests in the future
    df = load_manifest(config.manifest)
    df = df[config.task_id :: config.num_tasks]
    print(f"Processing {len(df)} files\n")

    # Setup
    tracker = ProgressTracker()

    with setup_writer(config.manifest, config.model_name, config.task_id) as sink:
        for counter, row in enumerate(df.iter_rows(named=True)):
            file_id = str(row["file_id"])
            audio_filepath = str(row["audio_filepath"])
            # segments = cast(List[Tuple[int, int]], row["segments"])

            try:
                process_audio_file(
                    model=model,
                    file_id=file_id,
                    audio_filepath=audio_filepath,
                    # segments=segments,
                    tracker=tracker,
                    sink=sink,
                )
            except Exception as e:
                error_msg = str(e)
                # Skip F0 extraction errors (expected for some files)
                if "Cannot subsample F0" not in error_msg:
                    log_error(file_id, audio_filepath, error_msg[:100])

            logProgress(counter, len(df), tracker)

    logSummary(config, tracker)


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize audio files and write to webdataset tar shards"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="CSV manifest with file_id, audio_filepath, and duration columns",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Full model name (e.g., mhubert-base-vp_mls_cv_8lang/kmeans/2000)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Custom name for output folder (e.g., mhubert)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--task-id", type=int, default=0, help="Task ID for distributed processing"
    )
    parser.add_argument(
        "--num-tasks", type=int, default=1, help="Total number of parallel tasks"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )

    args = parser.parse_args()
    config = Config(**vars(args))
    model = spidr_base().to(args.device)

    with torch.no_grad():
        tokenize_manifest(config, model)

        # # FOR DEBUGGING
        # df = load_manifest(config.manifest)
        # df = df.iloc[config.task_id :: config.num_tasks].reset_index(drop=True)

        # tracker = ProgressTracker()

        # with setup_writer(config.manifest, config.model_name, config.task_id) as sink:
        #     for counter, row in enumerate(df.itertuples(index=False)):

        #         if counter > 10:
        #             break

        #         file_id = str(row.file_id)
        #         audio_filepath = str(row.audio_filepath)
        #         segments = cast(List[Tuple[int, int]], row.segments)

        #         try:
        #             process_audio_file(
        #                 model=model,
        #                 file_id=file_id,
        #                 audio_filepath=audio_filepath,
        #                 segments=segments,
        #                 tracker=tracker,
        #                 sink=sink,
        #             )
        #         except Exception as e:
        #             error_msg = str(e)
        #             if "Cannot subsample F0" not in error_msg:
        #                 log_error(file_id, audio_filepath, error_msg[:100])
