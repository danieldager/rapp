#!/usr/bin/env python3
"""
VAD Pipeline for audio processing with multiprocessing support.
Dataset-agnostic: processes any directory structure, stores absolute paths.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torchaudio
from ten_vad import TenVad
from vad_utils import get_task_shard


def get_runs(flags):
    """Return (start, end) pairs for periods of speech and non-speech."""
    if len(flags) == 0:
        return np.array([]), np.array([])

    first_flag = flags[0]
    last_index = len(flags)
    arr = np.flatnonzero(np.diff(flags))
    arr = np.r_[0, arr + 1, last_index]

    pairs = np.column_stack((arr[:-1], arr[1:]))
    odd = pairs[::2]
    even = pairs[1::2]

    if first_flag == 1:
        ones = odd
        zeros = even
    else:
        ones = even
        zeros = odd

    return ones, zeros


def runs_to_secs(runs, hop_size, sr):
    """Convert runs of speech and non-speech from frames to seconds."""
    if runs is None or len(runs) == 0:
        return np.array([], dtype=np.float32)

    frame_lengths = runs[:, 1] - runs[:, 0]
    return (frame_lengths * (hop_size / sr)).astype(np.float32, copy=False)


def find_splits(flags, hop_size, sr, target_interval=30.0):
    """
    Find optimal split points for long audio files.

    Looks for non-speech runs of at least 300ms starting around target_interval,
    and places split points at the middle of suitable non-speech segments.

    Args:
        flags: Array of VAD flags (0=non-speech, 1=speech)
        hop_size: Number of samples per frame
        sr: Sample rate

    Returns:
        List of frame indices where splits should occur
    """
    splits = []

    target_interval_frames = int(target_interval * sr / hop_size)
    min_silence_frames = int(0.3 * sr / hop_size)

    # Start looking for splits after the first 30 seconds
    current_pos = target_interval_frames
    total_frames = len(flags)

    while current_pos < total_frames - target_interval_frames:
        # Look for a suitable non-speech run starting from current_pos
        split_found = False

        # Search window: look ahead up to 10 seconds for a good split point
        search_end = min(current_pos + int(10.0 * sr / hop_size), total_frames)

        i = current_pos
        while i < search_end:
            if flags[i] == 0:  # Found start of non-speech
                # Check how long this non-speech run is
                silence_start = i
                while i < total_frames and flags[i] == 0:
                    i += 1
                silence_end = i
                silence_length = silence_end - silence_start

                # If silence is long enough (>=300ms), place split in the middle
                if silence_length >= min_silence_frames:
                    split_frame = silence_start + silence_length // 2
                    splits.append(split_frame)

                    # Move to next target position (30 seconds after this split)
                    current_pos = split_frame + target_interval_frames
                    split_found = True
                    break
            else:
                i += 1

        # If no suitable split found, move forward and try again
        if not split_found:
            current_pos += int(10.0 * sr / hop_size)  # Skip ahead 10 seconds

    return splits


def log_progress(
    i: int,
    total: int,
    start_time: float,
    recent_times: list[float],
) -> None:
    """Log processing progress with rolling average rate and ETA."""
    elapsed = time.time() - start_time

    # Use rolling average if we have enough samples
    if len(recent_times) >= 100:
        rolling_rate = len(recent_times) / sum(recent_times)
    else:
        rolling_rate = i / elapsed if elapsed > 0 else 0

    files_remaining = total - i
    seconds_remaining = files_remaining / rolling_rate if rolling_rate > 0 else 0

    # Format ETA
    if seconds_remaining < 3600:
        eta_str = f"ETA {seconds_remaining / 60:.0f}m"
    else:
        eta_str = f"ETA {seconds_remaining / 3600:.1f}h"

    progress_line = f"[{i:>6}/{total}] {rolling_rate:.1f} files/s | {eta_str}"

    print(progress_line)


def get_log_interval(files_processed: int) -> int:
    """Get adaptive reporting interval based on progress."""
    if files_processed < 10000:
        return 1000
    elif files_processed < 50000:
        return 5000
    elif files_processed < 100000:
        return 10000
    else:
        return 20000


def create_error_record(path: str, error: str) -> dict:
    """Create an error record matching the schema of vad_pyannote.py."""
    return {
        "success": False,
        "path": str(path),
        "file_id": Path(path).stem,
        "duration": 0.0,
        "original_sr": 0,
        "speech_ratio": 0.0,
        "speech_max": 0.0,
        "speech_min": 0.0,
        "speech_sum": 0.0,
        "speech_num": 0,
        "speech_avg": 0.0,
        "nospch_max": 0.0,
        "nospch_min": 0.0,
        "nospch_sum": 0.0,
        "nospch_num": 0,
        "nospch_avg": 0.0,
        "speech_durations": [],
        "nospch_durations": [],
        "speech_segments": [],
        "error": error,
    }


def compute_segment_stats(durations: list[float]) -> dict:
    if not durations:
        return {"max": 0.0, "min": 0.0, "avg": 0.0, "sum": 0.0, "num": 0}

    # helper for list -> stats
    arr = np.array(durations)
    return {
        "max": float(arr.max()),
        "min": float(arr.min()),
        "sum": float(arr.sum()),
        "num": int(len(durations)),
        "avg": float(arr.mean()),
    }


def process_single_wav(args):
    """Process a single WAV file - designed for multiprocessing."""
    wav_path, hop_size, threshold = args

    try:
        # Each process gets its own instance
        TV = TenVad(hop_size=hop_size, threshold=threshold)
    except Exception as e:
        print(f"ERROR: TenVad init failed for {wav_path}: {e}", file=sys.stderr)
        return create_error_record(str(wav_path), f"TenVad init failed: {str(e)}")

    try:
        # Read audio file with torchaudio
        waveform, sample_rate = torchaudio.load(str(wav_path))
        original_sr = sample_rate

        # TODO: just take one channel
        if waveform.size(0) > 1:
            waveform = waveform[0:1, :]

        # Resample to 16kHz if needed (TenVAD expects 16kHz)
        TARGET_SR = 16000
        if sample_rate != TARGET_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=TARGET_SR,
            )
            waveform = resampler(waveform)
            sample_rate = TARGET_SR

        # TODO i'm very unsure about this chunk
        # Convert to int16 for TenVAD
        # torchaudio loads as float32 in [-1, 1]
        data = waveform.squeeze().numpy()
        data = (data * 32767).astype(np.int16)

        duration = len(data) / sample_rate

        # TenVad process
        num_frames = len(data) // hop_size
        if num_frames < hop_size:
            return create_error_record(
                str(wav_path), f"Audio too short: {len(data)} samples"
            )

        frames = data[: num_frames * hop_size].reshape(-1, hop_size)
        flags = np.empty(num_frames, dtype=np.uint8)

        process_func = TV.process
        for i in range(num_frames):
            _, flags[i] = process_func(frames[i])

        spch_ratio = float(flags.mean())

        # Calculate runs and durations
        ones, zeros = get_runs(flags)
        spoken_secs = runs_to_secs(ones, hop_size, sample_rate)
        nospch_secs = runs_to_secs(zeros, hop_size, sample_rate)

        # Convert runs (frame indices) to seconds [start, end]
        if ones.size > 0:
            speech_segments = (ones * hop_size / sample_rate).tolist()
        else:
            speech_segments = []

        # Calculate stats for output
        speech_durations = spoken_secs.tolist() if spoken_secs.size > 0 else []
        nospch_durations = nospch_secs.tolist() if nospch_secs.size > 0 else []

        speech_stats = compute_segment_stats(speech_durations)
        nospch_stats = compute_segment_stats(nospch_durations)

        speech_segments_detailed = []
        for seg in speech_segments:
            start_sec = seg[0]
            end_sec = seg[1]
            dur_sec = end_sec - start_sec

            # Calculate frames using original_sr to match pyannote output
            start_frames = int(start_sec * original_sr)
            end_frames = int(end_sec * original_sr)
            dur_frames = int(dur_sec * original_sr)

            speech_segments_detailed.append(
                {
                    "start_sec": start_sec,
                    "start_frames": start_frames,
                    "end_sec": end_sec,
                    "end_frames": end_frames,
                    "duration_sec": dur_sec,
                    "duration_frames": dur_frames,
                }
            )

        return {
            "success": True,
            "path": str(wav_path),
            "file_id": Path(wav_path).stem,
            "duration": duration,
            "original_sr": int(original_sr),
            "speech_ratio": spch_ratio,
            "speech_durations": speech_durations,
            "nospch_durations": nospch_durations,
            "speech_segments": speech_segments_detailed,
            **{f"speech_{k}": v for k, v in speech_stats.items()},
            **{f"nospch_{k}": v for k, v in nospch_stats.items()},
            "error": "",
        }

    except Exception as e:
        return create_error_record(str(wav_path), str(e))


def process_wavs_parallel(wavs, hop_size, threshold, max_workers):
    """Process WAV files in parallel across multiple workers."""

    args_list = [(wav, hop_size, threshold) for wav in wavs]

    results = []
    completed = 0
    errors = 0
    total = len(wavs)
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_wav = {
            executor.submit(process_single_wav, args): args[0] for args in args_list
        }

        # Collect results as they complete
        for future in as_completed(future_to_wav):
            wav_path = future_to_wav[future]
            completed += 1

            # Adaptive logging
            log_interval = get_log_interval(completed)
            if completed % log_interval == 0 or completed == total:
                log_progress(completed, total, start_time, [])

            try:
                result = future.result()
                if result is not None:
                    if "error" in result:
                        errors += 1
                        print(
                            f"WARNING: Error processing {wav_path.name}: {result['error']}",
                            file=sys.stderr,
                        )
                    results.append(result)
            except Exception as e:
                errors += 1
                print(f"ERROR: Exception with {wav_path}: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"Completed processing {len(results)}/{total} files in {elapsed:.1f}s")

    if errors > 0:
        print(
            f"WARNING: Encountered {errors} errors during processing", file=sys.stderr
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="VAD Pipeline for audio processing")
    parser.add_argument(
        "manifest",
        type=str,
        help="Path to manifest file (.txt, .csv, .parquet)",
    )
    parser.add_argument(
        "--hop_size",
        type=int,
        default=256,
        help="Hop size for VAD processing (default: 256)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="VAD threshold (default: 0.5)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect from CPUs)",
    )

    args = parser.parse_args()

    # Determine files to process
    total, chunk, wavs = get_task_shard(args.manifest, 0, 1)
    wavs = [Path(p) for p in wavs]

    # Auto-detect workers
    if args.workers is None:
        args.workers = mp.cpu_count()
    print(f"Using {args.workers} parallel workers")

    # Determine output files
    manifest_path = Path(args.manifest)
    output_dir = Path("metadata") / manifest_path.stem / "ten"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "shard_0.parquet"

    print(f"Processing {len(wavs)} files")

    # Process files
    try:
        if not wavs:
            print("No files to process in this shard.")
            sys.exit(0)

        results = process_wavs_parallel(
            wavs,
            args.hop_size,
            args.threshold,
            args.workers,
        )

        if not results:
            print(
                "ERROR: No results generated - all files failed processing",
                file=sys.stderr,
            )
            sys.exit(1)

        # Save Parquet
        pl.DataFrame(results).write_parquet(output_file, compression="zstd")
        print(f"Results saved to {output_file}")

        # Report statistics
        df = pl.read_parquet(output_file)
        if "error" in df.columns:
            successful = df.filter(pl.col("success") == True)
            errors_df = df.filter(pl.col("success") == False)
            print(f"Successfully processed: {len(successful)}/{len(df)} files")
            if len(errors_df) > 0:
                print(f"WARNING: Failed files: {len(errors_df)}", file=sys.stderr)
                # Print first 5 errors using polars
                for row in errors_df.head(5).iter_rows(named=True):
                    path_str = row.get("path", "unknown")
                    print(
                        f"  {Path(path_str).name}: {row['error']}",
                        file=sys.stderr,
                    )
        else:
            print(f"Successfully processed: {len(df)} files")

    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
