import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load(str(path))
            return {"path": path, "waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            return {"path": path, "error": str(e)}


def load_env():
    """Load .env file if it exists."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def setup_model_and_pipeline(hf_token: str, device: torch.device):
    """Load model and instantiate VAD pipeline."""

    print("\nLoading model from HuggingFace...")
    model = Model.from_pretrained("pyannote/segmentation-3.0", token=hf_token)
    model = model.to(device)  # type: ignore

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for faster computation on Ampere+ GPUs (compute capability >= 8.0)
        gpu_capability = torch.cuda.get_device_capability(0)
        if gpu_capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled (Ampere+ GPU detected)")

    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate(
        {
            "min_duration_on": 0.0,  # remove speech regions shorter than this
            "min_duration_off": 0.3,  # fill non-speech regions shorter than this
        }
    )

    print("Pipeline ready")

    return pipeline


def parse_manifest(manifest_path: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in manifest_path.read_text().splitlines()
        if line.strip()
    ]


def setup_output_dirs(manifest_path: Path, array_id: int) -> dict[str, Path]:
    """Create output directories and return task-specific file paths."""
    output_dir = Path("metadata") / manifest_path.stem
    metadata_dir = output_dir / "metadata"
    segments_dir = output_dir / "segments"
    durations_dir = output_dir / "durations"

    # Create all directories
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    segments_dir.mkdir(exist_ok=True)
    durations_dir.mkdir(exist_ok=True)

    # Construct task-specific file paths
    task_metadata_file = metadata_dir / f"task_{array_id:03d}.jsonl"
    task_durations_file = durations_dir / f"task_{array_id:03d}.jsonl"
    task_segments_file = segments_dir / f"task_{array_id:03d}.jsonl"

    # Clean up any existing files to ensure fresh writes
    for task_file in (task_metadata_file, task_durations_file, task_segments_file):
        if task_file.exists():
            task_file.unlink()

    return {
        "output_dir": output_dir,
        "task_metadata_file": task_metadata_file,
        "task_durations_file": task_durations_file,
        "task_segments_file": task_segments_file,
    }


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


def compute_segment_stats(segments: np.ndarray) -> dict:
    """Compute statistics for speech or non-speech segments."""
    if len(segments) == 0:
        return {
            "total": 0.0,
            "count": 0,
            "max": 0.0,
            "min": 0.0,
            "avg": 0.0,
            "durations": [],
        }

    durations = segments[:, 1] - segments[:, 0]
    return {
        "total": float(durations.sum()),
        "count": len(durations),
        "max": float(durations.max()),
        "min": float(durations.min()),
        "avg": float(durations.mean()),
        "durations": durations.tolist(),
    }


def compute_nospch_segments(speech_segments: np.ndarray, duration: float) -> np.ndarray:
    """Compute non-speech segments (gaps between speech)."""
    if len(speech_segments) == 0:
        return np.array([[0, duration]], dtype=np.float32)

    gaps = []
    # Gap before first speech
    if speech_segments[0, 0] > 0:
        gaps.append([0, speech_segments[0, 0]])

    # Gaps between speech segments
    for i in range(len(speech_segments) - 1):
        gap_start = speech_segments[i, 1]
        gap_end = speech_segments[i + 1, 0]
        if gap_end > gap_start:
            gaps.append([gap_start, gap_end])

    # Gap after last speech
    if speech_segments[-1, 1] < duration:
        gaps.append([speech_segments[-1, 1], duration])

    return (
        np.array(gaps, dtype=np.float32) if gaps else np.empty((0, 2), dtype=np.float32)
    )


def process_single_file(
    audio_tuple: tuple[Path, torch.Tensor, int], pipeline: VoiceActivityDetection
) -> tuple[dict, dict[str, list[float]], list[list[float]]]:
    """Process one file and return metadata, durations, and speech segments."""

    try:
        audio_path, waveform, sample_rate = audio_tuple

        # TODO: get torchcodec to work
        duration = waveform.shape[1] / sample_rate

        # Pass pre-loaded audio as dict to avoid AudioDecoder
        vad = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # # Free waveform tensor immediately after processing
        # del waveform

        # # Extract speech segments
        # speech_segments = [
        #     [seg.start, seg.end]
        #     for seg, _, label in vad.itertracks(yield_label=True)
        #     if label == "SPEECH"
        # ]
        speech_segments = [[seg.start, seg.end] for seg in vad.get_timeline().support()]

        # # Free VAD result
        # del vad

        speech_segments = (
            np.array(speech_segments, dtype=np.float32)
            if speech_segments
            else np.empty((0, 2), dtype=np.float32)
        )

        file_id = audio_path.stem

        # Compute statistics
        speech_stats = compute_segment_stats(speech_segments)
        nospch_segments = compute_nospch_segments(speech_segments, duration)
        nospch_stats = compute_segment_stats(nospch_segments)

        metadata = {
            "file_id": file_id,
            "audio_filepath": str(audio_path),
            "duration": float(duration),
            "max-speech": speech_stats["max"],
            "min-speech": speech_stats["min"],
            "avg-speech": speech_stats["avg"],
            "total-speech": speech_stats["total"],
            "count-speech": speech_stats["count"],
            "max-nospch": nospch_stats["max"],
            "min-nospch": nospch_stats["min"],
            "avg-nospch": nospch_stats["avg"],
            "total-nospch": nospch_stats["total"],
            "count-nospch": nospch_stats["count"],
            "spch-ratio": speech_stats["total"] / duration if duration > 0 else 0.0,
        }

        durations = {
            "speech_durations": speech_stats["durations"],
            "nospch_durations": nospch_stats["durations"],
        }

        return metadata, durations, speech_segments.tolist()

    except Exception as e:
        return {"audio_filepath": str(audio_path), "error": str(e)}, {}, []


def process_files(
    paths: list[Path],
    pipeline,
    metadata_file_path: Path,
    durations_file_path: Path,
    segments_file_path: Path,
    task_id: int,
) -> tuple[int, int]:
    """Process all files and write per-task shard files."""

    # Create dataset and dataloader
    dataset = AudioDataset(paths)
    dataloader = DataLoader(dataset, num_workers=8, batch_size=None, pin_memory=True)

    processed = 0
    errors = 0
    window_size = 1000
    recent_times = []
    start_time = time.time()

    with (
        open(metadata_file_path, "w") as metadata_file,
        open(durations_file_path, "w") as durations_file,
        open(segments_file_path, "w") as segments_file,
        torch.no_grad(),
    ):
        for i, audio_file in enumerate(dataloader, 1):
            file_start = time.time()
            path = audio_file["path"]

            # # Periodic GPU memory cleanup to prevent fragmentation
            # if i % 500 == 0 and torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            if "error" in audio_file:
                print(f"ERROR {path.name}: {audio_file['error']}", file=sys.stderr)
                metadata_file.write(
                    json.dumps(
                        {
                            "audio_filepath": str(path),
                            "error": audio_file["error"],
                            "task_id": task_id,
                        }
                    )
                    + "\n"
                )
                errors += 1
                continue

            waveform, sample_rate = audio_file["waveform"], audio_file["sample_rate"]
            duration = waveform.shape[1] / sample_rate

            # Adaptive progress reporting with increasing intervals
            log_interval = get_log_interval(i)
            should_log = (i % log_interval == 0) or (i == len(paths))

            if should_log:
                log_progress(i, len(paths), start_time, recent_times)

            metadata, durations, speech_segments = process_single_file(path, pipeline)

            # Track processing time for rolling average
            file_elapsed = time.time() - file_start
            recent_times.append(file_elapsed)
            if len(recent_times) > window_size:
                recent_times.pop(0)

            # Always write metadata record
            record = {**metadata, "task_id": task_id}
            metadata_file.write(json.dumps(record) + "\n")
            # metadata_file.flush()

            # Successfully processed
            processed += 1
            file_id = metadata["file_id"]
            audio_filepath = metadata["audio_filepath"]

            duration_record = {
                "file_id": file_id,
                "audio_filepath": audio_filepath,
                "speech_durations": durations.get("speech_durations", []),
                "nospch_durations": durations.get("nospch_durations", []),
            }
            durations_file.write(json.dumps(duration_record) + "\n")

            segment_record = {
                "file_id": file_id,
                "audio_filepath": audio_filepath,
                "speech_segments": speech_segments,
            }
            segments_file.write(json.dumps(segment_record) + "\n")

    elapsed = time.time() - start_time
    print(f"Completed {processed} files in {elapsed:.1f}s")

    return processed, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAD Pipeline for audio processing")
    parser.add_argument("manifest", type=str, help="Path to the manifest .txt file")
    parser.add_argument("--array-id", type=int, default=0, help="SLURM array task ID")
    parser.add_argument("--array-count", type=int, default=1, help="Total array tasks")
    args = parser.parse_args()

    print(f"Task {args.array_id}/{int(args.array_count) - 1} starting")

    # Load environment
    load_env()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError(
            "HF_TOKEN not found. Copy .env.example to .env and add token, "
            "or: export HF_TOKEN='your_token_here'"
        )

    # Setup model and pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = setup_model_and_pipeline(hf_token, device)

    # Parse manifest and shard by array ID
    manifest_path = Path(args.manifest)
    all_paths = parse_manifest(manifest_path)

    # Each task gets a contiguous chunk instead of round-robin
    # Maximizes NFS cache hits since files are sorted by subdir
    # Last task gets any remainder
    total = len(all_paths)
    chunk_size = total // args.array_count
    start_idx = args.array_id * chunk_size
    end_idx = total if args.array_id == args.array_count - 1 else start_idx + chunk_size
    paths = all_paths[start_idx:end_idx]

    print(f"Processing {len(paths)} files")
    print(f"Total files: {len(all_paths)} files")

    # Setup output directories and file paths
    output_paths = setup_output_dirs(manifest_path, args.array_id)
    print(f"\nOutput: {output_paths['output_dir']}")

    # Process files
    processed, errors = process_files(
        paths,
        pipeline,
        output_paths["task_metadata_file"],
        output_paths["task_durations_file"],
        output_paths["task_segments_file"],
        args.array_id,
    )

    if errors > 0:
        print(f"WARNING: {errors} errors encountered", file=sys.stderr)

    print(f"Task {args.array_id} complete")
