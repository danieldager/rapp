#!/usr/bin/env python3
"""scan.py

One-time dataset scan to produce a lightweight manifest of audio file paths.

Writes one absolute file path per line, with a small commented header.
"""

import argparse
import os
from pathlib import Path
from typing import List
from multiprocessing import Pool, cpu_count


AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
}


def scan_directory_recursive(dirpath: str) -> List[Path]:
    """Recursively scan a directory tree for audio files (single-threaded per tree)."""
    results = []
    try:
        for root, _, files in os.walk(dirpath):
            for filename in files:
                if Path(filename).suffix.lower() in AUDIO_EXTENSIONS:
                    results.append(Path(root, filename).resolve())
    except (PermissionError, OSError):
        pass  # Skip directories we can't read
    return results


def iter_audio_files_parallel(root: Path, num_workers: int | None) -> List[Path]:
    """Parallel scan by distributing top-level subdirectories across workers."""
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Cap at 8 to avoid overwhelming NFS

    # Get immediate subdirectories to distribute across workers
    print(
        f"Finding top-level directories to distribute across {num_workers} workers..."
    )
    try:
        subdirs = [
            str(Path(root, entry.name)) for entry in os.scandir(root) if entry.is_dir()
        ]
    except (PermissionError, OSError):
        subdirs = []

    # If no subdirs, just scan root directly
    if not subdirs:
        print(f"No subdirectories found, scanning root directly...")
        return scan_directory_recursive(str(root))

    print(f"Found {len(subdirs)} top-level directories, scanning in parallel...")

    # Scan each subtree in parallel
    all_files = []
    with Pool(num_workers) as pool:
        for i, results in enumerate(
            pool.imap_unordered(scan_directory_recursive, subdirs, chunksize=1), 1
        ):
            all_files.extend(results)
            print(
                f"  Completed {i}/{len(subdirs)} top-level directories, found {len(all_files)} files so far..."
            )

    return all_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset root directory")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.exists() or not dataset.is_dir():
        raise SystemExit(
            f"ERROR: dataset does not exist or is not a directory: {dataset}"
        )

    manifests_dir = Path("manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)
    output_path = (manifests_dir / f"{dataset.name}.txt").resolve()

    print(f"Scanning {dataset}...")
    paths = iter_audio_files_parallel(dataset, num_workers=args.workers)

    print(f"\nFound {len(paths):,} total files")
    print(
        f"Sorting by (directory, filename) for NFS cache locality during VAD processing..."
    )
    # Sort by directory first, then filename
    # This ensures sequential reads during VAD stay in same directory → NFS cache hits
    paths.sort(key=lambda p: (str(p.parent), p.name))

    print(f"Writing {len(paths):,} paths to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(paths, 1):
            f.write(str(p) + "\n")
            if i % 100000 == 0:
                print(f"  Written {i:,}/{len(paths):,} paths...")

    print(f"\nWrote {len(paths):,} paths to {output_path} (sorted by directory)")
    print(
        f"VAD tasks using contiguous chunks will benefit from NFS directory cache hits."
    )
