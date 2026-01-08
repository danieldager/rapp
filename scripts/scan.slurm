#!/bin/bash
#SBATCH --job-name=scan
#SBATCH --output=logs/scan_%j.out
#SBATCH --error=logs/scan_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --partition=erc-dupoux,erc-cristia

if [ -z "$1" ]; then
    echo "Usage: sbatch scan.slurm <DATASET_DIR>"
    echo "Example: sbatch scan.slurm /store/projects/lexical-benchmark/audio/"
    exit 1
fi

DATASET="$1"

if [ ! -d "$DATASET" ]; then
    echo "ERROR: Dataset directory not found: $DATASET"
    exit 1
fi

echo "Scan Job | Job ID: $SLURM_JOB_ID | CPUs: $SLURM_CPUS_PER_TASK"
echo "Dataset: $DATASET"
echo "Started: $(date)"
echo ""

# Use all allocated CPUs for parallel scanning
# -u flag disables Python stdout buffering for real-time logging
uv run python -u scan.py "$DATASET" --workers "$SLURM_CPUS_PER_TASK"

echo ""
echo "Completed: $(date)"
