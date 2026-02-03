#!/bin/bash
#SBATCH --job-name=swuggy
#SBATCH --output=logs/swuggy/%x_%j.out
#SBATCH --error=logs/swuggy/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=erc-dupoux

# Usage: sbatch scripts/swuggy/run.slurm <script_name>
#   script_name: utils, evaluate, or analysis (without .py)

if [ -z "$1" ]; then
    echo "Usage: sbatch scripts/swuggy/run.slurm <script_name>"
    echo "  script_name: utils, evaluate, or analysis"
    exit 1
fi

SCRIPT_NAME="$1"
SCRIPT_PATH="scripts/swuggy/${SCRIPT_NAME}.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found: $SCRIPT_PATH"
    exit 1
fi

echo "sWuggy: ${SCRIPT_NAME}.py"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

uv run --frozen "$SCRIPT_PATH"

echo ""
echo "Completed: $(date)"
