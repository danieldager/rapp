#!/bin/bash
#SBATCH --job-name=encode_array
#SBATCH --output=logs/encode/encode_%A_%a.out
#SBATCH --error=logs/encode/encode_%A_%a.err
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=erc-dupoux

# Validate inputs
if [ -z "$1" ]; then
    echo "Usage: sbatch scripts/encode/run.slurm <manifest_path>"
    exit 1
fi

MANIFEST_PATH="$1"
if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: Manifest not found: $MANIFEST_PATH"
    exit 1
fi

# Hardcoded model configuration
MODEL="spidr_base"
MODEL_NAME="spidr_base"

NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "Tokenize Waveforms"
echo "Job ID: $SLURM_JOB_ID | Task ID: $TASK_ID / $((NUM_TASKS - 1))"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Manifest: $MANIFEST_PATH"
echo "Model: $MODEL"
echo "Token Directory: tokens / {dataset}_$MODEL_NAME"
echo "Started: $(date)"

PYTHONUNBUFFERED=1 \
uv run --frozen scripts/encode/encode.py \
    --manifest "$MANIFEST_PATH" \
    --model "$MODEL" \
    --model-name "$MODEL_NAME" \
    --task-id "$TASK_ID" \
    --num-tasks "$NUM_TASKS" \
    --device cuda

echo ""
echo "Task $TASK_ID completed: $(date)"
