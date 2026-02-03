#!/bin/bash
#SBATCH --job-name=debug_lstm
#SBATCH --output=logs/train/debug_%j.out
#SBATCH --error=logs/train/debug_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2

# Usage: sbatch scripts/train/debug.slurm [test_number]
#   test_number: 0=all (default), 1=data stats, 2=loss calc, 3=gradients, 4=training, 5=precision

TEST_NUM=${1:-0}

echo "LSTM Debugging Script"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Test: $TEST_NUM (0=all, 1=data, 2=loss, 3=grad, 4=train, 5=precision)"
echo "Started: $(date)"

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

uv run --frozen scripts/train/debug_lstm.py --test "$TEST_NUM"

echo ""
echo "Completed: $(date)"
