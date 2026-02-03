#!/bin/bash
#SBATCH --job-name=cmp_models
#SBATCH --output=logs/train/compare_%j.out
#SBATCH --error=logs/train/compare_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2

echo "LSTM vs GPT2 Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

uv run --frozen scripts/train/debug_compare_models.py

echo
echo "Completed: $(date)"
