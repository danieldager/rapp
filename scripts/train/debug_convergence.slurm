#!/bin/bash
#SBATCH --job-name=debug_conv
#SBATCH --output=logs/train/convergence_%j.out
#SBATCH --error=logs/train/convergence_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1

echo "LSTM Convergence Variance Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo

# Export Python path for imports
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

uv run --frozen scripts/train/debug_convergence.py

echo
echo "Completed: $(date)"
