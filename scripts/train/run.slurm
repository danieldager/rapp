#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/train/train_%j.out
#SBATCH --error=logs/train/train_%j.err
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Setup environment
# source ~/.bashrc
# conda activate amela_train

# Extract number of GPUs from SLURM gres parameter
NUM_GPUS=$(echo $SLURM_STEP_GPUS | tr "," "\n" | wc -l)

# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Add current working directory to Python Path
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

uv run --frozen torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    scripts/train/train.py \

echo "Completed at: $(date)"