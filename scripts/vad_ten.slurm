#!/bin/bash
#SBATCH --job-name=vad-ten
#SBATCH --output=logs/vad-ten_%j.out
#SBATCH --error=logs/vad-ten_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu,erc-dupoux,gpu-p1,gpu-p2

# Validate dataset
if [ -z "$1" ]; then
    echo "Usage: sbatch vad_ten.slurm <MANIFEST_FILE>"
    echo "Example: sbatch vad_ten.slurm manifests/mydataset.txt"
    exit 1
fi

MANIFEST="$1"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    exit 1
fi

mkdir -p logs

echo "TenVAD Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Manifest: $MANIFEST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}"
echo "Started: $(date)"

export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/llvm-17.0.6-ibcxgmojxutzsgeoyywyraq5fmvycsfl/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH

module purge
module load ffmpeg llvm

# Use .venv directly to skip uv checks
PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Error: Virtual environment python not found at $PYTHON"
    exit 1
fi

PYTHONUNBUFFERED=1 \
$PYTHON scripts/vad_ten.py "$MANIFEST" \
    --workers "$SLURM_CPUS_PER_TASK"

echo ""
echo "VAD completed: $(date)"
