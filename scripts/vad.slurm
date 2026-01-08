#!/bin/bash
#SBATCH --job-name=vad
#SBATCH --output=logs/vad_%A_%a.out
#SBATCH --error=logs/vad_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=erc-dupoux,erc-cristia
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

if [ -z "$1" ]; then
    echo "Usage: sbatch vad.slurm <MANIFEST_FILE>"
    echo "Example: sbatch vad.slurm manifests/mydataset.txt"
    echo "Edit --array=0-3 in script to control number of parallel jobs"
    exit 1
fi

MANIFEST="$1"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    exit 1
fi

echo "VAD Pipeline | Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Manifest: $MANIFEST"
echo "Started: $(date)"

export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:$LD_LIBRARY_PATH

module purge
module load ffmpeg

echo ""
echo "Setting up environment..."
echo ""
echo "Node: $(hostname)"
echo "Python: $(which python3)"
echo "UV: $(which uv)"
echo "UV version: $(uv --version)"
echo "Working dir: $(pwd)"
echo "Venv exists: $([ -d .venv ] && echo 'yes' || echo 'no')"
echo "Venv lock: $([ -f .venv/.lock ] && echo 'exists' || echo 'missing')"

# uv run python -c "import sys; print('Python sys.path:', sys.path, sys.executable)"

echo ""
echo "Running VAD Pipeline..."
echo ""
PYTHONUNBUFFERED=1 UV_VERBOSE=1 uv run --no-sync vad.py "$MANIFEST" --array-id "$SLURM_ARRAY_TASK_ID" --array-count "$SLURM_ARRAY_TASK_COUNT"
EXIT_CODE=$?

echo ""
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
