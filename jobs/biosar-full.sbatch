#!/usr/bin/env bash
# Submitted by jobs/submit_vulcan.sh after syncing the current local source.
#SBATCH --job-name=biosar-full
#SBATCH --output=jobs/%x-%j.out
#SBATCH --error=jobs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
# Vulcan's documented GPU model is L40S; a100_1g.5gb is a Narval MIG type.
#SBATCH --gpus-per-node=l40s:1

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/datasets/BioSARDdBScene}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
MODEL="${MODEL:-attn-cnn}"
CHECKPOINT_DIR="$PROJECT_DIR/models/ckpt"

cd "$PROJECT_DIR"
module load StdEnv/2023 python/3.12
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Missing Python environment: $VENV_DIR" >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"
export OMP_NUM_THREADS=1

METADATA="$DATA_DIR/metadata.csv"
if [[ ! -f "$METADATA" ]]; then
    echo "Missing BioSAR metadata: $METADATA" >&2
    exit 1
fi

mkdir -p "$CHECKPOINT_DIR"
CHECKPOINT="$CHECKPOINT_DIR/biosar-full-${SLURM_JOB_ID}-$(date +%y%m%d).pt"

echo "Python executable: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
nvidia-smi -L || true
python - <<'PY'
import os
import sys
import torch
import torchaudio

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    torch.cuda.init()
    print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    raise RuntimeError(
        "PyTorch could not initialize CUDA. Check the GPU visibility and driver "
        "diagnostics above, then confirm this environment's PyTorch CUDA build "
        "is compatible with Vulcan's installed NVIDIA driver."
    ) from exc
PY

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python -u "$PROJECT_DIR/binary-biosar-full.py" \
    --metadata "$METADATA" \
    --data "$DATA_DIR" \
    --model "$MODEL" \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --output "$CHECKPOINT"
