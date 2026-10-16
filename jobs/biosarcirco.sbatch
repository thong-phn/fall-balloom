#!/usr/bin/env bash
# Submit from the project root. If needed, pass your Alliance account:
#   sbatch --account=def-<PI> jobs/biosarcirco.sbatch
# Create the environment once on Narval with:
#   module load StdEnv/2023 python/3.14
#   virtualenv --no-download .venv
#   source .venv/bin/activate
#   pip install --no-index torch torchaudio torchcodec soundfile numpy
#
#SBATCH --job-name=biosarcirco
#SBATCH --output=jobs/%x-%j.out
#SBATCH --error=jobs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=a100_1g.5gb:1
#SBATCH --account=def-sartmu

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="$PROJECT_DIR/datasets/BioSARdBScene-CirCo"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
CHECKPOINT_DIR="$PROJECT_DIR/models/ckpt"

cd "$PROJECT_DIR"
module load StdEnv/2023 python/3.14

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Missing Python environment: $VENV_DIR" >&2
    echo "Create it on Narval with the same Python module, then install torch and torchaudio." >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"
# Keep each worker's math libraries single-threaded to avoid oversubscribing CPUs.
export OMP_NUM_THREADS=1

if [[ ! -f "$DATA_DIR/metadata.csv" ]]; then
    echo "Missing CirCo metadata: $DATA_DIR/metadata.csv" >&2
    exit 1
fi

mkdir -p "$CHECKPOINT_DIR"
CHECKPOINT="$CHECKPOINT_DIR/biosarcirco-${SLURM_JOB_ID}.pt"

python -c 'import torch, torchaudio; print("PyTorch:", torch.__version__); print("CUDA:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"); assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"'

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python -u "$PROJECT_DIR/binary-heart-circo.py" \
    --metadata "$DATA_DIR/metadata.csv" \
    --data "$DATA_DIR" \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --output "$CHECKPOINT"
