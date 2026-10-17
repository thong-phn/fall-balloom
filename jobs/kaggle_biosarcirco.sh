#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/thong-phn/fall-balloom.git"
PROJECT_DIR="/kaggle/working/fall-balloom"
DATA_DIR="/kaggle/input/datasets/thongp/biosardbscene-circo/BioSARdBScene-CirCo"

# Check for git dir
if [[ ! -d "$PROJECT_DIR/.git" ]]; then
    git clone --depth 1 "$REPO_URL" "$PROJECT_DIR"
fi

# Get git commit 
COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD)

# Run the script
python "$PROJECT_DIR/jobs/kaggle_run.py" \
    --metadata "$DATA_DIR/metadata.csv" \
    --data "$DATA_DIR" \
    --epochs 100 \
    --num-workers 4 \
    --output /kaggle/working/biosarcirco-kaggle-${COMMIT}.pt
