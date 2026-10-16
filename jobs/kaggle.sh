#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/thong-phn/fall-balloom.git"
PROJECT_DIR="/kaggle/working/fall-balloom"
DATA_DIR="/kaggle/input/datasets/thongp/biosardbscene-circo/BioSARdBScene-CirCo"

if [[ ! -d "$PROJECT_DIR/.git" ]]; then
    git clone --depth 1 "$REPO_URL" "$PROJECT_DIR"
fi

python "$PROJECT_DIR/jobs/kaggle_run.py" \
    --metadata "$DATA_DIR/metadata.csv" \
    --data "$DATA_DIR" \
    --epochs 1 \
    --num-workers 4 \
    --output /kaggle/working/biosarcirco-kaggle.pt
