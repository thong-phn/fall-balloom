"""Kaggle entry point. A Kaggle script kernel gets no command line, so set the arguments here."""
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/thong-phn/fall-balloom.git"
BRANCH = "manual-biosar-exp"
PROJECT_DIR = Path("/kaggle/working/fall-balloom")
DATASET_SLUG = "biosardbscence-bmd"

# Edit these, then run `kaggle kernels push -p jobs/kaggle`.
ARGS = [
    "--dataset", "biosar-bmd",
    "--model", "simple-cnn",
    "--epochs", "100",
    "--num-workers", "4",
    "--output", "/kaggle/working/heart-biosar-bmd.pt",
]

if not (PROJECT_DIR / ".git").is_dir():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(PROJECT_DIR)], check=True
    )

# Kaggle mounts the dataset at /kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug>.
# Search for it by slug so both layouts work.
metadata = next(p for p in Path("/kaggle/input").rglob("metadata.csv") if DATASET_SLUG in p.parts)
subprocess.run(
    [sys.executable, "-u", str(PROJECT_DIR / "biosar-heart.py"),
     "--metadata", str(metadata), "--data", str(metadata.parent), *ARGS],
    check=True,
    cwd=PROJECT_DIR,
)
