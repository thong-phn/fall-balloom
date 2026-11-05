"""Kaggle entry point. A Kaggle script kernel gets no command line, so set the arguments here."""
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/thong-phn/fall-balloom.git"
BRANCH = "manual-biosar-exp"
PROJECT_DIR = Path("/kaggle/working/fall-balloom")
METADATA_SLUG = "biosardbscence-bmd"  # holds the metadata.csv (rows, labels, split)
AUDIO_SLUG = "bmd-original"  # holds the clean recordings in train/

# Edit these, then run `kaggle kernels push -p jobs/kaggle`.
ARGS = [
    "--dataset", "biosar-bmd",  # train and val on the BioSAR mixtures
    "--test-dataset", "bmd",  # test on the clean recordings
    "--model", "simple-cnn",
    "--epochs", "100",
    "--num-workers", "4",
    "--output", "/kaggle/working/heart-biosar-bmd-test-bmd.pt",
]

if not (PROJECT_DIR / ".git").is_dir():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(PROJECT_DIR)], check=True
    )

# Kaggle mounts a dataset at /kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug>.
# Search by slug so both layouts work.
inputs = Path("/kaggle/input")
metadata = next(p for p in inputs.rglob("metadata.csv") if METADATA_SLUG in p.parts)
audio = next(p for p in inputs.rglob("train") if p.is_dir() and AUDIO_SLUG in p.parts)
subprocess.run(
    [sys.executable, "-u", str(PROJECT_DIR / "biosar-heart.py"),
     "--metadata", str(metadata), "--data", str(metadata.parent), "--test-data", str(audio), *ARGS],
    check=True,
    cwd=PROJECT_DIR,
)
