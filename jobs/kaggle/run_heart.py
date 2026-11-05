"""Kaggle entry point. A Kaggle script kernel gets no command line, so set the arguments here."""
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/thong-phn/fall-balloom.git"
BRANCH = "manual-biosar-exp"
PROJECT_DIR = Path("/kaggle/working/fall-balloom")
METADATA_SLUG = "biosardbscence-bmd"  # holds the metadata.csv (rows, labels, split)
AUDIO_SLUG = "bmd-original"  # holds the clean recordings in train/

TRAIN = "bmd"  # train and val on this dataset: "bmd" (clean) or "biosar-bmd" (mixtures)
TEST = "biosar-bmd"  # test on this dataset

# Edit these, then run `kaggle kernels push -p jobs/kaggle`.
ARGS = [
    "--dataset", TRAIN,
    "--test-dataset", TEST,
    "--model", "simple-cnn",
    "--epochs", "100",
    "--num-workers", "4",
    "--output", f"/kaggle/working/heart-{TRAIN}-test-{TEST}.pt",
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
folders = {"biosar-bmd": metadata.parent, "bmd": audio}
subprocess.run(
    [sys.executable, "-u", str(PROJECT_DIR / "biosar-heart.py"),
     "--metadata", str(metadata), "--data", str(folders[TRAIN]), "--test-data", str(folders[TEST]), *ARGS],
    check=True,
    cwd=PROJECT_DIR,
)
