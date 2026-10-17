"""
Python runner script.
"""

import argparse
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/thong-phn/fall-balloom.git"
PROJECT_DIR = Path("/kaggle/working/fall-balloom")
DATA_DIR = Path("/kaggle/input/datasets/thongp/biosardbscene-circo/BioSARdBScene-CirCo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DATA_DIR / "metadata.csv")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not (PROJECT_DIR / ".git").is_dir():
        PROJECT_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(PROJECT_DIR)],
            check=True,
        )

    project_dir = PROJECT_DIR
    data_dir = args.data or args.metadata.parent
    train_script = project_dir / "binary-heart-biosar.py"
    commit = subprocess.run(
        ["git", "-C", str(project_dir), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output_path = args.output or Path(f"/kaggle/working/biosarcirco-kaggle-{commit}.pt")

    subprocess.run(
        [
            sys.executable, str(train_script),
            "--metadata", str(args.metadata),
            "--data", str(data_dir),
            "--epochs", str(args.epochs),
            "--num-workers", str(args.num_workers),
            "--output", str(output_path),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
