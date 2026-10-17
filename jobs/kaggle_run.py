"""
Python runner script.
"""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path,
                        help="Path to the attached CirCo metadata.csv")
    parser.add_argument("--data", type=Path,
                        help="Dataset root (defaults to the metadata folder)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/biosarcirco-kaggle.pt"))
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    data_dir = args.data or args.metadata.parent
    train_script = project_dir / "binary-heart-biosar.py"

    subprocess.run(
        [
            sys.executable, str(train_script),
            "--metadata", str(args.metadata),
            "--data", str(data_dir),
            "--epochs", str(args.epochs),
            "--num-workers", str(args.num_workers),
            "--output", str(args.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
