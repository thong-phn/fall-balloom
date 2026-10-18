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
# Clean CirCo recordings (override with --data)
CIRCO_DATA_DIR = Path("/kaggle/input/datasets/thongp/my-circo/CirCo/training_data/training_data")
# Full BioSARdB scene dataset (override with --data/--metadata)
FULL_DATA_DIR = Path("/kaggle/input/datasets/thongp/biosardbscene/BioSARDdBScene")
SCRIPTS = {
    "biosar-full": "binary-heart-biosar-full.py",  # BioSARdB mixtures, full dataset
    "biosar": "binary-heart-biosar.py",
    "circo": "binary-heart-circo.py",
    "mix": "binary-heart-mix.py",  # train on clean CirCo, test on BioSARdB
}
MULTI_GPU_SCRIPT = "dual-gpu.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, help="metadata.csv (defaults per --dataset)")
    parser.add_argument("--dataset", choices=SCRIPTS, default="biosar")
    parser.add_argument("--data", type=Path, help="audio root (defaults per --dataset)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4, help="total, split across GPUs")
    parser.add_argument("--gpus", type=int, default=1, help="2 = DDP over both T4s (dual-gpu.py)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.dataset in ("mix", "biosar-full") and args.gpus > 1:
        parser.error(f"--dataset {args.dataset} only supports --gpus 1 (dual-gpu.py has no {args.dataset} mode)")

    if not (PROJECT_DIR / ".git").is_dir():
        PROJECT_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(PROJECT_DIR)],
            check=True,
        )

    project_dir = PROJECT_DIR
    if args.dataset == "biosar-full":
        default_data = FULL_DATA_DIR
    elif args.dataset in ("circo", "mix"):
        default_data = CIRCO_DATA_DIR
    else:
        default_data = DATA_DIR
    data_dir = args.data or default_data
    default_metadata = (FULL_DATA_DIR if args.dataset == "biosar-full" else DATA_DIR) / "metadata.csv"
    metadata = args.metadata or default_metadata
    commit = subprocess.run(
        ["git", "-C", str(project_dir), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output_path = args.output or Path(f"/kaggle/working/{args.dataset}-kaggle-{commit}.pt")

    train_args = [
        "--metadata", str(metadata),
        "--data", str(data_dir),
        "--epochs", str(args.epochs),
        "--output", str(output_path),
    ]
    if args.dataset == "mix":
        train_args += ["--test-data", str(DATA_DIR)]
    if args.gpus > 1:
        command = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={args.gpus}",
            str(project_dir / MULTI_GPU_SCRIPT),
            "--dataset", args.dataset,
            "--num-workers", str(max(1, args.num_workers // args.gpus)),
            *train_args,
        ]
    else:
        command = [
            sys.executable, str(project_dir / SCRIPTS[args.dataset]),
            "--num-workers", str(args.num_workers),
            *train_args,
        ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
