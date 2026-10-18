"""Kaggle entry point: runs a training script with Kaggle paths and defaults filled in.

  python kaggle_run.py [--script binary-biosar-full.py] [--data-dir DIR] [--output PT] [--cache-dir DIR | --no-cache] [-- extra script args]

--data-dir holds metadata.csv and audio/ (read-only under /kaggle/input). The decoded 4 kHz waveforms are cached under --cache-dir (default /kaggle/working/cache) and
pre-built in parallel on the first run; cached outputs are identical to the uncached pipeline.
Anything after `--`
is passed to the script unchanged, so e.g. `-- --seed 7 --early-stop 10` works.
"""
import argparse
import csv
import importlib.util
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = "/kaggle/input/datasets/thongp/biosardb-full/BioSARDdBScene"


def _cache_one(path):
    from utils.preprocess import load_waveform
    load_waveform(path)


def build_cache(data, cache_dir):
    """Decode every wav once (all cores). A .complete marker makes later runs skip this."""
    marker = Path(cache_dir) / ".complete"
    if marker.exists():
        return
    with open(data / "metadata.csv", newline="") as f:
        paths = sorted({str(data / row["audio_path"]) for row in csv.DictReader(f)})
    print(f"building waveform cache: {len(paths)} files -> {cache_dir}", flush=True)
    with mp.Pool(os.cpu_count() or 2) as pool:
        for i, _ in enumerate(pool.imap_unordered(_cache_one, paths, chunksize=64), 1):
            if i % 5000 == 0:
                print(f"  cached {i}/{len(paths)}", flush=True)
    marker.touch()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", default="binary-biosar-full.py")
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--output", default="/kaggle/working/ckpt/biosar-8class-full.pt")
    p.add_argument("--cache-dir", default="/kaggle/working/cache")
    p.add_argument("--no-cache", action="store_true")
    args, extra = p.parse_known_args()
    extra = [a for a in extra if a != "--"]

    data = Path(args.data_dir)
    if not (data / "metadata.csv").is_file():
        sys.exit(f"metadata.csv not found in {data}; check the dataset is attached to the notebook")

    # Kaggle images ship torch/torchaudio; only pull what is missing.
    for mod, pkg in (("soundfile", "soundfile"), ("torchcodec", "torchcodec")):
        if importlib.util.find_spec(mod) is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

    import torch
    if not torch.cuda.is_available():
        sys.exit("no GPU: enable a GPU accelerator in the Kaggle notebook settings")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(HERE), OMP_NUM_THREADS="1")
    if not args.no_cache:
        os.environ["BIOSAR_CACHE_DIR"] = env["BIOSAR_CACHE_DIR"] = args.cache_dir
        sys.path.insert(0, str(HERE))
        build_cache(data, args.cache_dir)
    cmd = [sys.executable, "-u", str(HERE / args.script),
           "--metadata", str(data / "metadata.csv"), "--data", str(data),
           "--num-workers", str(os.cpu_count() or 2), "--output", args.output, *extra]
    sys.exit(subprocess.run(cmd, env=env, cwd=HERE).returncode)


if __name__ == "__main__":
    main()
