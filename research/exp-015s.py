"""exp-015s: OPERA-CT environment / load / front-end / timing smoke on the GPU (Stage 0 of exp-015). No training result.

Prints: checkpoint sha256 and key audit (strict load of the vendored encoder), a check that the waveform crop is the
student's crop (mel of the raw crop == utils.preprocess features under the same RNG state), embedding statistics of 64
real training crops for f_max 8000 and 2000, fp32-vs-bf16 embedding agreement, inference timing and peak memory, a short
train-mode fwd+bwd timing (batch 32, bf16 autocast, AdamW; on a copy, nothing is saved) to size exp-015a-c, and one
`curl -sI https://huggingface.co` to document compute-node internet access (not needed by the primary path).
Accepts the hpc/job.sbatch.tmpl arguments (--metadata --data --num-workers --output); --output is not written.
"""
import argparse
import copy
import hashlib
import os
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.datasets import BioSARMulti
from utils.opera_htsat import OPERA_CKPT_SHA256, OPERA_REVISION, OperaCTEncoder, load_opera_ct, opera_embed
from utils.preprocess import CLIP_SAMPLES, load_waveform, mel, preprocess
from utils.utils import set_seed


def wave_clips(path, train=False):
    """utils.preprocess.preprocess line for line, returning the 4 kHz waveform clips [n_clips, 1, 20000] (no mel)."""
    audio = load_waveform(path)
    length = audio.shape[-1]
    if length < CLIP_SAMPLES:
        audio = F.pad(audio, (0, CLIP_SAMPLES - length))
        length = CLIP_SAMPLES
    if train:
        start = torch.randint(length - CLIP_SAMPLES + 1, ()).item()
        clips = audio[:, start: start + CLIP_SAMPLES].unsqueeze(0)
    else:
        last_start = length - CLIP_SAMPLES
        starts = list(range(0, last_start + 1, CLIP_SAMPLES))
        if starts[-1] != last_start:
            starts.append(last_start)
        clips = torch.stack([audio[:, start: start + CLIP_SAMPLES] for start in starts])
    return clips


class BioSARMultiWave(BioSARMulti):
    def __getitem__(self, index):
        item = self.items[index]
        return wave_clips(str(item["audio_path"]), train=self.train), torch.tensor(item["label"], dtype=torch.long), item["mixture_id"]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def stats(name, e):
    print(f"emb_stats[{name}]: shape={tuple(e.shape)} finite={bool(torch.isfinite(e).all())} mean={e.mean():.4f} std={e.std():.4f} "
          f"min={e.min():.4f} max={e.max():.4f} per_dim_std_mean={e.std(dim=0).mean():.4f} "
          f"mean_pairwise_cos={F.cosine_similarity(e[:, None], e[None], dim=-1).mean():.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="exp-015s: OPERA-CT smoke")
    ap.add_argument("--metadata", default="./datasets/BioSARDdBScene/metadata.csv")
    ap.add_argument("--data", default="./datasets/BioSARDdBScene/")
    ap.add_argument("--output", default="models/ckpt/exp-015s.pt", help="accepted for the job template; nothing is written")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--ckpt", default="models/foundation/encoder-operaCT.ckpt")
    ap.add_argument("--n-crops", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=20, help="train-mode fwd+bwd timing steps (0 = skip)")
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--random-init", type=int, choices=[0, 1], default=0, help="CPU test only")
    ap.add_argument("--skip-curl", type=int, choices=[0, 1], default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"exp-015s: torch={torch.__version__} device={device} "
          f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else '-'} opera_code={OPERA_REVISION}", flush=True)

    # --- checkpoint + key audit
    if args.random_init:
        enc = OperaCTEncoder().eval()
        sha, rep = "random-init", {}
        print("encoder: RANDOM INIT (test mode)")
    else:
        h = hashlib.sha256()
        with open(args.ckpt, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        sha = h.hexdigest()
        print(f"ckpt_sha256: {sha} expected={OPERA_CKPT_SHA256} match={sha == OPERA_CKPT_SHA256} bytes={os.path.getsize(args.ckpt)}")
        assert sha == OPERA_CKPT_SHA256, "checkpoint hash mismatch"
        enc, rep = load_opera_ct(args.ckpt)
        print(f"key_audit: loaded={rep['loaded']} missing={len(rep['missing'])} unexpected_other={len(rep['unexpected_other'])} "
              f"ignored_extractor_projection={rep['unexpected_ignored']} shape_mismatch={len(rep['shape_mismatch'])} ckpt_keys={rep['n_ckpt']}")
    print(f"encoder_params: {sum(p.numel() for p in enc.parameters()):,}")
    enc.to(device).eval()

    # --- student's crop: raw 4 kHz waveform crop with the same single random call as utils.preprocess
    ds = BioSARMultiWave(data_path=args.data, metadata_path=args.metadata, split="train", train=True)
    ref = BioSARMulti(data_path=args.data, metadata_path=args.metadata, split="train", train=True)
    bad = 0
    for i in range(0, min(len(ds), 800), 100):
        torch.manual_seed(1000 + i)
        clips = wave_clips(str(ds.items[i]["audio_path"]), train=True)
        feat = torch.log(mel(clips) + 1e-6)
        torch.manual_seed(1000 + i)
        want = preprocess(str(ref.items[i]["audio_path"]), train=True)
        bad += int(not torch.equal(feat, want))
    print(f"crop_parity_with_preprocess: items=8 mismatches={bad}")
    assert bad == 0
    set_seed(args.seed)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=max(0, args.num_workers), drop_last=True)
    waves, labels = [], []
    for clips, lab, _ in loader:
        waves.append(clips.squeeze(2).squeeze(1))
        labels.append(lab)
        if sum(w.shape[0] for w in waves) >= args.n_crops:
            break
    waves = torch.cat(waves)[: args.n_crops]
    labels = torch.cat(labels)[: args.n_crops]
    print(f"crops: {tuple(waves.shape)} (4 kHz, 5 s) classes={sorted(set(labels.tolist()))} wave_std={waves.std():.4f} wave_absmax={waves.abs().max():.3f}")

    # --- inference: stats, timing, memory, fp32 vs bf16
    res = {}
    for fm in (8000, 2000):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            opera_embed(enc, waves[:args.batch_size].to(device), f_max=fm)  # warm-up
            sync(device)
            t0 = time.time()
            embs = [opera_embed(enc, waves[i: i + args.batch_size].to(device), f_max=fm).float().cpu() for i in range(0, len(waves), args.batch_size)]
            sync(device)
            dt = time.time() - t0
        e = torch.cat(embs)
        res[fm] = e
        mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
        print(f"infer[f_max={fm}]: {len(e)} crops in {dt:.2f}s = {dt / len(e) * 1000:.1f} ms/crop, batch {args.batch_size}, peak_gpu_mem_mb={mem:.0f}", flush=True)
        stats(f"fp32_{fm}", e)
        if device.type == "cuda":
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                eb = torch.cat([opera_embed(enc, waves[i: i + args.batch_size].to(device), f_max=fm).float().cpu() for i in range(0, len(waves), args.batch_size)])
            cos = F.cosine_similarity(e, eb, dim=-1)
            print(f"bf16_vs_fp32[{fm}]: max_abs_diff={(e - eb).abs().max():.4e} min_cosine={cos.min():.6f}")
        with torch.no_grad():
            e2 = torch.cat([opera_embed(enc, waves[i: i + args.batch_size].to(device), f_max=fm).float().cpu() for i in range(0, len(waves), args.batch_size)])
        print(f"repeat_identical[{fm}]: {torch.equal(e, e2)} max_abs_diff={(e - e2).abs().max():.2e}")
    print(f"emb_diff_fmax8000_vs_2000: mean_cosine={F.cosine_similarity(res[8000], res[2000], dim=-1).mean():.4f}")

    # --- train-mode fwd+bwd timing on a copy (sizes exp-015a-c)
    step_s, train_mem = float("nan"), float("nan")
    if args.train_steps > 0:
        m = copy.deepcopy(enc).train()
        for p in m.parameters():
            p.requires_grad_(True)
        head = torch.nn.Linear(768, 8).to(device)
        opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=3e-5, weight_decay=0.05)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        for s in range(args.train_steps):
            idx = torch.arange(s * args.batch_size, (s + 1) * args.batch_size) % len(waves)
            x, y = waves[idx].to(device), labels[idx].to(device)
            sync(device)
            t0 = time.time()
            opt.zero_grad()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(head(opera_embed(m, x, f_max=8000)).float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sync(device)
            times.append(time.time() - t0)
        step_s = sum(times[2:]) / max(1, len(times) - 2)
        train_mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
        n_train = len(ds)
        print(f"train_step: batch {args.batch_size} bf16 autocast, {step_s:.3f} s/step (steps 3..{len(times)}), peak_gpu_mem_mb={train_mem:.0f}, "
              f"final_loss={loss.item():.3f}; estimate per epoch of {n_train} recordings = {n_train / args.batch_size * step_s / 60:.1f} min "
              f"(+ data loading, + val), 12 epochs = {12 * n_train / args.batch_size * step_s / 3600:.2f} h GPU time", flush=True)
        del m, head, opt

    # --- compute-node internet
    status = "skipped"
    if not args.skip_curl:
        try:
            r = subprocess.run(["curl", "-sI", "-m", "15", "https://huggingface.co"], capture_output=True, text=True, timeout=30)
            first = r.stdout.splitlines()[0].strip() if r.stdout.strip() else "(no response)"
            status = f"rc={r.returncode} first_line={first!r}"
        except Exception as ex:  # curl missing, timeout, ...
            status = f"failed: {type(ex).__name__}: {ex}"
    print(f"curl_huggingface: {status}")

    print("---")
    print(f"ckpt_sha256_match: {int(sha == OPERA_CKPT_SHA256)}")
    print(f"key_audit_missing: {len(rep.get('missing', []))}")
    print(f"key_audit_unexpected_other: {len(rep.get('unexpected_other', []))}")
    print(f"emb_std_8000: {res[8000].std():.6f}")
    print(f"emb_std_2000: {res[2000].std():.6f}")
    print(f"train_step_seconds: {step_s:.4f}")
    print(f"train_peak_mem_mb: {train_mem:.0f}")


if __name__ == "__main__":
    main()
