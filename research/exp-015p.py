"""exp-015p: frozen OPERA-CT probe on the student's 4 kHz crop (design a of exp-015, run once for information).

Pipeline: BioSAR 4 kHz waveform clips (the evaluation tiling of utils.preprocess: all non-overlapping 5 s clips of a
recording plus a final overlapping clip when the length is not a multiple of 5 s) -> resample 4 -> 16 kHz -> OPERA mel
front-end (f_max 8000 and 2000) -> frozen vendored OPERA-CT encoder (utils/opera_htsat.py) -> 768-d embedding per clip.
Heads trained on the train clip embeddings (label = the recording's label): (linear) Linear(768, 8) and (mlp)
Linear(768,256)-ReLU-Dropout(0.2)-Linear(256,8); class-weighted CE (balance power 0.5, counts by recording), Adam 1e-3,
cosine to 1e-5, 30 epochs, batch 256. Evaluation = mean of the clip logits per recording. Decisions use VAL only; TEST is
printed next to it. Saves <output stem>-emb.pt (embeddings) and <output stem>-<fmax>-<head>-vallogits.pt (exp-018 format,
val_logits [1, N, 8]; use `exp-018.py --epoch 0`).
"""
import argparse
import csv
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from utils.datasets import CLASS_NAMES, ORGANS, BioSARMulti
from utils.opera_htsat import (OPERA_CKPT_SHA256, OPERA_REVISION, OperaCTEncoder, load_opera_ct, opera_embed)
from utils.preprocess import CLIP_SAMPLES, load_waveform
from utils.utils import binary_f1_auroc, set_seed

NUM_CLASSES = len(CLASS_NAMES)
MIN_STRATUM = 5


# ---------------------------------------------------------------- data: raw 4 kHz clips with the exact tiling of utils.preprocess
def wave_clips(path, train=False):
    """utils.preprocess.preprocess line for line, but returns the 4 kHz waveform clips [n_clips, 1, 20000] (no mel)."""
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
    """BioSARMulti returning the 4 kHz waveform clips instead of the log-mel features (utils untouched)."""

    def __getitem__(self, index):
        item = self.items[index]
        clips = wave_clips(str(item["audio_path"]), train=self.train)
        return clips, torch.tensor(item["label"], dtype=torch.long), item["mixture_id"]


def collate_recordings(batch):
    """Several recordings -> one flat clip batch: (clips [Nc, 20000], labels [B], ids [B], counts [B])."""
    clips = torch.cat([b[0].squeeze(1) for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return clips, labels, [b[2] for b in batch], torch.tensor([b[0].shape[0] for b in batch])


# ---------------------------------------------------------------- metrics (copied from exp-034 / exp-018)
def prob_auroc(logits, labels):
    p = logits.softmax(dim=1)
    vals = [binary_f1_auroc(p[:, c], (labels == c).float())[1]
            for c in range(NUM_CLASSES) if (labels == c).any() and (labels != c).any()]
    return sum(vals) / len(vals)


def multiclass_metrics(logits, labels):
    preds = logits.argmax(dim=1)
    recalls, f1s, aurocs = [], [], []
    for c in range(NUM_CLASSES):
        is_c, pred_c = labels == c, preds == c
        if not is_c.any():
            continue
        tp = (is_c & pred_c).sum().item()
        fp = (~is_c & pred_c).sum().item()
        fn = (is_c & ~pred_c).sum().item()
        recalls.append(tp / (tp + fn))
        f1s.append(2 * tp / (2 * tp + fp + fn))
        if (~is_c).any():
            aurocs.append(binary_f1_auroc(logits[:, c], is_c.float())[1])
    return {"acc": (preds == labels).float().mean().item(), "bal_acc": sum(recalls) / len(recalls),
            "macro_f1": sum(f1s) / len(f1s), "macro_auroc": sum(aurocs) / len(aurocs) if aurocs else float("nan")}


def class_weights(counts, power):
    w = counts.clamp(min=1) ** -power
    return w * counts.sum() / (w * counts).sum()


def corpus_key(r):
    return r["primary_source"].split("/")[2].lower().replace("abnormal", "").replace("normal", "")


def pooled_auroc(score, y, corp, strata):
    """sum_s P_s N_s AUROC_s / sum_s P_s N_s over the strata with both classes (exp-018)."""
    num = den = 0.0
    for s in strata:
        m = corp == s
        pos, neg = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if pos == 0 or neg == 0:
            continue
        num += pos * neg * binary_f1_auroc(score[m], y[m].float())[1]
        den += pos * neg
    return num / den if den > 0 else float("nan")


def within_corpus_report(logits, labels, corpora):
    """Heart pooled within-corpus abnormality AUROC and Lung within-cough AUROC (exp-018 definitions: score =
    p(true organ, abn) / (p(true organ, normal) + p(true organ, abn)); strata with >= 5 pos and >= 5 neg)."""
    p = logits.softmax(dim=1)
    organ = labels // 2
    idx = torch.arange(len(labels))
    s = p[idx, 2 * organ + 1] / (p[idx, 2 * organ] + p[idx, 2 * organ + 1])
    y = labels % 2
    names = sorted(set(corpora))
    cid = torch.tensor([names.index(c) for c in corpora])
    out = {}
    for o in ("Heart", "Lung"):
        rows = organ == ORGANS.index(o)
        strata = [ci for ci in range(len(names))
                  if int(((y == 1) & rows & (cid == ci)).sum()) >= MIN_STRATUM and int(((y == 0) & rows & (cid == ci)).sum()) >= MIN_STRATUM]
        out[f"{o.lower()}_overall"] = binary_f1_auroc(s[rows], y[rows].float())[1]
        out[f"{o.lower()}_pooled_within_corpus"] = pooled_auroc(s[rows], y[rows], cid[rows], strata)
    if "coughdataset" in names:
        m = (organ == ORGANS.index("Lung")) & (cid == names.index("coughdataset"))
        out["lung_within_cough"] = binary_f1_auroc(s[m], y[m].float())[1] if int(m.sum()) > 0 else float("nan")
    else:
        out["lung_within_cough"] = float("nan")
    return out


# ---------------------------------------------------------------- embeddings
@torch.no_grad()
def extract_split(encoder, data, fmaxes, device, args, tag):
    """Returns {fmax: dict(emb [Nc,768] cpu fp32, rec_index [Nc], labels [Nr], mixture_ids)} for one split."""
    loader = DataLoader(data, batch_size=args.rec_batch, shuffle=False, num_workers=max(0, args.num_workers),
                        collate_fn=collate_recordings, pin_memory=device.type == "cuda")
    embs = {fm: [] for fm in fmaxes}
    rec_index, labels, ids = [], [], []
    t0 = time.time()
    n_rec = 0
    for clips, lab, mid, counts in loader:
        clips = clips.to(device)
        for fm in fmaxes:
            for i in range(0, clips.shape[0], args.clip_batch):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(args.amp) and device.type == "cuda"):
                    e = opera_embed(encoder, clips[i: i + args.clip_batch], f_max=fm)
                embs[fm].append(e.float().cpu())
        rec_index.append(torch.arange(len(lab)).repeat_interleave(counts) + n_rec)
        labels.append(lab)
        ids += mid
        n_rec += len(lab)
    rec_index = torch.cat(rec_index)
    labels = torch.cat(labels)
    print(f"embeddings[{tag}]: recordings={n_rec} clips={rec_index.numel()} fmax={list(fmaxes)} seconds={time.time() - t0:.1f}", flush=True)
    return {fm: {"emb": torch.cat(embs[fm]), "rec_index": rec_index, "labels": labels, "mixture_ids": ids} for fm in fmaxes}


# ---------------------------------------------------------------- heads
def make_head(kind):
    if kind == "linear":
        return nn.Linear(768, NUM_CLASSES)
    return nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES))


def record_logits(head, d, device):
    """Mean of the clip logits per recording -> [Nr, 8]."""
    head.eval()
    with torch.no_grad():
        z = head(d["emb"].to(device)).cpu()
    nr = d["labels"].numel()
    out = torch.zeros(nr, NUM_CLASSES).index_add_(0, d["rec_index"], z)
    return out / torch.bincount(d["rec_index"], minlength=nr).unsqueeze(1).float()


def train_head(kind, train_d, weights, args, device, seed):
    set_seed(seed)
    head = make_head(kind).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    x = train_d["emb"].to(device)
    y = train_d["labels"][train_d["rec_index"]].to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    gen = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(n, generator=gen).to(device)
        tot = 0.0
        for i in range(0, n, args.head_batch):
            b = perm[i: i + args.head_batch]
            opt.zero_grad()
            loss = loss_fn(head(x[b]), y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * b.numel()
        sched.step()
        if ep == 0 or (ep + 1) % 10 == 0 or ep + 1 == args.epochs:
            print(f"  {kind} epoch {ep + 1}/{args.epochs} train_loss={tot / n:.4f}", flush=True)
    return head


def main():
    parser = argparse.ArgumentParser(description="exp-015p: frozen OPERA-CT probe on BioSAR")
    parser.add_argument("--metadata", default="./datasets/BioSARDdBScene/metadata.csv")
    parser.add_argument("--data", default="./datasets/BioSARDdBScene/")
    parser.add_argument("--output", default="models/ckpt/exp-015p.pt")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ckpt", default="models/foundation/encoder-operaCT.ckpt")
    parser.add_argument("--f-max", default="8000,2000", help="comma-separated upper mel edges")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--head-batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balance-power", type=float, default=0.5)
    parser.add_argument("--rec-batch", type=int, default=8, help="recordings per loader batch (embedding extraction)")
    parser.add_argument("--clip-batch", type=int, default=32, help="clips per encoder forward")
    parser.add_argument("--amp", type=int, choices=[0, 1], default=0, help="bf16 autocast for the encoder (default 0 = fp32)")
    parser.add_argument("--save-emb", type=int, choices=[0, 1], default=1)
    parser.add_argument("--random-init", type=int, choices=[0, 1], default=0, help="CPU test only: random encoder, no checkpoint")
    parser.add_argument("--max-recordings", type=int, default=0, help="CPU test only: random subsample of this many recordings per split")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fmaxes = [int(v) for v in args.f_max.split(",")]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"exp-015p: device={device} torch={torch.__version__} fmax={fmaxes} seed={args.seed} opera_code={OPERA_REVISION}", flush=True)

    if args.random_init:
        encoder = OperaCTEncoder().eval()
        print("encoder: RANDOM INIT (test mode)")
    else:
        encoder, rep = load_opera_ct(args.ckpt)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.to(device)

    with open(args.metadata, newline="") as f:
        rows = list(csv.DictReader(f))
    by_mid = {r["mixture_id"]: r for r in rows}

    data = {}
    for split in ("train", "val", "test"):
        ds = BioSARMultiWave(data_path=args.data, metadata_path=args.metadata, split=split, train=False)
        if args.max_recordings and len(ds.items) > args.max_recordings:
            ds.items = random.Random(0).sample(ds.items, args.max_recordings)
        data[split] = ds
    print("recordings: " + " ".join(f"{s}={len(d)}" for s, d in data.items()), flush=True)

    t0 = time.time()
    emb = {s: extract_split(encoder, data[s], fmaxes, device, args, s) for s in data}
    extract_seconds = time.time() - t0
    if device.type == "cuda":
        print(f"peak_gpu_mem_mb: {torch.cuda.max_memory_allocated() / 2**20:.0f}")
    if args.save_emb:
        torch.save({"emb": {fm: {s: emb[s][fm] for s in emb} for fm in fmaxes}, "opera_code": OPERA_REVISION,
                    "ckpt_sha256": OPERA_CKPT_SHA256, "random_init": bool(args.random_init)},
                   out.with_name(out.stem + "-emb.pt"))

    # recording-level metadata per split
    meta = {}
    for s in data:
        mids = emb[s][fmaxes[0]]["mixture_ids"]
        meta[s] = {"labels": emb[s][fmaxes[0]]["labels"], "corpora": [corpus_key(by_mid[m]) for m in mids],
                   "groups": [by_mid[m]["source_id"] for m in mids], "mids": mids}

    counts = torch.tensor([Counter(meta["train"]["labels"].tolist()).get(c, 0) for c in range(NUM_CLASSES)], dtype=torch.float)
    weights = class_weights(counts, args.balance_power)
    print("class weights (bp %.2f, counts by recording): " % args.balance_power
          + " ".join(f"{n}={int(c)}/{w:.2f}" for n, c, w in zip(CLASS_NAMES, counts.tolist(), weights.tolist())), flush=True)

    summary = {}
    for fm in fmaxes:
        for kind in ("linear", "mlp"):
            t1 = time.time()
            head = train_head(kind, emb["train"][fm], weights, args, device, args.seed)
            res = {}
            for s in ("val", "test"):
                z = record_logits(head, emb[s][fm], device)
                y = meta[s]["labels"]
                m = multiclass_metrics(z, y)
                w = within_corpus_report(z, y, meta[s]["corpora"])
                res[s] = {"auroc_prob": prob_auroc(z, y), "f1": m["macro_f1"], "bal_acc": m["bal_acc"], "logits": z, **w}
            v, t = res["val"], res["test"]
            print(f"probe[{fm},{kind}] val_auroc_prob={v['auroc_prob']:.6f} val_f1={v['f1']:.4f} val_bal_acc={v['bal_acc']:.4f} "
                  f"val_heart_within={v['heart_pooled_within_corpus']:.4f} val_lung_within_cough={v['lung_within_cough']:.4f} "
                  f"| TEST auroc_prob={t['auroc_prob']:.6f} f1={t['f1']:.4f} bal_acc={t['bal_acc']:.4f} "
                  f"heart_within={t['heart_pooled_within_corpus']:.4f} lung_within_cough={t['lung_within_cough']:.4f} "
                  f"seconds={time.time() - t1:.1f}", flush=True)
            torch.save({"val_logits": v["logits"].unsqueeze(0), "val_labels": meta["val"]["labels"], "val_groups": meta["val"]["groups"],
                        "val_mixture_ids": meta["val"]["mids"], "test_logits": t["logits"], "test_labels": meta["test"]["labels"],
                        "test_mixture_ids": meta["test"]["mids"], "f_max": fm, "head": kind, "seed": args.seed,
                        "opera_code": OPERA_REVISION, "ckpt_sha256": OPERA_CKPT_SHA256},
                       out.with_name(f"{out.stem}-{fm}-{kind}-vallogits.pt"))
            summary[(fm, kind)] = res

    if 8000 in fmaxes and 2000 in fmaxes:
        d = summary[(2000, "linear")]["val"]["auroc_prob"] - summary[(8000, "linear")]["val"]["auroc_prob"]
        rule = d > 0.003
        print(f"fmax_rule: L-probe val prob-AUROC f_max 2000 - 8000 = {d:+.6f} (> 0.003: {rule}) -> fine-tune f_max = {2000 if rule else 8000}")
    else:
        rule, d = False, float("nan")

    print("---")
    for (fm, kind), res in summary.items():
        for s in ("val", "test"):
            r = res[s]
            print(f"{s}_auroc_prob_{fm}_{kind}: {r['auroc_prob']:.6f}")
            print(f"{s}_macro_f1_{fm}_{kind}: {r['f1']:.6f}")
            print(f"{s}_heart_within_{fm}_{kind}: {r['heart_pooled_within_corpus']:.6f}")
            print(f"{s}_lung_within_cough_{fm}_{kind}: {r['lung_within_cough']:.6f}")
    print(f"fmax_diff_linear_2000_minus_8000: {d:.6f}")
    print(f"fmax_2000_selected: {int(rule)}")
    print(f"extract_seconds: {extract_seconds:.1f}")
    print(f"seed: {args.seed}")


if __name__ == "__main__":
    main()
