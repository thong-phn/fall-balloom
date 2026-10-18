import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


SR = 4000
CLIP_SEC = 5
CLIP_SAMPLES = SR * CLIP_SEC

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=512,
    win_length=512,
    hop_length=128,
    n_mels=64,
    f_min=20,
    f_max=1950,
)


def _cache_file(path):
    """Waveform cache entry for an audio file, or None when BIOSAR_CACHE_DIR is unset."""
    cache_dir = os.environ.get("BIOSAR_CACHE_DIR")
    if not cache_dir:
        return None
    return Path(cache_dir) / ("__".join(Path(path).parts[-3:]) + ".npy")


def load_waveform(path):
    """
    Load audio
    Convert to mono
    Resample to SR
    Remove DC
    Result is cached as float32 .npy when BIOSAR_CACHE_DIR is set (outputs are identical).
    """
    cache = _cache_file(path)
    if cache is not None and cache.exists():
        return torch.from_numpy(np.load(cache))

    audio, sample_rate = torchaudio.load(path)
    audio = audio.mean(dim=0, keepdim=True)

    if sample_rate != SR:
        audio = torchaudio.functional.resample(audio, sample_rate, SR)

    audio = audio - audio.mean()

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + f".{os.getpid()}.tmp")
        np.save(tmp, audio.numpy())  # np.save appends .npy
        os.replace(str(tmp) + ".npy", cache)
    return audio


def preprocess(path, train=True):
    """
    Load audio (see load_waveform)
    Split into 5-sec chunk (padding if less then 5s)
    Apply log-mel features.
    """
    audio = load_waveform(path)
    length = audio.shape[-1]

    if length < CLIP_SAMPLES:
        audio = F.pad(audio, (0, CLIP_SAMPLES - length))
        length = CLIP_SAMPLES

    if train:
        start = torch.randint(length - CLIP_SAMPLES + 1, ()).item()
        clips = audio[:, start : start + CLIP_SAMPLES].unsqueeze(0)
    else:
        last_start = length - CLIP_SAMPLES
        starts = list(range(0, last_start + 1, CLIP_SAMPLES))
        if starts[-1] != last_start:
            starts.append(last_start)
        clips = torch.stack(
            [audio[:, start : start + CLIP_SAMPLES] for start in starts]
        )

    return torch.log(mel(clips) + 1e-6)
