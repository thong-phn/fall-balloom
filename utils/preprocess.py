
import torch
import torch.nn.functional as F
import torchaudio

SR = 4000
WINDOW = 4 * SR

# typical selection
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,   # 4000 Hz
    n_fft=512,        # FFT window size
    win_length=512,   # analysis window size
    hop_length=128,   # shift between windows
    n_mels=64,        # output frequency bands
    f_min=20,
    f_max=1950
)

def preprocess(path, train=True):
    audio, sr = torchaudio.load(path)

    # Convert to mono
    audio = audio.mean(dim=0, keepdim=True)

    # Resample
    if sr != SR:
        audio = torchaudio.functional.resample(
            audio, sr, SR
        )

    # Remove DC offset
    audio = audio - audio.mean()

    # Pad only if shorter than 4 seconds
    if audio.shape[-1] < WINDOW:
        audio = F.pad(
            audio, (0, WINDOW - audio.shape[-1])
        )

    length = audio.shape[-1]

    if train:
        start = torch.randint(
            0, length - WINDOW + 1, (1,)
        ).item()
        chunks = audio[:, start:start + WINDOW].unsqueeze(0)
    else:
        starts = list(range(0, length - WINDOW + 1, WINDOW))
        if starts[-1] != length - WINDOW:
            starts.append(length - WINDOW)

        chunks = torch.stack([
            audio[:, s:s + WINDOW] for s in starts
        ])

    # [N, 1, 64, 126]
    features = torch.log(mel(chunks) + 1e-6)

    return features
