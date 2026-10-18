import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Small classifier for [batch, 1, mel bands, time] inputs."""
    def __init__(self, num_classes=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.ndim == 5 and x.shape[1] == 1:
            # The dataset returns [1, 1, mel, time] per training item.
            x = x.squeeze(1)
        if x.ndim == 3:
            # Also accept one unbatched spectrogram [1, mel, time].
            x = x.unsqueeze(0)
        x = self.features(x)
        # num_classes == 1 -> [batch] logits, otherwise [batch, num_classes]
        return self.head(x).squeeze(1) if self.head[-1].out_features == 1 else self.head(x)


def separable_conv(in_ch, out_ch):
    """Depthwise 3x3 followed by pointwise 1x1 (replaces a dense 3x3 conv)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
        nn.Conv2d(in_ch, out_ch, 1),
    )


class SeparableCNN(SimpleCNN):
    """SimpleCNN with depthwise-separable convs (first conv stays dense: 1 input channel)."""
    def __init__(self, num_classes=1):
        super().__init__(num_classes)

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            separable_conv(16, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            separable_conv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            separable_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )


class MambaBlock(nn.Module):
    """Minimal pure-PyTorch Mamba block (selective SSM, sequential scan)."""
    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state = d_inner, d_state
        self.dt_rank = max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        # S4D-real init: A = -[1..N]
        A = torch.arange(1, d_state + 1).float().repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x):  # x: [B, L, d_model]
        B, L, _ = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2)
        x = F.silu(x)

        dt, Bm, Cm = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))              # [B, L, d_inner]
        A = -torch.exp(self.A_log)                     # [d_inner, N]

        # dA/dBx are built per step: peak memory is O(B*d_inner*N), not O(B*L*d_inner*N)
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1)              # [B, d_inner, 1]
            h = torch.exp(dt_t * A) * h + dt_t * Bm[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + x * self.D
        return self.out_proj(y * F.silu(z))


class TinyMamba(nn.Module):
    """Mamba classifier for [batch, 1, mel bands, time] inputs (time is the sequence)."""
    def __init__(self, n_mels=None, d_model=48, n_layers=2, d_state=8, pool=4, num_classes=1):
        super().__init__()
        # strided conv shortens the sequence so the sequential scan stays cheap
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels or 64, d_model, pool, stride=pool),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.blocks = nn.ModuleList(MambaBlock(d_model, d_state) for _ in range(n_layers))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.2), nn.Linear(d_model, num_classes))

    def forward(self, x):
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.stem(x.squeeze(1)).transpose(1, 2)    # [B, L, d_model]
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        out = self.head(x.mean(dim=1))
        return out.squeeze(1) if out.shape[1] == 1 else out


class EdgeMamba(nn.Module):
    """~10k-param Mamba for edge devices.

    Parameter-free average pooling over time shortens the sequence (no stem conv
    weights), a single Linear maps mel bands to d_model, and the SSM state is
    tiny (d_state=4), so inference is O(L) with a fixed-size recurrent state.
    """
    def __init__(self, n_mels=64, d_model=24, n_layers=2, d_state=4, pool=8, num_classes=1):
        super().__init__()
        self.pool = pool
        self.stem = nn.Linear(n_mels, d_model)
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.blocks = nn.ModuleList(MambaBlock(d_model, d_state) for _ in range(n_layers))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.1), nn.Linear(d_model, num_classes))

    def forward(self, x):
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = F.avg_pool1d(x.squeeze(1), self.pool, ceil_mode=True)  # [B, mel, L/pool]
        x = F.silu(self.stem(x.transpose(1, 2)))                   # [B, L, d_model]
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        out = self.head(x.mean(dim=1))
        return out.squeeze(1) if out.shape[1] == 1 else out
