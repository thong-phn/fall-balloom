import torch.nn as nn


class SimpleCNN(nn.Module):
    """Small classifier for [batch, 1, mel bands, time] inputs."""
    def __init__(self):
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
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.ndim == 5 and x.shape[1] == 1:
            # The dataset returns [1, 1, mel, time] per training item.
            x = x.squeeze(1)
        if x.ndim == 3:
            # Also accept one unbatched spectrogram [1, mel, time].
            x = x.unsqueeze(0)
        x = self.features(x)
        return self.head(x).squeeze(1)
