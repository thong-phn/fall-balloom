import torch


class SpecAugment:
    """Random frequency and time masks on log-mel batches [B, ..., mel, time].

    Each sample gets its own masks. Masked cells take the sample's mean value,
    because log-mel values are negative and 0 would be an outlier.
    Call it on training batches only.
    """
    def __init__(self, freq_mask=8, time_mask=20, n_freq=2, n_time=2):
        self.freq_mask, self.time_mask = freq_mask, time_mask
        self.n_freq, self.n_time = n_freq, n_time

    @staticmethod
    def _band(x, max_width, axis):
        """Bool mask [B, ..., mel, time]: one random band of width 0..max_width along axis (-2 mel, -1 time)."""
        batch, length = x.shape[0], x.shape[axis]
        width = torch.randint(0, max_width + 1, (batch,), device=x.device)
        start = (torch.rand(batch, device=x.device) * (length - width + 1)).long()
        idx = torch.arange(length, device=x.device)
        band = (idx >= start[:, None]) & (idx < (start + width)[:, None])  # [B, length]
        shape = [batch] + [1] * (x.ndim - 1)
        shape[axis] = length
        return band.view(shape)

    def __call__(self, x):
        mask = torch.zeros_like(x, dtype=torch.bool)
        for _ in range(self.n_freq):
            mask |= self._band(x, self.freq_mask, -2)
        for _ in range(self.n_time):
            mask |= self._band(x, self.time_mask, -1)
        return torch.where(mask, x.mean(dim=(-2, -1), keepdim=True), x)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(32, 1, 1, 64, 157) - 8
    out = SpecAugment()(x)
    assert out.shape == x.shape
    changed = (out != x).float().mean().item()
    assert 0 < changed < 0.3, changed  # some cells masked, most kept
    assert torch.equal(SpecAugment(n_freq=0, n_time=0)(x), x)  # no masks = no change
    assert (SpecAugment(freq_mask=0, n_time=0)(x) == x).all()  # width 0 = no change
    print(f"ok: masked {changed:.1%} of cells")
