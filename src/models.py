"""Baseline architectures for hyperspectral patch classification.

Every model takes a patch of shape (B, C_in, P, P) where ``C_in`` is the number
of spectral components after PCA (PCA is fitted on training pixels only) and
predicts the label of the centre pixel.

Parameter counts and FLOPs are reported by ``model_cost`` so that accuracy can
be read against model size, which is the primary axis of this study.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN2D(nn.Module):
    """Plain spatial CNN over PCA components."""

    def __init__(self, c_in: int, n_classes: int, width: int = 32):
        super().__init__()
        # The conventional two-stage 2-D CNN used as a baseline in the HSI
        # literature: two convolution blocks each followed by 2x2 pooling.
        self.body = nn.Sequential(
            nn.Conv2d(c_in, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(width * 2, n_classes))

    def forward(self, x):
        return self.head(self.body(x))


class CNN3D(nn.Module):
    """Spectral-spatial 3-D CNN."""

    def __init__(self, c_in: int, n_classes: int, patch: int = 11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, (5, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, (3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n = self.features(torch.zeros(1, 1, c_in, patch, patch)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)))


class HybridSN(nn.Module):
    """Roy et al. (2020) HybridSN: three 3-D convolutions followed by a 2-D one."""

    def __init__(self, c_in: int, n_classes: int, patch: int = 11):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, (5, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, (3, 3, 3)),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            z = self.conv3d(torch.zeros(1, 1, c_in, patch, patch))
        ch2d = z.shape[1] * z.shape[2]
        self.conv2d = nn.Sequential(nn.Conv2d(ch2d, 64, 3), nn.ReLU(inplace=True))
        with torch.no_grad():
            n = self.conv2d(z.reshape(1, ch2d, z.shape[3], z.shape[4])).numel()
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(n, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        z = self.conv3d(x.unsqueeze(1))
        z = z.reshape(z.shape[0], z.shape[1] * z.shape[2], z.shape[3], z.shape[4])
        return self.head(self.conv2d(z))


class SSFTT(nn.Module):
    """Spectral-spatial feature tokenisation transformer, in the style of Sun et al. (2022).

    A 3-D convolution compresses the spectral axis, a 2-D convolution produces a
    feature map, the map is tokenised and passed through transformer encoder
    layers, and a class token carries the prediction.
    """

    def __init__(self, c_in: int, n_classes: int, patch: int = 11, dim: int = 64,
                 depth: int = 2, heads: int = 4):
        super().__init__()
        self.conv3d = nn.Sequential(nn.Conv3d(1, 8, (7, 3, 3)), nn.BatchNorm3d(8), nn.ReLU(True))
        with torch.no_grad():
            z = self.conv3d(torch.zeros(1, 1, c_in, patch, patch))
        ch2d = z.shape[1] * z.shape[2]
        self.conv2d = nn.Sequential(nn.Conv2d(ch2d, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(True))
        n_tok = z.shape[3] * z.shape[4]
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        z = self.conv3d(x.unsqueeze(1))
        z = z.reshape(z.shape[0], z.shape[1] * z.shape[2], z.shape[3], z.shape[4])
        z = self.conv2d(z).flatten(2).transpose(1, 2)
        z = torch.cat([self.cls.expand(z.shape[0], -1, -1), z], 1) + self.pos
        return self.head(self.norm(self.enc(z))[:, 0])


def model_cost(model: nn.Module, c_in: int, patch: int) -> dict:
    """Trainable parameter count and forward FLOPs for one patch."""
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = float("nan")
    try:
        from torch.utils.flop_counter import FlopCounterMode

        was_training = model.training
        model.eval()
        with torch.no_grad():
            counter = FlopCounterMode(display=False)
            with counter:
                model(torch.zeros(1, c_in, patch, patch))
            flops = float(counter.get_total_flops())
        if was_training:
            model.train()
    except Exception:
        pass
    return {"params": int(n_par), "flops": flops}


BASELINES = {
    "CNN2D": CNN2D,
    "CNN3D": CNN3D,
    "HybridSN": HybridSN,
    "SSFTT": SSFTT,
}


def build(name: str, c_in: int, n_classes: int, patch: int, **kw) -> nn.Module:
    cls = BASELINES[name] if name in BASELINES else MODELS[name]
    if cls is CNN2D:
        return cls(c_in, n_classes, **kw)
    return cls(c_in, n_classes, patch=patch, **kw)


# Filled in by proposed.py once the study's own architecture is registered.
MODELS: dict[str, type[nn.Module]] = dict(BASELINES)
