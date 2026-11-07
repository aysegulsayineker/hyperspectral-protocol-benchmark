"""ASCN, an asymmetric spectral-context network for hyperspectral classification.

The design follows from what spatially disjoint evaluation shows: a patch-based
model cannot rely on its neighbourhood carrying the answer, because under a
leakage-free split the neighbourhood no longer contains training pixels of the
same object.  What remains reliable is the spectrum of the pixel being
classified, with the neighbourhood acting only as weak context.

ASCN therefore spends its capacity asymmetrically:

* the **centre pixel** is encoded at full spectral resolution by a small 1-D
  convolutional encoder, so absorption features keep their shape;
* the **neighbourhood** is reduced to ``n_groups`` band averages before any
  spatial mixing, so context costs a fraction of the centre path;
* a **gate** decides per sample how much of the context to admit.

The spatial receptive field is exact and set by ``radius``: each 3x3 depthwise
convolution extends it by one pixel, and the centre cell of the resulting map is
read out.  ``radius=0`` removes the context path and leaves a purely spectral
classifier, which is the natural ablation floor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEncoder(nn.Module):
    """Small strided 1-D CNN over the wavelength axis of a single spectrum."""

    def __init__(self, dim: int = 64, widths: tuple[int, int] = (16, 32), n_tokens: int = 4):
        super().__init__()
        w1, w2 = widths
        self.n_tokens = n_tokens
        self.net = nn.Sequential(
            nn.Conv1d(1, w1, 7, stride=2, padding=3),
            nn.BatchNorm1d(w1),
            nn.GELU(),
            nn.Conv1d(w1, w2, 5, stride=2, padding=2),
            nn.BatchNorm1d(w2),
            nn.GELU(),
            nn.Conv1d(w2, dim, 3, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            # Pooling to a few tokens rather than one keeps coarse information
            # about WHERE along the spectrum a feature sits, which a single
            # global average would discard.
            nn.AdaptiveAvgPool1d(n_tokens),
        )
        self.merge = nn.Linear(dim * n_tokens, dim) if n_tokens > 1 else nn.Identity()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (N, L) -> (N, dim)
        z = self.net(spec.unsqueeze(1))
        return self.merge(z.flatten(1)) if self.n_tokens > 1 else z.squeeze(-1)


class ContextEncoder(nn.Module):
    """Band-grouped neighbourhood path with an exact receptive radius."""

    def __init__(self, n_groups: int, dim: int, radius: int):
        super().__init__()
        self.radius = radius
        self.proj = nn.Conv2d(n_groups, dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=0, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )
            for _ in range(radius)
        )

    def forward(self, grouped: torch.Tensor) -> torch.Tensor:
        # grouped: (N, G, P, P) -> (N, dim) read at the centre cell
        z = self.proj(grouped)
        for blk in self.blocks:
            z = blk(z)  # each 3x3 depthwise conv shrinks the map by one ring
        c = z.shape[-1] // 2
        return z[:, :, c, c]


class ASCN(nn.Module):
    """Asymmetric Spectral-Context Network.

    Args:
        c_in: number of input bands (full spectrum, or PCA components).
        n_classes: number of classes.
        patch: spatial side of the input patch.
        dim: embedding width shared by both paths.
        n_groups: band groups used by the context path.
        radius: exact spatial receptive radius of the context path.
        fusion: ``gate``, ``sum`` or ``concat``.
        symmetric: if true the full spectral encoder is applied to every pixel
            of the patch instead of the centre only.  This is the costly control
            condition used in the ablation, not the proposed configuration.
    """

    def __init__(
        self,
        c_in: int,
        n_classes: int,
        patch: int = 11,
        dim: int = 64,
        n_groups: int = 16,
        radius: int = 2,
        fusion: str = "gate",
        symmetric: bool = False,
    ):
        super().__init__()
        if radius > patch // 2:
            raise ValueError(f"radius {radius} exceeds patch radius {patch // 2}")
        self.patch = patch
        self.dim = dim
        self.n_groups = min(n_groups, c_in)
        self.radius = radius
        self.fusion = fusion
        self.symmetric = symmetric

        self.spectral = SpectralEncoder(dim)
        self.context = ContextEncoder(self.n_groups, dim, radius) if radius > 0 else None
        edges = torch.linspace(0, c_in, self.n_groups + 1).round().long()
        A = torch.zeros(self.n_groups, c_in)
        for g in range(self.n_groups):
            lo, hi = int(edges[g]), max(int(edges[g + 1]), int(edges[g]) + 1)
            A[g, lo:hi] = 1.0 / (hi - lo)
        self.register_buffer("group_matrix", A)

        if radius > 0 and fusion == "gate":
            self.gate = nn.Linear(2 * dim, dim)
        elif radius > 0 and fusion == "concat":
            self.mix = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def _group_bands(self, x: torch.Tensor) -> torch.Tensor:
        # Contiguous band averaging expressed as a fixed 1x1 convolution, which
        # is both cheaper and clearer than reshaping to pool per pixel.
        return F.conv2d(x, self.group_matrix[:, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, p, _ = x.shape
        mid = p // 2
        if self.symmetric:
            spec = x.permute(0, 2, 3, 1).reshape(n * p * p, c)
            z_all = self.spectral(spec).reshape(n, p, p, self.dim).permute(0, 3, 1, 2)
            z_c = z_all[:, :, mid, mid]
        else:
            z_c = self.spectral(x[:, :, mid, mid])

        if self.context is None:
            return self.head(self.norm(z_c))

        # The centre readout of the context path depends only on the ring of
        # radius `radius`, so cropping to it first is exact and much cheaper.
        lo, hi = mid - self.radius, mid + self.radius + 1
        z_s = self.context(self._group_bands(x[:, :, lo:hi, lo:hi]))
        if self.fusion == "gate":
            g = torch.sigmoid(self.gate(torch.cat([z_c, z_s], 1)))
            z = g * z_c + (1.0 - g) * z_s
        elif self.fusion == "sum":
            z = z_c + z_s
        else:
            z = self.mix(torch.cat([z_c, z_s], 1))
        return self.head(self.norm(z))


class LSSA(nn.Module):
    """Lightweight Spectral-Spatial Attention network.

    Design history, stated plainly because the motivating premise did not
    survive a larger training budget.  Under a 20-epoch budget the
    spectral-spatial transformer baseline led a spectral SVM by 4.97 points and
    a small 2-D CNN by 5.14 points, which suggested that combining the two paths
    under attention was what mattered.  At 60 epochs that gap closes: in the
    full 280-run protocol cross, mean overall accuracy over four scenes and five
    seeds is 0.8416 for the 2-D CNN against 0.8331 for the transformer, a
    difference of 0.0085 that is not significant (Wilcoxon p = 0.1769, 0.2185
    after Benjamini-Hochberg correction).  Under-training had favoured the
    transformer.

    LSSA is therefore best read as a compression of the transformer baseline
    rather than as an accuracy claim: it keeps the spectral-spatial attention
    structure and removes the parameters that are not carrying it.

    * the 3-D spectral stem is kept, since it costs almost nothing;
    * the dense 2-D projection that dominates the baseline's parameter count is
      replaced by a depthwise-separable pair;
    * the token grid is pooled before attention, so the sequence is short;
    * one narrow encoder layer replaces two wide ones.

    Args:
        c_in: input channels (PCA components).
        n_classes: number of classes.
        patch: spatial side of the input patch.
        dim: token width.
        depth: number of transformer encoder layers.
        heads: attention heads.
        pool: token-grid pooling factor; 1 keeps every spatial cell.
        separable: use the depthwise-separable projection (the proposed form).
    """

    def __init__(
        self,
        c_in: int,
        n_classes: int,
        patch: int = 11,
        dim: int = 48,
        depth: int = 1,
        heads: int = 4,
        pool: int = 2,
        separable: bool = True,
    ):
        super().__init__()
        self.conv3d = nn.Sequential(nn.Conv3d(1, 8, (7, 3, 3)), nn.BatchNorm3d(8), nn.GELU())
        with torch.no_grad():
            z = self.conv3d(torch.zeros(1, 1, c_in, patch, patch))
        ch2d = z.shape[1] * z.shape[2]
        if separable:
            self.proj = nn.Sequential(
                nn.Conv2d(ch2d, ch2d, 3, padding=1, groups=ch2d, bias=False),
                nn.BatchNorm2d(ch2d),
                nn.GELU(),
                nn.Conv2d(ch2d, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(ch2d, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.GELU()
            )
        self.pool = nn.AvgPool2d(pool) if pool > 1 else nn.Identity()
        with torch.no_grad():
            g = self.pool(self.proj(z.reshape(1, ch2d, z.shape[3], z.shape[4])))
        n_tok = g.shape[2] * g.shape[3]

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.1,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv3d(x.unsqueeze(1))
        z = z.reshape(z.shape[0], z.shape[1] * z.shape[2], z.shape[3], z.shape[4])
        z = self.pool(self.proj(z)).flatten(2).transpose(1, 2)
        z = torch.cat([self.cls.expand(z.shape[0], -1, -1), z], 1) + self.pos
        return self.head(self.norm(self.enc(z))[:, 0])


def register() -> None:
    """Make the study's own models visible to ``models.build``."""
    import models as M

    M.MODELS["ASCN"] = ASCN
    M.MODELS["LSSA"] = LSSA
