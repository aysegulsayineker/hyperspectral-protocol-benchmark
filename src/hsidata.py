"""Data handling and leakage-free splitting for hyperspectral image classification.

Two sampling protocols are provided side by side so the same model can be
evaluated under both:

``random``    the conventional protocol.  Labelled pixels are drawn at random,
              so a test pixel's patch window routinely overlaps the windows of
              training pixels.

``disjoint``  spatial blocks are assigned whole to train / val / test, and any
              labelled pixel whose patch window reaches into a block belonging
              to another split is discarded.  The discarded ring acts as a
              buffer, so no test patch can contain a training pixel.

Band standardisation is always fitted on training pixels only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import ndimage

# Scene name -> (cube file, ground-truth file).
DATASETS: dict[str, tuple[str, str]] = {
    "IndianPines": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat"),
    "PaviaU": ("PaviaU.mat", "PaviaU_gt.mat"),
    "Salinas": ("Salinas_corrected.mat", "Salinas_gt.mat"),
    "PaviaC": ("Pavia.mat", "Pavia_gt.mat"),
    "Botswana": ("Botswana.mat", "Botswana_gt.mat"),
    "KSC": ("KSC.mat", "KSC_gt.mat"),
    "Houston": ("Houston.mat", "Houston_gt.mat"),
}

SPLIT_NAMES = ("train", "val", "test")


def _first_array(mat: dict) -> np.ndarray:
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        keys = sorted(keys, key=lambda k: np.asarray(mat[k]).size, reverse=True)
    return np.asarray(mat[keys[0]])


def load_cube(name: str, root: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)``: ``X`` float32 (H, W, B) raw radiance, ``y`` int16 (H, W).

    Label 0 means unlabelled; classes are numbered from 1.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown scene {name!r}; known: {sorted(DATASETS)}")
    root = Path(root)
    fx, fy = DATASETS[name]
    X = _first_array(sio.loadmat(root / fx)).astype(np.float32)
    y = _first_array(sio.loadmat(root / fy)).astype(np.int16)
    if X.ndim != 3:
        raise ValueError(f"{name}: expected a 3-D cube, got {X.shape}")
    if X.shape[:2] != y.shape:
        raise ValueError(f"{name}: cube {X.shape[:2]} and ground truth {y.shape} disagree")
    return X, y


@dataclass
class Split:
    """Pixel index sets for one scene under one protocol."""

    scene: str
    protocol: str
    patch: int
    seed: int
    train: np.ndarray  # (N, 2) int32 of (row, col)
    val: np.ndarray
    test: np.ndarray
    split_map: np.ndarray | None = None  # (H, W) int8: 0 train, 1 val, 2 test
    n_dropped_buffer: int = 0
    buffer: int = 0
    block: int = 0
    classes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def random_split(
    y: np.ndarray,
    patch: int,
    seed: int = 0,
    train_frac: float = 0.10,
    val_frac: float = 0.05,
    scene: str = "",
) -> Split:
    """Conventional stratified random split over labelled pixels.

    Patch windows may overlap across splits; this is the protocol whose optimism
    the study measures.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y[y > 0])
    tr, va, te = [], [], []
    for c in classes:
        rr, cc = np.where(y == c)
        pick = np.stack([rr, cc], 1)[rng.permutation(len(rr))]
        n_tr = max(1, int(round(train_frac * len(rr))))
        n_va = max(1, int(round(val_frac * len(rr))))
        tr.append(pick[:n_tr])
        va.append(pick[n_tr : n_tr + n_va])
        te.append(pick[n_tr + n_va :])
    return Split(
        scene=scene,
        protocol="random",
        patch=patch,
        seed=seed,
        train=np.concatenate(tr).astype(np.int32),
        val=np.concatenate(va).astype(np.int32),
        test=np.concatenate(te).astype(np.int32),
        classes=classes,
    )


def _assign_blocks(
    n_blocks: int, fracs: tuple[float, float, float], rng: np.random.Generator
) -> np.ndarray:
    """Label each block 0 (train), 1 (validation) or 2 (test)."""
    n_tr = max(1, int(round(fracs[0] * n_blocks)))
    n_va = max(1, int(round(fracs[1] * n_blocks)))
    n_te = max(1, n_blocks - n_tr - n_va)
    labels = np.concatenate(
        [np.zeros(n_tr, np.int8), np.ones(n_va, np.int8), np.full(n_te, 2, np.int8)]
    )[:n_blocks]
    rng.shuffle(labels)
    return labels


def disjoint_split(
    y: np.ndarray,
    patch: int,
    seed: int = 0,
    block: int | None = None,
    fracs: tuple[float, float, float] = (0.25, 0.15, 0.60),
    min_per_class: int = 5,
    min_val_per_class: int = 2,
    max_tries: int = 200,
    scene: str = "",
    buffer: int | None = None,
    fixed_classes: np.ndarray | None = None,
) -> Split:
    """Spatially disjoint split with a buffer ring around every split boundary.

    The image is tiled into square blocks of side ``block`` and each block is
    assigned whole to one split.  A labelled pixel survives only if every cell
    within ``buffer`` of it belongs to its own split, which makes the patch
    windows of different splits disjoint by construction.

    ``buffer`` defaults to ``patch // 2``.  Set it to the largest patch radius
    used anywhere in the study and hold it fixed, so that one split serves every
    patch size and a spatial-radius sweep is not confounded by a changing test
    set.

    Block assignment is redrawn until every class has at least ``min_per_class``
    surviving pixels in both train and test, or ``max_tries`` is reached; the
    best assignment seen is kept and classes that never met the quota are
    excluded from ``Split.classes`` and from the index sets.
    """
    H, W = y.shape
    r = patch // 2 if buffer is None else int(buffer)
    if block is None:
        # Blocks must be comfortably larger than the buffer ring, or the ring
        # eats the whole block; they must also be numerous enough to partition.
        block = max(6 * r + 1, min(H, W) // 8)
    gh, gw = int(np.ceil(H / block)), int(np.ceil(W / block))
    n_blocks = gh * gw
    all_classes = (
        np.unique(y[y > 0]) if fixed_classes is None else np.asarray(fixed_classes, dtype=np.int16)
    )

    rr, cc = np.where(y > 0)
    interior = (rr >= r) & (rr < H - r) & (cc >= r) & (cc < W - r)
    rr, cc = rr[interior], cc[interior]
    labels = y[rr, cc]

    best: Split | None = None
    best_score = -1
    rng = np.random.default_rng(seed)

    for _ in range(max_tries):
        blk = _assign_blocks(n_blocks, fracs, rng).reshape(gh, gw)
        split_map = np.repeat(np.repeat(blk, block, axis=0), block, axis=1)[:H, :W]

        own = split_map[rr, cc]
        # A window is pure exactly when its minimum equals its maximum, which a
        # separable sliding filter computes for the whole image at once.
        size = 2 * r + 1
        lo = ndimage.minimum_filter(split_map, size=size, mode="nearest")
        hi = ndimage.maximum_filter(split_map, size=size, mode="nearest")
        keep = lo[rr, cc] == hi[rr, cc]

        kept_classes = []
        for c in all_classes:
            o = own[keep & (labels == c)]
            if (
                (o == 0).sum() >= min_per_class
                and (o == 1).sum() >= min_val_per_class
                and (o == 2).sum() >= min_per_class
            ):
                kept_classes.append(c)
        if len(kept_classes) > best_score:
            idx = np.stack([rr, cc], 1)[keep]
            o = own[keep]
            best = Split(
                scene=scene,
                protocol="disjoint",
                patch=patch,
                seed=seed,
                train=idx[o == 0].astype(np.int32),
                val=idx[o == 1].astype(np.int32),
                test=idx[o == 2].astype(np.int32),
                split_map=split_map.astype(np.int8),
                n_dropped_buffer=int((~keep).sum()),
                buffer=r,
                block=block,
                classes=np.array(kept_classes, dtype=np.int16),
            )
            best_score = len(kept_classes)
        if best_score == len(all_classes):
            break

    assert best is not None
    if fixed_classes is not None:
        # The class set is pinned so that seeds remain comparable; the search
        # above still prefers assignments that meet the quota for all of them.
        best.classes = np.sort(np.asarray(fixed_classes, dtype=np.int16))
    keep_lab = {int(c) for c in best.classes}
    for attr in SPLIT_NAMES:
        idx = getattr(best, attr)
        if len(idx):
            lab = y[idx[:, 0], idx[:, 1]]
            m = np.isin(lab, list(keep_lab))
            setattr(best, attr, idx[m])
    return best


def audit_leakage(split: Split, strict: bool = True) -> dict:
    """Measure how much spatial context test patches share with train/val pixels.

    Under ``disjoint`` every count must be zero.  Under ``random`` the counts
    quantify the shared context that inflates reported accuracy.
    """
    r = split.patch // 2
    out = {
        "scene": split.scene,
        "protocol": split.protocol,
        "patch": split.patch,
        "seed": split.seed,
        "test_patches": int(len(split.test)),
        "touching_train": 0,
        "touching_val": 0,
        "shared_train_pixels": 0,
    }
    if len(split.test) == 0 or len(split.train) == 0:
        return out

    hi_r = int(max(split.train[:, 0].max(), split.test[:, 0].max())) + r + 1
    hi_c = int(max(split.train[:, 1].max(), split.test[:, 1].max())) + r + 1
    if len(split.val):
        hi_r = max(hi_r, int(split.val[:, 0].max()) + r + 1)
        hi_c = max(hi_c, int(split.val[:, 1].max()) + r + 1)
    mark = np.zeros((hi_r + 1, hi_c + 1), dtype=np.int8)
    mark[split.train[:, 0], split.train[:, 1]] = 1
    if len(split.val):
        mark[split.val[:, 0], split.val[:, 1]] = 2

    for row, col in split.test:
        lo_r, lo_c = max(0, row - r), max(0, col - r)
        win = mark[lo_r : row + r + 1, lo_c : col + r + 1]
        n_tr = int((win == 1).sum())
        out["touching_train"] += n_tr > 0
        out["touching_val"] += int((win == 2).any())
        out["shared_train_pixels"] += n_tr
    out["touching_train"] = int(out["touching_train"])
    out["touching_val"] = int(out["touching_val"])
    out["frac_touching_train"] = out["touching_train"] / out["test_patches"]

    if strict and split.protocol == "disjoint" and (out["touching_train"] or out["touching_val"]):
        raise AssertionError(f"disjoint split leaked: {out}")
    return out


def fit_standardiser(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean and standard deviation from training pixels only."""
    vals = X[train_idx[:, 0], train_idx[:, 1], :]
    mu = vals.mean(0).astype(np.float32)
    sd = vals.std(0).astype(np.float32)
    sd[sd == 0] = 1.0
    return mu, sd


def apply_standardiser(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32)


def extract_patches(X: np.ndarray, idx: np.ndarray, patch: int) -> np.ndarray:
    """Return (N, B, patch, patch) float32 patches centred on ``idx``.

    The cube is reflect-padded so border pixels keep a full window.
    """
    r = patch // 2
    Xp = np.pad(X, ((r, r), (r, r), (0, 0)), mode="reflect")
    out = np.empty((len(idx), X.shape[2], patch, patch), dtype=np.float32)
    for i, (row, col) in enumerate(idx):
        out[i] = np.transpose(Xp[row : row + patch, col : col + patch, :], (2, 0, 1))
    return out


def labels_of(y: np.ndarray, idx: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map ground-truth labels onto contiguous 0..C-1 indices."""
    lut = np.full(int(classes.max()) + 1, -1, dtype=np.int64)
    lut[np.asarray(classes, dtype=np.int64)] = np.arange(len(classes))
    return lut[y[idx[:, 0], idx[:, 1]]]


def class_table(y: np.ndarray, split: Split) -> np.ndarray:
    """Rows = classes in ``split.classes``, columns = (train, val, test) counts."""
    out = np.zeros((len(split.classes), 3), dtype=np.int64)
    for j, idx in enumerate((split.train, split.val, split.test)):
        if len(idx) == 0:
            continue
        lab = y[idx[:, 0], idx[:, 1]]
        for i, c in enumerate(split.classes):
            out[i, j] = int((lab == c).sum())
    return out


def make_split(y: np.ndarray, protocol: str, patch: int, seed: int, scene: str = "", **kw) -> Split:
    if protocol == "random":
        return random_split(y, patch=patch, seed=seed, scene=scene, **kw)
    if protocol == "disjoint":
        return disjoint_split(y, patch=patch, seed=seed, scene=scene, **kw)
    raise ValueError(f"unknown protocol {protocol!r}")
