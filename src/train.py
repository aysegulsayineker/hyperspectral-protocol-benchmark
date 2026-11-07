"""Training and evaluation harness shared by every model and both protocols."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score

import hsidata as hd
import models as M


@dataclass
class Prepared:
    scene: str
    protocol: str
    seed: int
    patch: int
    n_pca: int
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    n_classes: int
    split: hd.Split


def prepare(
    cube: tuple[np.ndarray, np.ndarray],
    scene: str,
    protocol: str,
    seed: int,
    patch: int = 11,
    n_pca: int = 30,
    block: int | None = None,
    buffer: int = 7,
    n_per_class: int | None = None,
    fixed_classes=None,
    n_train: int | None = None,
    n_val: int | None = None,
    test_cap: int | None = None,
) -> Prepared:
    """Build patch tensors for one (scene, protocol, seed) configuration.

    Standardisation and PCA are both fitted on training pixels only.  When
    ``n_per_class`` is given, the training set is subsampled to that many pixels
    per class for the low-label regime; validation and test are untouched.
    """
    X, y = cube
    if protocol in ("disjoint", "block_nobuffer"):
        # 'block_nobuffer' is the control that separates the two things the
        # buffered block split changes at once: it keeps the identical block
        # geography but removes the buffer ring, so patches near a block
        # boundary do overlap the neighbouring split again.  The contrast
        # between the two isolates patch overlap from spatial distribution.
        eff_buffer = 0 if protocol == "block_nobuffer" else buffer
        sp = hd.disjoint_split(y, patch=patch, buffer=eff_buffer, seed=seed, block=block,
                               scene=scene, fixed_classes=fixed_classes)
        sp.protocol = protocol
        hd.audit_leakage(sp, strict=(protocol == "disjoint"))
    else:
        # When a budget is supplied, draw the random split's pool large enough
        # that the stratified match below can hit the target exactly; the
        # default fractions would leave Houston short of it.
        n_lab = int((np.isin(y, fixed_classes) if fixed_classes is not None else y > 0).sum())
        tf = 0.10 if n_train is None else min(0.80, 1.6 * n_train / max(1, n_lab))
        vf = 0.05 if n_val is None else min(0.15, 1.6 * n_val / max(1, n_lab))
        sp = hd.random_split(y, patch=patch, seed=seed, scene=scene,
                             train_frac=tf, val_frac=vf)
        if fixed_classes is not None:
            # The same class set under both protocols, so the comparison is not
            # confounded by a different label space.
            keep = np.asarray(fixed_classes, dtype=np.int16)
            sp.classes = np.sort(keep)
            for attr in ("train", "val", "test"):
                idx = getattr(sp, attr)
                lab = y[idx[:, 0], idx[:, 1]]
                setattr(sp, attr, idx[np.isin(lab, keep)])
    # Match the fitting budget of the buffered split so a comparison isolates
    # the protocol rather than training-set size.  This applies to every
    # protocol that is handed a budget, which is all of them except the
    # buffered split itself, since that is where the budget comes from.
    # Subsampling is stratified by class so the label distribution is kept.
    rng0 = np.random.default_rng(seed + 777)
    for attr, target in (("train", n_train), ("val", n_val)):
        idx = getattr(sp, attr)
        if target is None or target >= len(idx):
            continue
        lab = y[idx[:, 0], idx[:, 1]]
        share = target / len(idx)
        pick = []
        for c in np.unique(lab):
            pos = np.where(lab == c)[0]
            take = max(1, int(round(share * len(pos))))
            pick.append(rng0.choice(pos, size=min(take, len(pos)), replace=False))
        setattr(sp, attr, idx[np.sort(np.concatenate(pick))])

    if test_cap is not None and len(sp.test) > test_cap:
        # Stratified subsample of the test set; this costs precision, not bias.
        rng1 = np.random.default_rng(seed + 31337)
        lab = y[sp.test[:, 0], sp.test[:, 1]]
        share = test_cap / len(sp.test)
        pick = []
        for c in np.unique(lab):
            pos = np.where(lab == c)[0]
            take = max(1, int(round(share * len(pos))))
            pick.append(rng1.choice(pos, size=min(take, len(pos)), replace=False))
        sp.test = sp.test[np.sort(np.concatenate(pick))]

    train_idx = sp.train
    if n_per_class is not None:
        rng = np.random.default_rng(seed + 4242)
        lab = y[train_idx[:, 0], train_idx[:, 1]]
        pick = []
        for c in sp.classes:
            pos = np.where(lab == c)[0]
            if len(pos) == 0:
                continue
            take = min(n_per_class, len(pos))
            pick.append(rng.choice(pos, size=take, replace=False))
        train_idx = train_idx[np.concatenate(pick)]

    mu, sd = hd.fit_standardiser(X, train_idx)
    Xs = hd.apply_standardiser(X, mu, sd)

    if n_pca is None:
        # Full spectrum: the band axis keeps its wavelength ordering, which a
        # spectral convolution needs and PCA destroys.
        Xp = Xs
    else:
        flat_train = Xs[train_idx[:, 0], train_idx[:, 1], :]
        pca = PCA(n_components=min(n_pca, Xs.shape[2]), random_state=seed).fit(flat_train)
        H, W, _ = Xs.shape
        Xp = pca.transform(Xs.reshape(-1, Xs.shape[2])).reshape(H, W, -1).astype(np.float32)

    out = {}
    for name, idx in (("tr", train_idx), ("va", sp.val), ("te", sp.test)):
        out["X" + name] = hd.extract_patches(Xp, idx, patch)
        out["y" + name] = hd.labels_of(y, idx, sp.classes)
    return Prepared(scene=scene, protocol=protocol, seed=seed, patch=patch,
                    n_pca=Xp.shape[2], n_classes=len(sp.classes), split=sp, **out)


def _loader(X: np.ndarray, y: np.ndarray, bs: int, shuffle: bool):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    # A trailing batch of one breaks BatchNorm in training mode, so drop it when
    # training; evaluation keeps every sample.
    drop_last = shuffle and len(X) > bs and len(X) % bs == 1
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                       num_workers=0, drop_last=drop_last)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    oa = float((y_true == y_pred).mean())
    per = []
    for c in range(n_classes):
        m = y_true == c
        if m.sum():
            per.append(float((y_pred[m] == c).mean()))
    return {"OA": oa, "AA": float(np.mean(per)), "kappa": float(cohen_kappa_score(y_true, y_pred))}


def run_torch_model(
    name: str,
    data: Prepared,
    seed: int = 0,
    epochs: int = 80,
    batch: int = 128,
    lr: float = 1e-3,
    patience: int = 15,
    model_kw: dict | None = None,
) -> dict:
    """Train one model and return metrics, cost and timing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = M.build(name, data.n_pca, data.n_classes, data.patch, **(model_kw or {}))
    cost = M.model_cost(model, data.n_pca, data.patch)  # counted on CPU
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    tr = _loader(data.Xtr, data.ytr, batch, True)
    va = _loader(data.Xva, data.yva, 512, False)

    best_oa, best_state, bad, t0 = -1.0, None, 0, time.time()
    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in va:
                preds.append(model(xb.to(dev)).argmax(1).cpu().numpy())
        oa = float((np.concatenate(preds) == data.yva).mean()) if len(data.yva) else 0.0
        if oa > best_oa:
            best_oa, bad = oa, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    train_s = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    t1 = time.time()
    with torch.no_grad():
        for xb, _ in _loader(data.Xte, data.yte, 512, False):
            preds.append(model(xb.to(dev)).argmax(1).cpu().numpy())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    infer_s = time.time() - t1
    yp = np.concatenate(preds)
    out = {"model": name, "scene": data.scene, "protocol": data.protocol, "seed": seed,
           "patch": data.patch, "n_classes": data.n_classes,
           "n_train": len(data.ytr), "n_test": len(data.yte),
           "val_OA": best_oa, "train_s": train_s, "infer_s": infer_s,
           "infer_us_per_patch": 1e6 * infer_s / max(1, len(data.yte)), **cost}
    out.update(metrics(data.yte, yp, data.n_classes))
    return out


def run_svm(data: Prepared, seed: int = 0, c_grid=(1, 10, 100), g_grid=("scale",)) -> dict:
    """Spectral-only RBF SVM on the centre pixel, tuned on the validation split."""
    from sklearn.svm import SVC

    r = data.patch // 2
    tr = data.Xtr[:, :, r, r]
    va = data.Xva[:, :, r, r]
    te = data.Xte[:, :, r, r]
    best, best_oa = None, -1.0
    t0 = time.time()
    for C in c_grid:
        for g in g_grid:
            m = SVC(C=C, gamma=g, kernel="rbf", random_state=seed).fit(tr, data.ytr)
            oa = float((m.predict(va) == data.yva).mean()) if len(data.yva) else 0.0
            if oa > best_oa:
                best, best_oa = m, oa
    train_s = time.time() - t0
    t1 = time.time()
    yp = best.predict(te)
    infer_s = time.time() - t1
    out = {"model": "SVM", "scene": data.scene, "protocol": data.protocol, "seed": seed,
           "patch": data.patch, "n_classes": data.n_classes,
           "n_train": len(data.ytr), "n_test": len(data.yte),
           "val_OA": best_oa, "train_s": train_s, "infer_s": infer_s,
           "infer_us_per_patch": 1e6 * infer_s / max(1, len(data.yte)),
           "params": int(best.support_vectors_.size), "flops": float("nan")}
    out.update(metrics(data.yte, yp, data.n_classes))
    return out
