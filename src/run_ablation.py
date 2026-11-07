"""Ablation study for ASCN.

Each variant changes one design choice at a time against the proposed
configuration (full spectrum, 16 band groups, radius 2, gated fusion).  Results
are appended after every run so a partial sweep stays usable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hsidata as hd  # noqa: E402
import proposed  # noqa: E402
import train as T  # noqa: E402

proposed.register()

BASE = dict(radius=2, n_groups=16, dim=64, fusion="gate", symmetric=False)


def variants() -> list[tuple[str, str, dict, object]]:
    """(axis, label, model kwargs, n_pca) for every ablation cell."""
    out: list[tuple[str, str, dict, object]] = [("base", "proposed", dict(BASE), None)]
    for r in (0, 1, 3, 4):
        out.append(("radius", f"radius={r}", {**BASE, "radius": r}, None))
    for g in (4, 8, 32):
        out.append(("n_groups", f"groups={g}", {**BASE, "n_groups": g}, None))
    for f in ("sum", "concat"):
        out.append(("fusion", f"fusion={f}", {**BASE, "fusion": f}, None))
    out.append(("spectral_path", "symmetric", {**BASE, "symmetric": True}, None))
    out.append(("input", "PCA-30", dict(BASE), 30))
    for d in (32, 128):
        out.append(("width", f"dim={d}", {**BASE, "dim": d}, None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="protocol_config.json")
    ap.add_argument("--out", default="results_ablation.csv")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--axes", default="")
    ap.add_argument("--patch", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--test-cap", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = json.load(open(args.config))
    blocks = cfg["bench_blocks"]
    fixed = {k: np.array(v) for k, v in cfg["fixed_classes"].items()}
    scenes = args.scenes.split(",") if args.scenes else list(blocks)
    seeds = [int(s) for s in args.seeds.split(",")]
    axes = set(args.axes.split(",")) if args.axes else None

    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = {(r.label, r.scene, r.seed) for r in prev.itertuples()}
        print(f"resuming, {len(done)} runs already present", flush=True)

    for scene in scenes:
        cube = hd.load_cube(scene)
        for seed in seeds:
            cache: dict[object, T.Prepared] = {}
            for axis, label, kw, n_pca in variants():
                if axes is not None and axis not in axes:
                    continue
                if (label, scene, seed) in done:
                    continue
                if n_pca not in cache:
                    cache[n_pca] = T.prepare(
                        cube, scene, "disjoint", seed=seed, patch=args.patch, n_pca=n_pca,
                        block=blocks[scene], fixed_classes=fixed[scene], test_cap=args.test_cap,
                    )
                data = cache[n_pca]
                t0 = time.time()
                try:
                    row = T.run_torch_model("ASCN", data, seed=seed, epochs=args.epochs,
                                            batch=args.batch, patience=args.patience,
                                            model_kw=kw)
                except Exception as exc:
                    print(f"FAILED {label} {scene} seed{seed}: {exc}", flush=True)
                    continue
                row.update({"axis": axis, "label": label, "wall_s": time.time() - t0,
                            "n_pca_in": data.n_pca, **{f"kw_{k}": v for k, v in kw.items()}})
                pd.DataFrame([row]).to_csv(
                    args.out, mode="a", header=not os.path.exists(args.out), index=False
                )
                print(f"{label:12s} {scene:9s} seed{seed} OA={row['OA']:.4f} "
                      f"AA={row['AA']:.4f} params={row['params']} "
                      f"MFLOPs={row['flops'] / 1e6:.2f} ({row['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
