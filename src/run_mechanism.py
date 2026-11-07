"""Mechanism experiments behind the measurement result.

A  spatial window sweep crossed with protocol.  The block partition uses a
   fixed buffer of 7, so every patch size up to 15 is evaluated on the same
   pixels and the sweep is not confounded by a changing test set.
B  capacity sweep of the 2-D CNN under the leakage-free protocol.
C  training-budget sweep, which is what flipped the ranking between the early
   20-epoch runs and the 60-epoch ones.
D  low-label regime.

Each experiment appends to its own CSV after every run.
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
FULL_SPECTRUM = {"ASCN"}

# Models whose stacked 3x3 convolutions need room: three 3-D convolutions take
# a patch of side p down to p-6, so anything below 7 cannot be built.
NEEDS_ROOM = {"CNN3D", "HybridSN"}


def append(path: str, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def done_keys(path: str, cols: list[str]) -> set:
    if not os.path.exists(path):
        return set()
    prev = pd.read_csv(path)
    return {tuple(r) for r in prev[cols].itertuples(index=False)}


def run_one(name, data, seed, epochs, patience, batch, model_kw=None):
    if name == "SVM":
        return T.run_svm(data, seed=seed)
    return T.run_torch_model(name, data, seed=seed, epochs=epochs, batch=batch,
                             patience=patience, model_kw=model_kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="protocol_config.json")
    ap.add_argument("--only", default="A,B,C,D")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--test-cap", type=int, default=20000)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    blocks = cfg["bench_blocks"]
    fixed = {k: np.array(v) for k, v in cfg["fixed_classes"].items()}
    scenes = list(blocks)
    seeds = [int(s) for s in args.seeds.split(",")]
    only = set(args.only.split(","))
    cubes = {s: hd.load_cube(s) for s in scenes}

    def prep(scene, protocol, seed, patch, n_pca, **kw):
        return T.prepare(cubes[scene], scene, protocol, seed=seed, patch=patch, n_pca=n_pca,
                         block=blocks[scene], fixed_classes=fixed[scene],
                         test_cap=args.test_cap, **kw)

    # ---------------- A: window size x protocol ----------------
    if "A" in only:
        out = "results_window_sweep.csv"
        have = done_keys(out, ["model", "scene", "protocol", "seed", "patch"])
        for scene in scenes:
            for seed in seeds:
                for patch in (3, 5, 7, 9, 11, 15):
                    ref = prep(scene, "disjoint", seed, patch, 30)
                    budget = {"n_train": len(ref.ytr), "n_val": len(ref.yva)}
                    for protocol in ("disjoint", "random"):
                        data = ref if protocol == "disjoint" else prep(
                            scene, protocol, seed, patch, 30, **budget)
                        for name in ("SVM", "CNN2D", "SSFTT"):
                            if (name, scene, protocol, seed, patch) in have:
                                continue
                            if name in NEEDS_ROOM and patch < 7:
                                continue
                            t0 = time.time()
                            try:
                                row = run_one(name, data, seed, args.epochs, args.patience,
                                              args.batch)
                            except Exception as exc:
                                print(f"FAILED A {name} {scene} {protocol} s{seed} p{patch}: {exc}",
                                      flush=True)
                                continue
                            row["wall_s"] = time.time() - t0
                            append(out, row)
                            print(f"A {name:6s} {scene:8s} {protocol:8s} s{seed} p{patch:2d} "
                                  f"OA={row['OA']:.4f}", flush=True)

    # ---------------- B: capacity of the 2-D CNN ----------------
    if "B" in only:
        out = "results_capacity.csv"
        have = done_keys(out, ["scene", "seed", "width"])
        for scene in scenes:
            for seed in seeds:
                data = prep(scene, "disjoint", seed, 11, 30)
                for width in (8, 16, 32, 64, 128):
                    if (scene, seed, width) in have:
                        continue
                    t0 = time.time()
                    row = run_one("CNN2D", data, seed, args.epochs, args.patience, args.batch,
                                  model_kw={"width": width})
                    row.update(width=width, wall_s=time.time() - t0)
                    append(out, row)
                    print(f"B CNN2D w={width:3d} {scene:8s} s{seed} OA={row['OA']:.4f} "
                          f"params={row['params']}", flush=True)

    # ---------------- C: training budget ----------------
    if "C" in only:
        out = "results_budget.csv"
        have = done_keys(out, ["model", "scene", "seed", "epochs"])
        for scene in scenes:
            for seed in seeds:
                data = prep(scene, "disjoint", seed, 11, 30)
                for epochs in (10, 20, 40, 60, 100):
                    for name in ("CNN2D", "SSFTT", "HybridSN"):
                        if (name, scene, seed, epochs) in have:
                            continue
                        t0 = time.time()
                        row = run_one(name, data, seed, epochs, 10 ** 6, args.batch)
                        row.update(epochs=epochs, wall_s=time.time() - t0)
                        append(out, row)
                        print(f"C {name:9s} ep={epochs:3d} {scene:8s} s{seed} "
                              f"OA={row['OA']:.4f}", flush=True)

    # ---------------- D: low-label regime ----------------
    if "D" in only:
        out = "results_low_label.csv"
        have = done_keys(out, ["model", "scene", "seed", "n_per_class"])
        for scene in scenes:
            for seed in seeds:
                for n_per_class in (5, 10, 25):
                    cache = {}
                    for name in ("SVM", "CNN2D", "SSFTT", "LSSA"):
                        if (name, scene, seed, n_per_class) in have:
                            continue
                        n_pca = None if name in FULL_SPECTRUM else 30
                        if n_pca not in cache:
                            cache[n_pca] = prep(scene, "disjoint", seed, 11, n_pca,
                                                n_per_class=n_per_class)
                        t0 = time.time()
                        row = run_one(name, cache[n_pca], seed, args.epochs, args.patience,
                                      args.batch)
                        row.update(n_per_class=n_per_class, wall_s=time.time() - t0)
                        append(out, row)
                        print(f"D {name:6s} n={n_per_class:2d} {scene:8s} s{seed} "
                              f"OA={row['OA']:.4f}", flush=True)


if __name__ == "__main__":
    main()
