"""Run the protocol cross: every model, scene, protocol and seed.

Results are appended to the output CSV after each run, so a partial grid is
still usable if the process is interrupted.
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

# Models that take the full spectrum rather than the PCA-reduced input.
FULL_SPECTRUM = {"ASCN"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="protocol_config.json")
    ap.add_argument("--out", default="results_protocol_cross.csv")
    ap.add_argument("--models", default="SVM,CNN2D,CNN3D,HybridSN,SSFTT")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--protocols", default="disjoint,random")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--patch", type=int, default=11)
    ap.add_argument("--pca", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--test-cap", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = json.load(open(args.config))
    blocks = cfg["bench_blocks"]
    fixed = {k: np.array(v) for k, v in cfg["fixed_classes"].items()}
    scenes = args.scenes.split(",") if args.scenes else list(blocks)
    models = args.models.split(",")
    protocols = args.protocols.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = {(r.model, r.scene, r.protocol, r.seed) for r in prev.itertuples()}
        print(f"resuming, {len(done)} runs already present", flush=True)

    for scene in scenes:
        cube = hd.load_cube(scene)
        for seed in seeds:
            # The disjoint split sets the fitting budget that the random
            # protocol is then matched to.
            ref = T.prepare(cube, scene, "disjoint", seed=seed, patch=args.patch,
                            n_pca=args.pca, block=blocks[scene], fixed_classes=fixed[scene],
                            test_cap=args.test_cap)
            budget = {"n_train": len(ref.ytr), "n_val": len(ref.yva)}
            for protocol in protocols:
                cache: dict[object, T.Prepared] = {}
                if protocol == "disjoint":
                    cache[args.pca] = ref
                for name in models:
                    key = (name, scene, protocol, seed)
                    if key in done:
                        continue
                    # ASCN reads the full spectrum; every other model takes the
                    # conventional PCA-reduced input.
                    want = None if name in FULL_SPECTRUM else args.pca
                    if want not in cache:
                        cache[want] = T.prepare(
                            cube, scene, protocol, seed=seed, patch=args.patch, n_pca=want,
                            block=blocks[scene], fixed_classes=fixed[scene],
                            test_cap=args.test_cap,
                            **({} if protocol == "disjoint" else budget),
                        )
                    data = cache[want]
                    t0 = time.time()
                    try:
                        if name == "SVM":
                            row = T.run_svm(data, seed=seed)
                        else:
                            row = T.run_torch_model(name, data, seed=seed, epochs=args.epochs,
                                                    batch=args.batch, patience=args.patience)
                    except Exception as exc:  # keep the grid alive
                        print(f"FAILED {key}: {exc}", flush=True)
                        continue
                    row["wall_s"] = time.time() - t0
                    row["n_pca"] = data.n_pca
                    pd.DataFrame([row]).to_csv(
                        args.out, mode="a", header=not os.path.exists(args.out), index=False
                    )
                    print(
                        f"{name:9s} {scene:9s} {protocol:8s} seed{seed} "
                        f"OA={row['OA']:.4f} AA={row['AA']:.4f} k={row['kappa']:.4f} "
                        f"({row['wall_s']:.0f}s)",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
