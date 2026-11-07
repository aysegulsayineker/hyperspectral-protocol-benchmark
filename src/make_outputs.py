"""Build the study's result tables and figures from the experiment CSVs.

Run inside the environment that carries the figure-style helpers, since the
figures are deliverables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_ORDER = ["SVM", "CNN2D", "CNN3D", "HybridSN", "SSFTT", "ASCN"]
PRETTY = {
    "SVM": "spectral SVM",
    "CNN2D": "2-D CNN",
    "CNN3D": "3-D CNN",
    "HybridSN": "HybridSN",
    "SSFTT": "SSFTT",
    "ASCN": "ASCN (ours)",
}
SCENE_PRETTY = {
    "PaviaU": "Pavia University",
    "PaviaC": "Pavia Centre",
    "Salinas": "Salinas",
    "Houston": "Houston 2013",
}


def protocol_cross_table(cross: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of OA under each protocol, with the paired drop."""
    g = cross.groupby(["scene", "model", "protocol"])["OA"]
    tab = g.agg(["mean", "std", "count"]).reset_index()
    wide = tab.pivot(index=["scene", "model"], columns="protocol",
                     values=["mean", "std", "count"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide["drop"] = wide["mean_random"] - wide["mean_disjoint"]
    # Paired by seed, so the per-seed difference carries the uncertainty.
    pairs = cross.pivot_table(index=["scene", "model", "seed"], columns="protocol",
                              values="OA").dropna()
    d = (pairs["random"] - pairs["disjoint"]).groupby(level=[0, 1])
    wide["drop_sd"] = d.std()
    wide["drop_min"] = d.min()
    wide["drop_max"] = d.max()
    return wide.reset_index()


def cost_table(cross: pd.DataFrame) -> pd.DataFrame:
    """Accuracy under the leakage-free protocol against model size."""
    dis = cross[cross.protocol == "disjoint"]
    out = dis.groupby("model").agg(
        OA=("OA", "mean"), OA_sd=("OA", "std"),
        AA=("AA", "mean"), kappa=("kappa", "mean"),
        params=("params", "median"), flops=("flops", "median"),
        infer_us=("infer_us_per_patch", "mean"),
    )
    out["MFLOPs"] = out["flops"] / 1e6
    out = out.drop(columns="flops")
    return out.reindex([m for m in MODEL_ORDER if m in out.index]).reset_index()


def paired_drop_test(cross: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank test of the random-minus-disjoint difference per model."""
    from scipy.stats import wilcoxon

    pairs = cross.pivot_table(index=["scene", "model", "seed"], columns="protocol",
                              values="OA").dropna().reset_index()
    rows = []
    for model, g in pairs.groupby("model"):
        diff = (g["random"] - g["disjoint"]).to_numpy()
        row = {"model": model, "n_pairs": len(diff), "mean_drop": float(diff.mean()),
               "median_drop": float(np.median(diff)),
               "min_drop": float(diff.min()), "max_drop": float(diff.max())}
        if len(diff) >= 6 and np.ptp(diff) > 0:
            stat, p = wilcoxon(diff)
            row["wilcoxon_stat"], row["p_value"] = float(stat), float(p)
        else:
            row["wilcoxon_stat"], row["p_value"] = np.nan, np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.set_index("model").reindex(
        [m for m in MODEL_ORDER if m in out.model.values]).reset_index()


def ablation_table(abl: pd.DataFrame) -> pd.DataFrame:
    base = abl[abl.label == "proposed"].groupby("scene")["OA"].mean()
    g = abl.groupby(["axis", "label"]).agg(
        OA=("OA", "mean"), OA_sd=("OA", "std"),
        params=("params", "median"), flops=("flops", "median")).reset_index()
    g["MFLOPs"] = g["flops"] / 1e6
    g = g.drop(columns="flops")
    per_scene = abl.pivot_table(index=["axis", "label"], columns="scene", values="OA")
    delta = per_scene.sub(base, axis=1).mean(axis=1).rename("delta_vs_proposed")
    return g.merge(delta.reset_index(), on=["axis", "label"], how="left")


def main(cross_csv="results_protocol_cross.csv", abl_csv="results_ablation.csv") -> dict:
    import os

    out = {}
    cross = pd.read_csv(cross_csv)
    t1 = protocol_cross_table(cross)
    t1.to_csv("table_protocol_cross.csv", index=False)
    t2 = cost_table(cross)
    t2.to_csv("table_cost.csv", index=False)
    t3 = paired_drop_test(cross)
    t3.to_csv("table_drop_test.csv", index=False)
    out.update(protocol_cross=t1, cost=t2, drop_test=t3)
    if os.path.exists(abl_csv):
        t4 = ablation_table(pd.read_csv(abl_csv))
        t4.to_csv("table_ablation.csv", index=False)
        out["ablation"] = t4
    return out


if __name__ == "__main__":
    tables = main()
    for name, t in tables.items():
        print("===", name)
        print(t.to_string(index=False))
