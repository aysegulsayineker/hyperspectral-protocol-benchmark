"""Generate every table the manuscript reports, from the experiment CSVs.

No number in the manuscript is typed by hand: each table is written here as a
CSV and as a Markdown block in tables.md.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hsidata as hd  # noqa: E402

PRETTY = {"SVM": "spectral SVM", "CNN2D": "2-D CNN", "CNN3D": "3-D CNN",
          "HybridSN": "HybridSN", "SSFTT": "SSFTT", "ASCN": "ASCN", "LSSA": "LSSA"}
SCENE = {"PaviaU": "Pavia University", "PaviaC": "Pavia Centre",
         "Salinas": "Salinas", "Houston": "Houston 2013"}
CANDIDATES = ["PaviaU", "PaviaC", "Salinas", "Houston", "IndianPines", "KSC", "Botswana"]


def fmt_p(p: float) -> str:
    """Format a p-value for print. A p-value is never reported as zero."""
    if pd.isna(p):
        return ""
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 1.0
    for rank, idx in zip(range(m, 0, -1), order[::-1]):
        running = min(running, p[idx] * m / rank)
        adj[idx] = running
    return adj


def t_scenes(cfg: dict) -> pd.DataFrame:
    """Scene inventory and whether each supports a three-way disjoint split."""
    rows = []
    for name in CANDIDATES:
        X, y = hd.load_cube(name)
        classes = np.unique(y[y > 0])
        block = cfg["bench_blocks"].get(name)
        if block is None:
            best = 0
            for blk in (18, 26, 33, 48, 64):
                sp = hd.disjoint_split(y, patch=15, buffer=7, seed=0, block=blk,
                                       max_tries=200, scene=name)
                best = max(best, len(sp.classes))
            kept, used, supported = best, None, False
        else:
            sp = hd.disjoint_split(y, patch=15, buffer=7, seed=0, block=block,
                                   max_tries=300, scene=name,
                                   fixed_classes=np.array(cfg["fixed_classes"][name]))
            kept, used, supported = len(sp.classes), block, True
        rows.append(dict(scene=SCENE.get(name, name), height=y.shape[0], width=y.shape[1],
                         bands=X.shape[2], classes=len(classes), labelled=int((y > 0).sum()),
                         smallest_class=int(np.bincount(y.ravel())[1:].min()),
                         block=used, classes_retained=kept,
                         supports_protocol="yes" if supported else "no"))
    return pd.DataFrame(rows)


def t_leakage(cfg: dict) -> pd.DataFrame:
    """Shared context between train and test under each protocol."""
    rows = []
    for scene, block in cfg["bench_blocks"].items():
        _, y = hd.load_cube(scene)
        fixed = np.array(cfg["fixed_classes"][scene])
        for patch in (5, 7, 9, 11, 15):
            a = hd.audit_leakage(hd.random_split(y, patch=patch, seed=0, scene=scene))
            rows.append(dict(scene=SCENE[scene], protocol="random", patch=patch,
                             frac_test_patches_touching_train=round(a["frac_touching_train"], 4),
                             mean_train_pixels_in_window=round(
                                 a["shared_train_pixels"] / a["test_patches"], 2)))
        sp = hd.disjoint_split(y, patch=15, buffer=7, seed=0, block=block, scene=scene,
                               fixed_classes=fixed)
        a = hd.audit_leakage(sp)
        rows.append(dict(scene=SCENE[scene], protocol="block-disjoint", patch=15,
                         frac_test_patches_touching_train=round(a["frac_touching_train"], 4),
                         mean_train_pixels_in_window=0.0))
    return pd.DataFrame(rows)


def t_cross(cross: pd.DataFrame) -> pd.DataFrame:
    g = cross.groupby(["model", "protocol"])["OA"].agg(["mean", "std"]).unstack()
    out = pd.DataFrame({
        "random_OA": g[("mean", "random")].round(4),
        "random_sd": g[("std", "random")].round(4),
        "disjoint_OA": g[("mean", "disjoint")].round(4),
        "disjoint_sd": g[("std", "disjoint")].round(4),
    })
    out["drop"] = (out.random_OA - out.disjoint_OA).round(4)
    dis = cross[cross.protocol == "disjoint"].groupby("model")
    out["AA"] = dis["AA"].mean().round(4)
    out["kappa"] = dis["kappa"].mean().round(4)
    out["params"] = dis["params"].median().astype(int)
    out["MFLOPs"] = (dis["flops"].median() / 1e6).round(2)
    out.index = [PRETTY[i] for i in out.index]
    return out.sort_values("disjoint_OA", ascending=False).reset_index(names="model")


def t_pairs(cross: pd.DataFrame) -> pd.DataFrame:
    dis = cross[cross.protocol == "disjoint"]
    w = dis.pivot_table(index=["scene", "seed"], columns="model", values="OA")
    models = list(dis.groupby("model")["OA"].mean().sort_values(ascending=False).index)
    rows = []
    for a, b in itertools.combinations(models, 2):
        d = (w[a] - w[b]).dropna()
        _, p = wilcoxon(d)
        rows.append(dict(model_a=PRETTY[a], model_b=PRETTY[b], diff=round(d.mean(), 4),
                         n_pairs=len(d), p=p))
    out = pd.DataFrame(rows)
    out["p_BH_num"] = bh(out.p.values)
    out["significant"] = np.where(out.p_BH_num < 0.05, "yes", "no")
    out = out.sort_values("p_BH_num").reset_index(drop=True)
    out["p_BH"] = out.p_BH_num.map(fmt_p)
    out["p"] = out.p.map(fmt_p)
    return out[["model_a", "model_b", "diff", "n_pairs", "p", "p_BH", "significant"]]


def t_window(win: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("CNN2D", "SSFTT"):
        base = 5 if model == "CNN2D" else 3
        for protocol in ("random", "disjoint"):
            s = win[(win.model == model) & (win.protocol == protocol)]
            p = s.pivot_table(index=["scene", "seed"], columns="patch", values="OA")
            d = (p[15] - p[base]).dropna()
            _, pv = wilcoxon(d)
            rows.append(dict(model=PRETTY[model], protocol=protocol, from_patch=base,
                             to_patch=15, gain=round(d.mean(), 4), n_pairs=len(d),
                             p=fmt_p(pv)))
    return pd.DataFrame(rows)


def t_capacity(cap: pd.DataFrame) -> pd.DataFrame:
    out = cap.groupby("width").agg(OA=("OA", "mean"), sd=("OA", "std"),
                                   params=("params", "median")).round(4)
    out["params"] = out.params.astype(int)
    p = cap.pivot_table(index=["scene", "seed"], columns="width", values="OA")
    step, pv = [np.nan], [""]
    widths = sorted(cap.width.unique())
    for a, b in zip(widths[:-1], widths[1:]):
        d = (p[b] - p[a]).dropna()
        step.append(round(d.mean(), 4))
        pv.append(fmt_p(wilcoxon(d)[1]))
    out["gain_vs_previous"] = step
    out["p"] = pv
    return out.reset_index()


def t_budget(bud: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ep in sorted(bud.epochs.unique()):
        s = bud[bud.epochs == ep].pivot_table(index=["scene", "seed"], columns="model",
                                              values="OA")
        d = (s["SSFTT"] - s["CNN2D"]).dropna()
        _, p = wilcoxon(d)
        row = dict(epochs=int(ep), diff_SSFTT_minus_CNN2D=round(d.mean(), 4),
                   p=fmt_p(p), n_pairs=len(d))
        for m in s.columns:
            row[PRETTY[m]] = round(s[m].mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def t_lowlabel(low: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in sorted(low.n_per_class.unique()):
        s = low[low.n_per_class == n].pivot_table(index=["scene", "seed"], columns="model",
                                                  values="OA")
        d = (s["SSFTT"] - s["CNN2D"]).dropna()
        _, p = wilcoxon(d)
        row = dict(labels_per_class=int(n), diff_SSFTT_minus_CNN2D=round(d.mean(), 4),
                   p=fmt_p(p), n_pairs=len(d))
        for m in s.columns:
            row[PRETTY[m]] = round(s[m].mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def t_decomposition(three: pd.DataFrame) -> pd.DataFrame:
    """Split the random-to-buffered gap into its two operational contrasts."""
    p = three.pivot_table(index="model", columns="protocol", values="OA")
    out = pd.DataFrame({
        "buffered": p["disjoint"].round(4),
        "block_no_buffer": p["block_nobuffer"].round(4),
        "random": p["random"].round(4),
    })
    # Named after the operation, not the mechanism: dropping the buffer also
    # returns the boundary pixels, so neither contrast is a causal estimate.
    out["buffered_to_unbuffered"] = (out.block_no_buffer - out.buffered).round(4)
    out["unbuffered_to_random"] = (out.random - out.block_no_buffer).round(4)
    out["total"] = (out.random - out.buffered).round(4)
    out["first_contrast_share"] = (out.buffered_to_unbuffered / out.total).round(3)
    out.index = [PRETTY[i] for i in out.index]
    return out.sort_values("buffered", ascending=False).reset_index(names="model")


def t_scene_cluster(cross: pd.DataFrame, three: pd.DataFrame, win: pd.DataFrame,
                    cap: pd.DataFrame, bud: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Scene-level bootstrap, since seeds within a scene are repeats."""
    dis = cross[cross.protocol == "disjoint"]
    w = dis.pivot_table(index=["scene", "seed"], columns="model", values="OA")
    scenes = sorted(dis.scene.unique())
    pairs = [("CNN2D", "HybridSN"), ("CNN2D", "CNN3D"), ("CNN2D", "SSFTT"),
             ("CNN2D", "ASCN"), ("SSFTT", "ASCN")]
    # Every patch-based model against the spectral baseline, so the claim that
    # all of them beat it is checked at the scene level and not only pooled.
    pairs += [(m, "SVM") for m in ("CNN2D", "SSFTT", "LSSA", "CNN3D", "HybridSN", "ASCN")]
    rng = np.random.default_rng(0)
    rows = []
    for a, b in pairs:
        per = np.array([(w.loc[s][a] - w.loc[s][b]).mean() for s in scenes])
        draws = per[rng.integers(0, len(per), size=(20000, len(per)))].mean(1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        row = dict(contrast=f"{PRETTY[a]} - {PRETTY[b]}", mean=round(per.mean(), 4),
                   ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                   scenes_positive=f"{int((per > 0).sum())}/{len(scenes)}")
        row.update({SCENE[s]: round(v, 4) for s, v in zip(scenes, per)})
        rows.append(row)

    # The mechanism experiments rest on the same 20 non-independent cells, so
    # each significance claim they carry is repeated here at the scene level.
    def contrast(name, frame, col, a_key, b_key, model=None, models=None):
        """Scene-level bootstrap of one within-scene paired difference."""
        f = frame if model is None else frame[frame.model == model]
        w = f.pivot_table(index=["scene", "seed"], columns=col, values="OA")
        if models is None:
            per = np.array([(w.loc[s][a_key] - w.loc[s][b_key]).mean() for s in scenes])
        else:
            w = f.pivot_table(index=["scene", "seed"], columns=["model", col], values="OA")
            per = np.array([(w.loc[s][(models[0], a_key)] - w.loc[s][(models[1], b_key)]).mean()
                            for s in scenes])
        draws = per[rng.integers(0, len(per), size=(20000, len(per)))].mean(1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        r = dict(contrast=name, mean=round(per.mean(), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                 scenes_positive=f"{int((per > 0).sum())}/{len(scenes)}")
        r.update({SCENE[s]: round(v, 4) for s, v in zip(scenes, per)})
        rows.append(r)

    # The two operational contrasts of the decomposition, pooled over models.
    for a_key, b_key, name in (("block_nobuffer", "disjoint", "dropping the buffer, pooled over models"),
                               ("random", "block_nobuffer",
                                "blocked to interleaved geography, pooled over models")):
        w = three.pivot_table(index=["scene", "seed", "model"], columns="protocol", values="OA")
        per = np.array([(w.loc[s][a_key] - w.loc[s][b_key]).mean() for s in scenes])
        draws = per[rng.integers(0, len(per), size=(20000, len(per)))].mean(1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        r = dict(contrast=name, mean=round(per.mean(), 4), ci_lo=round(lo, 4),
                 ci_hi=round(hi, 4), scenes_positive=f"{int((per > 0).sum())}/{len(scenes)}")
        r.update({SCENE[s]: round(v, 4) for s, v in zip(scenes, per)})
        rows.append(r)

    for proto, lab in (("disjoint", "buffered"), ("random", "random")):
        contrast(f"2-D CNN window 5 to 15, {lab}", win[win.protocol == proto], "patch",
                 15, 5, model="CNN2D")
        contrast(f"SSFTT window 3 to 15, {lab}", win[win.protocol == proto], "patch",
                 15, 3, model="SSFTT")
    for a, b_ in ((32, 16), (64, 32), (128, 64)):
        contrast(f"2-D CNN width {b_} to {a}", cap, "width", a, b_)
    for ep in (10, 60):
        contrast(f"SSFTT - 2-D CNN at {ep} epochs", bud[bud.epochs == ep], "model",
                 "SSFTT", "CNN2D")
    for n in (5, 10, 25):
        contrast(f"SSFTT - 2-D CNN at {n} labels per class", low[low.n_per_class == n],
                 "model", "SSFTT", "CNN2D")
    return pd.DataFrame(rows)


def t_random_pairs(cross: pd.DataFrame) -> pd.DataFrame:
    """Pairwise tests under the conventional protocol.

    Almost every pair separates there, but on differences small enough to be
    practically meaningless, which is the point the table is making.
    """
    ran = cross[cross.protocol == "random"]
    w = ran.pivot_table(index=["scene", "seed"], columns="model", values="OA")
    models = list(ran.groupby("model")["OA"].mean().sort_values(ascending=False).index)
    rows = []
    for a, b in itertools.combinations(models, 2):
        diff = (w[a] - w[b]).dropna()
        _, p = wilcoxon(diff)
        rows.append(dict(model_a=PRETTY[a], model_b=PRETTY[b],
                         diff=round(diff.mean(), 4), p=p))
    out = pd.DataFrame(rows)
    out["p_BH_num"] = bh(out.p.values)
    out["significant"] = np.where(out.p_BH_num < 0.05, "yes", "no")
    out = out.sort_values("p_BH_num").reset_index(drop=True)
    out["p_BH"] = out.p_BH_num.map(fmt_p)
    out["p"] = out.p.map(fmt_p)
    return out[["model_a", "model_b", "diff", "p", "p_BH", "significant"]]


def main() -> None:
    cfg = json.load(open("protocol_config.json"))
    three = pd.read_csv("results_three_protocol.csv")
    cross = three[three.protocol.isin(["disjoint", "random"])].copy()
    win = pd.read_csv("results_window_sweep.csv")
    cap = pd.read_csv("results_capacity.csv")
    bud = pd.read_csv("results_budget.csv")
    low = pd.read_csv("results_low_label.csv")
    tables = {
        "table1_scenes": t_scenes(cfg),
        "table2_leakage_audit": t_leakage(cfg),
        "table3_protocol_cross": t_cross(cross),
        "table4_random_pairs": t_random_pairs(cross),
        "table5_model_pairs": t_pairs(cross),
        "table6_scene_cluster": t_scene_cluster(cross, three, win, cap, bud, low),
        "table7_decomposition": t_decomposition(three),
        "table8_window_sweep": t_window(win),
        "table9_capacity": t_capacity(cap),
        "table10_training_budget": t_budget(bud),
        "table11_low_label": t_lowlabel(low),
    }
    def as_markdown(t: pd.DataFrame) -> str:
        cols = [str(c) for c in t.columns]
        head = "| " + " | ".join(cols) + " |"
        rule = "|" + "|".join("---" for _ in cols) + "|"
        body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
                for row in t.itertuples(index=False)]
        return "\n".join([head, rule, *body])

    md = []
    for name, t in tables.items():
        t.to_csv(f"{name}.csv", index=False)
        md.append(f"## {name}\n\n{as_markdown(t)}\n")
        print(f"=== {name} ===")
        print(t.to_string(index=False))
        print()
    open("tables.md", "w").write("\n".join(md))


if __name__ == "__main__":
    main()
