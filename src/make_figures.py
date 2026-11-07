"""Draw the two figures of the paper from the result CSVs.

Run from the repository root:

    python src/make_figures.py

Reads results/ and tables/ and writes figures/fig_protocol_and_cost.{png,pdf}
and figures/fig_mechanism.{png,pdf}.  Every number drawn here comes from the
same CSVs that generate the tables, so a figure cannot disagree with a table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
TABLES = Path("tables")
FIGURES = Path("figures")

RED, BLUE, GREY, BLACK = "#c0392b", "#2471a3", "#8d8d8d", "#1a1a1a"
PRETTY = {"SVM": "spectral SVM", "CNN2D": "2-D CNN", "CNN3D": "3-D CNN",
          "HybridSN": "HybridSN", "SSFTT": "SSFTT", "ASCN": "ASCN", "LSSA": "LSSA"}
COLOUR = {"CNN2D": RED, "SSFTT": BLUE, "HybridSN": GREY, "SVM": BLACK, "LSSA": "#8e44ad"}

STYLE = {
    "font.size": 8.0, "axes.titlesize": 8.0, "axes.labelsize": 8.0,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 7.0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "axes.titlelocation": "left", "axes.titlepad": 6.0,
    "lines.linewidth": 1.2, "axes.linewidth": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 200,
}


def figure_one() -> None:
    """Protocols, the decomposition of the gap, and accuracy against cost."""
    dec = pd.read_csv(TABLES / "table7_decomposition.csv")
    three = pd.read_csv(RESULTS / "results_three_protocol.csv")
    dis = three[three.protocol == "disjoint"].copy()
    dis["m"] = dis.model.map(PRETTY)
    g = dis.groupby("m").agg(OA=("OA", "mean"), sd=("OA", "std"), n=("OA", "size"),
                             params=("params", "median"))
    g["se"] = g.sd / np.sqrt(g.n)
    order = dec.model.tolist()
    yy = np.arange(len(order))[::-1]
    top = yy[0]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.15))

    ax = axes[0]
    for y, (_, r) in zip(yy, dec.iterrows()):
        ax.plot([r.buffered, r.random], [y, y], color="#cfcfcf", lw=2, zorder=1,
                solid_capstyle="round")
        ax.scatter(r.buffered, y, s=34, color=RED, zorder=3)
        ax.scatter(r.block_no_buffer, y, s=34, color=BLUE, marker="D", zorder=3)
        ax.scatter(r.random, y, s=34, color=GREY, zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels(order)
    ax.set_xlabel("overall accuracy")
    ax.set_title("Three protocols, same models")
    ax.set_ylim(-0.8, len(order) + 0.55)
    ax.annotate("buffered\nblocks", (dec.buffered.iloc[0], top + 0.25), xytext=(-2, 0),
                textcoords="offset points", ha="center", va="bottom", color=RED, fontsize=7)
    ax.annotate("blocks,\nno buffer", (dec.block_no_buffer.iloc[0], top + 0.25), xytext=(8, 0),
                textcoords="offset points", ha="center", va="bottom", color=BLUE, fontsize=7)
    ax.annotate("random", (dec.random.iloc[0], top + 0.25), xytext=(0, 0),
                textcoords="offset points", ha="center", va="bottom", color=GREY, fontsize=7)

    ax = axes[1]
    a = dec.buffered_to_unbuffered.values * 100
    b = dec.unbuffered_to_random.values * 100
    ax.barh(yy, a, color=RED, height=0.62)
    ax.barh(yy, b, left=a, color=BLUE, height=0.62)
    for y, aa, bb, share in zip(yy, a, b, dec.first_contrast_share.values):
        ax.text(aa + bb + 0.4, y, f"{share * 100:.0f}%", va="center", ha="left",
                fontsize=7, color=RED)
    ax.set_yticks(yy)
    ax.set_yticklabels([])
    ax.set_xlabel("accuracy the conventional protocol adds (pp)")
    lo, hi = dec.first_contrast_share.min(), dec.first_contrast_share.max()
    ax.set_title(f"Dropping the buffer: {lo * 100:.0f} to {hi * 100:.0f} per cent")
    ax.text(a[0] / 2, top + 0.30, "buffer\ndropped", ha="center", va="bottom",
            color=RED, fontsize=7)
    ax.text(a[0] + b[0] / 2, top + 0.30, "geography\nchanged", ha="center", va="bottom",
            color=BLUE, fontsize=7)
    ax.set_xlim(0, (a + b).max() * 1.14)
    ax.set_ylim(-0.8, len(order) + 0.55)

    ax = axes[2]
    for name, row in g.iterrows():
        focal = name == "2-D CNN"
        ax.errorbar(row.params, row.OA, yerr=row.se, fmt="o", ms=5,
                    color=RED if focal else GREY, ecolor=RED if focal else GREY,
                    capsize=2.5, lw=1.2, zorder=3 if focal else 2)
        ax.annotate(name, (row.params, row.OA), textcoords="offset points", xytext=(7, -1),
                    fontsize=7, color=RED if focal else "#4d4d4d", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("overall accuracy, buffered split")
    ax.set_title("Accuracy against model size")
    ax.set_xticks([1e4, 1e5, 1e6])
    ax.set_xticklabels(["10k", "100k", "1M"])
    ax.set_xlim(7e3, 1.4e7)
    ax.margins(y=0.12)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_protocol_and_cost.png", dpi=600)
    fig.savefig(FIGURES / "fig_protocol_and_cost.pdf")
    plt.close(fig)


def figure_two() -> None:
    """Window size, training budget and label budget, under the buffered split."""
    win = pd.read_csv(RESULTS / "results_window_sweep.csv")
    bud = pd.read_csv(RESULTS / "results_budget.csv")
    low = pd.read_csv(RESULTS / "results_low_label.csv")

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.15))

    ax = axes[0]
    for model in ("CNN2D", "SSFTT"):
        for protocol, ls, marker in (("disjoint", "-", "s"), ("random", "--", "o")):
            s = win[(win.model == model) & (win.protocol == protocol)]
            p = s.pivot_table(index=["scene", "seed"], columns="patch", values="OA")
            sides = [c for c in sorted(p.columns) if c >= 5]
            gain = [(p[c] - p[5]).dropna().mean() * 100 for c in sides]
            ax.plot(sides, gain, ls, marker=marker, ms=4, lw=1.4, color=COLOUR[model],
                    alpha=1.0 if protocol == "disjoint" else 0.55)
    ax.axhline(0, color="#cfcfcf", lw=0.8, zorder=0)
    ax.set_xlabel("patch side (pixels)")
    ax.set_ylabel("accuracy gain over a 5-pixel window (pp)")
    ax.set_title("Window size: the gain is\nmodel-dependent")
    ax.text(11.2, 2.35, "2-D CNN", color=RED, fontsize=7)
    ax.text(12.6, 1.05, "SSFTT", color=BLUE, fontsize=7)
    ax.text(6.2, 2.05, "random split", color="#7a7a7a", fontsize=7)
    ax.text(6.2, 0.35, "buffered split", color="#4d4d4d", fontsize=7)
    ax.margins(0.05)

    ax = axes[1]
    for model in ("CNN2D", "SSFTT", "HybridSN"):
        s = bud[bud.model == model].groupby("epochs").OA.mean()
        ax.plot(s.index, s.values, "-o", ms=4, lw=1.4, color=COLOUR[model])
        ax.annotate(PRETTY[model], (s.index[-1], s.values[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=7, color=COLOUR[model], va="center")
    ax.set_xlabel("training epochs")
    ax.set_ylabel("overall accuracy, buffered split")
    ax.set_title("Training budget decides\nthe ranking")
    ax.set_xlim(0, 132)

    ax = axes[2]
    xs = sorted(low.n_per_class.unique())
    offsets = {"SSFTT": 0, "CNN2D": 7, "SVM": -8}
    for model, offset in offsets.items():
        s = low[low.model == model].groupby("n_per_class").OA.mean().reindex(xs)
        ax.plot(range(len(xs)), s.values, "-o", ms=4, lw=1.4, color=COLOUR[model])
        ax.annotate(PRETTY[model], (len(xs) - 1, s.values[-1]), textcoords="offset points",
                    xytext=(6, offset), fontsize=7, color=COLOUR[model], va="center")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_xlabel("labelled pixels per class")
    ax.set_ylabel("overall accuracy, buffered split")
    ax.set_title("The transformer leads at every\nlabel budget tested")
    ax.set_xlim(-0.15, len(xs) - 0.25)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_mechanism.png", dpi=600)
    fig.savefig(FIGURES / "fig_mechanism.pdf")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    with mpl.rc_context(STYLE):
        figure_one()
        figure_two()
    print("wrote", *(sorted(p.name for p in FIGURES.glob("fig_*"))))


if __name__ == "__main__":
    main()
