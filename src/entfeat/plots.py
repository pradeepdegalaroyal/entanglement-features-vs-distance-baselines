"""Figures (Okabe-Ito palette, 200 dpi): hatched bars, empirical CDFs, phase contours."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
METHOD_STYLE = {
    "dist-LR": (OI["sky"], "//"),
    "dist-SVM": (OI["blue"], "\\\\"),
    "dist-quad-LR": (OI["green"], "xx"),
    "fidelity-kernel-SVM": (OI["orange"], ".."),
    "entanglement-SVM": (OI["vermillion"], "++"),
}
TASK_LABEL = {"coordination": "co-movement task (second-order signal)",
              "shift": "mean-shift task (first-order signal)"}
EFFECT_LABEL = {"coordination": "co-movement weight in class 1",
                "shift": "mean displacement norm of class 1 (template units)"}
DPI = 200


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_accuracy_bars(main: pd.DataFrame, n_train: int, path: Path) -> None:
    """Grouped hatched bars: accuracy of every method per effect size, mean +- std over seeds."""
    tasks = list(main.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = main[(main.task == task) & (main.n_train == n_train)]
        effects = sorted(sub.effect.unique())
        methods = [m for m in METHOD_STYLE if m in sub.method.unique()]
        w = 0.8 / len(methods)
        for k, m in enumerate(methods):
            g = sub[sub.method == m].groupby("effect").acc.agg(["mean", "std"]).reindex(effects)
            ax.bar(np.arange(len(effects)) + (k - (len(methods) - 1) / 2) * w, g["mean"], w,
                   yerr=g["std"], color=METHOD_STYLE[m][0], hatch=METHOD_STYLE[m][1],
                   edgecolor="black", linewidth=0.6, capsize=2, label=m)
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        ax.set_xticks(np.arange(len(effects)), [f"{e:g}" for e in effects])
        ax.set_xlabel(EFFECT_LABEL[task])
        ax.set_ylabel("test accuracy (fraction)")
        ax.set_ylim(0.3, 1.02)
        ax.set_title(f"{TASK_LABEL[task]}, {n_train} training samples", fontsize=9)
    axes[0][0].legend(fontsize=7, loc="upper left")
    _save(fig, path)


def _gain_grid(main: pd.DataFrame, task: str, ref: str) -> tuple[np.ndarray, list, list]:
    sub = main[main.task == task]
    piv = sub.pivot_table(index=["effect", "n_train", "seed"], columns="method", values="acc")
    ent = piv["entanglement-SVM"]
    base = piv[["dist-LR", "dist-SVM", "dist-quad-LR"]].max(axis=1) if ref == "best" else piv[ref]
    gain = (ent - base).groupby(level=["effect", "n_train"]).mean().unstack("n_train")
    return gain.values.T, list(gain.index), list(gain.columns)


def fig_phase(main: pd.DataFrame, path: Path) -> None:
    """Filled-contour phase diagram: accuracy gain of entanglement features over a baseline."""
    cmap = LinearSegmentedColormap.from_list("oi", [OI["vermillion"], "#ffffff", OI["blue"]])
    tasks = list(main.task.unique())
    fig, axes = plt.subplots(len(tasks), 2, figsize=(10.5, 3.9 * len(tasks)), squeeze=False)
    for r, task in enumerate(tasks):
        for c, (ref, name) in enumerate([("dist-SVM", "distance RBF-SVM"),
                                         ("best", "best of three classical baselines")]):
            z, xs, ys = _gain_grid(main, task, ref)
            ax = axes[r][c]
            lim = max(0.05, np.nanmax(np.abs(z)))
            cs = ax.contourf(xs, ys, z, levels=np.linspace(-lim, lim, 13), cmap=cmap, extend="both")
            ax.contour(xs, ys, z, levels=[0.0], colors="black", linewidths=1.5)
            ax.set_yscale("log")
            ax.set_yticks(ys, [str(v) for v in ys])
            ax.minorticks_off()
            ax.set_xlabel(EFFECT_LABEL[task], fontsize=8)
            ax.set_ylabel("training samples")
            ax.set_title(f"{task}: entanglement-SVM minus {name}", fontsize=8.5)
            fig.colorbar(cs, ax=ax, label="accuracy gain (fraction)")
    _save(fig, path)


def fig_snapshot_ecdf(errs: pd.DataFrame, path: Path) -> None:
    """Empirical CDF of the absolute feature error for each snapshot budget."""
    kinds = list(errs.feature.unique())
    snaps = sorted(errs.snapshots.unique())
    colors = [OI["sky"], OI["blue"], OI["green"], OI["orange"], OI["vermillion"], OI["purple"]]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.4 * len(kinds), 4.0), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for s, col in zip(snaps, colors):
            e = np.sort(errs[(errs.feature == kind) & (errs.snapshots == s)].err.values)
            ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=col, lw=1.8, label=f"{s} snapshots")
        ax.set_xscale("log")
        ax.set_xlabel(f"absolute error of {kind} (nats)")
        ax.set_ylabel("empirical CDF (fraction of estimates)")
        ax.set_title(f"estimation error of {kind}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    _save(fig, path)


def fig_cost(shots: pd.DataFrame, path: Path) -> None:
    """Accuracy versus circuit executions per classified sample (log axis)."""
    tasks = list(shots.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.0 * len(tasks), 4.0), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = shots[shots.task == task]
        for m, mk in (("entanglement-SVM", "o"), ("fidelity-kernel-SVM", "s")):
            fin = sub[(sub.method == m) & np.isfinite(sub.cost_per_sample)]
            g = fin.groupby("cost_per_sample").acc.agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], g["std"], marker=mk, capsize=3, lw=1.6,
                        color=METHOD_STYLE[m][0], label=f"{m} (finite shots)")
            ex = sub[(sub.method == m) & ~np.isfinite(sub.cost_per_sample)].acc.mean()
            ax.axhline(ex, color=METHOD_STYLE[m][0], ls="--", lw=1.0)
        ref = sub[sub.method == "dist-SVM"].acc.mean()
        ax.axhline(ref, color="black", ls="-.", lw=1.2, label="dist-SVM (no quantum executions)")
        ax.set_xscale("log")
        ax.set_xlabel("circuit executions per classified sample")
        ax.set_ylabel("test accuracy (fraction)")
        ax.set_title(TASK_LABEL[task] + "\ndashed: exact (infinite-shot) value", fontsize=8.5)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, loc="lower right")
    _save(fig, path)


def fig_noise(noise: pd.DataFrame, path: Path) -> None:
    """Accuracy versus two-qubit depolarising error (damping proxy)."""
    tasks = list(noise.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.0 * len(tasks), 4.0), squeeze=False)
    series = [("entanglement-SVM", "matched", OI["vermillion"], "o", "entanglement, train and test noisy"),
              ("entanglement-SVM", "mismatched", OI["purple"], "^", "entanglement, trained noiseless"),
              ("fidelity-kernel-SVM", "matched", OI["orange"], "s", "fidelity kernel, train and test noisy")]
    for ax, task in zip(axes[0], tasks):
        sub = noise[noise.task == task]
        for m, mode, col, mk, lab in series:
            g = sub[(sub.method == m) & (sub["mode"] == mode)].groupby("p2q").acc.agg(["mean", "std"])
            ax.errorbar(g.index * 100, g["mean"], g["std"], color=col, marker=mk, capsize=3, lw=1.6, label=lab)
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        ax.set_xlabel("two-qubit gate depolarising error (percent)")
        ax.set_ylabel("test accuracy (fraction)")
        ax.set_title(TASK_LABEL[task], fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, loc="lower left")
    _save(fig, path)


def fig_ablation(abl: pd.DataFrame, path: Path) -> None:
    """Hatched bars: feature groups and circuit depth of the entanglement features."""
    tasks = list(abl.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.0 * len(tasks), 4.0), squeeze=False)
    variants = [("depth 2, single-qubit entropies only", 2, "intra", OI["sky"], "//"),
                ("depth 2, mutual information only", 2, "inter", OI["green"], "xx"),
                ("depth 1, both", 1, "both", OI["yellow"], "\\\\"),
                ("depth 2, both", 2, "both", OI["vermillion"], "++"),
                ("depth 3, both", 3, "both", OI["purple"], "..")]
    for ax, task in zip(axes[0], tasks):
        sub = abl[abl.task == task]
        effects = sorted(sub.effect.unique())
        w = 0.8 / len(variants)
        for k, (lab, depth, groups, col, hatch) in enumerate(variants):
            g = sub[(sub.depth == depth) & (sub.groups == groups)].groupby("effect").acc.agg(["mean", "std"]).reindex(effects)
            ax.bar(np.arange(len(effects)) + (k - (len(variants) - 1) / 2) * w, g["mean"], w, yerr=g["std"],
                   color=col, hatch=hatch, edgecolor="black", linewidth=0.6, capsize=2, label=lab)
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        ax.set_xticks(np.arange(len(effects)), [f"{e:g}" for e in effects])
        ax.set_xlabel(EFFECT_LABEL[task])
        ax.set_ylabel("test accuracy (fraction)")
        ax.set_ylim(0.3, 1.02)
        ax.set_title(TASK_LABEL[task], fontsize=9)
    axes[0][0].legend(fontsize=6.5, loc="upper left")
    _save(fig, path)
