"""Headline tables derived from the raw result files (the only source of README numbers)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .study import CLASSICAL


def make(res: Path, n_focus: int = 40) -> None:
    """Write ``headline_n{n}.csv`` (mean +- std per method) and ``paired_gains.csv``."""
    main = pd.read_csv(res / "main_raw.csv")
    sub = main[main.n_train == n_focus]
    g = sub.groupby(["task", "effect", "method"]).acc.agg(["mean", "std"]).reset_index()
    g["cell"] = g["mean"].map("{:.3f}".format) + " +- " + g["std"].map("{:.3f}".format)
    g.pivot_table(index=["task", "effect"], columns="method", values="cell", aggfunc="first").reset_index().to_csv(
        res / f"headline_n{n_focus}.csv", index=False)
    piv = main.pivot_table(index=["task", "effect", "n_train", "seed"], columns="method", values="acc")
    piv["best_classical"] = piv[list(CLASSICAL)].max(axis=1)
    out = pd.DataFrame({
        "gain_vs_dist-SVM": piv["entanglement-SVM"] - piv["dist-SVM"],
        "gain_vs_best_classical": piv["entanglement-SVM"] - piv["best_classical"],
        "gain_vs_fidelity-kernel": piv["entanglement-SVM"] - piv["fidelity-kernel-SVM"],
    })
    agg = out.groupby(["task", "effect", "n_train"]).agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.reset_index().to_csv(res / "paired_gains.csv", index=False)
