"""Reproduce every result and figure.  ``python experiments/run_all.py [--quick]``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from entfeat import plots, study, tables  # noqa: E402

FULL = dict(
    effects={"coordination": [0.0, 0.3, 0.6, 0.8, 0.95], "shift": [0.0, 0.1, 0.2, 0.3, 0.4]},
    ntrains=[10, 20, 40, 80],
    seeds=[0, 1, 2, 3, 4],
    bar_n=40,
    op_points=[("coordination", 0.8), ("shift", 0.3)],
    abl_effects={"coordination": [0.6, 0.8, 0.95], "shift": [0.1, 0.2, 0.3]},
    p2q=[0.0, 0.002, 0.005, 0.01, 0.02],
    snapshots=[25, 100, 400, 1600, 6400],
    kernel_shots=[1, 4, 16, 64, 256],
    n_test=300,
    n_test_shots=150,
)
QUICK = dict(
    effects={"coordination": [0.0, 0.8], "shift": [0.0, 0.3]},
    ntrains=[10, 20],
    seeds=[0],
    bar_n=20,
    op_points=[("coordination", 0.8), ("shift", 0.3)],
    abl_effects={"coordination": [0.8], "shift": [0.3]},
    p2q=[0.0, 0.01],
    snapshots=[25, 400],
    kernel_shots=[4, 64],
    n_test=100,
    n_test_shots=60,
)


def mean_std(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.groupby(keys).acc.agg(["mean", "std", "count"]).reset_index()


def summarise(main, abl, noise, shots, cfg, res: Path) -> dict:
    """Key numbers quoted in the README (all derived from the result tables)."""
    out: dict = {"config": {k: v for k, v in cfg.items() if k != "effects"}, "effects": cfg["effects"]}
    piv = main.pivot_table(index=["task", "effect", "n_train", "seed"], columns="method", values="acc")
    piv["best_classical"] = piv[list(study.CLASSICAL)].max(axis=1)
    piv["gain_vs_distSVM"] = piv["entanglement-SVM"] - piv["dist-SVM"]
    piv["gain_vs_best"] = piv["entanglement-SVM"] - piv["best_classical"]
    piv["gain_vs_kernel"] = piv["entanglement-SVM"] - piv["fidelity-kernel-SVM"]
    cell = piv.groupby(["task", "effect", "n_train"]).mean()
    live = cell[cell.index.get_level_values("effect") > 0]
    out["cells_with_signal"] = int(len(live))
    for col in ("gain_vs_distSVM", "gain_vs_best", "gain_vs_kernel"):
        out[f"{col}_cells_positive"] = int((live[col] > 0).sum())
        out[f"{col}_mean_by_task"] = live[col].groupby("task").mean().round(4).to_dict()
        out[f"{col}_max_by_task"] = live[col].groupby("task").max().round(4).to_dict()
        out[f"{col}_min_by_task"] = live[col].groupby("task").min().round(4).to_dict()
    null = cell[cell.index.get_level_values("effect") == 0]
    out["null_mean_acc_by_method"] = null[list(study.METHODS)].mean().round(4).to_dict()
    out["g_choice_counts"] = (
        main[main.method == "entanglement-SVM"].tag.value_counts().to_dict()
    )
    ms = mean_std(main, ["task", "effect", "n_train", "method"])
    ms.to_csv(res / "main_summary.csv", index=False)
    mean_std(abl, ["task", "effect", "depth", "groups"]).to_csv(res / "ablation_summary.csv", index=False)
    mean_std(noise, ["task", "p2q", "lam", "mode", "method"]).to_csv(res / "noise_summary.csv", index=False)
    mean_std(shots, ["task", "method", "budget", "cost_per_sample"]).to_csv(res / "shots_summary.csv", index=False)
    # snapshot threshold: smallest S within 2 points of the exact-feature accuracy
    thr = {}
    for task in shots.task.unique():
        s = shots[(shots.task == task) & (shots.method == "entanglement-SVM")]
        exact = s[~np.isfinite(s.budget)].acc.mean()
        m = s[np.isfinite(s.budget)].groupby("budget").acc.mean()
        ok = m[m >= exact - 0.02]
        thr[task] = {"exact_acc": round(float(exact), 4),
                     "min_snapshots_within_2pts": (int(ok.index.min()) if len(ok) else None)}
    out["snapshot_threshold"] = thr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run (< 1 minute)")
    args = ap.parse_args()
    cfg = QUICK if args.quick else FULL
    res = ROOT / "results" / ("quick" if args.quick else "")
    res.mkdir(parents=True, exist_ok=True)
    fig_dir = res / "figures" if args.quick else ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    main_df = study.run_main(cfg["effects"], cfg["ntrains"], cfg["seeds"], cfg["n_test"], log=log)
    main_df.to_csv(res / "main_raw.csv", index=False)
    abl_df = study.run_ablation(cfg["abl_effects"], cfg["bar_n"], cfg["seeds"], cfg["n_test"], log=log)
    abl_df.to_csv(res / "ablation_raw.csv", index=False)
    noise_df = study.run_noise(cfg["op_points"], cfg["bar_n"], cfg["seeds"], cfg["p2q"], cfg["n_test"], log=log)
    noise_df.to_csv(res / "noise_raw.csv", index=False)
    shots_df, errs = study.run_shots(cfg["op_points"], cfg["bar_n"], cfg["seeds"], cfg["snapshots"],
                                     cfg["kernel_shots"], cfg["n_test_shots"], log=log)
    shots_df.to_csv(res / "shots_raw.csv", index=False)

    tables.make(res, cfg["bar_n"])
    summary = summarise(main_df, abl_df, noise_df, shots_df, cfg, res)
    q = errs.groupby(["task", "snapshots", "feature"]).err.agg(["median", lambda x: x.quantile(0.9)])
    q.columns = ["median_abs_err", "p90_abs_err"]
    q.reset_index().to_csv(res / "snapshot_error_quantiles.csv", index=False)
    with open(res / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    plots.fig_accuracy_bars(main_df, cfg["bar_n"], fig_dir / "fig1_accuracy_by_method.png")
    plots.fig_phase(main_df, fig_dir / "fig2_phase_diagram.png")
    plots.fig_snapshot_ecdf(errs, fig_dir / "fig3_snapshot_error_ecdf.png")
    plots.fig_cost(shots_df, fig_dir / "fig4_accuracy_vs_executions.png")
    plots.fig_noise(noise_df, fig_dir / "fig5_depolarising_noise.png")
    plots.fig_ablation(abl_df, fig_dir / "fig6_ablation.png")
    log("done")


if __name__ == "__main__":
    main()
