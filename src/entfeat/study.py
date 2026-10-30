"""Experiment drivers: main grid, ablations, snapshot-shot sweep and noise sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import circuit, data, features
from .models import View, select_and_score

G_GRID = (0.15, 0.3, 0.6)
G_FIXED = 0.3
METHODS = ("dist-LR", "dist-SVM", "dist-quad-LR", "fidelity-kernel-SVM", "entanglement-SVM")
CLASSICAL = ("dist-LR", "dist-SVM", "dist-quad-LR")


def make_pool(task: str, effect: float, seed: int, n_pool: int, n_test: int):
    """Draw a train pool and a test set; returns distances and labels for both."""
    rng = np.random.default_rng(
        10_000 * seed + int(round(effect * 1000)) + (7 if task == "shift" else 0)
    )
    lm, y = data.sample_landmarks(task, effect, n_pool + n_test, rng)
    d = data.pairwise_distances(lm)
    return d[:n_pool], y[:n_pool], d[n_pool:], y[n_pool:]


def subset(y: np.ndarray, n: int) -> np.ndarray:
    """First ``n/2`` indices of each class (stratified, order-stable)."""
    return np.sort(np.concatenate([np.flatnonzero(y == c)[: n // 2] for c in (0, 1)]))


def entanglement_view(psi_tr, psi_te, g, groups="both", lam=1.0, lam_train=None) -> View:
    """Entanglement-feature view from statevectors (exact Paulis, optional damping)."""
    lt = lam if lam_train is None else lam_train
    r_tr, t_tr = circuit.damp(*circuit.exact_paulis(psi_tr), lt)
    r_te, t_te = circuit.damp(*circuit.exact_paulis(psi_te), lam)
    return View(
        "rbf",
        features.entanglement_features(r_tr, t_tr, groups),
        features.entanglement_features(r_te, t_te, groups),
        f"g={g}",
    )


def evaluate_methods(
    d_tr, y_tr, d_te, y_te, seed, depth=2, g_grid=G_GRID, groups="both", methods=METHODS,
    lam=1.0, lam_train=None,
) -> dict[str, tuple[float, str]]:
    """Test accuracy (and chosen encoding scale) of each method on one train/test split."""
    out: dict[str, tuple[float, str]] = {}
    mu = d_tr.mean(axis=0)
    if "dist-LR" in methods:
        a, t, _ = select_and_score([View("lin", d_tr, d_te, "")], y_tr, y_te, seed)
        out["dist-LR"] = (a, t)
    if "dist-SVM" in methods:
        a, t, _ = select_and_score([View("rbf", d_tr, d_te, "")], y_tr, y_te, seed)
        out["dist-SVM"] = (a, t)
    if "dist-quad-LR" in methods:
        q_tr, q_te = features.quadratic_features(d_tr, mu), features.quadratic_features(d_te, mu)
        a, t, _ = select_and_score([View("lin", q_tr, q_te, "")], y_tr, y_te, seed)
        out["dist-quad-LR"] = (a, t)
    kviews, eviews = [], []
    for g in g_grid:
        psi_tr, psi_te = circuit.encode(d_tr, g, depth), circuit.encode(d_te, g, depth)
        if "fidelity-kernel-SVM" in methods:
            # compute-uncompute kernel circuit is twice as deep: damping lam**2
            k_tr, k_te = features.fidelity_kernel(psi_tr, psi_tr), features.fidelity_kernel(psi_te, psi_tr)
            kl = lam**2
            kviews.append(View("kernel", kl * k_tr + (1 - kl) / circuit.DIM,
                               kl * k_te + (1 - kl) / circuit.DIM, f"g={g}"))
        if "entanglement-SVM" in methods:
            eviews.append(entanglement_view(psi_tr, psi_te, g, groups, lam, lam_train))
    if kviews:
        a, t, _ = select_and_score(kviews, y_tr, y_te, seed)
        out["fidelity-kernel-SVM"] = (a, t)
    if eviews:
        a, t, _ = select_and_score(eviews, y_tr, y_te, seed)
        out["entanglement-SVM"] = (a, t)
    return out


def run_main(effects, ntrains, seeds, n_test=300, log=print) -> pd.DataFrame:
    """Main grid: task x effect x n_train x seed x method (``effects`` maps task -> list)."""
    rows = []
    for task, task_effects in effects.items():
        for eff in task_effects:
            for seed in seeds:
                d_p, y_p, d_te, y_te = make_pool(task, eff, seed, max(ntrains), n_test)
                for n in ntrains:
                    idx = subset(y_p, n)
                    res = evaluate_methods(d_p[idx], y_p[idx], d_te, y_te, seed)
                    for m, (acc, tag) in res.items():
                        rows.append(dict(task=task, effect=eff, n_train=n, seed=seed,
                                         method=m, acc=acc, tag=tag))
            log(f"main: {task} effect={eff} done")
    return pd.DataFrame(rows)


def run_ablation(effects, n_train, seeds, n_test=300, log=print) -> pd.DataFrame:
    """Ablation: entanglement feature groups and circuit depth (``effects`` maps task -> list)."""
    rows = []
    for task, task_effects in effects.items():
        for eff in task_effects:
            for seed in seeds:
                d_p, y_p, d_te, y_te = make_pool(task, eff, seed, n_train, n_test)
                idx = subset(y_p, n_train)
                for depth in (1, 2, 3):
                    for groups in ("intra", "inter", "both"):
                        if depth != 2 and groups != "both":
                            continue
                        r = evaluate_methods(d_p[idx], y_p[idx], d_te, y_te, seed, depth=depth,
                                             groups=groups, methods=("entanglement-SVM",))
                        rows.append(dict(task=task, effect=eff, seed=seed, depth=depth,
                                         groups=groups, acc=r["entanglement-SVM"][0]))
            log(f"ablation: {task} effect={eff} done")
    return pd.DataFrame(rows)


def run_noise(tasks_effects, n_train, seeds, p2q_grid, n_test=300, depth=2, log=print) -> pd.DataFrame:
    """Depolarising sweep: matched (train and test noisy) and mismatched (train noiseless)."""
    rows = []
    for task, eff in tasks_effects:
        for seed in seeds:
            d_p, y_p, d_te, y_te = make_pool(task, eff, seed, n_train, n_test)
            idx = subset(y_p, n_train)
            for p2q in p2q_grid:
                lam = circuit.lam_from_gate_error(p2q, depth)
                for mode, lt in (("matched", None), ("mismatched", 1.0)):
                    for method in ("entanglement-SVM", "fidelity-kernel-SVM"):
                        if mode == "mismatched" and method == "fidelity-kernel-SVM":
                            continue
                        r = evaluate_methods(d_p[idx], y_p[idx], d_te, y_te, seed,
                                             g_grid=(G_FIXED,), methods=(method,), lam=lam,
                                             lam_train=lt)
                        rows.append(dict(task=task, effect=eff, seed=seed, p2q=p2q, lam=lam,
                                         mode=mode, method=method, acc=r[method][0]))
            log(f"noise: {task} seed={seed} done")
    return pd.DataFrame(rows)


def run_shots(tasks_effects, n_train, seeds, snapshots, kernel_shots, n_test=150,
              depth=2, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Finite-measurement sweep.

    Entanglement features come from ``S`` random-Pauli snapshots per sample; the
    kernel uses ``s`` shots per Gram entry.  Returns accuracy rows and per-feature
    absolute estimation errors (for the empirical CDF figure).
    """
    rows, errs = [], []
    for task, eff in tasks_effects:
        for seed in seeds:
            rng = np.random.default_rng(777 + seed)
            d_p, y_p, d_te, y_te = make_pool(task, eff, seed, n_train, n_test)
            idx = subset(y_p, n_train)
            d_tr, y_tr = d_p[idx], y_p[idx]
            psi_tr = circuit.encode(d_tr, G_FIXED, depth)
            psi_te = circuit.encode(d_te, G_FIXED, depth)
            f_tr0 = features.entanglement_features(*circuit.exact_paulis(psi_tr))
            f_te0 = features.entanglement_features(*circuit.exact_paulis(psi_te))
            a0, _, _ = select_and_score([View("rbf", f_tr0, f_te0, "")], y_tr, y_te, seed)
            rows.append(dict(task=task, effect=eff, seed=seed, method="entanglement-SVM",
                             budget=np.inf, cost_per_sample=np.inf, acc=a0))
            ad, _, _ = select_and_score([View("rbf", d_tr, d_te, "")], y_tr, y_te, seed)
            rows.append(dict(task=task, effect=eff, seed=seed, method="dist-SVM",
                             budget=np.inf, cost_per_sample=0, acc=ad))
            for s in snapshots:
                f_tr = features.entanglement_features(*circuit.snapshot_paulis(psi_tr, s, rng))
                f_te = features.entanglement_features(*circuit.snapshot_paulis(psi_te, s, rng))
                a, _, _ = select_and_score([View("rbf", f_tr, f_te, "")], y_tr, y_te, seed)
                rows.append(dict(task=task, effect=eff, seed=seed, method="entanglement-SVM",
                                 budget=s, cost_per_sample=s, acc=a))
                e = np.abs(f_te - f_te0)
                for kind, cols in (("entropy S_i", slice(0, 6)), ("mutual information I_ij", slice(6, 21))):
                    errs.append(pd.DataFrame(dict(task=task, seed=seed, snapshots=s, feature=kind,
                                                  err=e[:, cols].ravel())))
            k_tr0 = features.fidelity_kernel(psi_tr, psi_tr)
            k_te0 = features.fidelity_kernel(psi_te, psi_tr)
            a0, _, _ = select_and_score([View("kernel", k_tr0, k_te0, "")], y_tr, y_te, seed)
            rows.append(dict(task=task, effect=eff, seed=seed, method="fidelity-kernel-SVM",
                             budget=np.inf, cost_per_sample=np.inf, acc=a0))
            for s in kernel_shots:
                k_tr = features.noisy_kernel(k_tr0, s, rng, True)
                k_te = features.noisy_kernel(k_te0, s, rng, False)
                a, _, _ = select_and_score([View("kernel", k_tr, k_te, "")], y_tr, y_te, seed)
                rows.append(dict(task=task, effect=eff, seed=seed, method="fidelity-kernel-SVM",
                                 budget=s, cost_per_sample=s * n_train, acc=a))
            log(f"shots: {task} seed={seed} done")
    return pd.DataFrame(rows), pd.concat(errs, ignore_index=True)
