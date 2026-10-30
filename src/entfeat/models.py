"""Classifiers with inner-CV model selection that is identical for every feature set."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

C_SVM = (0.3, 3.0, 30.0)
GAMMA_MULT = (0.3, 1.0, 3.0)
C_LIN = (0.03, 0.3, 3.0, 30.0)


@dataclass
class View:
    """One candidate representation (e.g. one encoding scale ``g``).

    ``kind`` is ``"rbf"`` (RBF-SVM on features), ``"lin"`` (logistic regression on
    features) or ``"kernel"`` (SVM on a precomputed kernel; ``xtr`` is ``(n, n)`` and
    ``xte`` is ``(m, n)``).
    """

    kind: str
    xtr: np.ndarray
    xte: np.ndarray
    tag: str = ""
    extra: dict = field(default_factory=dict)


def _grid(kind: str, dim: int) -> list[dict]:
    if kind == "rbf":
        return [{"C": c, "gm": g} for c in C_SVM for g in GAMMA_MULT]
    if kind == "lin":
        return [{"C": c} for c in C_LIN]
    return [{"C": c} for c in C_SVM]


def _fit_predict(kind, params, xtr, ytr, xte, tr_idx=None, te_idx=None):
    """Fit on rows ``tr_idx`` and predict rows ``te_idx`` (kernel: slice both axes)."""
    if kind == "kernel":
        m = SVC(C=params["C"], kernel="precomputed").fit(xtr[np.ix_(tr_idx, tr_idx)], ytr[tr_idx])
        return m.predict(xtr[np.ix_(te_idx, tr_idx)])
    sc = StandardScaler().fit(xtr[tr_idx])
    a, b = sc.transform(xtr[tr_idx]), sc.transform(xte if te_idx is None else xtr[te_idx])
    if kind == "rbf":
        gamma = params["gm"] / a.shape[1]
        return SVC(C=params["C"], kernel="rbf", gamma=gamma).fit(a, ytr[tr_idx]).predict(b)
    return LogisticRegression(C=params["C"], max_iter=2000).fit(a, ytr[tr_idx]).predict(b)


def select_and_score(
    views: list[View], ytr: np.ndarray, yte: np.ndarray, seed: int
) -> tuple[float, str, dict]:
    """Pick (view, hyper-parameters) by 3-fold CV on the training set; return test accuracy."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(ytr)), ytr))
    best = (-1.0, None, None)
    for v in views:
        for p in _grid(v.kind, v.xtr.shape[1] if v.kind != "kernel" else 0):
            acc = np.mean(
                [
                    np.mean(_fit_predict(v.kind, p, v.xtr, ytr, v.xte, tr, va) == ytr[va])
                    for tr, va in folds
                ]
            )
            if acc > best[0] + 1e-12:
                best = (acc, v, p)
    _, v, p = best
    if v.kind == "kernel":
        m = SVC(C=p["C"], kernel="precomputed").fit(v.xtr, ytr)
        pred = m.predict(v.xte)
    else:
        idx = np.arange(len(ytr))
        pred = _fit_predict(v.kind, p, v.xtr, ytr, v.xte, idx, None)
    return float(np.mean(pred == yte)), v.tag, p
