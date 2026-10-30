"""Feature builders: entanglement descriptors, classical distance features, quantum kernel."""

from __future__ import annotations

import numpy as np

from .data import PAIRS

_LN2 = np.log(2.0)


def _entropy(evals: np.ndarray) -> np.ndarray:
    """Von Neumann entropy (nats) from eigenvalues on the last axis."""
    e = np.clip(evals, 1e-12, None)
    e = e / e.sum(axis=-1, keepdims=True)
    return -(e * np.log(e)).sum(axis=-1)


def entanglement_features(
    r1: np.ndarray, t2: np.ndarray, groups: str = "both"
) -> np.ndarray:
    """Entanglement-index features from Pauli expectations.

    * ``intra`` (6): von Neumann entropy ``S_i`` of every single-qubit reduced state
      (how entangled landmark ``i`` is with the rest).
    * ``inter`` (15): mutual information ``I_ij = S_i + S_j - S_ij`` for every pair
      (coordination between landmarks ``i`` and ``j``).

    Estimated (snapshot) Paulis may be unphysical; the Bloch vector is clipped to the
    unit ball and each two-qubit state is projected onto the positive semidefinite cone.
    """
    b, n, _ = r1.shape
    norm = np.linalg.norm(r1, axis=2, keepdims=True)
    r1c = r1 / np.maximum(norm, 1.0)
    s1 = _entropy(np.stack([(1 + np.linalg.norm(r1c, axis=2)) / 2,
                            (1 - np.linalg.norm(r1c, axis=2)) / 2], axis=-1))  # (B, N)
    out = []
    if groups in ("both", "intra"):
        out.append(s1)
    if groups in ("both", "inter"):
        sig = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
        basis = np.concatenate([np.eye(2)[None].astype(complex), sig])  # I, X, Y, Z
        mi = np.zeros((b, len(PAIRS)))
        for k, (i, j) in enumerate(PAIRS):
            coef = np.zeros((b, 4, 4))
            coef[:, 0, 0] = 1.0
            coef[:, 1:, 0] = r1c[:, i]
            coef[:, 0, 1:] = r1c[:, j]
            coef[:, 1:, 1:] = t2[:, i, j]
            rho = 0.25 * np.einsum("bkl,kxy,lzw->bxzyw", coef, basis, basis).reshape(b, 4, 4)
            rho = (rho + rho.conj().transpose(0, 2, 1)) / 2
            ev = np.linalg.eigvalsh(rho)
            ev = np.clip(ev, 0.0, None)
            mi[:, k] = s1[:, i] + s1[:, j] - _entropy(ev)
        out.append(np.clip(mi, 0.0, None))
    return np.concatenate(out, axis=1)


def quadratic_features(dist: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Classical second-order baseline: squared deviation of every distance from ``mean``."""
    return (dist - mean[None]) ** 2


def fidelity_kernel(psi_a: np.ndarray, psi_b: np.ndarray) -> np.ndarray:
    """Fidelity quantum kernel ``|<a|b>|^2``: the standard linear-quantum-model baseline."""
    return np.abs(psi_a.conj() @ psi_b.T) ** 2


def noisy_kernel(
    k: np.ndarray, shots: int, rng: np.random.Generator, symmetric: bool
) -> np.ndarray:
    """Finite-shot kernel entries ``Binomial(shots, k)/shots`` (compute-uncompute test)."""
    est = rng.binomial(shots, np.clip(k, 0.0, 1.0)) / shots
    if symmetric:
        est = np.triu(est) + np.triu(est, 1).T
        w, v = np.linalg.eigh(est)
        est = (v * np.clip(w, 0.0, None)) @ v.T  # project to PSD for the SVM
    return est
