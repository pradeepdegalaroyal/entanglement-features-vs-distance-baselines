"""Batched six-qubit statevector simulation of the distance-to-state feature map.

Landmark distances ``d_ij`` set the angles of an Ising-type map: start from ``|+>^n``,
apply ``exp(-i * g * d_ij * Z_i Z_j)`` on every pair, and (for depth ``L > 1``)
alternate with an ``RX(mix)`` layer.  Everything is exact except the explicit
random-Pauli *snapshot* sampler, which mimics a finite measurement budget.
"""

from __future__ import annotations

import numpy as np

from .data import N_LANDMARKS, PAIRS

N = N_LANDMARKS
DIM = 2**N
_BITS = (np.arange(DIM)[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1
_Z = 1 - 2 * _BITS  # (DIM, N) eigenvalues of Z on every basis state
_ZZ = np.stack([_Z[:, i] * _Z[:, j] for i, j in PAIRS], axis=1)  # (DIM, 15)

_I2 = np.eye(2, dtype=complex)
_PAULI = np.array(
    [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex
)  # X, Y, Z
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
# basis-change unitaries so that a Z-measurement after V measures X, Y, Z
_BASIS = np.stack([_H, _H @ np.diag([1, -1j]), _I2])


def _apply_1q(psi: np.ndarray, u: np.ndarray, q: int) -> np.ndarray:
    """Apply one 2x2 matrix to qubit ``q`` of a batch ``(B, 2, ..., 2)``."""
    out = np.tensordot(u, psi, axes=([1], [q + 1]))
    return np.moveaxis(out, 0, q + 1)


def encode(dist: np.ndarray, g: float, depth: int = 2, mix: float = 0.8) -> np.ndarray:
    """Statevectors ``(B, 64)`` for normalised distances ``dist`` ``(B, 15)``."""
    b = dist.shape[0]
    psi = np.full((b, DIM), 1.0 / np.sqrt(DIM), dtype=complex)
    rx = np.cos(mix / 2) * _I2 - 1j * np.sin(mix / 2) * _PAULI[0]
    for layer in range(depth):
        psi = psi * np.exp(-1j * g * (dist @ _ZZ.T))
        if layer < depth - 1:
            t = psi.reshape((b,) + (2,) * N)
            for q in range(N):
                t = _apply_1q(t, rx, q)
            psi = t.reshape(b, DIM)
    return psi


def exact_paulis(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact one- and two-qubit Pauli expectations.

    Returns ``r1 (B, N, 3)`` and ``t2 (B, N, N, 3, 3)`` (only ``i < j`` filled).
    """
    b = psi.shape[0]
    t = psi.reshape((b,) + (2,) * N)
    r1 = np.zeros((b, N, 3))
    t2 = np.zeros((b, N, N, 3, 3))
    rho1 = []
    for q in range(N):
        m = np.moveaxis(t, q + 1, 1).reshape(b, 2, -1)
        rho1.append(np.einsum("bai,bci->bac", m, m.conj()))
    for q in range(N):
        r1[:, q] = np.einsum("bac,kca->bk", rho1[q], _PAULI).real
    for i, j in PAIRS:
        m = np.moveaxis(t, (i + 1, j + 1), (1, 2)).reshape(b, 4, -1)
        rho = np.einsum("bai,bci->bac", m, m.conj()).reshape(b, 2, 2, 2, 2)
        t2[:, i, j] = np.einsum("baxcy,kca,lyx->bkl", rho, _PAULI, _PAULI).real
    return r1, t2


def damp(r1: np.ndarray, t2: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Global-depolarising proxy: every non-identity Pauli expectation scales by ``lam``.

    Weight-1 terms scale by ``lam``; weight-2 terms by ``lam`` too (one global channel).
    """
    return r1 * lam, t2 * lam


def lam_from_gate_error(p2q: float, depth: int) -> float:
    """Damping factor after ``depth`` ZZ layers, 15 pair interactions of 2 CNOTs each."""
    return float((1.0 - p2q) ** (2 * len(PAIRS) * depth))


def snapshot_paulis(
    psi: np.ndarray, shots: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the same Pauli expectations from ``shots`` random-Pauli snapshots per state.

    Each snapshot measures every qubit in an independently drawn X/Y/Z basis; a Pauli
    string of weight ``k`` is estimated by the ``3^k``-rescaled matching snapshots.
    """
    b = psi.shape[0]
    r1 = np.zeros((b, N, 3))
    t2 = np.zeros((b, N, N, 3, 3))
    bit_shift = N - 1 - np.arange(N)
    for s in range(b):
        bases = rng.integers(0, 3, size=(shots, N))
        t = np.broadcast_to(psi[s].reshape((1,) + (2,) * N), (shots,) + (2,) * N).copy()
        for q in range(N):
            u = _BASIS[bases[:, q]]  # (shots, 2, 2)
            m = np.moveaxis(t, q + 1, 1).reshape(shots, 2, -1)
            m = np.einsum("nab,nbr->nar", u, m)
            t = np.moveaxis(m.reshape((shots, 2) + (2,) * (N - 1)), 1, q + 1)
        p = np.abs(t.reshape(shots, DIM)) ** 2
        cdf = np.cumsum(p, axis=1)
        idx = (cdf < rng.random((shots, 1)) * cdf[:, -1:]).sum(axis=1).clip(0, DIM - 1)
        sign = 1 - 2 * ((idx[:, None] >> bit_shift[None, :]) & 1)  # (shots, N)
        m1 = 3.0 * (bases[:, :, None] == np.arange(3)[None, None, :]) * sign[:, :, None]
        r1[s] = m1.mean(axis=0)
        t2[s] = np.triu(np.ones((N, N)), 1)[:, :, None, None] * (
            np.einsum("nia,njb->ijab", m1, m1) / shots
        )
    return r1, t2
