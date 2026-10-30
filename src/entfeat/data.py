"""Synthetic six-landmark configurations with two kinds of class signal.

Three regions of two landmarks each live on a fixed 2-D template.  A sample is the
template plus a random displacement of every landmark.

* ``coordination`` task: both classes have the same per-landmark displacement
  variance and zero mean.  Class 1 makes the two landmarks of each region move
  together (shared displacement with weight ``effect``).  The class signal is a
  second-order (co-movement) one, the situation an entanglement index is meant for.
* ``shift`` task: class 1 adds a fixed displacement pattern of size ``effect``.
  The class signal is a first-order mean shift, where plain distances should do well.
"""

from __future__ import annotations

import numpy as np

N_LANDMARKS = 6
REGION = np.array([0, 0, 1, 1, 2, 2])
TEMPLATE = np.array(
    [[-1.0, 1.0], [-0.4, 1.2], [0.4, 1.2], [1.0, 1.0], [-0.5, -0.8], [0.5, -0.8]]
)
SIGMA = 0.15
PAIRS = [(i, j) for i in range(N_LANDMARKS) for j in range(i + 1, N_LANDMARKS)]
# fixed class-1 mean pattern for the "shift" task (unit-norm over the 12 coordinates)
_PATTERN = np.array(
    [[0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [-1.0, 0.5], [1.0, 0.5]]
)
_PATTERN = _PATTERN / np.linalg.norm(_PATTERN)


def sample_landmarks(
    task: str, effect: float, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` balanced samples. Returns landmarks ``(n, 6, 2)`` and labels ``(n,)``."""
    if task not in ("coordination", "shift"):
        raise ValueError(f"unknown task {task!r}")
    y = np.arange(n) % 2
    rng.shuffle(y)
    eps = rng.normal(0.0, SIGMA, size=(n, N_LANDMARKS, 2))
    if task == "coordination":
        eta = rng.normal(0.0, SIGMA, size=(n, 3, 2))[:, REGION, :]
        w = np.where(y == 1, effect, 0.0)[:, None, None]
        disp = np.sqrt(1.0 - w) * eps + np.sqrt(w) * eta
    else:
        disp = eps + (y == 1)[:, None, None] * effect * _PATTERN[None]
    return TEMPLATE[None] + disp, y


def pairwise_distances(landmarks: np.ndarray) -> np.ndarray:
    """Normalised Euclidean distances ``(n, 15)`` (divided by the template mean distance)."""
    d = np.stack(
        [np.linalg.norm(landmarks[:, i] - landmarks[:, j], axis=1) for i, j in PAIRS], axis=1
    )
    ref = np.mean([np.linalg.norm(TEMPLATE[i] - TEMPLATE[j]) for i, j in PAIRS])
    return d / ref
