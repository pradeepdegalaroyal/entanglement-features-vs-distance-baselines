# Method

## Data (`src/entfeat/data.py`)

Six landmarks in three regions of two, placed on a fixed two-dimensional template. A sample is the template plus Gaussian displacement of every landmark (standard deviation 0.15). Features for every method start from the 15 pairwise Euclidean distances, divided by the template mean distance.

| Task | Class 0 | Class 1 | Signal type |
|---|---|---|---|
| co-movement | independent displacements | the two landmarks of each region share a displacement component with weight `w` (per-landmark variance unchanged) | second order |
| mean-shift | independent displacements | a fixed displacement pattern of norm `m` is added | first order |

Effect size 0 is a null control: all methods must sit at 50 percent.

## Feature map and entanglement descriptor (`circuit.py`, `features.py`)

One qubit per landmark. Start from the uniform superposition state, apply the Ising phase `exp(-i g d_ij Z_i Z_j)` on all 15 pairs, then (depth above 1) a single-qubit `RX(0.8)` layer, then the phase layer again. Depth 2 is the default; depth 1 and 3 are an ablation. The encoding scale `g` is chosen from {0.15, 0.3, 0.6} by cross-validation (CV), as is every classifier hyper-parameter.

Entanglement descriptor (21 features): the von Neumann entropy of each single-qubit reduced state (6, "intra") and the mutual information of every qubit pair (15, "inter"). Both come from one- and two-qubit Pauli expectations. Exact values are computed from the statevector. Hardware-style values are estimated from random-Pauli snapshots (each snapshot measures every qubit in a random X, Y or Z basis; a weight-k Pauli string is estimated by the 3^k-rescaled matching snapshots). Estimated Bloch vectors are clipped to the unit ball and two-qubit states are projected onto the positive semidefinite cone.

## Methods compared

| Name | Input | Learner | Role |
|---|---|---|---|
| dist-LR | 15 distances | logistic regression | classical, linear |
| dist-SVM | 15 distances | radial basis function (RBF) support vector machine (SVM) | classical, non-linear |
| dist-quad-LR | squared deviation of each distance from the training mean | logistic regression | strong classical second-order baseline |
| fidelity-kernel-SVM | same quantum state | SVM on kernel `abs(<a|b>)^2` | standard quantum baseline (linear quantum model) |
| entanglement-SVM | 21 entanglement features | RBF-SVM | the method under test |

All learners get the same 3-fold stratified CV model selection (regularisation, kernel width, and `g` for the quantum views). Features are standardised on the training fold only.

## Metrics, seeds, design

- Test accuracy on 300 held-out balanced samples (150 in the shot sweep). Mean and standard deviation over 5 seeds (0 to 4); every seed redraws data.
- Main grid: 2 tasks x 5 effect sizes x training sizes {10, 20, 40, 80} x 5 methods.
- Paired gain: entanglement-SVM minus (a) dist-SVM, (b) the best of the three classical baselines per seed (a conservative reference that is optimistically biased towards the baselines), (c) the fidelity kernel.
- The first 2-seed pilot (not reported) only fixed the effect-size ranges.

## Ablation

At 40 training samples: single-qubit entropies only, mutual information only, both (depth 2), and depth 1 and 3 with both.

## Noise and robustness sweeps (at 40 training samples, one operating point per task)

1. **Finite measurements.** Entanglement features from 25, 100, 400, 1600, 6400 snapshots per sample; kernel entries from 1, 4, 16, 64, 256 shots each (binomial, then projected onto positive semidefinite). Cost axis: circuit executions per classified sample (snapshots for the descriptor; training size times shots per entry for the kernel). Empirical cumulative distribution functions (CDF) of the absolute feature errors are reported.
2. **Gate noise.** Two-qubit depolarising error p in {0, 0.2, 0.5, 1, 2} percent, converted to a global damping factor `(1-p)^(2*15*depth)` of every non-identity Pauli expectation (2 controlled-NOT (CNOT) gates per pair interaction). The kernel circuit is twice as deep, so its damping is the square. This is a deliberately simple proxy, not a calibrated device model. "Matched" trains and tests under the same noise; "mismatched" trains on noiseless simulation and tests on the noisy device.

## Limits of this design

Six qubits, synthetic data, global-depolarising proxy, a hand-made feature map rather than the exact QIEI map of the source paper (not fully specified in the abstract), exact simulation of the baselines. The goal is a controlled test of the idea, not a re-implementation of any published pipeline.
