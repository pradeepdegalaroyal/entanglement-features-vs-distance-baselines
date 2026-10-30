# Research gap: do entanglement-entropy descriptors add anything beyond the distances they are computed from?

## State of the art (from the supplied literature)

**Quantum-inspired descriptors.** (Citations give the digital object identifier, DOI.) Rengasamy et al. (2027, doi:10.1007/978-3-032-37667-1_24) introduce the Quantum Information Entanglement Index (QIEI). It maps normalised landmark distances into a Hilbert-space feature map and summarises coordination by the von Neumann entropy of reduced states. They report consistent gains in facial-expression recognition and call the method deterministic and free of physical quantum assumptions. Xiao and Pedrycz (2023, doi:10.1109/TPAMI.2022.3167045) use a quantum mass function and its entropy for pattern classification. Sbert et al. (2027, doi:10.1007/978-3-032-37976-4_5) show that von Neumann entropy is the unique additive member of two quantum f-divergence entropy families, which is the formal reason entropy is the natural summary of a reduced state.

**Quantum machine learning (QML) baselines.** Jerbi et al. (2023, doi:10.1038/s41467-023-36159-y) place fidelity kernels, variational models and data re-uploading circuits in one family of linear quantum models and discuss their fit to noisy intermediate-scale quantum (NISQ) constraints. A feature map plus a kernel is therefore the standard quantum baseline for any new quantum-flavoured feature.

**Hardware realism.** Experiments now reach two-qubit fidelity 99.5 percent (Evered et al., 2023, doi:10.1038/s41586-023-06481-y), erasure conversion (Ma et al., 2023, doi:10.1038/s41586-023-06438-1) and error correction beyond break-even (Sivak et al., 2023, doi:10.1038/s41586-023-05782-6). The review of Fisher et al. (2023, doi:10.1146/annurev-conmatphys-031720-030658) shows that entanglement structure is measured in practice from local measurements, and the repeater review of Azuma et al. (2023, doi:10.1103/RevModPhys.95.045006) stresses loss and noise budgets. Das and Dash (2027, doi:10.1007/978-3-032-35516-4_32) report T-count costs of a Qiskit circuit, an example of the resource accounting that feature-based QML papers usually omit.

## Related work table

The brief tags every paper with the role "other". The roles below are my own assignment from publication date, citation count and article type.

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Rengasamy et al., doi:10.1007/978-3-032-37667-1_24 | 2027 | recent | QIEI: entropy of reduced states as a landmark-coordination descriptor | Controlled comparison with the distances it is computed from; measurement cost; noise; quantum-kernel baseline |
| Sbert et al., doi:10.1007/978-3-032-37976-4_5 | 2027 | recent | Structure and coarse-graining of quantum f-divergence entropies | Any learning task or estimation from finite samples |
| Das and Dash, doi:10.1007/978-3-032-35516-4_32 | 2027 | recent | Quantum Hamming encoder and decoder in Qiskit with T-gate count | Data-driven use of circuits; classification |
| Wang et al., doi:10.1016/j.disc.2026.115424 | 2027 | recent | Orthogonal arrays with repeated rows (experimental design) | Applications to quantum feature selection or benchmarking |
| Jerbi et al., doi:10.1038/s41467-023-36159-y | 2023 | landmark | Linear-quantum-model framework: kernels, re-uploading, resource scaling | Hand-designed entropy features; finite-snapshot estimation of them |
| Evered et al., doi:10.1038/s41586-023-06481-y | 2023 | landmark | 99.5 percent parallel two-qubit gates on neutral atoms | Use of such gates for learning tasks |
| Sivak et al., doi:10.1038/s41586-023-05782-6 | 2023 | landmark | Real-time error correction beyond break-even | Near-term feature extraction without error correction |
| Fisher et al., doi:10.1146/annurev-conmatphys-031720-030658 | 2023 | review | Entanglement dynamics and measurement-induced transitions in random circuits | Task-level value of entanglement summaries |
| Azuma et al., doi:10.1103/RevModPhys.95.045006 | 2023 | review | Quantum repeaters: loss and noise as the central obstacle | Learning workloads |
| Xiao and Pedrycz, doi:10.1109/TPAMI.2022.3167045 | 2023 | adjacent | Quantum mass function negation for multisource fusion and pattern classification | Comparison against plain classical features; hardware cost |
| Ma et al., doi:10.1038/s41586-023-06438-1 | 2023 | adjacent | Mid-circuit erasure conversion in an atomic qubit | Learning tasks |

## The gap

QIEI-type descriptors are reported to help, but (i) the evidence is on benchmark data where many factors move at once, so nobody has isolated whether the entropy structure carries information beyond the distances that feed it, (ii) the standard quantum baseline of Jerbi et al., a fidelity kernel on the same state, is absent, and (iii) the descriptor is posed as "deterministic" while any run on quantum hardware must estimate it from snapshots under gate noise, whose cost the papers do not report.

**Novelty in one sentence:** this study measures, on controlled synthetic landmark data with a known type of class signal, when an entanglement-entropy descriptor beats the classical distances and a fidelity quantum kernel, and what snapshot budget and gate error it needs to do so.

## Falsifiable hypotheses and success metric

Scale: statevector simulation of 6 qubits (one per landmark), 5 seeds, 10 to 80 training samples. This is simulation and says nothing about quantum advantage.

- **H1 (co-movement task):** the entanglement support vector machine (SVM) beats the best of three classical baselines by at least 3 percentage points of test accuracy in at least half of the cells that carry class signal.
- **H2 (mean-shift task):** it does not (gain at most 0), because first-order signal is already in the distances.
- **H3 (cost):** within 2 percentage points of the exact-feature accuracy is reached with at most 1600 random-Pauli snapshots per sample.
- **H4 (budget):** at equal circuit executions per classified sample, entanglement features reach higher accuracy than the shot-limited fidelity kernel.

**Success metric:** mean test accuracy over 5 seeds and the paired accuracy gain per (task, effect size, training size) cell. H1 and H2 are the main claims. A 2-seed pilot was used only to choose effect-size ranges; it already hinted that H1 might fail, so the full study is a real test of it.
