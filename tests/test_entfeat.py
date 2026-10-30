"""Fast tests: physics of the feature map, estimator consistency, data balance."""

import numpy as np

from entfeat import circuit, data, features, study


def _dist(n=6, seed=0, task="coordination", effect=0.5):
    lm, y = data.sample_landmarks(task, effect, n, np.random.default_rng(seed))
    return data.pairwise_distances(lm), y


def test_encode_is_normalised():
    d, _ = _dist()
    for depth in (1, 2, 3):
        psi = circuit.encode(d, 0.3, depth)
        assert np.allclose(np.linalg.norm(psi, axis=1), 1.0)


def test_exact_paulis_match_dense_operator():
    d, _ = _dist()
    psi = circuit.encode(d, 0.5, 2)
    r1, t2 = circuit.exact_paulis(psi)
    x = np.array([[0, 1], [1, 0]])
    z = np.diag([1, -1])
    eye = np.eye(2)

    def op(ops):
        m = np.array([[1.0]])
        for o in ops:
            m = np.kron(m, o)
        return m

    ops = [eye] * 6
    ops[1], ops[4] = x, z
    assert np.isclose((psi[2].conj() @ op(ops) @ psi[2]).real, t2[2, 1, 4, 0, 2])
    ops = [eye] * 6
    ops[3] = z
    assert np.isclose((psi[2].conj() @ op(ops) @ psi[2]).real, r1[2, 3, 2])


def test_product_state_has_zero_entanglement():
    psi = circuit.encode(np.zeros((1, 15)), 0.3, 2)  # |+>^6
    f = features.entanglement_features(*circuit.exact_paulis(psi))
    assert np.allclose(f, 0.0, atol=1e-6)


def test_entropy_bounds_and_mi_nonnegative():
    d, _ = _dist(n=8)
    f = features.entanglement_features(*circuit.exact_paulis(circuit.encode(d, 0.6, 2)))
    assert f.shape == (8, 21)
    assert np.all(f[:, :6] <= np.log(2) + 1e-9) and np.all(f >= 0)


def test_snapshots_converge_to_exact():
    d, _ = _dist(n=2)
    psi = circuit.encode(d, 0.3, 2)
    r1, t2 = circuit.exact_paulis(psi)
    rng = np.random.default_rng(0)
    e_small = np.abs(circuit.snapshot_paulis(psi, 50, rng)[0] - r1).mean()
    e_big = np.abs(circuit.snapshot_paulis(psi, 4000, rng)[0] - r1).mean()
    assert e_big < e_small and e_big < 0.1


def test_data_balanced_and_null_task_identical_classes():
    _, y = _dist(n=100)
    assert abs(y.mean() - 0.5) < 0.05
    lm, y = data.sample_landmarks("coordination", 0.0, 2000, np.random.default_rng(1))
    assert np.allclose(lm[y == 0].var(axis=0), lm[y == 1].var(axis=0), rtol=0.2)


def test_fidelity_kernel_is_psd_with_unit_diagonal():
    d, _ = _dist(n=10)
    psi = circuit.encode(d, 0.3, 2)
    k = features.fidelity_kernel(psi, psi)
    assert np.allclose(np.diag(k), 1.0) and np.linalg.eigvalsh(k).min() > -1e-9


def test_evaluate_methods_smoke():
    d_p, y_p, d_te, y_te = study.make_pool("shift", 0.4, 0, 20, 40)
    idx = study.subset(y_p, 20)
    res = study.evaluate_methods(d_p[idx], y_p[idx], d_te, y_te, 0, g_grid=(0.3,))
    assert set(res) == set(study.METHODS)
    assert res["dist-LR"][0] > 0.6
