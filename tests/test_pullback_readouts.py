"""Analytic-case validation of the readout-map pullback core
(neuralgeom.geometry: maps / output_metrics / PullbackMetric / RiemannianField).

These complement ``test_pullback_metric.py`` (the torch batched-Jacobian
path): here the maps are numpy ReadoutMaps with analytic or finite-difference
Jacobians, and the output metrics include the Fisher-Rao families.

Run:  python tests/test_pullback_readouts.py     (or: pytest)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuralgeom.geometry import (Euclidean, BernoulliFisher, CategoricalFisher,   # noqa: E402
                             LogScaleGaussian, GaussianMuLogSigmaFisher,
                             LinearReadout, FunctionReadout, PullbackMetric,
                             RiemannianField)


def test_linear_pullback_is_WtW():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 6))
    pm = PullbackMetric(LinearReadout(W), Euclidean())
    x = rng.standard_normal(6)
    g = pm.metric_matrix(x)
    assert np.allclose(g, W.T @ W, atol=1e-10)


def test_isometry_gives_identity():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 3))
    Q, _ = np.linalg.qr(A)            # Q: (5,3), Q^T Q = I_3
    pm = PullbackMetric(LinearReadout(Q), Euclidean())
    g = pm.metric_matrix(rng.standard_normal(3))
    assert np.allclose(g, np.eye(3), atol=1e-10)


def test_numeric_jacobian_matches_analytic():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((3, 4))
    f = FunctionReadout(lambda x: W @ x)      # no analytic jac -> finite diff
    x = rng.standard_normal(4)
    assert np.allclose(f.jacobian(x), W, atol=1e-6)


def test_scalar_output_metric_is_rank1_outer_gradient():
    # f(x) = ||x||^2 ; grad = 2x ; pullback (Euclidean) = 4 x x^T (rank 1)
    f = FunctionReadout(lambda x: np.array([x @ x]))
    pm = PullbackMetric(f, Euclidean())
    x = np.array([1.0, -2.0, 0.5])
    g = pm.metric_matrix(x)
    assert np.allclose(g, 4.0 * np.outer(x, x), atol=1e-4)
    assert pm.rank(x) == 1
    assert abs(pm.effective_rank(x) - 1.0) < 1e-6


def test_categorical_fisher_kernel_is_ones():
    # softmax Fisher diag(p)-pp^T annihilates the all-ones logit direction
    G = CategoricalFisher().matrix(np.array([0.3, -1.0, 2.0]))
    ones = np.ones(3)
    assert np.allclose(G @ ones, 0.0, atol=1e-12)


def test_bernoulli_fisher_value():
    G = BernoulliFisher().matrix(np.array([0.25]))
    assert np.allclose(G, np.array([[1.0 / (0.25 * 0.75)]]), atol=1e-9)


def test_log_scale_metric_is_weber():
    # equal PERCENTAGE steps are equal distances: g_Y = 1/(sigma y)^2
    m = LogScaleGaussian(sigma=0.5)
    for y in (0.2, 0.6, 1.8):
        G = m.matrix(np.array([y]))
        assert np.allclose(G, [[1.0 / (0.5 ** 2 * y ** 2)]], atol=1e-12)


def test_mu_logsigma_fisher():
    G = GaussianMuLogSigmaFisher().matrix(np.array([0.3, np.log(0.5)]))
    assert np.allclose(G, np.diag([1 / 0.25, 2.0]), atol=1e-12)


def test_flat_metric_geodesic_is_straight():
    field = RiemannianField(lambda x: np.eye(2), dim=2)
    x0 = np.array([0.0, 0.0]); x1 = np.array([1.0, 2.0])
    d = field.distance(x0, x1, n_nodes=12, maxiter=200)
    assert abs(d - np.linalg.norm(x1 - x0)) < 1e-3


def test_sphere_scalar_curvature():
    # 2-sphere of radius r: metric diag(r^2, r^2 sin^2 theta); scalar curvature = 2/r^2.
    r = 1.7

    def metric(u):
        theta = u[0]
        return np.array([[r ** 2, 0.0], [0.0, r ** 2 * np.sin(theta) ** 2]])

    field = RiemannianField(metric, dim=2, h1=1e-4, h2=1e-3)
    R = field.scalar_curvature(np.array([1.0, 0.4]))  # away from poles
    expected = 2.0 / r ** 2
    assert abs(R - expected) / expected < 0.02, (R, expected)


def test_fixture_ground_truth_direction():
    # the pullback's top eigenvector must recover the built-in latency direction
    from neuralgeom.synth import make_synthetic_readout, sample_states
    spec = make_synthetic_readout(dim=5, seed=0)
    pm = PullbackMetric(spec["latency_readout"], Euclidean())
    cos = []
    for x in sample_states(5, n=50, seed=1):
        _, V = pm.spectrum(x)
        cos.append(abs(V[:, 0] @ spec["w_lat"]))
    assert np.median(cos) > 0.99


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"{len(fns)}/{len(fns)} passed")
