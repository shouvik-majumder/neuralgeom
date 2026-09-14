"""Tests for the numpy/geomstats frame-based Grassmannian API merged into
``neuralgeom.geometry.grassmann``."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralgeom.geometry import grassmann as G

geomstats = pytest.importorskip("geomstats")


def _rand_frame(N, k, seed):
    rng = np.random.default_rng(seed)
    return G.orthonormalize(rng.standard_normal((N, k)))


def test_projector_roundtrip():
    U = _rand_frame(20, 3, 0)
    P = G.frame_to_projector(U)
    assert np.allclose(P, P.T)                      # symmetric
    assert np.allclose(P @ P, P, atol=1e-8)         # idempotent
    assert abs(np.trace(P) - 3) < 1e-8              # trace == k
    U2 = G.projector_to_frame(P, 3)
    # same subspace ⇒ zero principal angles
    assert np.max(G.frame_principal_angles(U, U2)) < 1e-6


def test_principal_angles_self_zero_and_range():
    U = _rand_frame(15, 2, 1)
    assert np.allclose(G.frame_principal_angles(U, U), 0.0, atol=1e-7)
    V = _rand_frame(15, 2, 2)
    th = G.frame_principal_angles(U, V)
    assert np.all(th >= -1e-9) and np.all(th <= np.pi / 2 + 1e-6)
    assert np.all(np.diff(th) >= -1e-9)             # ascending


def test_fast_distance_matches_geomstats():
    # the √2·arc-length convention regression guard
    err = G.check_frame_distance_against_geomstats(N=30, k=2, n_pairs=10, seed=0)
    assert err < 1e-8


def test_canonical_is_sqrt2_times_arclength():
    U, V = _rand_frame(12, 2, 3), _rand_frame(12, 2, 4)
    d_can = G.frame_distance(U, V, "sqrt2_principal_angle")
    d_arc = G.frame_distance(U, V, "principal_angle")
    assert abs(d_can - G._SQRT2 * d_arc) < 1e-10


def test_distance_matrix_symmetric_zero_diag():
    frames = np.stack([_rand_frame(10, 1, s) for s in range(6)])
    D = G.frame_distance_matrix(frames, "sqrt2_principal_angle")
    assert D.shape == (6, 6)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)


def test_manifold_log_exp_roundtrip():
    man = G.GrassmannManifold(10, 2)
    A, B = _rand_frame(10, 2, 5), _rand_frame(10, 2, 6)
    Pa, Pb = G.frame_to_projector(A), G.frame_to_projector(B)
    v = man.log(Pb, Pa)
    Pb2 = man.exp(v, Pa)
    assert man.dist(Pb, Pb2) < 1e-6
    # norm of the log equals the geodesic distance
    assert abs(man.norm(v, Pa) - man.dist(Pa, Pb)) < 1e-6
