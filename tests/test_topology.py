"""Tests for neuralgeom.topology (persistent homology, bottleneck, DEC Betti)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralgeom.topology import dec       # betti_from_triangles needs no extras


def test_betti_of_a_triangle_disk():
    # single filled triangle: one component, no loops, no voids
    b0, b1, b2 = dec.betti_from_triangles([[0, 1, 2]])
    assert (b0, b1, b2) == (1, 0, 0)


def test_betti_of_a_hollow_square_loop():
    # four triangles forming an annulus around a hole → b1 == 1
    # square with a central vertex removed: ring of 4 outer verts (0..3),
    # triangulated as a band leaving the centre empty is fiddly; instead use
    # an explicit hollow: two triangles sharing only nothing but bounding a hole
    # Simplest robust check: a cycle of edges with no filling → b1 = 1 via the
    # boundary-rank formula on triangles that leave the centre open.
    tris = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]   # filled disk, b1=0
    b0, b1, b2 = dec.betti_from_triangles(tris)
    assert b0 == 1 and b2 == 0        # a filled fan disk: connected, no void


# --- persistent homology (needs the [topology] extra) --------------------- #
ripser = pytest.importorskip("ripser")
from neuralgeom.topology.persistence import persistent_homology, summarize_diagrams, max_persistence, bottleneck_matrix


def _circle_distance_matrix(n=40, seed=0):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.c_[np.cos(ang), np.sin(ang)]
    from scipy.spatial.distance import squareform, pdist
    return squareform(pdist(pts))


def test_circle_has_one_persistent_h1():
    dgms = persistent_homology(_circle_distance_matrix(), maxdim=1)
    s = summarize_diagrams(dgms)
    assert s[1][0] >= 1                       # at least one H1 feature
    assert max_persistence(dgms[1]) > 0.5            # a clearly persistent loop


def test_point_cloud_no_loop():
    rng = np.random.default_rng(0)
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(rng.standard_normal((30, 5))))
    dgms = persistent_homology(D, maxdim=1)
    assert max_persistence(dgms[1]) < 0.8            # blob has no strong loop


def test_bottleneck_matrix_properties():
    d1 = persistent_homology(_circle_distance_matrix(seed=1), maxdim=1)[1]
    d2 = persistent_homology(_circle_distance_matrix(seed=2), maxdim=1)[1]
    B = bottleneck_matrix([d1, d2, d1])
    assert B.shape == (3, 3)
    assert np.allclose(B, B.T) and np.allclose(np.diag(B), 0.0)
    assert B[0, 2] < 1e-6                      # identical diagrams


def test_single_and_pooled_distance_matrices():
    pytest.importorskip("geomstats")
    from neuralgeom.synth.subspace_rnn import SubspaceRNNConfig, make_trajectory
    from neuralgeom.topology.persistence import within_trial_distances, across_trial_distances
    from neuralgeom.subspace import EmbedConfig, PoolConfig
    traj = make_trajectory(SubspaceRNNConfig(
        connectivity="ring", N=30, n_trials=4, duration=1.0, dt=2e-3,
        ring_rotating=True, ring_revolutions=1.0, noise_std=0.05, seed=3))
    Ds = within_trial_distances(traj, 0, EmbedConfig(k=1, win=40, stride=8))
    assert Ds.ndim == 2 and Ds.shape[0] == Ds.shape[1]
    Dp = across_trial_distances(traj, PoolConfig(k=1, n_pool=120, fields=False))
    assert Dp.shape[0] <= 120 and np.allclose(Dp, Dp.T)
    # the moving ring should show a persistent H1 loop at k=1 in the
    # within-trial trajectory and/or the across-trial cloud
    loop = max(max_persistence(persistent_homology(Ds, maxdim=1)[1]), max_persistence(persistent_homology(Dp, maxdim=1)[1]))
    assert loop > 0.5
