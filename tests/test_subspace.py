"""Tests for neuralgeom.subspace (embedding, kinematics, pooling)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("geomstats")   # kinematics/pooling use GrassmannManifold

from neuralgeom.synth.subspace_rnn import SubspaceRNNConfig, make_trajectory
from neuralgeom.subspace import (EmbedConfig, embed_trajectory,
                                 embed_from_trajectory, KinConfig,
                                 compute_kinematics, tangent_pca,
                                 chordal_vs_geodesic, PoolConfig, pool_frames,
                                 distance_from_start)


@pytest.fixture(scope="module")
def ring_traj():
    cfg = SubspaceRNNConfig(connectivity="ring", N=30, n_trials=4, duration=1.0,
                            dt=2e-3, ring_rotating=True, ring_revolutions=1.0,
                            noise_std=0.05, seed=3)
    return make_trajectory(cfg)


def test_embed_shapes_and_orthonormal(ring_traj):
    emb = embed_from_trajectory(ring_traj, 0, EmbedConfig(k=2, win=40, stride=8))
    F = emb["frames"]
    assert F.shape[1] == ring_traj.N and F.shape[2] == 2
    # columns orthonormal
    G = np.einsum("mni,mnj->mij", F, F)
    assert np.allclose(G, np.eye(2)[None], atol=1e-6)
    assert emb["win_times"].shape[0] == F.shape[0]
    assert np.all(emb["sv_gap"] > 0)


def test_kinematics_speed_identity(ring_traj):
    emb = embed_from_trajectory(ring_traj, 0, EmbedConfig(k=1, win=40, stride=8))
    kin = compute_kinematics(emb["frames"], emb["win_times"], KinConfig())
    err = np.nanmax(np.abs(kin["speed"] * kin["dt"] - kin["step_distance"]))
    assert err < 1e-6                      # ||v||·dt == geodesic step distance
    assert 0.0 <= kin["endpoint_to_path_length_ratio"] <= 1.0 + 1e-9


def test_tangent_pca_dimensionality(ring_traj):
    emb = embed_from_trajectory(ring_traj, 0, EmbedConfig(k=1, win=40, stride=8))
    tp = tangent_pca(emb["frames"])
    assert tp["coords"].shape[0] == emb["frames"].shape[0]
    assert np.all(np.diff(tp["cum_evr"]) >= -1e-9)   # monotone
    dim90 = int(np.searchsorted(tp["cum_evr"], 0.90) + 1)
    assert dim90 >= 1


def test_chordal_le_geodesic(ring_traj):
    emb = embed_from_trajectory(ring_traj, 0, EmbedConfig(k=1, win=40, stride=8))
    geo, cho = chordal_vs_geodesic(emb["frames"], max_pairs=500, seed=0)
    # chordal (Σsin²θ) ≤ arc-length (Σθ²) elementwise, up to numerical slack
    assert np.all(cho <= geo + 1e-6)


def test_pooling_shapes_and_fields(ring_traj):
    frames, fields, meta = pool_frames(ring_traj, PoolConfig(k=1, n_pool=80))
    assert frames.shape[0] <= 80 and frames.shape[1] == ring_traj.N
    for name in ("mean_squared_activity", "speed", "participation_ratio",
                 "input_magnitude", "variance_explained"):
        assert name in fields and len(fields[name]) == frames.shape[0]
        assert np.all(np.isfinite(fields[name]))       # NaNs were filled
    # at k=1 the dominant direction captures most of the state variance
    assert np.median(fields["variance_explained"]) > 0.5


def test_distance_from_start_is_zero_at_start(ring_traj):
    emb = embed_from_trajectory(ring_traj, 0, EmbedConfig(k=1, win=40, stride=8))
    d = distance_from_start(emb["frames"])
    # d[0] is a self-distance; arccos near 1 is numerically ~1e-8, not 0
    assert abs(d[0]) < 1e-6 and np.all(d >= -1e-9)
