"""Tests for the unified Trajectory contract (``neuralgeom.data.trajectory``)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralgeom.data import Trajectory, load_trajectory, load_trial, load_all_trials


def _toy(n_trials=3, T=20, N=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_trials, T, N))


def test_from_arrays_infers_dt_and_promotes_single_trial():
    X = _toy(1, 10, 5)[0]                      # (T, N) single trial
    tr = Trajectory.from_arrays(X, dt=0.01)
    assert tr.X.shape == (1, 10, 5)
    assert abs(tr.dt - 0.01) < 1e-12
    assert tr.meta["generator"] == "arrays"


def test_from_synth_dict_maps_fields():
    X = _toy(2, 15, 6)
    d = {"X": X, "time": np.arange(15) * 0.02, "W": np.eye(6),
         "onsets": np.array([0.1, 0.2]), "omega": 3.0,
         "aux": {"theta": np.arange(6)}, "config": {"tau": 0.1, "N": 6}}
    tr = Trajectory.from_synth_dict(d, generator="subspace_rnn")
    assert tr.n_trials == 2 and tr.T == 15 and tr.N == 6
    assert tr.W.shape == (6, 6)
    assert "onsets" in tr.aux and "theta" in tr.aux
    assert tr.meta["generator"] == "subspace_rnn"
    assert abs(tr.meta["omega"] - 3.0) < 1e-9
    assert abs(tr.tau - 0.1) < 1e-9


def test_hdf5_roundtrip_canonical(tmp_path):
    X = _toy(3, 12, 7)
    tr = Trajectory.from_arrays(
        X, time=np.arange(12) * 0.05, tau=0.1,
        inputs=np.zeros((3, 12, 2)), condition=np.array([0, 1, 0]),
        W=np.eye(7), aux={"foo": np.arange(4)}, generator="test")
    p = tmp_path / "t.h5"
    tr.save(str(p))
    tr2 = load_trajectory(str(p))
    assert np.allclose(tr2.X, tr.X)
    assert np.allclose(tr2.time, tr.time)
    assert np.allclose(tr2.condition, tr.condition)
    assert np.allclose(tr2.W, tr.W)
    assert np.allclose(tr2.aux["foo"], np.arange(4))
    assert tr2.meta["generator"] == "test"
    assert abs(tr2.tau - 0.1) < 1e-9


def test_load_trial_and_all(tmp_path):
    X = _toy(4, 10, 5)
    tr = Trajectory.from_arrays(X, time=np.arange(10) * 0.1)
    p = tmp_path / "t.h5"
    tr.save(str(p))
    Xi, ti, meta = load_trial(str(p), 2)
    assert Xi.shape == (10, 5) and ti.shape == (10,)
    Xa, ta = load_all_trials(str(p))
    assert Xa.shape == (4, 10, 5)
    assert np.allclose(Xa[2], Xi)


def test_legacy_rnn_schema(tmp_path):
    """A legacy ProjectiveSpaceModels rnn_{tag}.h5 (no 'meta' attr) loads with
    its onsets/phi0/omega/config folded into aux/meta."""
    import h5py

    X = _toy(2, 10, 6)
    p = tmp_path / "rnn_ring_moving.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("time", data=np.arange(10) * 0.001)
        f.create_dataset("onsets", data=np.array([np.nan, np.nan]))
        f.create_dataset("phi0", data=np.array([0.1, 0.2]))
        f.attrs["omega"] = 6.28
        f.attrs["connectivity"] = "ring"
        f.attrs["tau"] = 0.1
        f.attrs["b"] = "None"                    # legacy None sentinel
        f.create_dataset("aux/theta", data=np.arange(6))
    tr = load_trajectory(str(p))
    assert tr.meta["generator"] == "legacy_rnn"
    assert "phi0" in tr.aux and "theta" in tr.aux
    assert abs(tr.meta["omega"] - 6.28) < 1e-6
    assert tr.meta["config"]["connectivity"] == "ring"
    assert tr.meta["config"]["b"] is None        # sentinel decoded


def test_from_session_adapter():
    """A minimal duck-typed Session round-trips into a Trajectory."""
    class _Sess:
        pass
    s = _Sess()
    s.X = _toy(3, 8, 4)
    s.t = np.arange(8) * 0.02
    s.bin_s = 0.02
    s.lick = np.array([0.3, 0.4, 0.5])
    s.trials = {}
    s.meta = {"note": 1}
    s.session_id = "SMxx"
    tr = Trajectory.from_session(s, condition="lick")
    assert tr.n_trials == 3 and tr.N == 4
    assert np.allclose(tr.condition, s.lick)
    assert tr.meta["session_id"] == "SMxx"
    assert abs(tr.dt - 0.02) < 1e-9
