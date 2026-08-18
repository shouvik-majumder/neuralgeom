"""Data-layer test: the Session adapters, shared PCA state space (incl. the
split-half instrument), and the condition builders.

Run:  python tests/test_data_adapters.py     (or: pytest)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuralgeom.data import (from_arrays, from_synthetic, state_pca,           # noqa: E402
                         state_pca_split_half, lick_time_bins, epoch_masks,
                         trial_masks, lick_align)


def _fake_session(n=60, T=120, N=24, seed=0):
    rng = np.random.default_rng(seed)
    t = -0.6 + np.arange(T) * 0.02
    lick = np.clip(rng.gamma(6, 0.1, n), 0.05, None)
    X = rng.normal(size=(n, T, N))
    for i in range(n):                       # inject a lick-locked ramp so PCA has structure
        X[i] += np.outer(np.clip(t / max(lick[i], 1e-3), 0, 1.2), np.ones(N))
    trials = {"is_cue_trial": np.ones(n, int), "is_no_cue": np.zeros(n, int),
              "is_rewarded": (lick > np.median(lick)).astype(int)}
    return from_arrays(X, t, lick=lick, trials=trials, name="fake")


def test_from_arrays_contract():
    s = _fake_session()
    assert s.X.shape[0] == len(s.lick)
    assert s.n_trials == 60 and s.n_time == 120 and s.n_units == 24
    assert "first_lick_s" in s.trials
    assert s.cond["licked"].all()
    # the Session methods from the loader work on adapter-built sessions too
    assert s.prelick_mask().shape == (s.n_trials, s.n_time)
    b = s.lick_bins(4)
    assert b["n_bins"] >= 2


def test_from_synthetic_contract():
    n, T, N = 20, 40, 12
    d = dict(X=np.random.default_rng(1).normal(size=(n, T, N)),
             time=np.linspace(-0.5, 1.5, T), lick=np.full(n, 0.4),
             cond=np.arange(n) % 3, latent=np.zeros((n, T, 2)))
    s = from_synthetic(d)
    assert s.X.shape == (n, T, N) and "latent" in s.meta
    assert (s.trials["is_cue_trial"] == 1).all()
    assert (s.trials["cond_level"] == np.arange(n) % 3).all()


def test_lick_time_bins_and_epochs():
    s = _fake_session()
    b = lick_time_bins(s.lick, n_bins=4)
    assert b["counts"].sum() == len(s.lick) and np.all(np.diff(b["median"]) > 0)
    em = epoch_masks(s.t, s.lick)
    assert em["pre_cue"].sum() > 0 and em["ramp"].sum() > 0


def test_state_pca_and_split_half():
    s = _fake_session()
    Z = state_pca(s, n_pc=4, fit_window=(0.0, 1.0))
    assert Z.shape == (s.X.shape[0], s.X.shape[1], 4)
    Zall, Za, Zb = state_pca_split_half(s, n_pc=4, fit_window=(0.0, 1.0))
    # with an even unit count each half is scaled by 2, so they sum to twice the full state
    assert np.allclose(Za + Zb, 2 * Zall, atol=1e-6)
    # the two halves track the full state but carry (near-)independent observation noise
    for k in range(4):
        assert np.corrcoef(Za[:, :, k].ravel(), Zall[:, :, k].ravel())[0, 1] > 0.5


def test_trial_masks_and_lick_align():
    s = _fake_session()
    m = trial_masks(s)
    assert m["cue"].sum() == len(s.lick) and m["rewarded"].sum() > 0
    la = lick_align(s, window=(-0.2, 0.6))
    assert la.X.shape[0] > 0 and abs(la.t[0] + 0.2) < 0.05
    assert la.align == "lick" and "source_lick" in la.meta


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"{len(fns)}/{len(fns)} passed")
