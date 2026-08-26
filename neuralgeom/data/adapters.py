"""
neuralgeom.data.adapters — build the shared ``Session`` from any data source,
plus the shared PCA state space (and its split-half instrument).
=========================================================================

Every analysis in this repository consumes the same ``Session`` object
(defined in ``loader.py``). Real recordings get one from ``load_session``;
this module provides the other entry points, so downstream code never cares
where the data came from:

    from_arrays     plain numpy arrays -> Session
    from_synthetic  the plain dict emitted by ``neuralgeom.synth.generate``
                    -> Session (reads keys only; imports NOTHING from synth,
                    so the dynamics estimators never depend on the generator)
    from_trialdata  a ``TrialData`` container (``trial_data.py``) -> Session

Also here, because they are the shared state space every dynamics analysis
starts from:

    state_pca             z-score + PCA on a SHARED basis -> per-trial state
                          trajectories Z (n_trials, T, n_pc)
    state_pca_split_half  the same state estimated TWICE from disjoint halves
                          of the units — the instrument that removes the
                          finite-difference velocity bias in
                          ``neuralgeom.dynamics.lds.fit_lds(..., instrument=...)``

(These differ from ``reduce.py``: reducers implement the leakage-safe
fit/transform interface used inside cross-validation folds; ``state_pca`` is
the plain shared basis used when a single common state space is wanted.)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .loader import Session

__all__ = ["from_arrays", "from_synthetic", "from_trialdata",
           "state_pca", "state_pca_split_half"]


def _bin_s(t) -> float:
    return float(np.median(np.diff(np.asarray(t, float))))


def _cond_masks(lick: np.ndarray, trials: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The standard named boolean trial masks, tolerant of missing columns."""
    licked = np.isfinite(lick) & (lick > 0)
    rew = (np.asarray(trials["is_rewarded"]).astype(bool)
           if "is_rewarded" in trials else np.zeros(len(lick), bool))
    return {"rewarded": rew, "early_lick": licked & ~rew, "licked": licked}


def from_arrays(X, t, lick=None, trials=None, cond=None, name="session",
                group="synthetic", meta=None) -> Session:
    """Build a Session from plain arrays. X is (n_trials, T, N); t is (T,), cue at 0.

    ``trials`` is a dict of per-trial arrays (any length-n_trials vectors you
    want carried along, e.g. condition labels or ground-truth parameters);
    ``cond`` optionally overrides the named boolean masks.
    """
    X = np.asarray(X, float)
    t = np.asarray(t, float)
    n = X.shape[0]
    lick = np.full(n, np.nan) if lick is None else np.asarray(lick, float)
    trials = {k: np.asarray(v) for k, v in (trials or {}).items()}
    trials.setdefault("first_lick_s", lick)
    return Session(
        session_id=name, animal="n/a", date="n/a", group=group, coarse=group,
        training_day=-1, X=X, t=t, bin_s=_bin_s(t), lick=lick, trials=trials,
        units={}, cond=(cond if cond is not None else _cond_masks(lick, trials)),
        meta=dict(meta or {}), align="cue", path="")


def from_synthetic(d, name="synthetic") -> Session:
    """Wrap a synthetic dataset dict from ``neuralgeom.synth`` (read by KEY ONLY).

    Expected keys (all plain arrays — no import of the generator):
        X      (n_trials, T, N)  already-embedded 'neural' activity,
        time   (T,)              cue-relative seconds (cue at 0),
        lick   (n_trials,)       first post-cue lick time (NaN = no lick),
    and optionally ``cond`` (per-trial condition index), per-trial ground
    truth (``cue_amp``, ``cue_ang``, ``ay2``, ``rho``), and ``latent`` — the
    NOISELESS ground-truth state, kept in ``Session.meta['latent']`` for
    validation against the known dynamics.
    """
    X = np.asarray(d["X"], float)
    t = np.asarray(d["time"], float)
    lick = np.asarray(d["lick"], float)
    n = X.shape[0]
    trials: Dict[str, np.ndarray] = {"first_lick_s": lick,
                                     "is_cue_trial": np.ones(n, int),
                                     "is_no_cue": np.zeros(n, int),
                                     "is_rewarded": np.zeros(n, int)}
    if "cond" in d:
        trials["cond_level"] = np.asarray(d["cond"])
    for k in ("cue_amp", "cue_ang", "ay2", "rho"):
        if k in d and np.ndim(d[k]) == 1 and len(np.asarray(d[k])) == n:
            trials[k] = np.asarray(d[k], float)
    if "conditions" in d:
        trials.update({k: np.asarray(v) for k, v in d["conditions"].items()})
    meta = {k: d[k] for k in ("mechanism", "levels", "bin_s") if k in d}
    if "latent" in d:
        meta["latent"] = np.asarray(d["latent"], float)
    return from_arrays(X, t, lick=lick, trials=trials, name=name,
                       group="synthetic", meta=meta)


def from_trialdata(td, name="session") -> Session:
    """Wrap a ``TrialData`` container (``neuralgeom.data.trial_data``)."""
    lick = np.asarray(td.behavior.get("first_lick_s",
                                      np.full(td.X.shape[0], np.nan)), float)
    trials = {k: np.asarray(v) for k, v in td.conditions.items()
              if np.ndim(v) == 1 and len(np.asarray(v)) == td.X.shape[0]}
    trials.update({k: np.asarray(v, float) for k, v in td.behavior.items()
                   if np.ndim(v) == 1 and len(np.asarray(v)) == td.X.shape[0]})
    return from_arrays(td.X, td.time, lick=lick, trials=trials, name=name,
                       group=str(td.meta.get("source", "trialdata")),
                       meta=dict(td.meta))


# ------------------------------------------------------------------ state spaces
def _X_t(sess):
    """Accept a Session or any (X, t)-bearing object / dict."""
    if isinstance(sess, Session) or (hasattr(sess, "X") and hasattr(sess, "t")):
        return np.asarray(sess.X, float), np.asarray(sess.t, float)
    if isinstance(sess, dict):
        return (np.asarray(sess["X"], float),
                np.asarray(sess.get("t", sess.get("time")), float))
    raise TypeError("expected a Session, an object with .X/.t, or a dict")


def state_pca(sess, n_pc=6, fit_mask=None, fit_window=None, return_model=False):
    """z-score + PCA on a SHARED basis, returning per-trial state trajectories.

    The basis is fit on the (trial, time) samples selected by ``fit_mask``
    (n_trials, T bool) or ``fit_window`` (t0, t1) and then applied to every
    trial, so conditions are compared in one common state space. Returns
    Z (n_trials, T, n_pc); with ``return_model`` also (scaler, pca,
    var_explained).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X, t = _X_t(sess)
    n, T, N = X.shape
    if fit_mask is None:
        w = (np.ones(T, bool) if fit_window is None
             else (t >= fit_window[0]) & (t < fit_window[1]))
        fit_mask = np.tile(w, (n, 1))
    fit_mask = np.asarray(fit_mask, bool)
    Xf = X[fit_mask]
    if len(Xf) < max(10, n_pc + 1):
        raise ValueError(f"only {len(Xf)} samples to fit the PCA basis")
    sc = StandardScaler().fit(Xf)
    pca = PCA(n_pc, random_state=0).fit(sc.transform(Xf))
    Z = pca.transform(sc.transform(X.reshape(-1, N))).reshape(n, T, n_pc)
    if return_model:
        return Z, sc, pca, pca.explained_variance_ratio_
    return Z


def state_pca_split_half(sess, n_pc=6, fit_mask=None, fit_window=None, seed=0):
    """State estimated TWICE from disjoint halves of the units — an instrument
    for the finite-difference velocity bias.

    Observation (spike-count) noise is essentially independent across neurons,
    so a state reconstructed from a disjoint set of units is correlated with
    the true state but carries independent noise — the ideal instrument, with
    no time lag and hence no attenuation. The PCA basis is fit once on all
    units; each half reconstructs the same coordinates from its own units
    (scaled by N/|half| so the two are on a comparable scale).

    Returns (Z_all, Z_a, Z_b), each (n_trials, T, n_pc).

    USAGE RULE: each half instruments the OTHER half — build (Z, V) from one
    half's state and pass the other half's state (at the same samples) as
    ``instrument=`` to ``neuralgeom.dynamics.fit_lds``. Do NOT instrument Z_all
    with one of its own halves: they share units, so their observation noise
    is correlated and the bias survives. ``tests/test_lds.py`` demonstrates
    the correct pattern end to end.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X, t = _X_t(sess)
    n, T, N = X.shape
    if N < 4:
        raise ValueError(f"split-half instrumenting needs >=4 units, got {N}")
    if fit_mask is None:
        w = (np.ones(T, bool) if fit_window is None
             else (t >= fit_window[0]) & (t < fit_window[1]))
        fit_mask = np.tile(w, (n, 1))
    fit_mask = np.asarray(fit_mask, bool)
    sc = StandardScaler().fit(X[fit_mask])
    pca = PCA(n_pc, random_state=0).fit(sc.transform(X[fit_mask]))

    Xs = sc.transform(X.reshape(-1, N))
    Xc = Xs - pca.mean_
    Z_all = (Xc @ pca.components_.T).reshape(n, T, n_pc)

    perm = np.random.default_rng(seed).permutation(N)
    ha, hb = perm[: N // 2], perm[N // 2:]
    out = []
    for h in (ha, hb):
        w = np.zeros(N)
        w[h] = 1.0
        out.append(((Xc * w) @ pca.components_.T
                    * (N / max(len(h), 1))).reshape(n, T, n_pc))
    return Z_all, out[0], out[1]
