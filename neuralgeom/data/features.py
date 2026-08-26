"""Build single-trial (features -> behavior) design matrices from a Session (or TrialData).

For the lick-timing task: the input is the population firing rate in an early post-cue
window (strictly *before* the lick, so the lick is predictable from it), and the target is
the single-trial lick time. Trials that lick inside (or before) the window are excluded,
because their neural activity in the window is contaminated by / post-dates the action we
are trying to predict.
"""

from __future__ import annotations
import numpy as np

__all__ = ["cue_window_features", "window_features", "lick_history_regressors",
           "temporal_features"]


def _xtb(td, target):
    """Accept a Session or a TrialData: return (X, time, target array)."""
    t = np.asarray(getattr(td, "t", getattr(td, "time", None)), float)
    X = np.asarray(td.X, float)
    if hasattr(td, "behavior"):                       # TrialData
        y = np.asarray(td.behavior[target], float)
    else:                                             # Session
        src = td.trials if target in td.trials else {"first_lick_s": td.lick}
        y = np.asarray(src.get(target, td.lick), float)
    return X, t, y


def cue_window_features(td, window=(0.0, 0.20), target="first_lick_s",
                        min_lick_s=None, n_subbins=1):
    """Mean firing rate per unit in an early cue-relative window -> (X, y, keep).

    Parameters
    ----------
    td        : TrialData with X (n_trials, T, N) and behavior[target] (n_trials,).
    window    : (t0, t1) seconds, cue-relative, for the input window.
    target    : behavior key to predict (default 'first_lick_s').
    min_lick_s: exclude trials whose target < this. Defaults to the window end t1, i.e.
                only keep trials that lick strictly after the input window (the lick must be
                predictable, not already happening).
    n_subbins : split the window into this many equal sub-bins (features = N * n_subbins);
                1 = a single mean-rate feature per unit (the '200 ms bin').

    Returns
    -------
    X    : (n_kept, N * n_subbins) firing-rate features.
    y    : (n_kept,) target (lick time, s).
    keep : (n_trials,) boolean mask of retained trials.
    """
    Xall, t, y = _xtb(td, target)
    t0, t1 = window

    if n_subbins == 1:
        m = (t >= t0) & (t < t1)
        X = np.nanmean(Xall[:, m, :], axis=1)                      # (n_trials, N)
    else:
        edges = np.linspace(t0, t1, n_subbins + 1)
        feats = []
        for a in range(n_subbins):
            m = (t >= edges[a]) & (t < edges[a + 1])
            feats.append(np.nanmean(Xall[:, m, :], axis=1))
        X = np.concatenate(feats, axis=1)                          # (n_trials, N*n_subbins)

    thr = t1 if min_lick_s is None else float(min_lick_s)
    keep = np.isfinite(y) & (y >= thr) & np.all(np.isfinite(X), axis=1)
    return X[keep], y[keep], keep


def window_features(td, window, align="cue", reference="first_lick_s",
                    exclude_lick_in_window=True, require_after_cue=True):
    """Mean firing rate per unit in a window, cue-aligned or lick-aligned.

    Parameters
    ----------
    window   : (a, b) seconds. For align='cue', absolute cue-relative bounds. For
               align='lick', bounds relative to that trial's `reference` lick time
               (use negatives, e.g. (-0.25, -0.05) = the 200 ms ending 50 ms before lick).
    align    : 'cue' -> one fixed window for all trials (e.g. ITI (-0.4, 0) or early
               post-cue (0, 0.2)); 'lick' -> a per-trial window locked to the lick.
    reference: behavior key giving the per-trial event time and the prediction target.
    exclude_lick_in_window : (cue-aligned only) drop trials whose lick is before the
               window end, so the input strictly precedes the lick.
    require_after_cue      : (lick-aligned only) drop trials whose window would start
               before the cue (t < 0).

    Returns (X, y, keep) as in `cue_window_features`.

    Note on interpretation: a *lick-aligned pre-lick* window is causal (before the lick)
    but its position is set by the lick time, so it probes the pre-movement neural state
    rather than being a clean out-of-time predictor; compare it to fixed cue-aligned
    windows accordingly.
    """
    Xall, t, y = _xtb(td, reference)
    a, b = window
    ntr, _, N = Xall.shape
    keep = np.isfinite(y)

    if align == "cue":
        m = (t >= a) & (t < b)
        if m.sum() == 0:
            raise ValueError(f"empty cue-aligned window {window} for time axis.")
        X = np.nanmean(Xall[:, m, :], axis=1)
        if exclude_lick_in_window:
            keep &= (y >= b)
    elif align == "lick":
        X = np.full((ntr, N), np.nan)
        for i in range(ntr):
            if not np.isfinite(y[i]):
                continue
            lo, hi = y[i] + a, y[i] + b
            if require_after_cue and lo < t[0]:
                continue
            if require_after_cue and lo < 0.0:
                continue
            m = (t >= lo) & (t < hi)
            if m.sum() == 0:
                continue
            X[i] = np.nanmean(Xall[i, m, :], axis=0)
    else:
        raise ValueError("align must be 'cue' or 'lick'.")

    keep &= np.all(np.isfinite(X), axis=1)
    return X[keep], y[keep], keep


def lick_history_regressors(session_path, n_lags=2, target="first_lick_s",
                            min_lick_s=0.1):
    """Autoregressive trial-history design matrix, within the CUE-TRIAL sequence.

    No-cue (probe) trials are removed first (they have no meaningful lick/reward and would
    corrupt the notion of "previous trial"); lags are then computed within the ordered
    sequence of cue trials. Columns are [tau_{t-1..t-L}, R_{t-1..t-L}] (previous lick times
    then previous rewards); the bias term is left to the estimator (fit_intercept=True).

    Returns (X, y, cue_pos) where X is (n, 2*n_lags), y the current lick time, and cue_pos
    the original trial indices (into the full session) of the retained current trials, so X
    can be aligned with neural features from the same cue-trial selection.
    """
    import h5py
    with h5py.File(session_path, "r") as f:
        first = f["trials/first_lick_s"][:].astype(float)
        rew = f["trials/is_rewarded"][:].astype(float)
        cue = f["trials/is_cue_trial"][:].astype(bool)
    tau, R = first[cue], rew[cue]

    def lag(a, k):
        o = np.full(a.shape, np.nan); o[k:] = a[:-k]; return o

    cols = [lag(tau, k) for k in range(1, n_lags + 1)] + \
           [lag(R, k) for k in range(1, n_lags + 1)]
    X = np.column_stack(cols)
    y = tau
    keep = np.isfinite(y) & (y > min_lick_s) & np.all(np.isfinite(X), axis=1)
    return X[keep], y[keep], np.where(cue)[0][keep]


def temporal_features(td, window=(0.0, 0.4), reference="first_lick_s", flatten=True):
    """Strictly-causal temporal population features over a cue-aligned window [t0, t1).

    Uses every time bin in [t0, t1) and keeps only trials whose `reference` event (lick) is
    at or after t1, so the ENTIRE window precedes the event (no bin can reach the lick). Set
    the bin/smoothing via `load_session(target_bin_s=..., smooth_ms=...)`.

    Returns (X, y, keep, tbins) where X is (n, T*N) if flatten else (n, T, N), y the target,
    keep the trial mask, tbins the retained bin-centre times. Assertion guards causality.
    """
    Xall, t, y = _xtb(td, reference)
    t0, t1 = window
    m = (t >= t0) & (t < t1)
    if m.sum() == 0:
        raise ValueError(f"empty window {window}.")
    keep = np.isfinite(y) & (y >= t1) & np.all(np.isfinite(Xall[:, m, :]), axis=(1, 2))
    tb = t[m]
    if keep.any():
        assert tb.max() < y[keep].min() + 1e-12, "window reaches the lick — not causal"
    Xseq = Xall[np.ix_(keep)][:, m, :]                 # (n, T, N)
    X = Xseq.reshape(Xseq.shape[0], -1) if flatten else Xseq
    return X, y[keep], keep, tb
