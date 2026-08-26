"""Condition and epoch builders for the timing task, operating on a ``Session``.

These slice a session into the groups the analyses use:
  - ``lick_time_bins``  within-session quantile bins of first_lick_s, labelled by MEDIAN lick
                        time (the established convention -- never "condition N").
                        (``Session.lick_bins`` is the method form of the same thing.)
  - ``epoch_masks``     pre-cue / cue / ramp / peri-lick / post-lick time masks per trial.
  - ``trial_masks``     cue vs no-cue (the empirical zero-input control), rewarded vs
                        unrewarded, has-lick / no-lick, valid.
  - ``lick_align``      re-slice trials around each trial's own first lick (needed for the
                        reward contrast, which is otherwise collinear with lick time).
                        (``Session.lick_aligned`` is the method form; this adds a trial mask.)
  - ``describe_conditions``  trial counts + lick stats per condition (the descriptive check
                        shown BEFORE any model is fit).
"""

from __future__ import annotations

import numpy as np

from .loader import Session

__all__ = ["DEFAULT_EPOCHS", "lick_time_bins", "epoch_masks", "trial_masks", "lick_align",
           "describe_conditions"]


# Epoch definitions.  't' is cue-relative seconds; 'lick' is the trial's first-lick time.
# Each endpoint is a number, "cue", or "lick"(+/-offset).
DEFAULT_EPOCHS = {
    "pre_cue":   (-0.6, 0.0),          # spontaneous / autonomous: no cue input by construction
    "cue":       (0.0, 0.2),           # cue-driven window (the 0.2 s is a DISPLAY split only --
                                       #  no estimator may assume a cue duration)
    "ramp":      (0.2, "lick"),        # autonomous evolution toward the action
    "peri_lick": ("lick-0.15", "lick+0.15"),
    "post_lick": ("lick+0.05", "lick+1.5"),
}


def lick_time_bins(lick, n_bins=5, mask=None, min_per_bin=20):
    """Within-session quantile bins of first-lick time.

    Returns dict: label (n_trials,) int, -1 = excluded; median (n_bins,) median lick per bin
    (use THIS to label figures); edges; counts; n_bins.
    """
    lick = np.asarray(lick, float)
    ok = np.isfinite(lick) & (lick > 0)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    v = lick[ok]
    if len(v) == 0:
        return dict(label=np.full(len(lick), -1, int), median=np.zeros(0), edges=np.zeros(0),
                    counts=np.zeros(0, int), n_bins=0)
    if len(v) < n_bins * min_per_bin:
        n_bins = max(2, len(v) // max(min_per_bin, 1))
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))   # tied lick times can
    n_bins = max(len(edges) - 1, 1)                                    # collapse quantiles
    edges = (np.concatenate([[edges[0] - 1e-9], edges[1:-1], [edges[-1] + 1e-9]])
             if len(edges) > 1 else np.array([edges[0] - 1e-9, edges[0] + 1e-9]))
    label = np.full(len(lick), -1, int)
    label[ok] = np.clip(np.digitize(v, edges[1:-1]), 0, n_bins - 1)
    med = np.array([np.median(lick[label == b]) if np.any(label == b) else np.nan
                    for b in range(n_bins)])
    cnt = np.array([int(np.sum(label == b)) for b in range(n_bins)])
    return dict(label=label, median=med, edges=edges, counts=cnt, n_bins=n_bins)


def _resolve(spec, lick):
    """Resolve an epoch endpoint spec ('lick+0.2', 'cue', or a number) to seconds."""
    if isinstance(spec, (int, float, np.floating, np.integer)):
        return float(spec)
    s = str(spec).strip()
    if s == "cue":
        return 0.0
    if s.startswith("lick"):
        off = s[4:].strip()
        return float(lick) + (float(off) if off else 0.0)
    raise ValueError(f"unrecognized epoch endpoint {spec!r}")


def epoch_masks(time, lick, epochs=None):
    """Per-trial time masks for each epoch -> dict name -> (n_trials, T) boolean.

    Trials with no lick get all-False for any epoch whose endpoints reference the lick.
    """
    time = np.asarray(time, float); lick = np.asarray(lick, float)
    epochs = DEFAULT_EPOCHS if epochs is None else epochs
    out = {}
    for name, (a, b) in epochs.items():
        M = np.zeros((len(lick), len(time)), bool)
        for i, L in enumerate(lick):
            if (not np.isfinite(L)) and ("lick" in str(a) or "lick" in str(b)):
                continue
            t0, t1 = _resolve(a, L), _resolve(b, L)
            if t1 > t0:
                M[i] = (time >= t0) & (time < t1)
        out[name] = M
    return out


def trial_masks(sess: Session):
    """Standard per-trial boolean masks, tolerant of missing columns.

    ``cue`` / ``no_cue`` -- the no-cue probe trials are the empirical ZERO-INPUT control:
    a shared-field + input estimator must return ~0 input on them. (The loader's default
    ``valid_only=True`` already excludes no-cue trials; load with ``valid_only=False`` to
    keep them.)
    """
    c = sess.trials
    lick = np.asarray(sess.lick, float)
    n = len(lick)

    def col(name, default):
        return np.asarray(c[name]).astype(bool) if name in c else np.asarray(default, bool)

    cue = col("is_cue_trial", np.ones(n))
    no_cue = col("is_no_cue", ~cue)
    rew = col("is_rewarded", sess.cond.get("rewarded", np.zeros(n)))
    has_lick = np.isfinite(lick) & (lick > 0)
    valid = col("is_valid_for_analysis", cue & has_lick & (lick > 0.1))
    return dict(cue=cue, no_cue=no_cue, rewarded=rew & has_lick,
                unrewarded=cue & has_lick & ~rew, has_lick=has_lick,
                no_lick=cue & ~has_lick, valid=valid, all=np.ones(n, bool))


def lick_align(sess: Session, window=(-0.3, 1.5), mask=None) -> Session:
    """Re-slice every trial around ITS OWN first lick (time becomes lick-relative, lick=0).

    Needed for the reward contrast: cue-aligned, 'rewarded' is nearly a re-labelling of the
    lick-time axis (reward is defined by licking after the required delay), so that comparison
    is only meaningful on lick-aligned, post-lick dynamics.

    Like ``Session.lick_aligned`` but with an optional trial ``mask``; the original trial
    indices and lick times are kept in ``meta['source_trials']`` / ``meta['source_lick']``.
    """
    X = np.asarray(sess.X, float); t = np.asarray(sess.t, float)
    lick = np.asarray(sess.lick, float)
    dt = float(sess.bin_s)
    n_out = int(round((window[1] - window[0]) / dt))
    t_out = window[0] + (np.arange(n_out) + 0.5) * dt
    keep, Xo = [], []
    m_in = np.ones(len(lick), bool) if mask is None else np.asarray(mask, bool)
    for i, L in enumerate(lick):
        if not (m_in[i] and np.isfinite(L)):
            continue
        if (L + t_out[0]) < t[0] or (L + t_out[-1]) > t[-1]:
            continue                       # window falls outside the recording -> drop the trial
        idx = np.searchsorted(t, L + t_out)
        if idx.max() >= len(t):
            continue
        Xo.append(X[i, idx, :]); keep.append(i)
    keep = np.array(keep, int)
    if len(keep) == 0:
        raise ValueError("no trials survive lick alignment in this window")
    meta = dict(sess.meta)
    meta["source_trials"] = keep
    meta["source_lick"] = lick[keep]
    return Session(
        session_id=sess.session_id + "_lickaligned", animal=sess.animal,
        date=sess.date, group=sess.group, coarse=sess.coarse,
        training_day=sess.training_day, X=np.stack(Xo), t=t_out, bin_s=dt,
        lick=np.zeros(len(keep)),
        trials={k: np.asarray(v)[keep] for k, v in sess.trials.items()},
        units=sess.units,
        cond={k: np.asarray(v)[keep] for k, v in sess.cond.items()},
        meta=meta, align="lick", path=sess.path)


def describe_conditions(sess: Session, n_bins=5):
    """Trial counts and lick statistics per condition -- the descriptive check shown BEFORE
    any model is fit."""
    m = trial_masks(sess)
    lick = np.asarray(sess.lick, float)
    b = lick_time_bins(lick, n_bins=n_bins, mask=m["cue"])
    rows = [dict(name=k, n=int(v.sum()),
                 lick_median=float(np.nanmedian(lick[v])) if v.sum() else np.nan,
                 lick_10_90=[float(x) for x in np.nanpercentile(lick[v], [10, 90])]
                 if np.isfinite(lick[v]).sum() > 2 else [np.nan, np.nan])
            for k, v in m.items()]
    return dict(masks=rows, bins=b, n_trials=len(lick),
                t_range=(float(sess.t[0]), float(sess.t[-1])),
                bin_s=float(sess.bin_s))
