"""
neuralgeom.data.loader — read Neuropixels session HDF5 files into analysis-ready tensors.
==========================================================================================

File format (verified against the files themselves; there is no data
dictionary shipped with them):

    meta/          attrs only: session_id, animal, session_date, training_day,
                   exp_type_coarse / exp_type_fine (experimental group),
                   n_units_*, bin_size_s (0.005), alignment ("cue_onset"),
                   spike_window_pre_s / post_s (-3 / +6)
    spikes/counts       (n_units, n_trials, n_bins) spike counts per 5 ms bin
    spikes/firing_rate  same / bin size (Hz); NOT smoothed
    spikes/t_bins       (n_bins,) bin centres in seconds, cue onset at t = 0
    trials/        per-trial behaviour
    units/         per-unit metadata and quality metrics

The task is a timing task. The behavioural variable is ``first_lick_s`` —
*when* the animal licks after the cue. Reward requires withholding until a
required delay. ``response_type`` is 0 = early lick, 1 = rewarded,
2 = no response, 3 = no-cue catch trial; the loader verifies this coding
against the boolean flags on every load.

Verified facts that shape this loader
-------------------------------------
* ``is_valid_for_analysis`` already implies ``is_cue_trial`` AND a lick
  (checked: valid & ~cue == 0 in every session), so a separate cue filter is
  redundant. No-response and no-cue trials are excluded by validity.
* ``firing_rate == counts / bin_size_s`` exactly (ratio 200 for 5 ms bins).
* **``delay_duration_s`` is NOT the required delay and is not used here.** It
  is uncorrelated with lick time (|r| < 0.06 in every session), its median is
  2-3 s with a range to 28 s, and on ~98% of *rewarded* trials the lick
  precedes it. Per the experimenter it is probably a delay-period onset in
  absolute trial time, not cue-relative, and is not meaningful. It is
  deliberately dropped from the condition set so downstream code cannot use
  it by accident.
* In learning sessions the reward criterion drifts within the session (the
  bpod protocol has auto-learn: the required delay grows with performance).
  Example: SM259 day 1, the running minimum rewarded lick time climbs
  0.11 -> 0.42 s, so "rewarded" vs "early lick" is partly confounded with
  time in session. ``Session.trial_frac`` exposes normalized trial number so
  this confound can be regressed out or tested later.

Formatting conventions (kept compatible with the earlier neuro_core code)
-------------------------------------------------------------------------
* re-binning SUMS spike counts and divides by the effective bin width;
* smoothing is a CAUSAL boxcar by default, so activity is never smeared
  backwards in time (critical for a timing task);
* activity is returned as X (n_trials, T, N) — the same convention as the
  ``rnn`` package, so the same analysis code applies;
* lick-time conditions are quantile bins labelled by their MEDIAN lick time,
  never "condition 1, 2, 3".

Typical use
-----------
>>> from neuro.loader import load_session
>>> s = load_session("SampleData/SM259_20230417_g0.h5")
>>> s.X.shape                 # (trials, time, units)
>>> s.lick                    # (trials,) first lick time, s — the behaviour
>>> s.cond["rewarded"]        # boolean trial mask
>>> b = s.lick_bins(5)        # quantile bins labelled by median lick time
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError as _e:  # pragma: no cover
    raise ImportError("h5py is required: pip install h5py") from _e

__all__ = ["Session", "load_session", "list_sessions", "session_table",
           "RESPONSE_TYPES", "DROPPED_FIELDS"]

RESPONSE_TYPES = {0: "early_lick", 1: "rewarded", 2: "no_response",
                  3: "no_cue"}

#: Fields deliberately not exposed, with the reason.
DROPPED_FIELDS = {
    "delay_duration_s": "not the required delay; uncorrelated with lick time "
                        "and rewarded licks precede it (see module docstring)",
}


@dataclass
class Session:
    """One recording session, preprocessed for population analysis."""
    session_id: str
    animal: str
    date: str
    group: str                       # exp_type_fine
    coarse: str                      # exp_type_coarse
    training_day: int
    X: np.ndarray                    # (trials, time, units) firing rate, Hz
    t: np.ndarray                    # (time,) bin centres, s (0 = cue onset)
    bin_s: float
    lick: np.ndarray                 # (trials,) first_lick_s — the behaviour
    trials: Dict[str, np.ndarray]
    units: Dict[str, np.ndarray]
    cond: Dict[str, np.ndarray]      # named boolean trial masks
    meta: Dict[str, object]
    align: str = "cue"               # "cue" or "lick"
    path: str = ""

    # -- shape helpers ---------------------------------------------------- #
    @property
    def n_trials(self) -> int: return self.X.shape[0]

    @property
    def n_time(self) -> int: return self.X.shape[1]

    @property
    def n_units(self) -> int: return self.X.shape[2]

    @property
    def trial_frac(self) -> np.ndarray:
        """Normalized trial number in [0, 1] — the learning-drift regressor."""
        idx = self.trials.get("trial_index_0based",
                              np.arange(self.n_trials)).astype(float)
        return (idx - idx.min()) / max(1.0, (idx.max() - idx.min()))

    def window(self, t0: float, t1: float) -> slice:
        i0 = int(np.searchsorted(self.t, t0))
        i1 = int(np.searchsorted(self.t, t1))
        return slice(max(0, i0), max(i0 + 1, i1))

    def epoch_mean(self, t0: float, t1: float) -> np.ndarray:
        """(trials, units) mean rate in a window."""
        return self.X[:, self.window(t0, t1), :].mean(1)

    def zscored(self, baseline: Tuple[float, float] = (-0.5, -0.1)
                ) -> np.ndarray:
        """X z-scored per unit using a baseline window across all trials."""
        b = self.X[:, self.window(*baseline), :]
        mu = b.mean(axis=(0, 1), keepdims=True)
        sd = b.std(axis=(0, 1), keepdims=True)
        return (self.X - mu) / np.where(sd < 1e-6, 1.0, sd)

    def lick_bins(self, n_bins: int = 5, mask: Optional[np.ndarray] = None,
                  min_per_bin: int = 20) -> Dict[str, np.ndarray]:
        """Quantile bins of first-lick time, LABELLED BY MEDIAN LICK TIME.

        Returns dict with ``label`` (-1 = excluded), ``median`` (use this to
        label figures), ``edges``, ``counts``, ``n_bins``.
        """
        lick = np.asarray(self.lick, float)
        ok = np.isfinite(lick) & (lick > 0)
        if mask is not None:
            ok &= np.asarray(mask, bool)
        v = lick[ok]
        if v.size == 0:
            return dict(label=np.full(len(lick), -1, int),
                        median=np.zeros(0), edges=np.zeros(0),
                        counts=np.zeros(0, int), n_bins=0)
        if v.size < n_bins * min_per_bin:
            n_bins = max(2, v.size // max(min_per_bin, 1))
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
        n_bins = max(len(edges) - 1, 1)
        label = np.full(len(lick), -1, int)
        label[ok] = np.clip(np.digitize(v, edges[1:-1]), 0, n_bins - 1)
        med = np.array([np.median(lick[label == b]) if np.any(label == b)
                        else np.nan for b in range(n_bins)])
        cnt = np.array([int(np.sum(label == b)) for b in range(n_bins)])
        return dict(label=label, median=med, edges=edges, counts=cnt,
                    n_bins=n_bins)

    def prelick_mask(self, margin_s: float = 0.0) -> np.ndarray:
        """(trials, T) boolean mask of bins strictly BEFORE that trial's lick.

        This is the right way to define "pre-lick" for any analysis that pools
        (trial, bin) samples, because it needs no rectangular tensor and
        therefore discards NO trials. Every trial contributes however many
        bins precede its own lick.

        Contrast with ``prelick(window_s)``, which keeps a fixed window and so
        must drop trials that licked inside it — introducing a selection bias
        towards slow (and hence more often rewarded) trials. Use that only
        when a rectangular (trials, T, units) tensor is required.

        Because smoothing is a CAUSAL boxcar, the value in a bin centred at t
        depends only on spikes in [t - width, t]. A bin with t < lick therefore
        cannot contain any post-lick spikes, so no extra margin is needed;
        ``margin_s`` is available for extra conservatism (e.g. to guard
        against lick-detection jitter).
        """
        return (self.t[None, :] < (self.lick[:, None] - margin_s))

    def prelick(self, window_s: float = 0.20) -> "Session":
        """EPOCH A — a FIXED post-cue window, before any lick has occurred.

        Returns the raw (unwarped) traces from cue onset to ``window_s``,
        keeping only trials whose first lick comes AFTER the window end. Every
        retained trial therefore contributes real, identically-sampled data
        from the same absolute times, and the epoch contains no lick or
        outcome activity at all.

        No time warping is used: warping would rescale each trial's time axis
        differently and so distort exactly the geometry we want to measure.

        THE TRADE-OFF, which the caller must respect: a longer window keeps
        more of the timing dynamics but discards fast trials, and because
        slow trials are more often rewarded, the retained set becomes biased
        towards rewarded trials. For SM259 the fraction rewarded is 0.31
        overall, 0.30 at window 0.15 s, 0.37 at 0.20 s and 0.90 at 0.40 s.
        ``meta['prelick_frac_rewarded']`` records this for the chosen window
        so the bias is always visible.
        """
        i_cue = int(np.searchsorted(self.t, 0.0))
        i_end = int(np.searchsorted(self.t, window_s))
        if i_end - i_cue < 2:
            raise ValueError(f"window {window_s}s is under two bins wide")
        keep = np.isfinite(self.lick) & (self.lick > self.t[i_end - 1])
        idx = np.where(keep)[0]
        if len(idx) < 20:
            raise ValueError(f"window {window_s}s leaves only {len(idx)} "
                             "trials; shorten it")
        meta = dict(self.meta)
        meta["prelick_window_s"] = window_s
        meta["prelick_kept"] = int(len(idx))
        meta["prelick_dropped"] = int(self.n_trials - len(idx))
        meta["prelick_frac_rewarded"] = float(self.cond["rewarded"][idx].mean())
        meta["prelick_frac_rewarded_all"] = float(self.cond["rewarded"].mean())
        return Session(
            session_id=self.session_id, animal=self.animal, date=self.date,
            group=self.group, coarse=self.coarse,
            training_day=self.training_day,
            X=self.X[idx][:, i_cue:i_end, :], t=self.t[i_cue:i_end],
            bin_s=self.bin_s, lick=self.lick[idx],
            trials={k: v[idx] for k, v in self.trials.items()},
            units=self.units,
            cond={k: v[idx] for k, v in self.cond.items()},
            meta=meta, align="prelick_fixed", path=self.path)

    def lick_aligned(self, t_range: Tuple[float, float] = (-0.6, 0.4)
                     ) -> "Session":
        """Re-slice every trial around its own first lick.

        Returns a new Session with ``align='lick'`` and t = 0 at the lick.
        Trials whose window falls outside the recorded range are dropped.
        """
        n_pre = int(round(-t_range[0] / self.bin_s))
        n_post = int(round(t_range[1] / self.bin_s))
        T = n_pre + n_post
        centre = np.searchsorted(self.t, self.lick)
        keep = (centre - n_pre >= 0) & (centre + n_post <= self.n_time)
        keep &= np.isfinite(self.lick)
        idx = np.where(keep)[0]
        Xl = np.stack([self.X[i, centre[i] - n_pre:centre[i] + n_post, :]
                       for i in idx])
        return Session(
            session_id=self.session_id, animal=self.animal, date=self.date,
            group=self.group, coarse=self.coarse,
            training_day=self.training_day, X=Xl,
            t=(np.arange(T) - n_pre) * self.bin_s + self.bin_s / 2,
            bin_s=self.bin_s, lick=self.lick[idx],
            trials={k: v[idx] for k, v in self.trials.items()},
            units=self.units,
            cond={k: v[idx] for k, v in self.cond.items()},
            meta=self.meta, align="lick", path=self.path)

    def __repr__(self) -> str:
        return (f"Session({self.session_id}, {self.group}, day "
                f"{self.training_day}, align={self.align}: {self.n_trials} "
                f"trials x {self.n_time} bins x {self.n_units} units)")


# --------------------------------------------------------------------------- #
def list_sessions(folder: str | Path) -> List[Path]:
    return sorted(Path(folder).glob("*.h5"))


def session_table(folder: str | Path) -> List[Dict]:
    """Metadata-only scan of every session (no spike loading)."""
    out = []
    for p in list_sessions(folder):
        with h5py.File(p, "r") as f:
            m = f["meta"].attrs
            rt = f["trials/response_type"][:]
            valid = f["trials/is_valid_for_analysis"][:].astype(bool)
            qm = f["units/qm_pass"][:].astype(bool)
            fl = f["trials/first_lick_s"][:]
            out.append(dict(
                path=str(p), session_id=str(m["session_id"]),
                animal=str(m["animal"]), date=str(m["session_date"]),
                group=str(m["exp_type_fine"]),
                coarse=str(m["exp_type_coarse"]),
                training_day=int(m["training_day"]),
                n_units=int(m["n_units_total"]), n_units_qm=int(qm.sum()),
                n_trials=int(m["n_trials"]), n_valid=int(valid.sum()),
                n_rewarded=int((valid & (rt == 1)).sum()),
                n_early=int((valid & (rt == 0)).sum()),
                median_lick_rewarded=float(np.nanmedian(fl[rt == 1]))
                if (rt == 1).any() else np.nan,
                median_lick_early=float(np.nanmedian(fl[rt == 0]))
                if (rt == 0).any() else np.nan,
            ))
    return out


def _rebin_sum(x: np.ndarray, factor: int) -> np.ndarray:
    """Sum over non-overlapping blocks along the last axis."""
    if factor <= 1:
        return x
    n = (x.shape[-1] // factor) * factor
    return x[..., :n].reshape(*x.shape[:-1], n // factor, factor).sum(-1)


def _causal_boxcar(x: np.ndarray, width_bins: int) -> np.ndarray:
    """Trailing (causal) boxcar along the last axis — no backward smearing."""
    if width_bins <= 1:
        return x
    k = np.ones(width_bins) / width_bins
    flat = x.reshape(-1, x.shape[-1])
    out = np.empty_like(flat, dtype=float)
    for i in range(flat.shape[0]):
        out[i] = np.convolve(flat[i], k, "full")[:flat.shape[1]]
    return out.reshape(x.shape)


def load_session(
    path: str | Path,
    *,
    bin_ms: float = 20.0,
    smooth_ms: float = 60.0,
    t_range: Tuple[float, float] = (-0.5, 1.5),
    qm_only: bool = True,
    valid_only: bool = True,
    min_rate_hz: float = 0.5,
    pyramidal_only: bool = False,
    smoothing: str = "causal",
) -> Session:
    """Load one session into a (trials, time, units) firing-rate tensor.

    Parameters
    ----------
    bin_ms : re-bin the native 5 ms bins to this width by SUMMING counts.
    smooth_ms : boxcar width (``smoothing='causal'``, default) or Gaussian SD
        (``smoothing='gaussian'``). 0 disables.
    t_range : seconds relative to cue onset.
    qm_only : keep only units passing the quality metric.
    valid_only : keep only ``is_valid_for_analysis`` trials (this already
        implies cue trials with a lick).
    min_rate_hz : drop units below this mean rate inside the window.
    pyramidal_only : keep only ``is_pyramidal`` units.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        m = {k: (v.item() if isinstance(v, np.generic) else v)
             for k, v in f["meta"].attrs.items()}
        bin_s0 = float(m.get("bin_size_s", 0.005))
        t0 = f["spikes/t_bins"][:]
        trials = {k: f["trials"][k][:] for k in f["trials"].keys()}
        units = {k: f["units"][k][:] for k in f["units"].keys()}
        rt = trials["response_type"]

        # verify the response_type coding against the boolean flags
        for code, flag in {0: "is_early_lick", 1: "is_rewarded",
                           2: "is_no_response", 3: "is_no_cue"}.items():
            if flag in trials:
                agree = np.mean((rt == code) == trials[flag].astype(bool))
                if agree < 0.999:
                    raise ValueError(
                        f"{path.name}: response_type=={code} disagrees with "
                        f"{flag} ({agree:.3f}); the assumed coding is wrong "
                        "for this file.")

        tmask = np.ones(len(rt), dtype=bool)
        if valid_only and "is_valid_for_analysis" in trials:
            tmask &= trials["is_valid_for_analysis"].astype(bool)
            # sanity: validity should already imply a cue trial
            if "is_cue_trial" in trials:
                bad = int((tmask & ~trials["is_cue_trial"].astype(bool)).sum())
                if bad:
                    raise ValueError(f"{path.name}: {bad} valid non-cue "
                                     "trials — assumption violated.")

        umask = np.ones(len(units["qm_pass"]), dtype=bool)
        if qm_only:
            umask &= units["qm_pass"].astype(bool)
        if pyramidal_only and "is_pyramidal" in units:
            umask &= units["is_pyramidal"].astype(bool)

        # HARD RULE: always SMOOTH FIRST, THEN SLICE. Load a generous margin
        # (2x the smoothing width) beyond the requested window so the kernel
        # never averages in implicit zeros, then trim after smoothing.
        # Without this the causal boxcar depresses the leading bins — it
        # showed up in QC as a spurious dip at the start of the window.
        margin_s = 2.0 * max(0.0, smooth_ms / 1000.0) + 4 * bin_ms / 1000.0
        keep_t = (t0 >= t_range[0] - margin_s) & (t0 <= t_range[1] + margin_s)
        if not keep_t.any():
            raise ValueError(f"{path.name}: t_range {t_range} outside the "
                             f"recorded window {t0[0]:.2f}..{t0[-1]:.2f}s")
        i0 = int(np.argmax(keep_t))
        i1 = int(len(keep_t) - np.argmax(keep_t[::-1]))
        counts = f["spikes/counts"][:, :, i0:i1][umask][:, tmask, :]
        t = t0[i0:i1]

    factor = int(round(bin_ms / (bin_s0 * 1000.0)))
    if factor < 1:
        raise ValueError(f"bin_ms={bin_ms} finer than native "
                         f"{bin_s0*1000:.0f} ms")
    counts = _rebin_sum(np.nan_to_num(counts), factor)
    t = t[:(len(t) // factor) * factor].reshape(-1, factor).mean(-1)
    bin_s = bin_s0 * factor
    rates = counts / bin_s                                   # Hz

    if smooth_ms > 0:
        if smoothing == "causal":
            rates = _causal_boxcar(rates, max(1, int(round(
                (smooth_ms / 1000.0) / bin_s))))
        elif smoothing == "gaussian":
            sig = (smooth_ms / 1000.0) / bin_s
            r = int(np.ceil(3 * sig))
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
            k /= k.sum()
            pad = np.pad(rates, [(0, 0), (0, 0), (r, r)], mode="edge")
            rates = np.apply_along_axis(
                lambda v: np.convolve(v, k, "valid"), -1, pad)
        else:
            raise ValueError(f"unknown smoothing {smoothing!r}")

    # trim the smoothing margin off, leaving exactly the requested window
    inside = (t >= t_range[0]) & (t <= t_range[1])
    rates, t = rates[:, :, inside], t[inside]

    X = np.transpose(rates, (1, 2, 0)).astype(np.float64)    # (tr, T, N)

    keep_u = X.mean(axis=(0, 1)) >= min_rate_hz
    X = X[:, :, keep_u]
    units = {k: v[umask][keep_u] for k, v in units.items()}
    trials = {k: v[tmask] for k, v in trials.items()}
    for dead in DROPPED_FIELDS:
        trials.pop(dead, None)

    rt = trials["response_type"]
    lick = np.asarray(trials["first_lick_s"], float)
    cond = {
        "rewarded": rt == 1,
        "early_lick": rt == 0,
        "licked": (rt == 0) | (rt == 1),
    }

    return Session(
        session_id=str(m.get("session_id", path.stem)),
        animal=str(m.get("animal", "?")), date=str(m.get("session_date", "?")),
        group=str(m.get("exp_type_fine", "?")),
        coarse=str(m.get("exp_type_coarse", "?")),
        training_day=int(m.get("training_day", -1)),
        X=X, t=t, bin_s=bin_s, lick=lick, trials=trials, units=units,
        cond=cond, meta=m, align="cue", path=str(path),
    )
