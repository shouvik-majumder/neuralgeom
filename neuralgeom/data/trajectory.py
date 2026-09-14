"""
neuralgeom.data.trajectory — the shared Trajectory object and its HDF5 schema.
==============================================================================

Every analysis in ``neuralgeom`` consumes the same object: a batch of
high-dimensional state trajectories with enough provenance to interpret them.
This module defines that object and its HDF5 schema, so a synthetic generator,
a recording loader, a trained network and every analysis exchange one format.

Fields
------
A :class:`Trajectory` bundles::

    X          : (n_trials, T, N)      REQUIRED   hidden / pre-activation states
    time       : (T,)                  REQUIRED   sample times (seconds)
    dt, tau    : scalars               dt required (inferred from time if absent)
    inputs     : (n_trials, T, n_in)   optional   stimulus / drive
    condition  : (n_trials,)           optional   trial label (coherence, context,
                                                   heading, lick-time bin, …)
    outputs    : (n_trials, T, n_out)  optional   readout / target
    W          : (N, N)                optional   connectivity
    U, V       : (N, R)                optional   low-rank loadings (W = U Vᵀ / N)
    aux        : dict[str, ndarray]    optional   generator-specific extras
                                                   (loadings m, generator G, ring
                                                   angles θ, per-trial onsets, …)
    meta       : dict                  REQUIRED   generator id + full config

Anything downstream that needs only ``X`` and ``time`` (the subspace embedding,
the topology) is handed those arrays; anything that needs drive/condition labels
reads the optional fields. A trained-RNN repository satisfies the whole
framework simply by writing this schema (see :meth:`Trajectory.save`).

Interoperability
----------------
* :meth:`Trajectory.from_arrays`     — the minimal path (just states).
* :meth:`Trajectory.from_synth_dict` — adapts the dict emitted by the synthetic
  generators (:mod:`neuralgeom.synth`), including the subspace RNN generator.
* :meth:`Trajectory.from_session`    — adapts a recording
  :class:`~neuralgeom.data.loader.Session`.
* :meth:`Trajectory.save` / :func:`load_trajectory` — HDF5 round-trip, and the
  format the subspace/topology pipelines read.
* :func:`load_trial` / :func:`load_all_trials` — thin array front-ends (the only
  functions the subspace pipeline calls), reading either a Trajectory HDF5 file
  or the legacy ``rnn_{tag}.h5`` schema, so nothing breaks.

Only ``numpy`` is required to build a Trajectory; ``h5py`` is needed for the
HDF5 I/O (it is a core dependency).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "Trajectory",
    "load_trajectory",
    "load_trial",
    "load_all_trials",
]


# --------------------------------------------------------------------------- #
# The Trajectory object
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    """A batch of state trajectories plus provenance — the common interchange.

    See the module docstring for the field list. ``X`` and ``time`` are
    required; ``dt`` is inferred from ``time`` if not given; ``meta`` always
    carries at least a ``generator`` key so an analysis can report where the
    data came from.
    """

    X: np.ndarray                                   # (n_trials, T, N)
    time: np.ndarray                                # (T,)
    dt: float = None                                # inferred if None
    tau: Optional[float] = None
    inputs: Optional[np.ndarray] = None             # (n_trials, T, n_in)
    condition: Optional[np.ndarray] = None          # (n_trials,)
    outputs: Optional[np.ndarray] = None            # (n_trials, T, n_out)
    W: Optional[np.ndarray] = None                  # (N, N)
    U: Optional[np.ndarray] = None                  # (N, R) low-rank loadings
    V: Optional[np.ndarray] = None                  # (N, R)
    aux: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.X = np.asarray(self.X, float)
        if self.X.ndim != 3:
            raise ValueError(
                f"X must be (n_trials, T, N); got shape {self.X.shape}. "
                "For a single trial use X[None] to add the trial axis."
            )
        self.time = np.asarray(self.time, float).ravel()
        if self.time.shape[0] != self.X.shape[1]:
            raise ValueError(
                f"time length {self.time.shape[0]} != T={self.X.shape[1]}."
            )
        if self.dt is None:
            self.dt = (float(np.mean(np.diff(self.time)))
                       if self.time.size > 1 else 1.0)
        else:
            self.dt = float(self.dt)
        if "generator" not in self.meta:
            self.meta = {"generator": "unknown", **self.meta}

    # -- shape helpers ------------------------------------------------------ #
    @property
    def n_trials(self) -> int:
        return int(self.X.shape[0])

    @property
    def T(self) -> int:
        return int(self.X.shape[1])

    @property
    def N(self) -> int:
        return int(self.X.shape[2])

    def trial(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(X_i (T, N), time (T,))`` for one trial — the array pair the
        subspace embedding consumes."""
        return self.X[i], self.time

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        extras = [n for n in ("inputs", "condition", "outputs", "W", "U", "V")
                  if getattr(self, n) is not None]
        return (f"Trajectory(n_trials={self.n_trials}, T={self.T}, N={self.N}, "
                f"dt={self.dt:g}, generator={self.meta.get('generator')!r}, "
                f"has={extras}, aux={list(self.aux)})")

    # -- adapters ----------------------------------------------------------- #
    @classmethod
    def from_arrays(cls, X, time=None, *, dt=None, tau=None, inputs=None,
                    condition=None, outputs=None, W=None, U=None, V=None,
                    aux=None, generator="arrays", **meta) -> "Trajectory":
        """Build a Trajectory from raw arrays. ``X`` may be ``(n_trials, T, N)``
        or ``(T, N)`` (a single trial, promoted automatically). If ``time`` is
        omitted an integer index scaled by ``dt`` (default 1) is used."""
        X = np.asarray(X, float)
        if X.ndim == 2:
            X = X[None]
        if time is None:
            time = np.arange(X.shape[1]) * (1.0 if dt is None else dt)
        return cls(X=X, time=time, dt=dt, tau=tau, inputs=inputs,
                   condition=condition, outputs=outputs, W=W, U=U, V=V,
                   aux=dict(aux or {}), meta={"generator": generator, **meta})

    @classmethod
    def from_synth_dict(cls, d: dict, *, generator: str = "synth") -> "Trajectory":
        """Adapt the plain dict emitted by :mod:`neuralgeom.synth` generators.

        Recognised keys: ``X``, ``time`` (required); ``W``, ``b``, ``aux``,
        ``config`` and any of ``onsets``/``phi0``/``omega`` (folded into
        ``aux``/``meta``). This is how the subspace RNN generator
        (:mod:`neuralgeom.synth.subspace_rnn`) and the attractor/low-rank
        generators are converted.
        """
        d = dict(d)
        X = np.asarray(d["X"], float)
        time = np.asarray(d.get("time", np.arange(X.shape[1])), float)
        aux = {k: np.asarray(v) for k, v in dict(d.get("aux", {})).items()}
        for k in ("onsets", "phi0", "b"):
            if d.get(k) is not None:
                aux[k] = np.asarray(d[k])
        meta = {"generator": generator}
        if d.get("config") is not None:
            meta["config"] = d["config"]
        if d.get("omega") is not None and np.isfinite(np.atleast_1d(d["omega"])).all():
            meta["omega"] = float(np.atleast_1d(d["omega"]).ravel()[0])
        tau = None
        if isinstance(d.get("config"), dict):
            tau = d["config"].get("tau")
        return cls(X=X, time=time, tau=tau, inputs=d.get("inputs"),
                   condition=d.get("condition"), outputs=d.get("outputs"),
                   W=d.get("W"), U=d.get("U"), V=d.get("V"), aux=aux, meta=meta)

    @classmethod
    def from_session(cls, session, *, condition: str = "lick") -> "Trajectory":
        """Adapt a recording :class:`~neuralgeom.data.loader.Session`.

        ``session.X`` (trials, time, units, in Hz) becomes ``X``; ``session.t``
        becomes ``time``; ``dt`` is ``session.bin_s``. ``condition`` selects the
        per-trial label to carry (default ``"lick"`` — first-lick time); pass a
        different key present on the session, or ``None`` for no label. The
        session id and any ``meta`` are preserved.
        """
        X = np.asarray(session.X, float)
        time = np.asarray(getattr(session, "t"), float)
        dt = float(getattr(session, "bin_s", np.mean(np.diff(time))))
        cond = None
        if condition is not None:
            cond = getattr(session, condition, None)
            if cond is None and isinstance(getattr(session, "trials", None), dict):
                cond = session.trials.get(condition)
        meta = {"generator": "session",
                "session_id": getattr(session, "session_id", None)}
        smeta = getattr(session, "meta", None)
        if isinstance(smeta, dict):
            meta["session_meta"] = {k: v for k, v in smeta.items()
                                    if np.isscalar(v)}
        return cls(X=X, time=time, dt=dt,
                   condition=None if cond is None else np.asarray(cond),
                   meta=meta)

    # -- HDF5 round-trip ---------------------------------------------------- #
    def save(self, path: str, *, compression: str = "gzip") -> str:
        """Write the trajectory to HDF5 in the canonical schema and return the
        path. Large arrays (``X``, ``inputs``, ``outputs``) are gzip-compressed;
        ``meta`` is stored as a JSON string attribute for lossless round-trip."""
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("X", data=self.X, compression=compression)
            f.create_dataset("time", data=self.time)
            f.attrs["dt"] = self.dt
            if self.tau is not None:
                f.attrs["tau"] = self.tau
            for name in ("inputs", "outputs"):
                val = getattr(self, name)
                if val is not None:
                    f.create_dataset(name, data=np.asarray(val),
                                     compression=compression)
            for name in ("condition", "W", "U", "V"):
                val = getattr(self, name)
                if val is not None:
                    f.create_dataset(name, data=np.asarray(val))
            for key, val in self.aux.items():
                f.create_dataset(f"aux/{key}", data=np.asarray(val))
            f.attrs["meta"] = json.dumps(_jsonable(self.meta))
            f.attrs["neuralgeom_trajectory"] = True
        return path


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def load_trajectory(path: str) -> Trajectory:
    """Read a :class:`Trajectory` from HDF5.

    Handles both the canonical schema written by :meth:`Trajectory.save` and the
    legacy ``rnn_{tag}.h5`` schema of earlier RNN generators (which stores
    ``onsets``/``phi0``/``omega`` and the config as loose attributes) — legacy
    extras are folded into ``aux``/``meta`` so old datasets load unchanged.
    """
    import h5py

    with h5py.File(path, "r") as f:
        X = np.asarray(f["X"])
        time = (np.asarray(f["time"]) if "time" in f
                else np.arange(X.shape[1], dtype=float))
        attrs = dict(f.attrs)
        dt = float(attrs.get("dt", np.mean(np.diff(time)) if time.size > 1 else 1.0))
        tau = attrs.get("tau")
        inputs = np.asarray(f["inputs"]) if "inputs" in f else None
        outputs = np.asarray(f["outputs"]) if "outputs" in f else None
        condition = np.asarray(f["condition"]) if "condition" in f else None
        W = np.asarray(f["W"]) if "W" in f else None
        U = np.asarray(f["U"]) if "U" in f else None
        V = np.asarray(f["V"]) if "V" in f else None
        aux = {k: np.asarray(f[f"aux/{k}"]) for k in f["aux"]} if "aux" in f else {}

        if "meta" in attrs:                       # canonical schema
            try:
                meta = json.loads(attrs["meta"])
            except (TypeError, json.JSONDecodeError):
                meta = {"generator": "unknown"}
        else:                                     # legacy rnn_{tag}.h5
            meta = {"generator": "legacy_rnn"}
            for k in ("onsets", "phi0", "b"):
                if k in f:
                    aux[k] = np.asarray(f[k])
            cfg = {k: _from_attr(v) for k, v in attrs.items()
                   if k not in ("omega",)}
            if cfg:
                meta["config"] = cfg
            if "omega" in attrs:
                meta["omega"] = _from_attr(attrs["omega"])
            if tau is None and isinstance(cfg.get("tau"), (int, float)):
                tau = float(cfg["tau"])

    return Trajectory(X=X, time=time, dt=dt,
                      tau=None if tau is None else float(tau),
                      inputs=inputs, outputs=outputs, condition=condition,
                      W=W, U=U, V=V, aux=aux, meta=meta)


def load_trial(path: str, trial: int = 0):
    """Return ``(X_trial (T, N), time (T,), meta)`` for one trial of a stored
    trajectory. This is the array front-end the subspace embedding calls — the
    only piece that changes when swapping in trained networks or real data."""
    traj = load_trajectory(path)
    return traj.X[trial], traj.time, traj.meta


def load_all_trials(path: str):
    """Return ``(X (n_trials, T, N), time (T,))`` for a stored trajectory."""
    traj = load_trajectory(path)
    return traj.X, traj.time


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _jsonable(obj):
    """Recursively convert numpy scalars/arrays so ``meta`` serialises to JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _from_attr(v):
    """Decode an HDF5 attribute, turning the sentinel string 'None' back into
    ``None`` (the legacy generator stored None-valued config that way)."""
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str) and v == "None":
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v
