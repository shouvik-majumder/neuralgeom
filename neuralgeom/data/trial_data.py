"""TrialData: the shared trial-tensor container and its HDF5 I/O.

A single container for population activity from *any* source -- recorded neurons or a
task-trained RNN -- so the same analysis code runs on both. The layout deliberately
matches the RNN-geometry pipeline's HDF5 contract (`X (n_trials, T, N)`, `time (T,)`,
`inputs`, connectivity) while also exposing the neuroscience-conventional
neuron x time x trial view.

Fields
------
X            : (n_trials, T, N) float  -- activity (firing rate or RNN state); N units.
time         : (T,) float              -- time axis (s), typically cue/event-relative.
behavior     : dict[str, array]        -- per-trial or per-trial-per-time targets/labels
                                          (e.g. 'first_lick_s' (n_trials,)). Arbitrary
                                          output types -- scalar, categorical, vector.
inputs       : (n_trials, T, n_in) or None -- task inputs (optional).
conditions   : dict[str, array]        -- per-trial condition labels (reward, delay, ...).
connectivity : dict[str, array]        -- for model data: {'W'} or low-rank {'U','V'}.
meta         : dict                    -- free-form provenance/attributes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TrialData:
    X: np.ndarray
    time: np.ndarray
    behavior: dict = field(default_factory=dict)
    inputs: Optional[np.ndarray] = None
    conditions: dict = field(default_factory=dict)
    connectivity: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.X = np.asarray(self.X, float)
        if self.X.ndim != 3:
            raise ValueError("X must be (n_trials, T, N).")
        self.time = np.asarray(self.time, float).ravel()
        if self.time.size != self.X.shape[1]:
            raise ValueError("time length must equal T (X.shape[1]).")

    # --- shape accessors ----------------------------------------------------
    @property
    def n_trials(self):
        return self.X.shape[0]

    @property
    def T(self):
        return self.X.shape[1]

    @property
    def n_units(self):
        return self.X.shape[2]

    @property
    def neuron_time_trial(self) -> np.ndarray:
        """Return activity as the neuroscience-conventional (N, T, n_trials) tensor."""
        return np.transpose(self.X, (2, 1, 0))

    def trial_mean(self) -> np.ndarray:
        """Condition-independent mean activity, (T, N)."""
        return self.X.mean(axis=0)

    # --- HDF5 I/O -----------------------------------------------------------
    def save(self, path):
        import h5py
        with h5py.File(path, "w") as h:
            h.create_dataset("X", data=self.X, compression="gzip")
            h.create_dataset("time", data=self.time)
            if self.inputs is not None:
                h.create_dataset("inputs", data=np.asarray(self.inputs, float),
                                  compression="gzip")
            for grp, d in (("behavior", self.behavior),
                           ("conditions", self.conditions),
                           ("connectivity", self.connectivity)):
                g = h.create_group(grp)
                for k, v in d.items():
                    g.create_dataset(str(k), data=np.asarray(v))
            for k, v in self.meta.items():
                try:
                    h.attrs[str(k)] = v
                except (TypeError, ValueError):
                    h.attrs[str(k)] = str(v)

    @classmethod
    def load(cls, path):
        import h5py
        with h5py.File(path, "r") as h:
            X = h["X"][:]
            time = h["time"][:]
            inputs = h["inputs"][:] if "inputs" in h else None

            def read_group(name):
                return ({k: h[name][k][:] for k in h[name]} if name in h else {})

            return cls(
                X=X, time=time, inputs=inputs,
                behavior=read_group("behavior"),
                conditions=read_group("conditions"),
                connectivity=read_group("connectivity"),
                meta={k: h.attrs[k] for k in h.attrs},
            )
