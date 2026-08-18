"""Continuous-time rate RNN generator (low-rank emphasis), emitting TrialData.

Architecture matches the sibling RNN-geometry pipeline's contract so trained-network
activity is interchangeable between projects:

    tau * xdot = -x + W @ tanh(x) + B @ u(t) + noise,     integrated by Euler(-Maruyama).

Connectivity is either full (`W`) or low-rank `W = (U V^T) / N` with U, V in R^{N x r}
(low-rank RNNs are the most analyzable substrate for connectivity<->computation<->geometry).

This module provides the *substrate and I/O contract*. Task-training (fitting U, V, B and
a readout to solve a specific task) is the next milestone; the readout used by
PullbackMetric can be any trained decoder mapping the state x(t) to the task output.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..data.trial_data import TrialData


@dataclass
class RNNConfig:
    N: int = 100
    rank: int = 2
    dt: float = 0.005          # s
    T: int = 200               # timesteps
    tau: float = 0.1           # s
    noise_std: float = 0.02
    gain: float = 1.2
    seed: int = 0


def make_lowrank_connectivity(N, rank, gain, rng):
    U = rng.standard_normal((N, rank))
    V = rng.standard_normal((N, rank))
    W = (U @ V.T) / N
    # scale to a target spectral radius
    radius = max(abs(np.linalg.eigvals(W)).max(), 1e-8)
    W = W * (gain / radius)
    return W, U, V


def simulate(cfg: RNNConfig, inputs=None, x0=None):
    """Integrate the rate RNN for one trial. inputs: (T, n_in) or None. Returns X (T, N)."""
    rng = np.random.default_rng(cfg.seed)
    W, U, V = make_lowrank_connectivity(cfg.N, cfg.rank, cfg.gain, rng)
    x = np.zeros(cfg.N) if x0 is None else np.asarray(x0, float)
    n_in = 0 if inputs is None else inputs.shape[1]
    B = rng.standard_normal((cfg.N, n_in)) / np.sqrt(cfg.N) if n_in else None
    X = np.zeros((cfg.T, cfg.N))
    sq = np.sqrt(cfg.dt)
    for t in range(cfg.T):
        drive = W @ np.tanh(x)
        if B is not None:
            drive = drive + B @ inputs[t]
        dx = (-x + drive) * (cfg.dt / cfg.tau)
        if cfg.noise_std > 0:
            dx = dx + cfg.noise_std * sq * rng.standard_normal(cfg.N)
        x = x + dx
        X[t] = x
    return X, (W, U, V)


def make_dataset(cfg: RNNConfig, n_trials: int = 40, input_fn=None) -> TrialData:
    """Generate a multi-trial dataset in the shared schema.

    input_fn(trial_index, time) -> (T, n_in) inputs, or None for autonomous dynamics.
    """
    rng = np.random.default_rng(cfg.seed)
    time = np.arange(cfg.T) * cfg.dt
    Xs, ins = [], []
    W = U = V = None
    for i in range(n_trials):
        u = None if input_fn is None else np.asarray(input_fn(i, time), float)
        c = RNNConfig(**{**cfg.__dict__, "seed": cfg.seed + i})
        Xi, (W, U, V) = simulate(c, inputs=u, x0=rng.standard_normal(cfg.N) * 0.1)
        Xs.append(Xi)
        if u is not None:
            ins.append(u)
    X = np.stack(Xs, 0)                                   # (n_trials, T, N)
    inputs = np.stack(ins, 0) if ins else None
    return TrialData(
        X=X, time=time, inputs=inputs,
        connectivity={"W": W, "U": U, "V": V},
        meta={"source": "rnn.make_lowrank", "N": cfg.N, "rank": cfg.rank,
              "tau": cfg.tau, "dt": cfg.dt, "gain": cfg.gain},
    )
