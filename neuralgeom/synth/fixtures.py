"""Synthetic latent -> behavior systems with known ground-truth geometry.

Used to validate that the pullback metric recovers the behaviorally-relevant directions
we built in, before touching real data. The generative model:

  - a low-D latent state x in R^d,
  - a scalar timing readout   latency(x) = tau0 + softplus(c - w_lat . x)   [seconds],
  - a binary reward readout    p_reward(x) = sigmoid(w_rew . x),

so the local sensitivity of behavior lives in span{w_lat, w_rew}. Directions orthogonal
to both are behaviorally irrelevant and should fall in the kernel of the pullback metric.

`ramp_trajectories` produces per-trial latent trajectories that ramp along w_lat (a
ramp-to-threshold timing motif), so the pullback metric can be tracked across a trial.
"""

from __future__ import annotations
import numpy as np

from ..geometry.maps import FunctionReadout
from ..data.trial_data import TrialData


def _unit(v):
    return v / np.linalg.norm(v)


def make_synthetic_readout(dim: int = 5, seed: int = 0, tau0: float = 0.15, c: float = 1.5):
    """Return ground-truth directions and readout maps for the synthetic system."""
    rng = np.random.default_rng(seed)
    w_lat = _unit(rng.standard_normal(dim))
    w_rew = rng.standard_normal(dim)
    w_rew = _unit(w_rew - (w_rew @ w_lat) * w_lat)   # orthogonalize for a clean test

    def latency(x):
        z = c - w_lat @ x
        return np.array([tau0 + np.log1p(np.exp(z))])

    def latency_and_reward(x):
        z = c - w_lat @ x
        lat = tau0 + np.log1p(np.exp(z))
        p = 1.0 / (1.0 + np.exp(-(w_rew @ x)))
        return np.array([lat, p])

    return {
        "dim": dim,
        "w_lat": w_lat,
        "w_rew": w_rew,
        "latency_readout": FunctionReadout(latency),
        "latency_reward_readout": FunctionReadout(latency_and_reward),
    }


def sample_states(dim: int, n: int = 400, seed: int = 1, scale: float = 0.7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)) * scale


def ramp_trajectories(spec, n_trials: int = 60, T: int = 40, dt: float = 0.025,
                      seed: int = 2, ramp_rate: float = 2.0, noise: float = 0.05):
    """Latent trajectories that ramp along w_lat, as a TrialData tensor.

    Each trial: x(t) = x0 + (rate_i * t) * w_lat + small isotropic noise, with a
    per-trial ramp rate (trial-to-trial timing variability). Behavior 'first_lick_s'
    is the time the projection onto w_lat crosses a threshold.
    """
    rng = np.random.default_rng(seed)
    d = spec["dim"]
    w = spec["w_lat"]
    time = np.arange(T) * dt
    X = np.zeros((n_trials, T, d))
    latency = np.zeros(n_trials)
    threshold = 1.5
    for i in range(n_trials):
        x0 = rng.standard_normal(d) * 0.3
        rate = ramp_rate * (1.0 + 0.3 * rng.standard_normal())
        proj = (w @ x0) + rate * time
        traj = x0[None, :] + (rate * time)[:, None] * w[None, :]
        traj += rng.standard_normal((T, d)) * noise
        X[i] = traj
        cross = np.where(proj >= threshold)[0]
        latency[i] = time[cross[0]] if cross.size else time[-1]
    return TrialData(
        X=X, time=time,
        behavior={"first_lick_s": latency},
        conditions={"ramp_rate": np.full(n_trials, ramp_rate)},
        meta={"source": "synthetic.ramp_trajectories", "threshold": threshold},
    )
