"""
neuralgeom.synth.subspace_rnn — connectivity-family rate RNNs (the subspace testbed).
=====================================================================================

Ported from the ProjectiveSpaceModels ``rnn_generator`` (Step 2 of its
pipeline). Simulates high-dimensional hidden-state trajectories of a
continuous-time (rate) recurrent network::

        τ ẋ = −x + W · tanh(x) + I(t) + noise

and returns them as a :class:`~neuralgeom.data.trajectory.Trajectory`, so the
subspace-embedding, kinematics and topology analyses (:mod:`neuralgeom.subspace`,
:mod:`neuralgeom.topology`) consume it exactly like a recording or a trained
network. This is a *testbed*: three connectivity families × two dynamical
regimes give data whose subspace geometry and topology are known in advance, so
the pipeline can be validated before it is trusted on unknown data.

Why it lives beside ``lowrank_rnn`` rather than replacing it
------------------------------------------------------------
``neuralgeom.synth.lowrank_rnn`` already provides a low-rank rate RNN aimed at
the *pullback / dynamics* estimators (it emits a
:class:`~neuralgeom.data.trial_data.TrialData`). This generator is a different
instrument: three connectivity families (``random`` chaotic, ``lowrank``
point/rotation, ``ring`` attractor) and two regimes (``settling`` / ``moving``)
chosen to exercise the *subspace-geometry / topology* lens (loops on the
Grassmannian, ℝP¹ structure, reliability failures). Its config class is named
:class:`SubspaceRNNConfig` to avoid colliding with ``lowrank_rnn.RNNConfig``.

Connectivity families
----------------------
* ``random``  — Gaussian ``W`` with entries ~ N(0, g²/N); gain ``g = 1.5`` ⇒ chaotic.
* ``lowrank`` — ``W = M G Mᵀ`` with orthonormal loadings ``M`` (N, R); ``G``
  symmetric ⇒ point/line attractors, or a block-rotation ⇒ oscillatory. Spectral
  radius is exactly ``lowrank_gain`` (not ~1/√N), so dynamics are genuinely
  low-rank-driven and confined to span(M).
* ``ring``    — cosine kernel ``W_ij = (J1/N) cos(θ_i − θ_j)``, itself rank-2 —
  a structural bridge to the low-rank case; a continuous ring attractor.

Integration is fixed-step RK4 when ``noise_std == 0`` and Euler–Maruyama when
``noise_std > 0`` (default 0.3). Only ``numpy`` is required.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..data.trajectory import Trajectory

__all__ = [
    "SubspaceRNNConfig",
    "make_random_connectivity",
    "make_lowrank_connectivity",
    "make_ring_connectivity",
    "make_input_direction",
    "build_trial_input",
    "simulate_trial",
    "make_dataset",
    "make_trajectory",
    "population_vector_angle",
    "build_specs",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SubspaceRNNConfig:
    """All hyperparameters for one connectivity-family rate RNN dataset."""

    # --- topology / size ---
    N: int = 100
    connectivity: str = "random"          # "random" | "lowrank" | "ring"

    # --- time ---
    dt: float = 1e-3                       # 1 ms
    duration: float = 2.0                  # seconds
    tau: float = 0.1                       # 100 ms time constant

    # --- dataset ---
    n_trials: int = 30
    ic_std: float = 1.0                    # std of x0 ~ N(0, ic_std²)

    # --- random connectivity ---
    g: float = 1.5                         # gain; >1 ⇒ chaotic

    # --- low-rank connectivity ---
    rank: int = 2
    lowrank_gain: float = 1.5              # effective gain of each low-rank mode
    lowrank_mode: str = "symmetric"        # "symmetric" | "rotation"
    lowrank_rot_angle: float = 0.5         # rotation angle per 2×2 block (radians)
    lowrank_random_g: float = 0.0          # optional random background (g/√N)

    # --- ring connectivity ---
    ring_J1: float = 4.0                   # cosine kernel amplitude
    ring_J0: float = 0.0                   # uniform component
    ring_moving: bool = False              # rotating stimulus bump ⇒ single-trial loop
    ring_stim_amp: float = 1.0
    ring_revolutions: float = 1.0

    # --- input pulse (random & lowrank only) ---
    use_pulse: bool = True
    pulse_amp: float = 2.0
    pulse_dur: float = 0.1                 # 100 ms
    pulse_onset_range: tuple = (0.2, 0.5)  # seconds, uniform per trial

    # --- noise (0 ⇒ deterministic RK4; >0 ⇒ Euler–Maruyama) ---
    noise_std: float = 0.3

    # --- reproducibility ---
    seed: int = 0

    @property
    def n_steps(self) -> int:
        return int(round(self.duration / self.dt)) + 1

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.n_steps) * self.dt


# --------------------------------------------------------------------------- #
# Connectivity builders
# --------------------------------------------------------------------------- #
def make_random_connectivity(cfg: SubspaceRNNConfig, rng) -> np.ndarray:
    """Gaussian ``W`` with entries ~ N(0, g²/N) (chaotic for g > 1)."""
    return rng.standard_normal((cfg.N, cfg.N)) * (cfg.g / np.sqrt(cfg.N))


def make_lowrank_connectivity(cfg: SubspaceRNNConfig, rng):
    """Rank-R connectivity ``W = M G Mᵀ`` confined to an R-dim plane.

    ``M`` (N, R) are orthonormal loadings; ``G`` (R, R) is the in-plane
    generator — ``lowrank_gain·I`` (real spectrum ⇒ point/line attractors) or
    block-rotations (complex spectrum ⇒ oscillatory). Guarantees
    ``spectral_radius(W) = lowrank_gain``. Returns ``(W, {'m', 'G'})``; ``m[:,0]``
    doubles as the aligned input direction.
    """
    A = rng.standard_normal((cfg.N, cfg.rank))
    M, _ = np.linalg.qr(A)
    M = M[:, : cfg.rank]

    if cfg.lowrank_mode == "symmetric":
        G = cfg.lowrank_gain * np.eye(cfg.rank)
    elif cfg.lowrank_mode == "rotation":
        G = np.zeros((cfg.rank, cfg.rank))
        c, s = np.cos(cfg.lowrank_rot_angle), np.sin(cfg.lowrank_rot_angle)
        block = cfg.lowrank_gain * np.array([[c, -s], [s, c]])
        r = 0
        while r + 1 < cfg.rank:
            G[r:r + 2, r:r + 2] = block
            r += 2
        if r < cfg.rank:
            G[r, r] = cfg.lowrank_gain
    else:
        raise ValueError(f"Unknown lowrank_mode: {cfg.lowrank_mode!r}")

    W = M @ G @ M.T
    if cfg.lowrank_random_g > 0:
        W = W + rng.standard_normal((cfg.N, cfg.N)) * (cfg.lowrank_random_g / np.sqrt(cfg.N))
    return W, {"m": M, "G": G}


def make_ring_connectivity(cfg: SubspaceRNNConfig):
    """Continuous ring attractor ``W_ij = (1/N)[J0 + J1 cos(θ_i − θ_j)]``.

    ``θ_i = 2πi/N`` are preferred angles; the cosine kernel is rank-2 in
    (cos θ, sin θ). Returns ``(W, {'theta'})``.
    """
    theta = 2.0 * np.pi * np.arange(cfg.N) / cfg.N
    diff = theta[:, None] - theta[None, :]
    W = (cfg.ring_J0 + cfg.ring_J1 * np.cos(diff)) / cfg.N
    return W, {"theta": theta}


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def make_input_direction(cfg: SubspaceRNNConfig, rng, aux: dict) -> Optional[np.ndarray]:
    """Fixed unit input direction ``b``. For low-rank nets it is aligned to the
    first loading ``m₁`` so the pulse steers the state onto the low-rank plane."""
    if not cfg.use_pulse or cfg.connectivity == "ring":
        return None
    if cfg.connectivity == "lowrank" and "m" in aux:
        b = aux["m"][:, 0].copy()
    else:
        b = rng.standard_normal(cfg.N)
    return b / (np.linalg.norm(b) + 1e-12)


def build_trial_input(cfg: SubspaceRNNConfig, rng, aux: dict, b: Optional[np.ndarray]):
    """Per-trial input current ``I`` of shape ``(T, N)`` plus an info dict.

    * pulse (random/lowrank): rectangular ``pulse_amp·b`` over
      ``[onset, onset + pulse_dur]``, onset ~ U(pulse_onset_range).
    * moving ring: a cosine bump ``A·cos(θ − φ(t))`` rotating at
      ``ω = 2π·revolutions/duration`` from a random initial phase φ₀.
    * otherwise: zeros.
    """
    T, N = cfg.n_steps, cfg.N
    t = cfg.time
    I = np.zeros((T, N))
    info: dict = {"onset": np.nan, "phi0": np.nan, "omega": np.nan}

    if cfg.connectivity == "ring" and cfg.ring_moving:
        theta = aux["theta"]
        phi0 = rng.uniform(0.0, 2 * np.pi)
        omega = 2 * np.pi * cfg.ring_revolutions / cfg.duration
        phi = phi0 + omega * t
        I = cfg.ring_stim_amp * np.cos(theta[None, :] - phi[:, None])
        info.update(phi0=phi0, omega=omega)
    elif b is not None:
        lo, hi = cfg.pulse_onset_range
        onset = rng.uniform(lo, hi)
        mask = (t >= onset) & (t < onset + cfg.pulse_dur)
        I[mask] = cfg.pulse_amp * b[None, :]
        info["onset"] = onset

    return I, info


# --------------------------------------------------------------------------- #
# Integrators
# --------------------------------------------------------------------------- #
def _drift(x, W, I_t):
    """RHS before dividing by τ:  f(x) = −x + W·tanh(x) + I_t."""
    return -x + W @ np.tanh(x) + I_t


def simulate_trial(cfg: SubspaceRNNConfig, W, I, x0, rng) -> np.ndarray:
    """Integrate one trial with input current ``I`` (T, N). Returns ``X`` (T, N).

    Euler–Maruyama when ``cfg.noise_std > 0``, else fixed-step RK4 on ``ẋ = f/τ``.
    """
    T, N, dt, tau = cfg.n_steps, cfg.N, cfg.dt, cfg.tau
    X = np.empty((T, N))
    X[0] = x0
    x = x0.copy()

    if cfg.noise_std > 0:
        sig = cfg.noise_std * np.sqrt(dt)
        for k in range(1, T):
            f = _drift(x, W, I[k - 1])
            x = x + (dt / tau) * f + sig * rng.standard_normal(N)
            X[k] = x
    else:
        for k in range(1, T):
            I0, I1 = I[k - 1], I[k]
            Im = 0.5 * (I0 + I1)
            k1 = _drift(x, W, I0) / tau
            k2 = _drift(x + 0.5 * dt * k1, W, Im) / tau
            k3 = _drift(x + 0.5 * dt * k2, W, Im) / tau
            k4 = _drift(x + dt * k3, W, I1) / tau
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            X[k] = x
    return X


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def make_dataset(cfg: SubspaceRNNConfig) -> dict:
    """Generate the full dataset as a plain dict (the pre-contract form).

    Keys: ``X`` (n_trials, T, N), ``inputs`` (n_trials, T, N), ``onsets``,
    ``phi0``, ``omega``, ``W``, ``b``, ``aux``, ``time``, ``config``. Prefer
    :func:`make_trajectory` unless you specifically want the raw dict.
    """
    rng = np.random.default_rng(cfg.seed)

    aux: dict = {}
    if cfg.connectivity == "random":
        W = make_random_connectivity(cfg, rng)
    elif cfg.connectivity == "lowrank":
        W, aux = make_lowrank_connectivity(cfg, rng)
    elif cfg.connectivity == "ring":
        W, aux = make_ring_connectivity(cfg)
    else:
        raise ValueError(f"Unknown connectivity: {cfg.connectivity!r}")

    b = make_input_direction(cfg, rng, aux)

    T = cfg.n_steps
    X = np.empty((cfg.n_trials, T, cfg.N))
    inputs = np.empty((cfg.n_trials, T, cfg.N))
    onsets = np.full(cfg.n_trials, np.nan)
    phi0 = np.full(cfg.n_trials, np.nan)
    omega = np.nan

    for i in range(cfg.n_trials):
        x0 = cfg.ic_std * rng.standard_normal(cfg.N)
        I, info = build_trial_input(cfg, rng, aux, b)
        inputs[i] = I
        onsets[i] = info["onset"]
        phi0[i] = info["phi0"]
        omega = info["omega"]
        X[i] = simulate_trial(cfg, W, I, x0, rng)

    return {
        "X": X, "inputs": inputs, "onsets": onsets, "phi0": phi0,
        "omega": omega, "W": W, "b": b, "aux": aux,
        "time": cfg.time, "config": asdict(cfg),
    }


def make_trajectory(cfg: SubspaceRNNConfig) -> Trajectory:
    """Generate a dataset and return it as a :class:`Trajectory` (the contract).

    The per-trial input current is kept as ``Trajectory.inputs``; ``W`` and the
    connectivity-specific extras (loadings ``m``, generator ``G``, ring angles
    ``theta``, per-trial ``onsets``/``phi0``) travel in ``W``/``aux``; the full
    config and ``omega`` are recorded in ``meta``.
    """
    d = make_dataset(cfg)
    return Trajectory.from_synth_dict(d, generator="subspace_rnn")


# --------------------------------------------------------------------------- #
# Diagnostics / regime presets
# --------------------------------------------------------------------------- #
def population_vector_angle(X_final: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Ring networks: angle of the population activity bump per trial, from the
    final-state population vector. Returns ``(n_trials,)``."""
    r = np.tanh(X_final)
    c = r @ np.cos(theta)
    s = r @ np.sin(theta)
    return np.arctan2(s, c)


def build_specs(regime: str) -> dict:
    """Return ``{name: SubspaceRNNConfig}`` for the three families in a regime.

    ``regime="settling"``: symmetric low-rank + static-bump ring (point
    attractors). ``regime="moving"``: rotation low-rank + rotating-stimulus ring
    (genuinely 2-D single-trial loops; reliable k=2).
    """
    random_cfg = SubspaceRNNConfig(connectivity="random", n_trials=30, seed=1)
    if regime == "settling":
        lowrank_cfg = SubspaceRNNConfig(connectivity="lowrank", n_trials=30, rank=2,
                                        lowrank_mode="symmetric", seed=2)
        ring_cfg = SubspaceRNNConfig(connectivity="ring", n_trials=60,
                                     ring_moving=False, seed=3)
    elif regime == "moving":
        lowrank_cfg = SubspaceRNNConfig(connectivity="lowrank", n_trials=30, rank=2,
                                        lowrank_mode="rotation", lowrank_rot_angle=0.5,
                                        seed=2)
        ring_cfg = SubspaceRNNConfig(connectivity="ring", n_trials=60, ring_moving=True,
                                     ring_revolutions=1.0, ring_stim_amp=1.0, seed=3)
    else:
        raise ValueError(regime)
    return {"random": random_cfg, "lowrank": lowrank_cfg, "ring": ring_cfg}
