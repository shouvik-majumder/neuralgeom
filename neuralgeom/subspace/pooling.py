"""
neuralgeom.subspace.pooling — pool subspace frames across trials.
=================================================================

Kinematics (:mod:`neuralgeom.subspace.kinematics`) are single-trial; some
structure only appears *across* trials (e.g. a ring's settling loop lives in the
across-trial cloud, not within one trajectory). This module pools the
Grassmannian frames of every trial of a
:class:`~neuralgeom.data.trajectory.Trajectory` into one point cloud, optionally
subsampled, and attaches per-frame scalar fields used by the DEC layer and as
colourings elsewhere:

    energy       instantaneous state energy  ``mean_i x_i(t)²``
    speed        Riemannian subspace speed ``dist(P_t, P_{t+1})/Δt``
    part_ratio   participation ratio of the window (effective dimensionality)
    input_drive  drive at the window centre (from ``inputs`` if present, else
                 reconstructed from the generator config/onsets — see below)
    selfcapture  ``‖Uᵀx‖² / ‖x‖²`` = fraction of state energy the frame captures
                 (= 1 − chordal_dist² to the state)

This logic was factored out of the original ProjectiveSpaceModels DEC module so
that the topology layer (persistent homology on the pooled cloud) does not
depend on the optional DEC/``dxtr`` extra.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.grassmann import frame_distance
from .embed import EmbedConfig, embed_trajectory

__all__ = ["PoolConfig", "pool_frames", "reconstruct_drive"]


@dataclass
class PoolConfig:
    """Pooling options.

    k        : subspace dimension for the embedding.
    embed    : the :class:`~neuralgeom.subspace.embed.EmbedConfig` to use
               (its ``k`` is overridden by ``k`` here for convenience).
    n_pool   : target number of pooled vertices (random subsample; None = keep all).
    seed     : RNG seed for the subsample.
    fields   : whether to compute the scalar fields (set False for topology-only).
    """

    k: int = 1
    embed: EmbedConfig = None
    n_pool: int = 350
    seed: int = 0
    fields: bool = True


def reconstruct_drive(traj, trial, t_center, inputs_trial=None) -> float:
    """Scalar input-drive at time ``t_center`` for one trial.

    Prefers the explicit ``inputs`` tensor if the trajectory carries one (mean
    absolute drive across units at that time); otherwise reconstructs from the
    generator's config + per-trial onset/phase stored in ``meta``/``aux`` — the
    original ProjectiveSpaceModels behaviour (bump phase for a moving ring, pulse
    amplitude within the pulse window, else 0).
    """
    if inputs_trial is not None:
        idx = int(np.argmin(np.abs(np.asarray(traj.time) - t_center)))
        return float(np.mean(np.abs(inputs_trial[idx])))

    cfg = traj.meta.get("config", {}) if isinstance(traj.meta, dict) else {}
    conn = cfg.get("connectivity", "")
    if conn == "ring" and cfg.get("ring_moving", False):
        phi0 = float(traj.aux.get("phi0", np.zeros(traj.n_trials))[trial])
        omega = float(traj.meta.get("omega", 0.0) or 0.0)
        return float((phi0 + omega * t_center) % (2 * np.pi))
    onsets = traj.aux.get("onsets")
    if onsets is not None and np.isfinite(onsets[trial]):
        amp = float(cfg.get("pulse_amp", 0.0) or 0.0)
        dur = float(cfg.get("pulse_dur", 0.0) or 0.0)
        onset = float(onsets[trial])
        return amp if (onset <= t_center < onset + dur) else 0.0
    return 0.0


def pool_frames(traj, cfg: PoolConfig = None):
    """Pool the subspace frames of every trial of ``traj`` into one cloud.

    Returns ``(frames, fields, meta)`` where ``frames`` is ``(n_pool, N, k)``,
    ``fields`` is a dict of ``(n_pool,)`` scalar arrays (empty if
    ``cfg.fields`` is False), and ``meta`` is the trajectory's metadata. NaN
    field values at trajectory ends are replaced by the field median.
    """
    cfg = cfg or PoolConfig()
    emb_cfg = cfg.embed or EmbedConfig(k=cfg.k, win=50, stride=10, center=False)
    emb_cfg = EmbedConfig(k=cfg.k, win=emb_cfg.win, stride=emb_cfg.stride,
                          center=emb_cfg.center, metric=emb_cfg.metric)

    time = np.asarray(traj.time)
    field_names = ["energy", "speed", "part_ratio", "input_drive", "selfcapture"]
    frames = []
    fields = {n: [] for n in field_names} if cfg.fields else {}

    for tr in range(traj.n_trials):
        Xtr = traj.X[tr]
        emb = embed_trajectory(Xtr, emb_cfg)
        F = emb["frames"]
        centers = emb["win_centers"]
        tcen = time[centers]
        inputs_tr = traj.inputs[tr] if traj.inputs is not None else None

        if cfg.fields:
            dt = float(np.mean(np.diff(tcen))) if len(tcen) > 1 else 1.0
            speed = np.full(len(F), np.nan)
            for m in range(len(F) - 1):
                speed[m] = frame_distance(F[m], F[m + 1], "canonical") / dt

        for m in range(len(F)):
            frames.append(F[m])
            if not cfg.fields:
                continue
            U = F[m]
            x = Xtr[centers[m]]
            nx2 = float(x @ x) + 1e-12
            proj2 = float(np.sum((U.T @ x) ** 2))
            Wd = Xtr[emb["starts"][m]:emb["starts"][m] + emb_cfg.win]
            Wc = Wd - Wd.mean(0, keepdims=True)
            ev = np.clip(np.linalg.eigvalsh(Wc.T @ Wc / Wd.shape[0]), 0, None)
            pr = (ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12)
            fields["energy"].append(float(np.mean(x ** 2)))
            fields["speed"].append(float(speed[m]))
            fields["part_ratio"].append(float(pr))
            fields["input_drive"].append(
                reconstruct_drive(traj, tr, tcen[m], inputs_tr))
            fields["selfcapture"].append(proj2 / nx2)

    frames = np.array(frames)
    fields = {n: np.array(v) for n, v in fields.items()}

    rng = np.random.default_rng(cfg.seed)
    if cfg.n_pool and len(frames) > cfg.n_pool:
        idx = rng.choice(len(frames), cfg.n_pool, replace=False)
        frames = frames[idx]
        fields = {n: v[idx] for n, v in fields.items()}
    for n, v in fields.items():
        m = np.isnan(v)
        if m.any():
            v[m] = np.nanmedian(v)
    return frames, fields, (traj.meta if isinstance(traj.meta, dict) else {})
