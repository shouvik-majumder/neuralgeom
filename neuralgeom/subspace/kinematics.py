"""
neuralgeom.subspace.kinematics — Riemannian kinematics of a subspace trajectory.
================================================================================

Given a Grassmannian trajectory (frames from :mod:`neuralgeom.subspace.embed`),
describe how the subspace *moves*, intrinsically, using the manifold's own
log/exp/parallel-transport. Two layers, both ported from ProjectiveSpaceModels
(Steps 4 and 4b):

Basic kinematics (:func:`compute_kinematics`)
    velocity      ``v_t = Log_{P_t}(P_{t+1}) / Δt``      (tangent at P_t)
    speed         ``‖v_t‖ = dist(P_t, P_{t+1}) / Δt``
    acceleration  ``a_t = [v_t − PT_{t-1→t}(v_{t-1})] / Δt``  (covariant; 0 on a
                  geodesic, so ``‖a_t‖`` measures how the subspace *curves*)
    curvature     ``κ_t = ‖a_⊥‖ / speed²``
    efficiency    ``endpoint_dist / path_len``

Richer kinematics
    :func:`karcher_mean`          Fréchet mean on the manifold (iterated log/exp)
    :func:`tangent_pca`           "Grassmannian PCA": log-map to the Karcher-mean
                                  tangent space, PCA ⇒ modes of subspace variation
                                  + explained variance ⇒ intrinsic dimensionality
    :func:`chordal_geodesic`      chordal ``√Σsin²θ`` vs geodesic ``√Σθ²`` scatter
    :func:`transported_velocities` velocity field parallel-transported to the
                                  Karcher mean for a comparable tangent-plane flow

Manifold-agnostic by construction
----------------------------------
Every routine calls **only** the :ref:`Manifold interface` methods
(``dist``/``log``/``exp``/``norm``/``inner_product``/``parallel_transport``), so
the same code runs on the Grassmannian (default —
:class:`~neuralgeom.geometry.grassmann.GrassmannManifold`) or on any other
manifold implementing that interface (e.g. an SPD covariance manifold — the
planned companion lens). Pass your own ``manifold=`` to reuse it elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..geometry.grassmann import (GrassmannManifold, frame_to_projector,
                                   frame_principal_angles)

__all__ = ["KinConfig", "compute_kinematics", "karcher_mean", "tangent_pca",
           "chordal_geodesic", "transported_velocities"]


@dataclass
class KinConfig:
    """Kinematics options.

    accel  : "covariant" (parallel-transport the previous velocity, the
             intrinsic choice) or "coordinate" (naive log-vector difference).
    smooth : optional moving-average window on the velocity field (0 = none).
    """

    accel: str = "covariant"
    smooth: int = 0


def _moving_average(v, w):
    if w <= 1:
        return v
    kernel = np.ones(w) / w
    flat = v.reshape(len(v), -1)
    sm = np.vstack([np.convolve(flat[:, j], kernel, mode="same")
                    for j in range(flat.shape[1])]).T
    return sm.reshape(v.shape)


def _default_manifold(frames):
    _M, N, k = frames.shape
    return GrassmannManifold(N, k)


def compute_kinematics(frames: np.ndarray, win_times: np.ndarray,
                       cfg: KinConfig = None, *, manifold=None) -> dict:
    """Riemannian velocity / speed / covariant acceleration / curvature along a
    frame trajectory.

    frames    : (M, N, k) orthonormal frames.
    win_times : (M,) times (s) of each frame (sets Δt).
    manifold  : a Manifold implementing dist/log/exp/norm/inner_product/
                parallel_transport (default: Grassmannian on Gr(N, k)).

    Returns a dict of per-time arrays (NaN where undefined at the ends) plus the
    global summaries ``path_len``, ``endpoint_dist``, ``efficiency``, ``dt``.
    Identity check used by the tests: ``speed · dt == step_dist``.
    """
    cfg = cfg or KinConfig()
    frames = np.asarray(frames, float)
    M, N, k = frames.shape
    man = manifold if manifold is not None else _default_manifold(frames)
    P = np.stack([frame_to_projector(U) for U in frames])
    dt = float(np.mean(np.diff(win_times)))

    V = np.full((M, N, N), np.nan)
    speed = np.full(M, np.nan)
    step_dist = np.full(M, np.nan)
    for t in range(M - 1):
        v = man.log(P[t + 1], P[t]) / dt
        V[t] = v
        speed[t] = man.norm(v, P[t])
        step_dist[t] = man.dist(P[t], P[t + 1])

    if cfg.smooth > 1:
        V = _moving_average(V, cfg.smooth)

    acc_mag = np.full(M, np.nan)
    curvature = np.full(M, np.nan)
    for t in range(1, M - 1):
        if cfg.accel == "covariant":
            v_prev = man.parallel_transport(V[t - 1], P[t - 1], P[t])
            a = (V[t] - v_prev) / dt
        else:
            a = (V[t] - V[t - 1]) / dt
        acc_mag[t] = man.norm(a, P[t])
        s = speed[t]
        if s > 1e-9:
            vhat = V[t] / s
            a_par = man.inner_product(a, vhat, P[t]) * vhat
            a_perp = a - a_par
            curvature[t] = man.norm(a_perp, P[t]) / (s ** 2)

    path_len = float(np.nansum(step_dist))
    endpoint_dist = man.dist(P[0], P[-1])
    return {
        "time": np.asarray(win_times, float),
        "speed": speed, "step_dist": step_dist,
        "acc_mag": acc_mag, "curvature": curvature,
        "path_len": path_len, "endpoint_dist": endpoint_dist,
        "efficiency": endpoint_dist / (path_len + 1e-12), "dt": dt,
    }


def karcher_mean(projectors, manifold=None, iters: int = 30, tol: float = 1e-9):
    """Fréchet (Karcher) mean of points (projectors) on the manifold, via
    iterated tangent-averaging log/exp. ``manifold`` defaults to the Grassmannian
    inferred from the projector size."""
    projectors = list(projectors)
    if manifold is None:
        N = projectors[0].shape[0]
        # infer k from trace of the projector (tr P = k)
        k = int(round(float(np.trace(projectors[0]))))
        manifold = GrassmannManifold(N, max(k, 1))
    Mpt = np.asarray(projectors[0], float).copy()
    for _ in range(iters):
        tang = np.mean([manifold.log(P, Mpt) for P in projectors], axis=0)
        step = manifold.norm(tang, Mpt)
        Mpt = manifold.exp(tang, Mpt)
        if step < tol:
            break
    return Mpt


def tangent_pca(frames: np.ndarray, manifold=None, n_comp: int = 6) -> dict:
    """"Grassmannian PCA": log-map frames to the Karcher-mean tangent space and
    PCA the tangent vectors.

    Returns a dict with the mean projector ``mean``, per-frame PCA ``coords``
    (n, n_comp), the explained-variance ratio ``evr``, its cumulative sum
    ``cum_evr`` (⇒ intrinsic dimensionality = #components for a variance target),
    and the projector stack ``P`` (for :func:`transported_velocities`).
    """
    frames = np.asarray(frames, float)
    man = manifold if manifold is not None else _default_manifold(frames)
    P = np.stack([frame_to_projector(U) for U in frames])
    Mpt = karcher_mean(P, man)
    logs = np.array([np.asarray(man.log(Pi, Mpt)).ravel() for Pi in P])
    L = logs - logs.mean(0)
    Uc, S, _ = np.linalg.svd(L, full_matrices=False)
    coords = np.real(Uc * S)
    evr = S ** 2 / (np.sum(S ** 2) + 1e-12)
    return dict(mean=Mpt, coords=coords[:, :n_comp], evr=evr[:n_comp],
                cum_evr=np.cumsum(evr), P=P)


def chordal_geodesic(frames: np.ndarray, max_pairs: int = 4000, seed: int = 0):
    """Sampled ``(geodesic arc-length, chordal)`` distance pairs between frames,
    to show where the small-angle chordal approximation departs from the true
    geodesic. Returns two 1-D arrays ``(geo, cho)``."""
    frames = np.asarray(frames, float)
    rng = np.random.default_rng(seed)
    n = len(frames)
    ii = rng.integers(0, n, max_pairs)
    jj = rng.integers(0, n, max_pairs)
    geo, cho = [], []
    for i, j in zip(ii, jj):
        if i == j:
            continue
        th = frame_principal_angles(frames[i], frames[j])
        geo.append(np.sqrt(np.sum(th ** 2)))
        cho.append(np.sqrt(np.sum(np.sin(th) ** 2)))
    return np.array(geo), np.array(cho)


def transported_velocities(frames: np.ndarray, tp: dict, manifold=None) -> np.ndarray:
    """Velocity field ``v_t = log(P_{t+1}, P_t)`` parallel-transported to the
    Karcher mean and expressed in the top-2 tangent-PCA directions — a
    comparable tangent-plane flow field. ``tp`` is the dict from
    :func:`tangent_pca`. Returns ``(M-1, 2)`` arrow coordinates."""
    man = manifold if manifold is not None else _default_manifold(frames)
    P = tp["P"]
    Mpt = tp["mean"]
    logs = np.array([np.asarray(man.log(Pi, Mpt)).ravel() for Pi in P])
    L = logs - logs.mean(0)
    _, _, Vt = np.linalg.svd(L, full_matrices=False)
    basis = Vt[:2]
    arrows = []
    for t in range(len(P) - 1):
        v = man.log(P[t + 1], P[t])
        vt = man.parallel_transport(v, P[t], Mpt)
        arrows.append(np.real(basis @ np.asarray(vt).ravel()))
    return np.real(np.array(arrows))
