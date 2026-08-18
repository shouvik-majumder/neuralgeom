"""
neuralgeom.subspace.embed — sliding-window Grassmannian embedding (Step 3).
===========================================================================

Turns a single high-dimensional state trajectory ``X`` (T, N) into a *trajectory
of points on the Grassmannian* ``Gr(k, N)``: over each sliding window the top-k
right singular vectors of the windowed states form an orthonormal frame (N, k),
i.e. the k-dimensional subspace the activity locally occupies. Tracking that
subspace — rather than the raw state — is the central move of the
ProjectiveSpaceModels lens (``Gr(1, N) = ℝPᴺ⁻¹``, hence "projective space").

Load-bearing conventions (carried over verbatim — changing them silently is a
regression):

* ``center=False`` **is the default and matters**: an *uncentered* window SVD
  captures the subspace the activity *occupies* (stable); a *centered* window
  captures the local tangent/velocity subspace (noise-dominated at rest).
* ``sv_gap = σ_k / σ_{k+1}`` is the **frame-reliability diagnostic** — a k-frame
  is only trustworthy where this is well above 1. Report it alongside every
  kinematic/topology result.
* ``k`` acts as a *topological filter*: different ``k`` expose different
  structure, so analyse multiple ``k``.

The only data-dependent entry point is a plain array (or a
:class:`~neuralgeom.data.trajectory.Trajectory`), so the same embedding runs on
synthetic, trained-network, or real data unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.grassmann import frame_distance, frame_distance_matrix

__all__ = ["EmbedConfig", "embed_trajectory", "embed_from_trajectory",
           "frames_distance_matrix", "subspace_drift"]


@dataclass
class EmbedConfig:
    """Sliding-window embedding parameters.

    k       : subspace dimension (default 1 ⇒ projective space ℝPᴺ⁻¹).
    win     : window length in samples.
    stride  : step between windows in samples.
    center  : mean-center each window over time before the SVD (default False;
              see module docstring — False = occupied subspace).
    metric  : "canonical" (geomstats, √2 × arc-length) or "principal_angle".
    """

    k: int = 1
    win: int = 50
    stride: int = 10
    center: bool = False
    metric: str = "canonical"


def embed_trajectory(X: np.ndarray, cfg: EmbedConfig) -> dict:
    """Sliding-window SVD embedding of a single trial ``X`` (T, N).

    Returns a dict with:
        frames      (M, N, k)  orthonormal frames (points on Gr(k, N))
        win_centers (M,)       sample index of each window centre
        starts      (M,)       window start indices
        N           int        ambient dimension
        evr         (M,)       top-k variance fraction per window
        sv_gap      (M,)       σ_k / σ_{k+1} frame reliability per window
    """
    X = np.asarray(X, float)
    T, N = X.shape
    starts = np.arange(0, T - cfg.win + 1, cfg.stride)
    frames = np.empty((len(starts), N, cfg.k))
    centers = np.empty(len(starts), dtype=int)
    evr = np.empty(len(starts))
    sv_gap = np.empty(len(starts))

    for m, s in enumerate(starts):
        Wd = X[s:s + cfg.win]
        if cfg.center:
            Wd = Wd - Wd.mean(axis=0, keepdims=True)
        _, sv, Vt = np.linalg.svd(Wd, full_matrices=False)
        frames[m] = Vt[: cfg.k].T                        # (N, k), orthonormal
        centers[m] = s + cfg.win // 2
        sv2 = sv ** 2
        evr[m] = sv2[: cfg.k].sum() / (sv2.sum() + 1e-12)
        sv_gap[m] = (sv[cfg.k - 1] / (sv[cfg.k] + 1e-12)
                     if len(sv) > cfg.k else np.inf)

    return {"frames": frames, "win_centers": centers, "starts": starts,
            "N": N, "evr": evr, "sv_gap": sv_gap}


def embed_from_trajectory(traj, trial: int, cfg: EmbedConfig) -> dict:
    """Embed one trial of a :class:`~neuralgeom.data.trajectory.Trajectory`.

    Returns the same dict as :func:`embed_trajectory` plus ``win_times`` — the
    window-centre times in seconds — which the kinematics and topology layers
    use directly.
    """
    X, time = traj.trial(trial)
    emb = embed_trajectory(X, cfg)
    emb["win_times"] = np.asarray(time)[emb["win_centers"]]
    return emb


def frames_distance_matrix(frames: np.ndarray, metric: str = "canonical") -> np.ndarray:
    """Pairwise geodesic distance matrix (M, M) of a stack of frames.

    A convenience re-export of
    :func:`neuralgeom.geometry.grassmann.frame_distance_matrix`, the input to
    the topology layer and to MDS visualisations.
    """
    return frame_distance_matrix(frames, metric=metric)


def subspace_drift(frames: np.ndarray, metric: str = "canonical") -> np.ndarray:
    """Geodesic distance of each frame from the first frame — the "subspace
    drift" curve d(frame₀, frameₜ) that separates settling from moving/looping
    dynamics."""
    return np.array([frame_distance(frames[0], f, metric) for f in frames])
