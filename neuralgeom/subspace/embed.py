"""
neuralgeom.subspace.embed — sliding-window Grassmannian embedding.
==================================================================

Turns a single high-dimensional state trajectory ``X`` (T, N) into a sequence
of points on the Grassmannian ``Gr(k, N)``: over each sliding window the top-k
right singular vectors of the windowed states form an orthonormal frame (N, k),
i.e. the k-dimensional principal subspace of the activity in that window.
Tracking that subspace, rather than the raw state, is the construction the
rest of :mod:`neuralgeom.subspace` and :mod:`neuralgeom.topology` build on
(``Gr(1, N) = ℝPᴺ⁻¹``, real projective space).

Conventions
-----------
* ``center=False`` is the default. An uncentered window SVD returns the
  principal subspace of the second-moment matrix, i.e. the subspace the
  activity itself lies in. A centered window SVD returns the principal
  subspace of the window covariance, i.e. the local velocity / tangent
  subspace, which is noise-dominated when the state is nearly stationary.
* ``sv_gap = σ_k / σ_{k+1}`` is the singular-value (spectral) gap of each
  window. It bounds how well the k-dimensional subspace is determined
  (Davis–Kahan): a k-frame is well conditioned only where the gap is well
  above 1. Report it alongside every kinematic or topological result.
* The detected structure depends on ``k``. A rotating line produces a loop on
  ``Gr(1, N)`` but not on ``Gr(2, N)`` if the containing plane is static, so
  analyse several values of ``k``.

The entry points take a plain array or a
:class:`~neuralgeom.data.trajectory.Trajectory`, so the same embedding runs on
synthetic, trained-network, or recorded data unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.grassmann import frame_distance

__all__ = ["EmbedConfig", "embed_trajectory", "embed_from_trajectory",
           "distance_from_start"]


@dataclass
class EmbedConfig:
    """Sliding-window embedding parameters.

    k       : subspace dimension (default 1 ⇒ projective space ℝPᴺ⁻¹).
    win     : window length in samples.
    stride  : step between windows in samples.
    center  : mean-center each window over time before the SVD (default
              False: principal subspace of the second-moment matrix; see
              the module docstring).
    metric  : "sqrt2_principal_angle" (geomstats, √2 × arc-length) or "principal_angle".
    """

    k: int = 1
    win: int = 50
    stride: int = 10
    center: bool = False
    metric: str = "sqrt2_principal_angle"


def embed_trajectory(X: np.ndarray, cfg: EmbedConfig) -> dict:
    """Sliding-window SVD embedding of a single trial ``X`` (T, N).

    Returns a dict with:
        frames      (M, N, k)  orthonormal frames (points on Gr(k, N))
        win_centers (M,)       sample index of each window centre
        starts      (M,)       window start indices
        N           int        ambient dimension
        evr         (M,)       top-k variance fraction per window
        sv_gap      (M,)       σ_k / σ_{k+1} singular-value gap per window
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


def distance_from_start(frames: np.ndarray, metric: str = "sqrt2_principal_angle") -> np.ndarray:
    """Geodesic distance of each frame from the first frame, d(frame₀, frameₜ).

    A curve that rises and stays up indicates a subspace that moves to a new
    orientation; one that returns toward zero indicates a closed orbit."""
    return np.array([frame_distance(frames[0], f, metric) for f in frames])
