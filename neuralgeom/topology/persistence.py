"""
neuralgeom.topology.persistence — persistent homology of subspace trajectories.
================================================================================

Runs persistent homology on a precomputed distance matrix, so it is
independent of the metric and the manifold: any distance matrix (Grassmannian
geodesic, SPD, Euclidean state distance, …) yields birth / death diagrams.
Homology is computed over 𝔽₂ (``coeff=2``), which detects real-projective /
non-orientable structure (``Gr(1, N) = ℝPᴺ⁻¹``).

    persistent_homology(D)     ripser on a precomputed distance matrix
    summarize_diagrams(dgms)   per-dimension feature counts and top lifetimes
    max_persistence(dgm)       largest finite lifetime in one diagram
    bottleneck_matrix(h1s)     pairwise H1 bottleneck distances between
                               conditions
    within_trial_distances     Gr(k, N) geodesic distances along one trial
    across_trial_distances     geodesic distances of the pooled frame cloud

Two point clouds are relevant: the frames of a single trial (loops traversed
within a trial) and the frames of all trials pooled (structure that only
appears across trials). Persistent homology needs no embedding, which is why
it is the primary topological result and the DEC layer is secondary.

``ripser`` and ``persim`` are the optional ``[topology]`` extra and are
imported lazily. Building the distance matrices needs only the core geometry
layer.
"""
from __future__ import annotations

import numpy as np

from ..geometry.grassmann import frame_distance_matrix
from ..subspace.embed import EmbedConfig, embed_trajectory
from ..subspace.pooling import PoolConfig, pool_frames

__all__ = ["persistent_homology", "summarize_diagrams", "max_persistence", "bottleneck_matrix",
           "within_trial_distances", "across_trial_distances"]


def _ripser():
    try:
        from ripser import ripser
        return ripser
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Persistent homology needs the optional 'topology' extra: "
            "pip install 'neuralgeom[topology]' (ripser, persim)."
        ) from e


def _bottleneck():
    try:
        from persim import bottleneck
        return bottleneck
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "bottleneck distance needs the optional 'topology' extra: "
            "pip install 'neuralgeom[topology]' (ripser, persim)."
        ) from e


def persistent_homology(D: np.ndarray, maxdim: int = 2, coeff: int = 2):
    """Persistent homology of a **precomputed distance matrix** ``D`` (M, M).

    Returns the list of persistence diagrams ``[H0, H1, …]`` (each an array of
    ``[birth, death]`` rows). ``coeff=2`` (𝔽₂) is the default because it
    detects non-orientable / projective structure.
    """
    ripser = _ripser()
    return ripser(np.asarray(D, float), distance_matrix=True,
                  maxdim=maxdim, coeff=coeff)["dgms"]


def summarize_diagrams(dgms) -> dict:
    """Per dimension → ``(n_finite_features, [top-2 lifetimes])``."""
    out = {}
    for dim, dg in enumerate(dgms):
        if len(dg) == 0:
            out[dim] = (0, [])
            continue
        life = dg[:, 1] - dg[:, 0]
        life = life[np.isfinite(life)]
        top = np.sort(life)[::-1][:2]
        out[dim] = (int(len(life)), [round(float(x), 3) for x in top])
    return out


def max_persistence(dgm) -> float:
    """Largest finite lifetime in one diagram (0 if empty)."""
    if dgm is None or len(dgm) == 0:
        return 0.0
    life = dgm[:, 1] - dgm[:, 0]
    life = life[np.isfinite(life)]
    return float(np.max(life)) if len(life) else 0.0


def bottleneck_matrix(h1_list) -> np.ndarray:
    """Symmetric matrix of pairwise H1 bottleneck distances between diagrams,
    a topological dissimilarity between conditions."""
    bottleneck = _bottleneck()
    n = len(h1_list)
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = bottleneck(h1_list[i], h1_list[j])
            B[i, j] = B[j, i] = d
    return B


def within_trial_distances(traj, trial: int = 0, cfg: EmbedConfig = None,
                           metric: str = "sqrt2_principal_angle") -> np.ndarray:
    """Geodesic distance matrix of the frames along one trial's Gr(k, N)
    trajectory."""
    cfg = cfg or EmbedConfig()
    X, _ = traj.trial(trial)
    emb = embed_trajectory(X, cfg)
    return frame_distance_matrix(emb["frames"], metric=metric)


def across_trial_distances(traj, cfg: PoolConfig = None,
                     metric: str = "sqrt2_principal_angle") -> np.ndarray:
    """Geodesic distance matrix of the frames of all trials pooled into one
    point cloud."""
    cfg = cfg or PoolConfig(fields=False)
    frames, _, _ = pool_frames(traj, cfg)
    return frame_distance_matrix(frames, metric=metric)
