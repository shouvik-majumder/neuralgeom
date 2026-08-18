"""
neuralgeom.topology.persistence — persistent homology of subspace trajectories.
================================================================================

The topological half of the subspace lens (ProjectiveSpaceModels Step 6). Runs
persistent homology directly on a **precomputed geodesic distance matrix**, so
it is metric- and manifold-agnostic: give it any distance matrix (Grassmannian
geodesic, SPD, Euclidean state distance …) and it returns the birth/death
diagrams. Homology is computed over the field 𝔽₂ (``coeff=2``), the field that
exposes real-projective / non-orientable structure (``Gr(1, N) = ℝPᴺ⁻¹``).

    ph(D)                     ripser on a precomputed distance matrix → diagrams
    summarize(dgms)           per-dimension feature counts + top lifetimes
    top_life(dgm)             the single most persistent lifetime
    bottleneck_matrix(h1s)    pairwise H1 bottleneck distances (a topological
                              fingerprint dissimilarity between conditions)
    single_trial_distances    Gr(k,N) geodesic distances for one trial
    pooled_distances          across-trial pooled-cloud geodesic distances

Two scopes matter (per the conventions): ``single`` (a per-trial trajectory —
within-trial loops) and ``pooled`` (the across-trial cloud — loops that only
live across trials). Persistent homology needs no embedding, which is why it —
not the DEC layer — is the load-bearing topology.

``ripser`` and ``persim`` are the optional ``[topology]`` extra; they are
imported lazily with a clear error if missing. Building the distance matrices
needs only the (core) geometry layer.
"""
from __future__ import annotations

import numpy as np

from ..geometry.grassmann import frame_distance_matrix
from ..subspace.embed import EmbedConfig, embed_trajectory
from ..subspace.pooling import PoolConfig, pool_frames

__all__ = ["ph", "summarize", "top_life", "bottleneck_matrix",
           "single_trial_distances", "pooled_distances"]


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


def ph(D: np.ndarray, maxdim: int = 2, coeff: int = 2):
    """Persistent homology of a **precomputed distance matrix** ``D`` (M, M).

    Returns the list of persistence diagrams ``[H0, H1, …]`` (each an array of
    ``[birth, death]`` rows). ``coeff=2`` (𝔽₂) is the convention here — it
    exposes non-orientable / projective structure.
    """
    ripser = _ripser()
    return ripser(np.asarray(D, float), distance_matrix=True,
                  maxdim=maxdim, coeff=coeff)["dgms"]


def summarize(dgms) -> dict:
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


def top_life(dgm) -> float:
    """Largest finite lifetime in one diagram (0 if empty)."""
    if dgm is None or len(dgm) == 0:
        return 0.0
    life = dgm[:, 1] - dgm[:, 0]
    life = life[np.isfinite(life)]
    return float(np.max(life)) if len(life) else 0.0


def bottleneck_matrix(h1_list) -> np.ndarray:
    """Symmetric matrix of pairwise H1 bottleneck distances between diagrams —
    a topological-fingerprint dissimilarity between conditions."""
    bottleneck = _bottleneck()
    n = len(h1_list)
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = bottleneck(h1_list[i], h1_list[j])
            B[i, j] = B[j, i] = d
    return B


def single_trial_distances(traj, trial: int = 0, cfg: EmbedConfig = None,
                           metric: str = "canonical") -> np.ndarray:
    """Geodesic distance matrix of one trial's Gr(k, N) trajectory (the
    ``single`` scope)."""
    cfg = cfg or EmbedConfig()
    X, _ = traj.trial(trial)
    emb = embed_trajectory(X, cfg)
    return frame_distance_matrix(emb["frames"], metric=metric)


def pooled_distances(traj, cfg: PoolConfig = None,
                     metric: str = "canonical") -> np.ndarray:
    """Geodesic distance matrix of the across-trial pooled frame cloud (the
    ``pooled`` scope)."""
    cfg = cfg or PoolConfig(fields=False)
    frames, _, _ = pool_frames(traj, cfg)
    return frame_distance_matrix(frames, metric=metric)
