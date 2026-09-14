"""
neuralgeom.topology.direct — persistent homology of the raw state manifold.
===========================================================================

Optional comparison. Estimates the neural state manifold directly in activity
space (PCA / Isomap for display) and runs the same persistent homology on the
Euclidean state-distance matrix, then compares the top H1 persistence and a
scale-normalised bottleneck distance against the Grassmannian ``k = 1``
diagram of the same trial.

A loop can be a property of the state trajectory, of the subspace trajectory,
or both. Agreement between the two metrics (extrinsic Euclidean state distance
and intrinsic Grassmannian geodesic distance) supports a detected loop and
guards against artefacts of either metric.

Dependencies: ``scikit-learn`` (core) for PCA / Isomap; persistent homology
and the bottleneck distance need the ``[topology]`` extra (ripser, persim).
The embeddings are used only for visualisation; the homology runs on
distances.
"""
from __future__ import annotations

import numpy as np

from .persistence import persistent_homology, max_persistence
from .persistence import within_trial_distances

__all__ = ["state_embeddings", "state_distances", "compare_state_and_subspace_homology"]


def state_embeddings(X_trial: np.ndarray, sub: int = 10, n_neighbors: int = 10):
    """PCA(3) and Isomap(2) embeddings of one trial's subsampled raw states.

    Returns ``(pca3, iso2, states_sub)``.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap

    states = np.asarray(X_trial, float)[::sub]
    pca3 = PCA(n_components=3).fit_transform(states - states.mean(0))
    iso2 = Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(states)
    return pca3, iso2, states


def state_distances(X_trial: np.ndarray, sub: int = 10) -> np.ndarray:
    """Euclidean state-distance matrix of one trial's subsampled states."""
    from scipy.spatial.distance import pdist, squareform

    states = np.asarray(X_trial, float)[::sub]
    return squareform(pdist(states))


def _scale(dgm):
    if len(dgm) == 0:
        return dgm
    finite = dgm[np.isfinite(dgm[:, 1])]
    m = np.nanmax(finite[:, 1]) if len(finite) else 1.0
    return dgm / (m + 1e-12)


def compare_state_and_subspace_homology(traj, trial: int = 0, sub: int = 10,
                                maxdim: int = 2) -> dict:
    """Compare the H1 diagrams of the Euclidean state distances and of the
    Grassmannian ``k=1`` geodesic distances for one trial.

    Returns ``{h1_state, h1_subspace, bottleneck, dgms_state, dgms_subspace}``:
    the two ``h1_*`` entries are the maximum H1 persistence under each metric,
    the ``dgms_*`` entries the full diagram lists, and ``bottleneck`` is the
    distance between the two H1 diagrams after each is scaled to unit maximum
    death (the metrics differ, so scales are normalised before comparison).
    """
    from persim import bottleneck

    from ..subspace.embed import EmbedConfig

    X, _ = traj.trial(trial)
    D_direct = state_distances(X, sub=sub)
    dgms_state = persistent_homology(D_direct, maxdim=maxdim, coeff=2)

    D_grass = within_trial_distances(traj, trial, EmbedConfig(k=1))
    dgms_subspace = persistent_homology(D_grass, maxdim=maxdim, coeff=2)

    h1_state = dgms_state[1] if len(dgms_state) > 1 else np.empty((0, 2))
    h1_subspace = dgms_subspace[1] if len(dgms_subspace) > 1 else np.empty((0, 2))
    bn = bottleneck(_scale(h1_state), _scale(h1_subspace))
    return {"h1_state": max_persistence(h1_state), "h1_subspace": max_persistence(h1_subspace),
            "bottleneck": float(bn), "dgms_state": dgms_state,
            "dgms_subspace": dgms_subspace}
