"""
neuralgeom.topology.direct — direct state-manifold topology cross-check.
========================================================================

**Optional cross-check** (ProjectiveSpaceModels Step 3b). Estimates the neural
state manifold *directly* in activity space (PCA / Isomap) and runs the **same**
persistent homology on the Euclidean state-distance matrix, then compares the
top-H1 persistence and a scale-normalised bottleneck distance against the
Grassmannian ``k = 1`` diagram of the same trial.

Purpose: a loop can be a state-space phenomenon, a subspace (Grassmannian)
phenomenon, or both. Agreement between the two very different metrics (extrinsic
Euclidean state distance vs intrinsic Grassmannian geodesic distance) is a strong
cross-validation of a detected loop and guards against metric artifacts.

Dependencies: ``scikit-learn`` (core) for PCA/Isomap; persistent homology and
the bottleneck need the ``[topology]`` extra (ripser, persim). UMAP is
deliberately not required — the persistent homology runs on distances, not on
the embedding, so the embedding is only for visualisation.
"""
from __future__ import annotations

import numpy as np

from .persistence import ph, top_life
from .persistence import single_trial_distances

__all__ = ["state_embeddings", "direct_distances", "compare_direct_vs_grassmann"]


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


def direct_distances(X_trial: np.ndarray, sub: int = 10) -> np.ndarray:
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


def compare_direct_vs_grassmann(traj, trial: int = 0, sub: int = 10,
                                maxdim: int = 2) -> dict:
    """Compare the direct state-space and Grassmannian ``k=1`` H1 diagrams for
    one trial.

    Returns ``{h1_direct, h1_grass, bottleneck, dgms_direct, dgms_grass}`` where
    the two ``h1_*`` are the top H1 lifetimes and ``bottleneck`` is between the
    scale-normalised H1 diagrams (the two use different metrics, so scales are
    normalised before comparison).
    """
    from persim import bottleneck

    from ..subspace.embed import EmbedConfig

    X, _ = traj.trial(trial)
    D_direct = direct_distances(X, sub=sub)
    dgms_direct = ph(D_direct, maxdim=maxdim, coeff=2)

    D_grass = single_trial_distances(traj, trial, EmbedConfig(k=1))
    dgms_grass = ph(D_grass, maxdim=maxdim, coeff=2)

    h1_direct = dgms_direct[1] if len(dgms_direct) > 1 else np.empty((0, 2))
    h1_grass = dgms_grass[1] if len(dgms_grass) > 1 else np.empty((0, 2))
    bn = bottleneck(_scale(h1_direct), _scale(h1_grass))
    return {"h1_direct": top_life(h1_direct), "h1_grass": top_life(h1_grass),
            "bottleneck": float(bn), "dgms_direct": dgms_direct,
            "dgms_grass": dgms_grass}
