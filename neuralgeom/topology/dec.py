"""
neuralgeom.topology.dec — Discrete Exterior Calculus on the pooled subspace manifold.
=====================================================================================

**Optional layer** (ProjectiveSpaceModels Step 5). Builds a 2-D simplicial
complex approximating the Grassmannian submanifold a network explores (pooled
across trials), then computes the intrinsic exterior calculus of scalar 0-fields
on it with ``dxtr``:

    df   = exterior_derivative(f)   gradient 1-form (per edge)
    Δf   = laplacian(f)             Laplace–de Rham 0-form (sources / sinks)
    b0,b1,b2  simplicial Betti numbers from oriented boundary-matrix ranks

plus the **cross-projection matrix** ``R[t, j] = ‖Uⱼᵀx(t)‖² / ‖x(t)‖²`` whose
diagonal is self-capture and whose off-diagonal stripes image loop recurrence.

Honest caveat (kept from the source repo): the MDS embedding used to build the
complex is non-isometric, so Laplacian magnitudes are inflated and readable only
*qualitatively*; ``df`` and the Betti numbers are robust. The load-bearing
topology is persistent homology (:mod:`neuralgeom.topology.persistence`), which
needs no embedding.

Dependencies: the complex construction (:func:`build_complex`) and
:func:`betti_from_triangles` need only ``scipy`` + ``scikit-learn`` (core). The
DEC operators (:func:`dec_scalar`, :func:`make_manifold`) need ``dxtr`` — the
optional ``[dec]`` extra — imported lazily with a clear error if missing.

Gotcha carried over: the PyPI package ``pydec`` is **not** the
discrete-exterior-calculus PyDEC — it is an unrelated ML library. Use ``dxtr``;
never add ``pydec``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DECConfig", "build_complex", "betti_from_triangles",
           "make_manifold", "dec_scalar", "cross_projection"]


@dataclass
class DECConfig:
    """Complex-construction options.

    n_pool    : pooled vertices (subsample); passed through to pooling.
    mds_dim   : MDS embedding dimension (2 for a triangulable disk).
    prune_pct : long-edge prune percentile for the secondary "hole hint"
                (threshold-sensitive; the DEC operators always run on the full
                Delaunay disk). None disables pruning.
    seed      : RNG / MDS seed.
    """

    n_pool: int = 350
    mds_dim: int = 2
    prune_pct: float = 92.0
    seed: int = 0


def _dxtr():
    try:
        import dxtr  # noqa: F401
        from dxtr import SimplicialManifold, Cochain
        from dxtr.operators import exterior_derivative, laplacian
        try:
            dxtr.logger.setLevel("ERROR")
        except Exception:
            pass
        return SimplicialManifold, Cochain, exterior_derivative, laplacian
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The DEC layer needs the optional 'dec' extra: "
            "pip install 'neuralgeom[dec]' (dxtr). Note: 'pydec' on PyPI is a "
            "DIFFERENT, unrelated package — do not install it."
        ) from e


def build_complex(frames, cfg: DECConfig = None):
    """Geodesic distances → metric MDS(2-D) → Delaunay triangulation.

    Returns ``(XY, tris_full, tris_pruned, D, stress)``. Needs only scipy +
    scikit-learn (core dependencies).
    """
    from scipy.spatial import Delaunay
    from sklearn.manifold import MDS

    from ..geometry.grassmann import frame_distance_matrix

    cfg = cfg or DECConfig()
    D = frame_distance_matrix(frames, metric="canonical")
    mds = MDS(n_components=cfg.mds_dim, dissimilarity="precomputed",
              random_state=cfg.seed, normalized_stress="auto",
              n_init=2, max_iter=300)
    XY = mds.fit_transform(D)
    stress = float(mds.stress_)
    tris_full = Delaunay(XY).simplices

    tris_pruned = tris_full
    if cfg.prune_pct is not None:
        def tri_maxedge(T):
            e = [np.linalg.norm(XY[T[:, i]] - XY[T[:, j]], axis=1)
                 for i, j in [(0, 1), (1, 2), (2, 0)]]
            return np.max(e, axis=0)
        maxe = tri_maxedge(tris_full)
        thr = np.percentile(maxe, cfg.prune_pct)
        tris_pruned = tris_full[maxe <= thr]
    return XY, tris_full, tris_pruned, D, stress


def betti_from_triangles(triangles):
    """``(b0, b1, b2)`` from oriented simplicial boundary-matrix ranks.

    Self-contained (numpy only): ``b0 = V − rank(∂1)``,
    ``b1 = E − rank(∂1) − rank(∂2)``, ``b2 = F − rank(∂2)``.
    """
    tris = [tuple(sorted(int(v) for v in t)) for t in triangles]
    used = sorted({v for t in tris for v in t})
    remap = {v: i for i, v in enumerate(used)}
    tris = [tuple(remap[v] for v in t) for t in tris]
    V, F = len(used), len(tris)

    edges = sorted({tuple(sorted((t[a], t[b])))
                    for t in tris for a, b in [(0, 1), (1, 2), (0, 2)]})
    eidx = {e: i for i, e in enumerate(edges)}
    E = len(edges)

    d1 = np.zeros((V, E))
    for e, (i, j) in enumerate(edges):
        d1[i, e], d1[j, e] = -1.0, 1.0
    d2 = np.zeros((E, F))
    for f, (i, j, k) in enumerate(tris):
        d2[eidx[(i, j)], f] += 1.0
        d2[eidx[(i, k)], f] += -1.0
        d2[eidx[(j, k)], f] += 1.0

    r1 = int(np.linalg.matrix_rank(d1)) if E else 0
    r2 = int(np.linalg.matrix_rank(d2)) if F else 0
    return V - r1, E - r1 - r2, F - r2


def make_manifold(verts2d, triangles):
    """Build a ``dxtr.SimplicialManifold`` from 2-D vertices (embedded at z=0),
    reindexed to the vertices actually used. Returns ``(manifold, used)``."""
    SimplicialManifold, _, _, _ = _dxtr()
    used = np.unique(triangles)
    remap = {int(v): i for i, v in enumerate(used)}
    tris = [[remap[int(v)] for v in t] for t in triangles]
    V = np.c_[np.asarray(verts2d)[used], np.zeros(len(used))]
    return SimplicialManifold(tris, V), used


def dec_scalar(manifold, f_vals):
    """Exterior derivative and Laplacian of a scalar 0-field on a dxtr manifold.

    Returns ``(df_cochain, df_array, laplacian_array)``.
    """
    _, Cochain, exterior_derivative, laplacian = _dxtr()
    f = Cochain(manifold, 0, np.asarray(f_vals, float))
    df = exterior_derivative(f)
    df_arr = np.asarray(df.toarray(), float).ravel()
    try:
        lap_arr = np.asarray(laplacian(f).toarray(), float).ravel()
    except Exception:
        lap_arr = np.full(len(f_vals), np.nan)
    return df, df_arr, lap_arr


def cross_projection(traj, trial: int = 0, k: int = 1, cfg=None) -> dict:
    """Cross-projection matrix ``R[t, j] = ‖Uⱼᵀx(t)‖² / ‖x(t)‖²`` for one trial.

    The diagonal is self-capture; off-diagonal stripes image loop recurrence.
    Returns ``{R, win_times, diagonal, centrality}``. Needs no dxtr.
    """
    from ..subspace.embed import EmbedConfig, embed_trajectory

    cfg = cfg or EmbedConfig(k=k)
    X, time = traj.trial(trial)
    emb = embed_trajectory(X, cfg)
    F = emb["frames"]
    tc = np.asarray(time)[emb["win_centers"]]
    Xc = X[emb["win_centers"]]
    nrm2 = np.sum(Xc ** 2, axis=1) + 1e-12
    proj2 = np.stack([np.sum((Xc @ F[j]) ** 2, axis=1) for j in range(len(F))]).T
    R = proj2 / nrm2[:, None]
    return {"R": R, "win_times": tc,
            "diagonal": np.diag(R), "centrality": R.mean(axis=0)}
