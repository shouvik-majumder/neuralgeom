"""
neuralgeom.subspace — the subspace-trajectory lens (Grassmannian embedding + kinematics).
=========================================================================================

Track the k-dimensional subspace a high-dimensional trajectory locally occupies
as a point on the Grassmannian ``Gr(k, N)`` (``Gr(1, N) = ℝPᴺ⁻¹``), and study how
that point *moves*. Ported from ProjectiveSpaceModels, refactored onto the
:class:`~neuralgeom.data.trajectory.Trajectory` contract and the shared
Grassmannian geometry in :mod:`neuralgeom.geometry.grassmann`.

    embed.py       sliding-window SVD → frames on Gr(k, N); frame reliability
                   (sv_gap); subspace-drift curves
    kinematics.py  Riemannian speed / covariant acceleration / curvature;
                   Karcher mean; tangent-PCA (intrinsic dimensionality);
                   chordal-vs-geodesic; transported velocity field
    pooling.py     pool frames + scalar fields across trials (feeds topology/DEC)

Everything here is manifold-agnostic (it calls only the Manifold interface), so
the same pipeline applies to any manifold — the Grassmannian by default, an SPD
covariance manifold as the planned companion lens.

The topological half of this lens (persistent homology, bottleneck, DEC) lives
in :mod:`neuralgeom.topology`.
"""
from .embed import (EmbedConfig, embed_trajectory, embed_from_trajectory,
                    frames_distance_matrix, subspace_drift)
from .kinematics import (KinConfig, compute_kinematics, karcher_mean,
                         tangent_pca, chordal_geodesic, transported_velocities)
from .pooling import PoolConfig, pool_frames, reconstruct_drive

__all__ = [
    "EmbedConfig", "embed_trajectory", "embed_from_trajectory",
    "frames_distance_matrix", "subspace_drift",
    "KinConfig", "compute_kinematics", "karcher_mean", "tangent_pca",
    "chordal_geodesic", "transported_velocities",
    "PoolConfig", "pool_frames", "reconstruct_drive",
]
