"""
neuralgeom.subspace — subspace trajectories on the Grassmannian.
================================================================

Track the k-dimensional subspace that a high-dimensional trajectory locally
occupies as a point on the Grassmannian ``Gr(k, N)`` (``Gr(1, N) = ℝPᴺ⁻¹``)
and describe how that point moves. Consumes a
:class:`~neuralgeom.data.trajectory.Trajectory` and the Grassmannian geometry
in :mod:`neuralgeom.geometry.grassmann`.

    embed.py       sliding-window SVD → frames on Gr(k, N); singular-value
                   gap; geodesic distance from the initial subspace
    kinematics.py  Riemannian speed / covariant acceleration / curvature;
                   Karcher mean; tangent PCA (intrinsic dimensionality);
                   chordal vs geodesic distances; transported velocity field
    pooling.py     pool frames and scalar fields across trials

The routines call only the generic manifold interface (dist / log / exp /
norm / inner_product / parallel_transport), so they apply to any Riemannian
manifold that exposes it; the Grassmannian is the default.

Topological analysis of these trajectories (persistent homology, bottleneck
distances, discrete exterior calculus) lives in :mod:`neuralgeom.topology`.
"""
from ..geometry.grassmann import frame_distance_matrix
from .embed import (EmbedConfig, embed_trajectory, embed_from_trajectory,
                    distance_from_start)
from .kinematics import (KinConfig, compute_kinematics, karcher_mean,
                         tangent_pca, chordal_vs_geodesic, transported_velocities)
from .pooling import PoolConfig, pool_frames, input_magnitude_at

__all__ = [
    "EmbedConfig", "embed_trajectory", "embed_from_trajectory",
    "frame_distance_matrix", "distance_from_start",
    "KinConfig", "compute_kinematics", "karcher_mean", "tangent_pca",
    "chordal_vs_geodesic", "transported_velocities",
    "PoolConfig", "pool_frames", "input_magnitude_at",
]
