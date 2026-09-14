"""
neuralgeom.geometry — geometry of differentiable maps and of their metrics.
==========================================================================

Given a differentiable map this package computes the induced (pullback)
geometry; given the resulting matrices it computes distances between them.
It is independent of any data source or task.

    jacobian.py        per-sample Jacobians, g = J^T J, volume elements, spectra
    manifold.py        the same for a LOW-dimensional domain: volume element,
                       anisotropy, Gaussian curvature, noise-weighted codomain
    spd.py             distances/means between metric tensors (SPD & PSD manifolds)
    grassmann.py       distances/means between SUBSPACES (Grassmannian)
    output_metrics.py  codomain metrics g_Y to pull back: Euclidean, scaled
                       Gaussian, and the Fisher-Rao families (Bernoulli,
                       categorical, Gaussian mean+var, (mu, log sigma),
                       log-scale/Weber), plus block composition
    maps.py            ReadoutMap wrappers f: X -> Y with Jacobians (linear,
                       any callable via finite differences, sklearn MLP with
                       the exact analytic Jacobian)
    torch_readouts.py  torch readouts with exact autograd Jacobians (the
                       heteroscedastic (mu, log sigma) MLP)
    pullback.py        PullbackMetric: g(x) = J^T g_Y J from a ReadoutMap +
                       OutputMetric, with spectral read-outs
    riemann.py         RiemannianField: Christoffels, geodesics, curvature of
                       any smooth metric field (finite differences)

Direction of the map
--------------------
rank(g) = dim(domain). A scalar-output map (e.g. decoding one behavioural
variable) gives rank-1 g: one direction, one magnitude, no volume and no
curvature. To obtain richer geometry put a low-dimensional manifold in the
DOMAIN and the high-dimensional representation in the codomain — see
``manifold.py``.
"""
from .jacobian import (ModelPullbackGeometry, batch_jacobian, log_volume_element,
                       metric_spectrum, euclidean_pullback_metric, volume_element)
from .manifold import (anisotropy, gaussian_curvature_2d, metric_summary,
                       noise_whitener, pullback_metric_field, whiten_map)
from .manifold import volume_element as domain_volume_element
from .spd import (affine_invariant_distance, bures_wasserstein_distance,
                  log_euclidean_distance, pairwise_spd_distance,
                  psd_fixed_rank_distance, regularize, spd_distance,
                  spd_frechet_mean, spectral_ratio_distance)
from .grassmann import (ModelSubspaceGeometry, grassmann_distance,
                        grassmann_frechet_mean, pairwise_grassmann_distance,
                        principal_angles, tangent_subspaces,
                        # numpy / geomstats frame-based API
                        frame_to_projector, projector_to_frame,
                        orthonormalize, frame_principal_angles, frame_distance,
                        frame_distance_matrix, GrassmannManifold,
                        check_frame_distance_against_geomstats)
from .output_metrics import (OutputMetric, Euclidean, ScaledGaussian,
                             BernoulliFisher, GaussianMeanVarFisher,
                             CategoricalFisher, LogScaleGaussian,
                             GaussianMuLogSigmaFisher, BlockOutputMetric)
from .maps import (ReadoutMap, LinearReadout, FunctionReadout, MLPReadout,
                   numeric_jacobian)
from .pullback import PullbackMetric
from .riemann import RiemannianField

__all__ = [
    "batch_jacobian", "euclidean_pullback_metric", "volume_element",
    "log_volume_element", "metric_spectrum", "ModelPullbackGeometry",
    "pullback_metric_field", "domain_volume_element", "anisotropy",
    "gaussian_curvature_2d", "noise_whitener", "whiten_map", "metric_summary",
    "affine_invariant_distance", "log_euclidean_distance",
    "bures_wasserstein_distance", "spectral_ratio_distance", "spd_distance",
    "pairwise_spd_distance", "spd_frechet_mean", "psd_fixed_rank_distance",
    "regularize",
    "principal_angles", "grassmann_distance", "pairwise_grassmann_distance",
    "grassmann_frechet_mean", "tangent_subspaces", "ModelSubspaceGeometry",
    "frame_to_projector", "projector_to_frame", "orthonormalize",
    "frame_principal_angles", "frame_distance", "frame_distance_matrix",
    "GrassmannManifold", "check_frame_distance_against_geomstats",
    "OutputMetric", "Euclidean", "ScaledGaussian", "BernoulliFisher",
    "GaussianMeanVarFisher", "CategoricalFisher", "LogScaleGaussian",
    "GaussianMuLogSigmaFisher", "BlockOutputMetric",
    "ReadoutMap", "LinearReadout", "FunctionReadout", "MLPReadout",
    "numeric_jacobian", "PullbackMetric", "RiemannianField",
]
