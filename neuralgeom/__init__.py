"""
neuralgeom — differential geometry and topology of neural population activity.
==============================================================================

Tools for describing how neural representations and dynamics are organised
geometrically: the pullback metric of a differentiable map, the Grassmannian
trajectory of the subspace that activity occupies, and the persistent homology
of that trajectory. Every analysis consumes one data object, the
:class:`neuralgeom.data.Trajectory`, a batch of high-dimensional state
trajectories with their time axis and metadata.

Submodules:

    neuralgeom.geometry   pullback metrics of differentiable maps (Jacobians,
                          output / Fisher–Rao metrics, readout maps, geodesics
                          and curvature, volume and anisotropy) and distances
                          between metric tensors (SPD) and between subspaces
                          (Grassmannian), for torch models and NumPy arrays
    neuralgeom.subspace   sliding-window Grassmannian embedding of a
                          trajectory, Riemannian kinematics (speed, covariant
                          acceleration, curvature), Karcher mean, tangent PCA,
                          across-trial pooling
    neuralgeom.topology   persistent homology (𝔽₂) on precomputed distance
                          matrices, bottleneck distances; optional discrete
                          exterior calculus and a comparison with state-space
                          homology
    neuralgeom.dynamics   geometry of recurrent dynamics (recurrent and input
                          Jacobians, fixed points, spectra) and estimation of
                          linear dynamics from data (instrumental-variable
                          correction, input inference, Helmholtz decomposition)
    neuralgeom.data       loaders and adapters (arrays, synthetic data,
                          recordings), the shared PCA state space, pluggable
                          dimensionality reduction, condition builders, and the
                          Trajectory object with its HDF5 schema
    neuralgeom.synth      synthetic data with known generating dynamics: a
                          two-attractor model, low-rank rate RNNs, and
                          connectivity-family rate RNNs
    neuralgeom.tasks      cognitive tasks and supervised RNN training
    neuralgeom.viz        plotting style, metric-field renderings, PDF reports

plus three small shared modules:

    neuralgeom.paths      data and output locations
    neuralgeom.fitting    differentiable readout maps and trial-wise
                          cross-validation
    neuralgeom.stats      permutation tests and shuffle constructors

Motivation
----------
For a differentiable map f, the pullback metric g = Jᵀ M J measures how f
distorts its domain: which directions are magnified, which are collapsed, and
how that varies from point to point. Since rank(g) = dim(domain), the choice
of what to place in the domain determines how rich the geometry can be.

Runnable analyses live in ``scripts/`` and tests in ``tests/``.
"""
from . import (data, dynamics, fitting, geometry, paths, stats, subspace,
               synth, tasks, topology, viz)

__all__ = ["geometry", "subspace", "topology", "data", "synth", "tasks",
           "dynamics", "viz", "paths", "fitting", "stats"]
__version__ = "0.5.0"
