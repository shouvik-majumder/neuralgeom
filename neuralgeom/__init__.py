"""
neuralgeom — geometry & topology of neural representations and dynamics.
========================================================================

A unified library merging two exploratory repositories: **PullbackMetric**
(the pullback-metric / dynamics engine) and **ProjectiveSpaceModels** (the
subspace / Grassmannian-trajectory + topology lens). Both are two views of one
object — a batch of high-dimensional neural state trajectories — sharing the
:class:`neuralgeom.data.Trajectory` contract.

Submodules:

    neuralgeom.geometry   map-agnostic geometry toolbox (Jacobians, pullback
                      metrics, output/Fisher-Rao metrics, readout maps,
                      geodesics & curvature, volume/anisotropy, SPD and
                      Grassmannian distances — incl. the closed-form frame /
                      projector Grassmannian API used by the subspace lens)
    neuralgeom.subspace   the subspace lens: sliding-window Grassmannian
                      embedding, Riemannian kinematics (speed / covariant
                      acceleration / curvature), Karcher mean, tangent-PCA,
                      across-trial pooling — manifold-agnostic
    neuralgeom.topology   persistent homology (𝔽₂) on precomputed distance
                      matrices, bottleneck fingerprints; optional DEC (dxtr)
                      and direct-manifold cross-check
    neuralgeom.data       self-sufficient loader for the recordings + adapters
                      from arrays / synthetic dicts / TrialData, the shared
                      PCA state space (incl. the split-half instrument),
                      pluggable dimensionality reduction, condition builders
    neuralgeom.synth      synthetic neural data from KNOWN dynamics: the paper
                      2-attractor timing model and a low-rank rate RNN, with
                      the noiseless latent kept as ground truth
    neuralgeom.tasks      self-sufficient cognitive-task generation and RNN
                      training
    neuralgeom.dynamics   geometry of recurrent dynamics (recurrent/input
                      Jacobians, fixed points, spectra) AND estimation of
                      dynamics from data (LDS / cubic field, IV-corrected;
                      input inference; Helmholtz split)
    neuralgeom.viz        plotting style, PDF reports, rolling dashboard

plus three small shared modules:

    neuralgeom.paths      where data and outputs live (single source of truth)
    neuralgeom.fitting    the differentiable maps + trial-wise cross-validation
    neuralgeom.stats      permutation tests and shuffle constructors

THE ONE IDEA
------------
For a differentiable map f, the pullback metric g = J^T M J measures how f
distorts its domain: which directions are magnified, which are collapsed, and
how that varies from point to point. rank(g) = dim(domain), so the choice of
what to put in the domain determines how rich the geometry can be.

Runnable analyses live in ``scripts/`` and tests in ``tests/``.
"""
from . import (data, dynamics, fitting, geometry, paths, stats, subspace,
               synth, tasks, topology, viz)

__all__ = ["geometry", "subspace", "topology", "data", "synth", "tasks",
           "dynamics", "viz", "paths", "fitting", "stats"]
__version__ = "0.5.0"
