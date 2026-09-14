"""
neuralgeom.topology — topology of subspace trajectories.
========================================================

Persistent homology and related tools for the Grassmannian trajectories
produced by :mod:`neuralgeom.subspace`.

    persistence.py  persistent homology on precomputed distance matrices
                    (𝔽₂ coefficients), pairwise bottleneck distances, and the
                    within-trial / across-trial distance matrices. This is the
                    primary topological result: it needs no embedding.
    dec.py          optional discrete exterior calculus on a simplicial
                    complex built from the pooled frames (needs the [dec]
                    extra, dxtr), plus the state–subspace projection matrix.
    direct.py       optional comparison against persistent homology of the
                    raw state-space distances (PCA / Isomap for display), as
                    a check against metric artefacts.

``persistence`` and ``direct`` need the [topology] extra (ripser, persim);
``dec`` additionally needs the [dec] extra (dxtr). All are imported lazily
with explicit errors, so ``import neuralgeom.topology`` works without them.
"""
from .persistence import (persistent_homology, summarize_diagrams, max_persistence, bottleneck_matrix,
                          within_trial_distances, across_trial_distances)
from . import dec
from . import direct

__all__ = ["persistent_homology", "summarize_diagrams", "max_persistence", "bottleneck_matrix",
           "within_trial_distances", "across_trial_distances", "dec", "direct"]
