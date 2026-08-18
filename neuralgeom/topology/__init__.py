"""
neuralgeom.topology — topology of subspace trajectories (persistent homology, DEC).
===================================================================================

The topological half of the subspace lens (ProjectiveSpaceModels Steps 5, 6,
3b), refactored onto the :class:`~neuralgeom.data.trajectory.Trajectory`
contract and the shared geometry layer.

    persistence.py  persistent homology on precomputed distance matrices
                    (metric-agnostic), 𝔽₂ coefficients, bottleneck fingerprints,
                    single-trial and pooled scopes — the load-bearing topology.
    dec.py          OPTIONAL discrete exterior calculus on the pooled manifold
                    (needs the [dec] extra: dxtr) + cross-projection matrix.
    direct.py       OPTIONAL direct state-manifold cross-check (PCA/Isomap + the
                    same persistent homology), guarding against metric artifacts.

``persistence`` and ``direct`` need the [topology] extra (ripser, persim);
``dec`` additionally needs the [dec] extra (dxtr). All are imported lazily with
clear errors, so ``import neuralgeom.topology`` works without any of them.
"""
from .persistence import (ph, summarize, top_life, bottleneck_matrix,
                          single_trial_distances, pooled_distances)
from . import dec
from . import direct

__all__ = ["ph", "summarize", "top_life", "bottleneck_matrix",
           "single_trial_distances", "pooled_distances", "dec", "direct"]
