"""
neuralgeom.synth — synthetic neural data with known generating dynamics.
====================================================================

Generate activity whose generating dynamics are known, so downstream methods
(neuralgeom.dynamics estimators, neuralgeom.geometry pullbacks) can be validated
against ground truth before touching real recordings.

    attractor.py    the two-attractor timing model
                    (Majumder et al.): a 2-D latent [cue mode, ramping mode]
                    with two Gaussian-well attractors and a brief cue pulse.
                    ``generate(mechanism, levels, ...)`` simulates, embeds
                    into n_neurons noisy 'neurons' (Gaussian or Poisson
                    counts, matchable to a target mean count), re-zeros time
                    to cue onset, and returns a plain dict ready for
                    ``neuralgeom.data.from_synthetic`` — with the noiseless
                    latent kept as ground truth. Mechanisms: ``input``
                    (fixed field, cue varies) and ``landscape`` (fixed cue,
                    field varies).
    lowrank_rnn.py  a continuous-time low-rank rate RNN emitting the shared
                    TrialData schema.
    fixtures.py     tiny latent -> behavior systems with known ground-truth
                    geometry, for validating the pullback metric.

Dependency rule: the dynamics
estimators consume a ``Session`` built by ``neuralgeom.data.from_synthetic`` from
the plain dict returned here — ``neuralgeom.dynamics`` never imports this
package, so recorded and synthetic data flow through the same interface.
(``tests/test_synth.py`` asserts the independence.)
"""

from .attractor import (generate, make_two_attractor_dataset, embed_rates,
                        simulate_batch, simulate_two_attractor_trial,
                        attractor_field, lick_attractor_pull_ratio_family,
                        CUE_AMPLITUDE_LEVELS, FIXED_CUE,
                        LICK_ATTRACTOR_X_PULL_LEVELS, LICK_ATTRACTOR_Y_PULL_LEVELS,
                        LICK_ATTRACTOR_PULL_RATIO_LEVELS,
                        LICK_ATTRACTOR_BEHAVIOUR_MATCHED_LEVELS,
                        T_CUE, CUE_DUR, FIELD_SCALE)
from .lowrank_rnn import RNNConfig, make_lowrank_connectivity, simulate, make_dataset
from .fixtures import make_synthetic_readout, sample_states, ramp_trajectories
# The connectivity-family rate-RNN generator.
# Exposed as a submodule alias to avoid name clashes with lowrank_rnn's
# RNNConfig/make_dataset — use ``neuralgeom.synth.subspace_rnn.make_trajectory``.
from . import subspace_rnn
from .subspace_rnn import SubspaceRNNConfig, build_specs

__all__ = [
    "generate", "make_two_attractor_dataset", "embed_rates", "simulate_batch",
    "simulate_two_attractor_trial", "attractor_field",
    "lick_attractor_pull_ratio_family",
    "CUE_AMPLITUDE_LEVELS", "FIXED_CUE",
    "LICK_ATTRACTOR_X_PULL_LEVELS", "LICK_ATTRACTOR_Y_PULL_LEVELS",
    "LICK_ATTRACTOR_PULL_RATIO_LEVELS", "LICK_ATTRACTOR_BEHAVIOUR_MATCHED_LEVELS",
    "T_CUE", "CUE_DUR", "FIELD_SCALE",
    "RNNConfig", "make_lowrank_connectivity", "simulate", "make_dataset",
    "make_synthetic_readout", "sample_states", "ramp_trajectories",
    "subspace_rnn", "SubspaceRNNConfig", "build_specs",
]
