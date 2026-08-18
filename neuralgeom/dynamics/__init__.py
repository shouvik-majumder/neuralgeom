"""
neuralgeom.dynamics — geometry of recurrent dynamics.
=================================================

For a recurrent system the interesting map is the one-step update
h_{t+1} = F(h_t, x_t), and it has two derivatives worth taking:

    J_rec = dF/dh   how the state evolves on its own  -> memory, attractors
    J_inp = dF/dx   how input enters the state        -> gain, selection

Both are full rank in the state dimension, so ``g = J^T J`` here is a genuine
field over state space rather than the rank-1 object a scalar decoder gives.

Also here: ESTIMATING dynamics from data (``lds.py``) — three comparable
linear model classes (global LDS, sliding-window LDS, cubic field +
linearization), shared-field + per-condition-input inference, the
metric-aware Helmholtz (potential/rotational) split, and the
instrumental-variables correction for the finite-difference velocity bias
(feed ``neuralgeom.data.state_pca_split_half`` to ``fit_lds(instrument=...)``) —
plus a held-out regression suite (``regression.py``) and the autograd tools
(``torch_geometry.py``: nonlinear Helmholtz potential fit, inverse-metric
learner). Self-tests: ``python -m neuralgeom.dynamics.lds`` and
``python -m neuralgeom.dynamics.torch_geometry``.
"""
from .rnn import (SlowPoints, find_slow_points, input_jacobian,
                  jacobian_spectrum, participation_ratio, readout_subspace,
                  recurrent_jacobian, state_pullback_metric,
                  subspace_alignment, trajectory_subspaces, input_subspace)
from .lds import (trial_velocities, fit_lds, fit_sliding_lds, fit_cubic_field,
                  cubic_predict, linearize_cubic, cubic_fixed_point,
                  helmholtz_split, metric_dependence, potential,
                  fit_shared_input_free, fit_shared_input_lowrank,
                  velocity_budget, behavior_relevant_fraction,
                  shuffle_null_r2, r2_score_multi)
from .regression import run_regression_suite, ttl_order_analysis
from . import lds

__all__ = ["recurrent_jacobian", "input_jacobian", "jacobian_spectrum",
           "state_pullback_metric", "find_slow_points", "SlowPoints",
           "participation_ratio", "trajectory_subspaces", "readout_subspace",
           "input_subspace", "subspace_alignment",
           "trial_velocities", "fit_lds", "fit_sliding_lds",
           "fit_cubic_field", "cubic_predict", "linearize_cubic",
           "cubic_fixed_point", "helmholtz_split", "metric_dependence",
           "potential", "fit_shared_input_free", "fit_shared_input_lowrank",
           "velocity_budget", "behavior_relevant_fraction", "shuffle_null_r2",
           "r2_score_multi", "run_regression_suite", "ttl_order_analysis"]
