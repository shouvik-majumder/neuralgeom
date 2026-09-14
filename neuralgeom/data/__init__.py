"""
neuralgeom.data — loaders, adapters, dimensionality reduction and the Trajectory object.
========================================================================================

    loader.py     HDF5 -> Session (trials x time x units), epoch masks,
                  lick-time conditions. Verifies the response_type coding on
                  every load and drops fields known to be meaningless.
    adapters.py   alternative constructors for a Session: plain arrays, the synthetic
                  generator's dict (neuralgeom.synth), or a TrialData container —
                  plus the shared PCA state space and its split-half version
                  (the instrumental variable used to correct the finite-
                  difference velocity bias).
    reduce.py     pluggable dimensionality reduction behind one interface
                  (full space, PCA at fixed k or fixed retained variance,
                  random projection as a null, LDA)
    conditions.py condition/epoch builders for the timing task (lick-time
                  bins labelled by median lick time, epoch masks, trial masks,
                  lick alignment, the pre-model descriptive check)
    features.py   single-trial (features -> behavior) design matrices
    trial_data.py the TrialData container + HDF5 I/O (RNN-generator schema)

Depends only on numpy + scipy + scikit-learn + h5py.
"""
from .loader import (DROPPED_FIELDS, RESPONSE_TYPES, Session, list_sessions,
                     load_session, session_table)
from .adapters import (from_arrays, from_synthetic, from_trialdata,
                       state_pca, state_pca_split_half)
from .reduce import (Identity, LDA, PCA, PCAVariance, RandomProjection,
                     REDUCERS, Reducer, make_reducer)
from .conditions import (DEFAULT_EPOCHS, describe_conditions, epoch_masks,
                         lick_align, lick_time_bins, trial_masks)
from .trial_data import TrialData
from .trajectory import (Trajectory, load_trajectory, load_trial,
                         load_all_trials)

__all__ = ["Session", "load_session", "list_sessions", "session_table",
           "RESPONSE_TYPES", "DROPPED_FIELDS",
           "from_arrays", "from_synthetic", "from_trialdata",
           "state_pca", "state_pca_split_half",
           "Reducer", "Identity", "PCA", "PCAVariance", "RandomProjection",
           "LDA", "make_reducer", "REDUCERS",
           "DEFAULT_EPOCHS", "lick_time_bins", "epoch_masks", "trial_masks",
           "lick_align", "describe_conditions", "TrialData",
           "Trajectory", "load_trajectory", "load_trial", "load_all_trials"]
