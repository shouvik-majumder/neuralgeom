"""
quickstart.py — the smallest end-to-end example.
================================================

Run:  python examples/quickstart.py
The subspace / topology part needs the optional extras:
    pip install 'neuralgeom[geom,topology]'
The pullback-metric part needs only the core install.
"""
import numpy as np

# --- the shared data object: Trajectory ------------------------------------ #
from neuralgeom.synth.subspace_rnn import SubspaceRNNConfig, make_trajectory

traj = make_trajectory(SubspaceRNNConfig(
    connectivity="ring", ring_rotating=True, N=40, n_trials=4,
    duration=1.0, dt=2e-3, noise_std=0.05, seed=3))
print("Trajectory:", traj)                       # one object every analysis consumes

# --- subspace trajectory: embed -> kinematics -> persistent homology ------- #
from neuralgeom.subspace import (EmbedConfig, embed_from_trajectory,
                                 compute_kinematics, tangent_pca)
from neuralgeom.topology.persistence import within_trial_distances, persistent_homology, max_persistence

emb = embed_from_trajectory(traj, 0, EmbedConfig(k=1, win=40, stride=8))
kin = compute_kinematics(emb["frames"], emb["win_times"])
tp = tangent_pca(emb["frames"])
D = within_trial_distances(traj, 0, EmbedConfig(k=1, win=40, stride=8))
print(f"subspace trajectory: endpoint/path length={kin['endpoint_to_path_length_ratio']:.2f}  "
      f"tangent-dim(90%)={int(np.searchsorted(tp['cum_evr'], 0.90) + 1)}  "
      f"H1 loop persistence={max_persistence(persistent_homology(D, maxdim=1)[1]):.2f}")

# --- pullback metric: feed-forward geometry of a small network ------------- #
import torch
import torch.nn as nn
from neuralgeom.geometry import ModelPullbackGeometry

net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 8))
geo = ModelPullbackGeometry(net)
X = torch.randn(16, 3)
vol = geo.volume_element(X)
print(f"pullback metric: mean local volume magnification √det g = "
      f"{float(vol.mean()):.3g}")
