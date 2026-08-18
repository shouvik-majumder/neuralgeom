"""
quickstart.py — the smallest end-to-end tour of both lenses.
============================================================

Run:  python examples/quickstart.py
Needs the optional extras for the subspace half:  pip install 'neuralgeom[geom,topology]'
(The pullback half needs only the core install.)
"""
import numpy as np

# --- the shared Trajectory contract ---------------------------------------- #
from neuralgeom.synth.subspace_rnn import SubspaceRNNConfig, make_trajectory

traj = make_trajectory(SubspaceRNNConfig(
    connectivity="ring", ring_moving=True, N=40, n_trials=4,
    duration=1.0, dt=2e-3, noise_std=0.05, seed=3))
print("Trajectory:", traj)                       # one object every analysis consumes

# --- subspace lens: embed -> kinematics -> topology ------------------------ #
from neuralgeom.subspace import (EmbedConfig, embed_from_trajectory,
                                 compute_kinematics, tangent_pca)
from neuralgeom.topology.persistence import single_trial_distances, ph, top_life

emb = embed_from_trajectory(traj, 0, EmbedConfig(k=1, win=40, stride=8))
kin = compute_kinematics(emb["frames"], emb["win_times"])
tp = tangent_pca(emb["frames"])
D = single_trial_distances(traj, 0, EmbedConfig(k=1, win=40, stride=8))
print(f"subspace lens: efficiency={kin['efficiency']:.2f}  "
      f"tangent-dim(90%)={int(np.searchsorted(tp['cum_evr'], 0.90) + 1)}  "
      f"H1 loop persistence={top_life(ph(D, maxdim=1)[1]):.2f}")

# --- pullback lens: feed-forward geometry of a tiny net -------------------- #
import torch
import torch.nn as nn
from neuralgeom.geometry import PullbackGeometry

net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 8))
geo = PullbackGeometry(net)
X = torch.randn(16, 3)
vol = geo.volume_element(X)
print(f"pullback lens: mean local volume magnification √det g = "
      f"{float(vol.mean()):.3g}")
