"""
demo_subspace_pipeline.py — end-to-end demo of the subspace / topology pipeline.
================================================================================

Runs generate → embed → kinematics → persistent homology on synthetic rate
RNNs. For each connectivity family (random / low-rank / ring) in each regime
(stationary / rotating) it:

  1. generates a synthetic rate-RNN trajectory
     (:mod:`neuralgeom.synth.subspace_rnn`),
  2. embeds each trial onto the Grassmannian ``Gr(k, N)``
     (:mod:`neuralgeom.subspace.embed`),
  3. computes Riemannian kinematics and tangent-PCA intrinsic dimension
     (:mod:`neuralgeom.subspace.kinematics`),
  4. runs persistent homology on the single-trial and pooled distance matrices
     (:mod:`neuralgeom.topology.persistence`),
  5. saves a 3-panel overview figure per condition and prints a summary table.

Expected qualitative result: the
*rotating ring* traces a persistent H1 loop at k=1 (ℝP¹); *stationary* ring loops
live only in the pooled (across-trial) cloud; chaotic *random* shows no dominant
loop. Run with ``--quick`` for a fast smoke run.

    python scripts/subspace/demo_subspace_pipeline.py            # default
    python scripts/subspace/demo_subspace_pipeline.py --quick    # tiny & fast

Requires the optional extras: ``pip install 'neuralgeom[geom,topology]'``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neuralgeom.paths import fig_dir
from neuralgeom.synth.subspace_rnn import SubspaceRNNConfig, build_specs
from neuralgeom.subspace import (EmbedConfig, embed_from_trajectory, KinConfig,
                                 compute_kinematics, tangent_pca, PoolConfig)
from neuralgeom.subspace.embed import distance_from_start
from neuralgeom.topology.persistence import (persistent_homology, max_persistence,
                                             within_trial_distances,
                                             across_trial_distances)

NETS = ["random", "lowrank", "ring"]
REGIMES = ["stationary", "rotating"]


def _shrink(cfg: SubspaceRNNConfig, quick: bool) -> SubspaceRNNConfig:
    """Make the presets small/fast for a demo (fewer units, shorter, coarser)."""
    cfg.N = 30 if quick else 50
    cfg.dt = 2e-3
    cfg.duration = 1.0 if quick else 1.5
    cfg.n_trials = 4 if quick else (cfg.n_trials // 2 or 4)
    cfg.noise_std = 0.05
    return cfg


def run_condition(net, regime, cfg, k, figdir, quick):
    traj = None
    from neuralgeom.synth.subspace_rnn import make_trajectory
    traj = make_trajectory(cfg)

    emb = embed_from_trajectory(traj, 0, EmbedConfig(k=k, win=40, stride=8))
    kin = compute_kinematics(emb["frames"], emb["win_times"], KinConfig())
    tp = tangent_pca(emb["frames"])
    dim90 = int(np.searchsorted(tp["cum_evr"], 0.90) + 1)

    Ds = within_trial_distances(traj, 0, EmbedConfig(k=k, win=40, stride=8))
    Dp = across_trial_distances(traj, PoolConfig(k=k, n_pool=(80 if quick else 150),
                                           fields=False))
    h1_single = max_persistence(persistent_homology(Ds, maxdim=1)[1])
    h1_pooled = max_persistence(persistent_homology(Dp, maxdim=1)[1])

    # 3-panel overview: state PCA(2) | subspace drift | single-trial recurrence
    tag = f"{net}_{regime}"
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    X0 = traj.X[0]
    from numpy.linalg import svd
    Xc = X0 - X0.mean(0)
    U, S, Vt = svd(Xc, full_matrices=False)
    pc = Xc @ Vt[:2].T
    sc = ax[0].scatter(pc[:, 0], pc[:, 1], c=traj.time, cmap="viridis", s=6)
    ax[0].plot(pc[:, 0], pc[:, 1], color="0.85", lw=0.5)
    fig.colorbar(sc, ax=ax[0], label="time (s)")
    ax[0].set_title(f"state PCA(2)  {tag}"); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")

    d0 = distance_from_start(emb["frames"])
    ax[1].plot(emb["win_times"], d0, color="C0")
    ax[1].set_title(f"distance from initial subspace  Gr({k},N)")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("geodesic dist from start")

    im = ax[2].imshow(Ds, origin="lower", cmap="magma",
                      extent=[emb["win_times"][0], emb["win_times"][-1],
                              emb["win_times"][0], emb["win_times"][-1]])
    fig.colorbar(im, ax=ax[2], label="geodesic dist")
    ax[2].set_title(f"recurrence  |  H1(single)={h1_single:.2f}")
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("time (s)")

    fig.suptitle(f"{tag}  |  Gr({k}, {traj.N})  |  endpoint/path length={kin['endpoint_to_path_length_ratio']:.2f}  "
                 f"|  tangent-dim(90%)={dim90}  |  H1 single={h1_single:.2f} "
                 f"pooled={h1_pooled:.2f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = figdir / f"subspace_demo_{tag}_k{k}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return dict(tag=tag, k=k, endpoint_to_path_length_ratio=kin["endpoint_to_path_length_ratio"], dim90=dim90,
                h1_single=h1_single, h1_pooled=h1_pooled,
                med_sv_gap=float(np.median(emb["sv_gap"])), fig=str(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny, fast smoke run")
    ap.add_argument("--k", type=int, default=1, help="subspace dimension")
    args = ap.parse_args()

    figdir = fig_dir("demos")
    print(f"{'condition':18s} {'k':>2s} {'end/path':>8s} {'tdim':>5s} "
          f"{'H1_single':>10s} {'H1_pooled':>10s} {'sv_gap':>7s}")
    rows = []
    for regime in REGIMES:
        specs = build_specs(regime)
        for net in NETS:
            cfg = _shrink(specs[net], args.quick)
            r = run_condition(net, regime, cfg, args.k, figdir, args.quick)
            rows.append(r)
            print(f"{r['tag']:18s} {r['k']:>2d} {r['endpoint_to_path_length_ratio']:8.2f} "
                  f"{r['dim90']:>5d} {r['h1_single']:10.2f} {r['h1_pooled']:10.2f} "
                  f"{r['med_sv_gap']:7.2f}  -> {Path(r['fig']).name}", flush=True)
    print(f"\nFigures written to {figdir}")
    return rows


if __name__ == "__main__":
    main()
