"""
verify_all.py — run every branch of neuralgeom and report PASS / SKIP / FAIL.
=============================================================================

A single script that exercises each functional branch of the merged library so
you can confirm, on your own machine, that everything works. Branches that need
an optional extra you have not installed are SKIPped (not failed), with the
reason shown. Nothing here needs the bundled datasets — all data is synthetic.

    python examples/verify_all.py            # run everything available
    python examples/verify_all.py --quick    # smaller/faster

Exit code is non-zero iff any branch FAILED (skips do not fail the run).
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

RESULTS = []


def branch(name):
    """Decorator: run a branch, catch ImportError→SKIP, other errors→FAIL."""
    def deco(fn):
        def wrapped(quick):
            t0 = time.time()
            try:
                detail = fn(quick)
                RESULTS.append((name, "PASS", time.time() - t0, detail or ""))
            except ImportError as e:
                RESULTS.append((name, "SKIP", time.time() - t0,
                                f"missing extra: {str(e).splitlines()[0][:60]}"))
            except Exception as e:                       # noqa: BLE001
                RESULTS.append((name, "FAIL", time.time() - t0,
                                f"{type(e).__name__}: {e}"))
                traceback.print_exc()
        wrapped._branch_name = name
        return wrapped
    return deco


# --------------------------------------------------------------------------- #
# shared fixtures
# --------------------------------------------------------------------------- #
def _ring_traj(quick):
    from neuralgeom.synth.subspace_rnn import SubspaceRNNConfig, make_trajectory
    N = 24 if quick else 40
    return make_trajectory(SubspaceRNNConfig(
        connectivity="ring", ring_moving=True, N=N, n_trials=4,
        duration=1.0, dt=2e-3, noise_std=0.05, seed=3))


# --------------------------------------------------------------------------- #
# branches
# --------------------------------------------------------------------------- #
@branch("data.Trajectory contract (roundtrip + adapters)")
def b_contract(quick):
    import tempfile, os
    import numpy as np
    from neuralgeom.data import Trajectory, load_trajectory, load_trial
    X = np.random.default_rng(0).standard_normal((3, 20, 6))
    tr = Trajectory.from_arrays(X, dt=0.02, condition=np.arange(3), generator="test")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.h5"); tr.save(p)
        tr2 = load_trajectory(p); Xi, ti, meta = load_trial(p, 1)
        assert np.allclose(tr2.X, tr.X) and Xi.shape == (20, 6)
    return f"n_trials={tr.n_trials} N={tr.N} roundtrip OK"


@branch("synth generators (subspace_rnn + attractor + lowrank_rnn)")
def b_synth(quick):
    from neuralgeom.synth.subspace_rnn import SubspaceRNNConfig, make_trajectory
    from neuralgeom import synth
    tr = make_trajectory(SubspaceRNNConfig(connectivity="lowrank", N=20,
                                           n_trials=3, duration=0.6, dt=2e-3,
                                           lowrank_mode="rotation", seed=2))
    d = synth.generate("input", synth.INPUT_COND, n_trials=20, poisson=False)
    lr = synth.make_dataset(synth.RNNConfig(N=20, T=40), n_trials=4)
    return f"subspace X{tr.X.shape} · attractor keys={len(d)} · lowrank TrialData ok"


@branch("geometry.pullback (feed-forward g = JᵀJ, volume, spectrum)")
def b_pullback(quick):
    import torch, torch.nn as nn
    from neuralgeom.geometry import PullbackGeometry
    net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 8))
    geo = PullbackGeometry(net)
    X = torch.randn(12, 3)
    vol = geo.volume_element(X); ev, rank = geo.spectrum(X)
    return f"mean√detg={float(vol.mean()):.3g} rank={int(rank[0])}"


@branch("geometry.spd (tensor distances + Fréchet mean)")
def b_spd(quick):
    import torch
    from neuralgeom.geometry import spd_distance, spd_frechet_mean
    torch.manual_seed(0)
    mats = []
    for _ in range(5):
        m = torch.randn(4, 4)
        mats.append(m @ m.T + 2.0 * torch.eye(4))   # SPD
    A = torch.stack(mats)
    d = float(spd_distance(A[0], A[1], metric="affine"))
    mu = spd_frechet_mean(A, metric="log_euclidean")
    return f"affine d={d:.3f} · log-euclidean Fréchet mean shape={tuple(mu.shape)}"


@branch("geometry.grassmann — torch model API")
def b_grass_torch(quick):
    import torch, torch.nn as nn
    from neuralgeom.geometry import tangent_subspaces, grassmann_distance
    net = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 6))
    X = torch.randn(8, 4)
    Q, s, rank = tangent_subspaces(net, X, which="row", k=2)
    d = grassmann_distance(Q[0], Q[1], metric="geodesic")
    return f"subspaces {tuple(Q.shape)} geodesic d={float(d):.3f}"


@branch("geometry.grassmann — numpy/geomstats frame API")
def b_grass_frames(quick):
    from neuralgeom.geometry import validate_fast_matches_geomstats
    err = validate_fast_matches_geomstats(N=30, k=2, n_pairs=8, seed=0)
    return f"fast-path vs geomstats max err = {err:.1e}"


@branch("subspace (embed → kinematics → tangent-PCA)")
def b_subspace(quick):
    import numpy as np
    from neuralgeom.subspace import (EmbedConfig, embed_from_trajectory,
                                     compute_kinematics, tangent_pca)
    tr = _ring_traj(quick)
    emb = embed_from_trajectory(tr, 0, EmbedConfig(k=1, win=40, stride=8))
    kin = compute_kinematics(emb["frames"], emb["win_times"])
    err = np.nanmax(np.abs(kin["speed"] * kin["dt"] - kin["step_dist"]))
    tp = tangent_pca(emb["frames"])
    dim = int(np.searchsorted(tp["cum_evr"], 0.90) + 1)
    assert err < 1e-6
    return f"speed·dt==step_dist(err {err:.0e}) eff={kin['efficiency']:.2f} tdim={dim}"


@branch("topology.persistence (single + pooled PH, 𝔽₂)")
def b_topology(quick):
    from neuralgeom.subspace import EmbedConfig, PoolConfig
    from neuralgeom.topology.persistence import (single_trial_distances,
                                                 pooled_distances, ph, top_life)
    tr = _ring_traj(quick)
    Ds = single_trial_distances(tr, 0, EmbedConfig(k=1, win=40, stride=8))
    Dp = pooled_distances(tr, PoolConfig(k=1, n_pool=100, fields=False))
    loop = max(top_life(ph(Ds, maxdim=1)[1]), top_life(ph(Dp, maxdim=1)[1]))
    assert loop > 0.5, f"expected a ring H1 loop, got {loop:.2f}"
    return f"moving-ring H1 persistence = {loop:.2f}"


@branch("topology.dec (complex + Betti; dxtr operators if installed)")
def b_dec(quick):
    from neuralgeom.subspace.pooling import pool_frames, PoolConfig
    from neuralgeom.topology import dec
    tr = _ring_traj(quick)
    frames, fields, _ = pool_frames(tr, PoolConfig(k=1, n_pool=90, fields=True))
    XY, tf, tp_, D, stress = dec.build_complex(frames, dec.DECConfig(n_pool=90))
    b = dec.betti_from_triangles(tf)
    note = f"betti_full={b} stress={stress:.2g}"
    try:                                       # dxtr operators (optional)
        man, used = dec.make_manifold(XY, tf)
        _, df_arr, _ = dec.dec_scalar(man, fields["energy"][used])
        note += f" · dxtr |df|edges={df_arr.shape[0]}"
    except ImportError:
        note += " · dxtr not installed (DEC operators skipped)"
    return note


@branch("topology.direct (state-manifold cross-check)")
def b_direct(quick):
    from neuralgeom.topology import direct
    tr = _ring_traj(quick)
    r = direct.compare_direct_vs_grassmann(tr, 0, sub=8, maxdim=1)
    return f"H1 direct={r['h1_direct']:.2f} grass={r['h1_grass']:.2f} bn={r['bottleneck']:.2f}"


@branch("dynamics (recurrent Jacobian + LDS fit)")
def b_dynamics(quick):
    import numpy as np
    from neuralgeom.dynamics.lds import fit_lds
    rng = np.random.default_rng(0)
    A = np.array([[0.0, 1.0], [-1.0, 0.0]]) * 0.1
    z = np.zeros((200, 2)); z[0] = [1, 0]
    for t in range(199):
        z[t + 1] = z[t] + A @ z[t] + 0.01 * rng.standard_normal(2)
    res = fit_lds(z[:-1], np.diff(z, axis=0))
    return f"fit_lds A shape={np.asarray(res['A']).shape}"


@branch("tasks (build task + model + a few train steps)")
def b_tasks(quick):
    from neuralgeom.tasks import make_task, make_model, train, evaluate
    task = make_task("perceptual_decision", dt=50, seed=0)
    model = make_model("vanilla", task.spec, hidden_size=32, dt=task.dt)
    train(model, task, steps=(20 if quick else 60), batch_size=32)
    acc = evaluate(model, task)
    return f"trained vanilla RNN, eval acc≈{float(acc):.2f}"


BRANCHES = [b_contract, b_synth, b_pullback, b_spd, b_grass_torch,
            b_grass_frames, b_subspace, b_topology, b_dec, b_direct,
            b_dynamics, b_tasks]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    print(f"neuralgeom — verify all branches {'(quick)' if args.quick else ''}\n" + "-" * 66)
    for b in BRANCHES:
        b(args.quick)
        name, status, dt, detail = RESULTS[-1]
        mark = {"PASS": "✓", "SKIP": "–", "FAIL": "✗"}[status]
        print(f"{mark} {status:4}  {name:<52} {dt:5.1f}s")
        if detail:
            print(f"        {detail}")
    n_fail = sum(1 for _, s, _, _ in RESULTS if s == "FAIL")
    n_skip = sum(1 for _, s, _, _ in RESULTS if s == "SKIP")
    n_pass = sum(1 for _, s, _, _ in RESULTS if s == "PASS")
    print("-" * 66)
    print(f"{n_pass} passed · {n_skip} skipped (missing extras) · {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
