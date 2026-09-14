"""
decoder_pullback_field.py — pullback metric field of a time-to-lick decoder.
============================================================================

The map
-------
A single decoder is fitted over all (trial, time-bin) pairs:

        f :  neural state h  ->  TIME REMAINING UNTIL THE LICK  (seconds)

Every (trial, bin) pair is one training sample. The pullback metric

        g(h) = grad f(h) grad f(h)^T          (k x k, rank 1)

is then a FIELD over neural state space: it has a value at every point the
population visits, not one value per time bin. That is exactly the object the
pullback-metric machinery is built for, and it can be drawn on the state
space (see the quiver panel).

Target
------
time_to_lick(i, t) = lick_time_i - t     (seconds, always > 0 in epoch A)

The network sees ONLY the neural state, never t, so it cannot read the answer
off the clock; it must infer remaining time from population activity.

Cross-validation
----------------
Time bins within one trial are strongly correlated (they are a smoothed
trajectory). If folds were split over SAMPLES, bins from the same trial would
appear in both train and test and R^2 would be inflated by memorising trials
rather than learning the map. Therefore:

  * folds are split BY TRIAL — every bin of a trial is in the same fold;
  * the REDUCER (PCA) is fit inside each fold on training trials only, so
    test trials never influence the coordinate system either;
  * predictions, R^2 and all Jacobians are out-of-fold.

Null controls (identical pipeline)
----------------------------------
  N1 shuffled lick times   permute lick_time across trials, keeping each
                           trial's activity and its internal time structure
                           intact. Breaks the state->behaviour link only.
  N2 random projection     k random orthonormal directions instead of PCs,
                           at matched k. Tests whether the result needs the
                           high-variance directions or only k dimensions.
  N3 trial-shifted target  assign trial i the lick time of trial i+1. A
                           weaker, more conservative null that preserves the
                           marginal distribution of targets exactly.

Run:  python scripts/neural/decoder_pullback_field.py [SESSION] [A|B]
"""
from __future__ import annotations

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
from neuralgeom.paths import DATA_DIR, fig_dir  # noqa: E402
from neuralgeom.data.loader import load_session                       # noqa: E402
from neuralgeom.data.reduce import PCA, RandomProjection              # noqa: E402
from neuralgeom.geometry.jacobian import batch_jacobian                  # noqa: E402

SESSION = sys.argv[1] if len(sys.argv) > 1 else "SM239_20230302_g0"
EPOCH = "B" if "B" in sys.argv[1:] else "A"
WINDOW, KDEF, NFOLD, STEPS = 0.20, 10, 5, 200
FIG = fig_dir("neural")
torch.set_default_dtype(torch.float64)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 145, "font.size": 8,
                     "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.fontsize": 7})


def save(fig, name):
    p = FIG / name
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p.name}")


def fit_mlp(H, y, hidden=32, steps=STEPS, lr=5e-3, wd=1e-4, seed=0,
            linear=False):
    """One hidden layer, 32 tanh units. Kept deliberately small: the map only
    needs enough capacity to be nonlinear, and a larger net costs 9x the time
    for no gain in out-of-fold R^2 on this data."""
    torch.manual_seed(seed)
    k = H.shape[1]
    net = (nn.Sequential(nn.Linear(k, 1)) if linear else
           nn.Sequential(nn.Linear(k, hidden), nn.Tanh(),
                         nn.Linear(hidden, 1)))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Ht, yt = torch.as_tensor(H), torch.as_tensor(y).reshape(-1, 1)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            ((net(Ht) - yt) ** 2).mean().backward()
            opt.step()
    net.eval()
    return net


def trialwise_cv(Z, y_trial, t_grid, k=KDEF, reducer="pca", linear=False,
                 nfold=NFOLD, seed=0):
    """Pool (trial, bin) samples; CV by TRIAL; PCA fit inside each fold.

    Z        (trials, T, units) z-scored activity
    y_trial  (trials,) lick time
    t_grid   (T,) bin centre times
    Returns dict with out-of-fold predictions, targets, R^2, Jacobians,
    latent states, and per-sample trial/time indices.
    """
    ntr, T, _ = Z.shape
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(ntr), nfold)
    tt = np.tile(t_grid, ntr)
    tr_id = np.repeat(np.arange(ntr), T)
    target = np.repeat(y_trial, T) - tt                    # time to lick
    pred = np.zeros_like(target)
    Jall = np.zeros((len(target), k))
    Pall = np.zeros((len(target), k))
    for f in range(nfold):
        # Sort the test-trial indices: the boolean mask used to scatter the
        # predictions back is in ASCENDING trial order, so the transformed
        # test states must be too, or predictions land on the wrong trials.
        te_tr = np.sort(folds[f])
        tr_tr = np.concatenate([folds[g] for g in range(nfold) if g != f])
        red = (PCA(k) if reducer == "pca"
               else RandomProjection(k, seed=seed + f))
        red.fit(Z[tr_tr])                                   # fold-internal fit
        Ptr = red.transform(Z[tr_tr]).reshape(-1, k)
        Pte = red.transform(Z[te_tr]).reshape(-1, k)
        ytr = (np.repeat(y_trial[tr_tr], T)
               - np.tile(t_grid, len(tr_tr)))
        net = fit_mlp(Ptr, ytr, seed=seed + f, linear=linear)
        with torch.no_grad():
            pte = net(torch.as_tensor(Pte)).numpy().ravel()
        Jte = batch_jacobian(net, torch.as_tensor(Pte)
                             ).squeeze(1).detach().numpy()
        mask = np.isin(tr_id, te_tr)          # ascending trial order
        assert mask.sum() == len(pte), "test alignment broken"
        pred[mask] = pte
        Jall[mask] = Jte
        Pall[mask] = Pte
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return dict(pred=pred, target=target, r2=1 - ss_res / ss_tot, J=Jall,
                P=Pall, tr_id=tr_id, t=tt)


# --------------------------------------------------------------------------- #
print("=" * 74)
s_full = load_session(DATA_DIR / f"{SESSION}.h5")
s = s_full.prelick(WINDOW) if EPOCH == "A" else s_full
Z = s.zscored(baseline=(s.t[0], s.t[0] + 0.1) if EPOCH == "A" else (-0.5, -0.1))
if EPOCH == "B":
    Z = Z[:, ::5, :]; tg = s.t[::5]                 # 100 ms grid for compute
else:
    tg = s.t
y = s.lick.astype(float)
print(f"decoder pullback field — {SESSION} ({s_full.group}, day {s_full.training_day})  "
      f"EPOCH {EPOCH}")
print(f"  activity {Z.shape} -> {Z.shape[0]*Z.shape[1]} pooled (trial,bin) "
      f"samples; target = time to lick")
print(f"  CV: {NFOLD}-fold BY TRIAL; PCA k={KDEF} fit inside each fold")
print("=" * 74)

res = trialwise_cv(Z, y, tg)
res_lin = trialwise_cv(Z, y, tg, linear=True)
rng = np.random.default_rng(0)
res_shuf = trialwise_cv(Z, rng.permutation(y), tg)
res_roll = trialwise_cv(Z, np.roll(y, 1), tg)
res_rand = [trialwise_cv(Z, y, tg, reducer="rand", seed=sd) for sd in range(3)]

print(f"\n#### Q1 decoding time-to-lick from state alone (out-of-fold)")
print(f"  MLP, PCA-{KDEF}          R^2 = {res['r2']:+.3f}")
print(f"  linear, PCA-{KDEF}       R^2 = {res_lin['r2']:+.3f}")
print(f"  N1 shuffled lick times   R^2 = {res_shuf['r2']:+.3f}")
print(f"  N3 trial-shifted target  R^2 = {res_roll['r2']:+.3f}")
print(f"  N2 random projection     R^2 = "
      f"{np.mean([r['r2'] for r in res_rand]):+.3f} +- "
      f"{np.std([r['r2'] for r in res_rand]):.3f}")

sens = np.linalg.norm(res["J"], axis=1)
print(f"\n#### Q2 pullback sensitivity |grad f| over the whole field")
print(f"  median {np.median(sens):.3f} s per unit state; "
      f"IQR [{np.percentile(sens,25):.3f}, {np.percentile(sens,75):.3f}]; "
      f"range [{sens.min():.3f}, {sens.max():.3f}]")
print(f"  fold-change across the field (p95/p5) = "
      f"{np.percentile(sens,95)/max(np.percentile(sens,5),1e-9):.1f}x  "
      f"(1.0 would mean a globally linear map)")

# does sensitivity depend on where you are, or when you are?
r_t = np.corrcoef(res["t"], sens)[0, 1]
r_ttl = np.corrcoef(res["target"], sens)[0, 1]
print(f"  corr(|grad f|, time in epoch)   = {r_t:+.3f}")
print(f"  corr(|grad f|, time to lick)    = {r_ttl:+.3f}")

# axis variability across the field
Jn = res["J"] / np.maximum(np.linalg.norm(res["J"], axis=1, keepdims=True), 1e-12)
mean_axis = Jn.mean(0); mean_axis /= np.linalg.norm(mean_axis)
cosang = np.clip(Jn @ mean_axis, -1, 1)
print(f"\n#### Q3 is the decoding AXIS constant over state space?")
print(f"  angle to the field-mean axis: median "
      f"{np.degrees(np.arccos(np.abs(cosang))).mean():.1f} deg, "
      f"p95 = {np.percentile(np.degrees(np.arccos(np.abs(cosang))),95):.1f} deg")
print(f"  (0 everywhere would mean the map is effectively LINEAR)")

# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(2, 3, figsize=(14, 7.4), constrained_layout=True)

a = ax[0, 0]
names = [f"MLP PCA-{KDEF}", "linear", "N1 shuffled", "N3 shifted",
         "N2 random proj"]
vals = [res["r2"], res_lin["r2"], res_shuf["r2"], res_roll["r2"],
        np.mean([r["r2"] for r in res_rand])]
errs = [0, 0, 0, 0, np.std([r["r2"] for r in res_rand])]
cols = ["C0", "C1", "0.6", "0.6", "C3"]
a.barh(range(len(names)), vals, xerr=errs, color=cols)
a.set_yticks(range(len(names)), names, fontsize=7)
a.axvline(0, color="k", lw=0.8)
a.set_xlabel("out-of-fold $R^2$ (time to lick)")
a.set_title("Q1 — ONE decoder, all bins pooled\nCV split BY TRIAL "
            "(bins within a trial never span folds)")

a = ax[0, 1]
sub = np.random.default_rng(0).choice(len(sens), min(4000, len(sens)),
                                      replace=False)
sc = a.scatter(res["target"][sub], res["pred"][sub], s=3, alpha=0.25,
               c=res["t"][sub], cmap="viridis")
fig.colorbar(sc, ax=a, label="time in epoch (s)")
lim = [0, np.percentile(res["target"], 99)]
a.plot(lim, lim, "k--", lw=1); a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("actual time to lick (s)"); a.set_ylabel("predicted (out-of-fold)")
a.set_title(f"Single-sample prediction, $R^2$ = {res['r2']:+.3f}")

a = ax[0, 2]
a.hist(sens, bins=60, color="C3")
a.set_xlabel("$|\\nabla f|$  (s of time-to-lick per unit state)")
a.set_ylabel("(trial, bin) samples")
a.set_title(f"Q2 — the METRIC FIELD $\\sqrt{{pseudo\\!-\\!\\det\\,g}}$\n"
            f"p95/p5 = {np.percentile(sens,95)/max(np.percentile(sens,5),1e-9):.1f}x "
            f"(1.0 = globally linear)")

a = ax[1, 0]
nb = min(12, len(np.unique(res["t"])))
tb = np.linspace(res["t"].min(), res["t"].max() + 1e-9, nb + 1)
grp = [sens[(res["t"] >= lo) & (res["t"] < hi)] for lo, hi in zip(tb[:-1], tb[1:])]
keepb = [i for i, g in enumerate(grp) if len(g) > 5]
med = [np.median(grp[i]) for i in keepb]
q1 = [np.percentile(grp[i], 25) for i in keepb]
q3 = [np.percentile(grp[i], 75) for i in keepb]
ctr = (0.5 * (tb[:-1] + tb[1:]))[keepb]
a.plot(ctr, med, "o-", color="C3"); a.fill_between(ctr, q1, q3, color="C3",
                                                   alpha=0.25)
a.set_xlabel("time in epoch (s)"); a.set_ylabel("$|\\nabla f|$")
a.set_title(f"Sensitivity across the epoch\ncorr with time = {r_t:+.2f}")

a = ax[1, 1]
ang = np.degrees(np.arccos(np.abs(cosang)))
a.hist(ang, bins=60, color="C4")
a.set_xlabel("angle between local decoding axis and field mean (deg)")
a.set_ylabel("samples")
a.set_title("Q3 — does the axis ROTATE across state space?\n"
            "(a spike at 0 would mean a linear map)")

a = ax[1, 2]
P = res["P"]
step = max(1, len(P) // 1200)
sc = a.scatter(P[::step, 0], P[::step, 1], c=sens[::step], s=5, cmap="magma",
               alpha=0.7)
fig.colorbar(sc, ax=a, label="$|\\nabla f|$")
q = a.quiver(P[::step * 4, 0], P[::step * 4, 1],
             res["J"][::step * 4, 0], res["J"][::step * 4, 1],
             color="k", alpha=0.5, width=2.5e-3, scale=3.0)
a.set_xlabel("PC1 of neural state"); a.set_ylabel("PC2")
a.set_title("THE METRIC FIELD ON STATE SPACE\ncolour = sensitivity, arrows = "
            "$\\nabla f$ (decoding axis)")
fig.suptitle(f"Pooled decoder f: state -> TIME TO LICK.  "
             f"{SESSION} ({s_full.group}), EPOCH {EPOCH}, PCA k={KDEF} fit "
             f"within folds.  {Z.shape[0]} trials x {Z.shape[1]} bins = "
             f"{Z.shape[0]*Z.shape[1]} samples.", fontsize=9.5)
save(fig, f"decoder_field_{EPOCH}.png")
print("\nDone.")
