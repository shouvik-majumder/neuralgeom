"""
STAGE 3 — The pullback metric of a behaviour decoder, on real neural data.
==========================================================================

THE MAP WE PULL BACK THROUGH
-----------------------------
The pullback metric needs a differentiable map. Here it is the DECODER:

        f :  neural state h  (k dims, after a reducer)  ->  lick time (s)

fit separately at each time bin of the epoch. Nothing is binned by behaviour;
every trial contributes its own state and its own lick time.

WHAT THE METRIC IS, FOR A SCALAR OUTPUT
---------------------------------------
J = df/dh is a 1 x k row vector (the gradient of predicted lick time with
respect to the neural state), so

        g = J^T J = grad(f) grad(f)^T        (k x k, RANK 1)

Rank 1 is not a defect; it is the geometry of scalar prediction. It means:

  * pseudo-volume  sqrt(pseudo-det g) = |grad f|
        = SENSITIVITY: how many seconds the predicted lick time changes per
          unit change in neural state. Large where the population is close to
          "deciding", small where state changes do not matter.
  * row space (1-dim) = grad f / |grad f|
        = the DECODING AXIS: the one direction in neural state space that
          changes the predicted lick time.
  * kernel of g (k-1 dims)
        = directions the behaviour is BLIND to. Activity can move freely
          there without changing predicted timing.

QUESTIONS THIS STAGE ASKS
-------------------------
 Q1 Does the state predict lick time at all, and from when? (cross-validated
    R^2 per time bin; a metric from a decoder that cannot decode is
    meaningless)
 Q2 How does SENSITIVITY |grad f| evolve within the epoch, and does it vary
    across trials in a structured way?
 Q3 Is the DECODING AXIS stable over time, or does it rotate? (principal
    angle between the axis at different time bins)
 Q4 Is the geometry a property of the data or of the dimensionality? (repeat
    everything under random projections at matched k -- the null reducer)
 Q5 Is it noise-driven? (shuffled lick times; and stability under trial
    subsampling)

DEFINITIONS OF THE VALIDATION QUANTITIES
----------------------------------------
* Cross-validated R^2: fit the decoder on 4/5 of trials, predict the held-out
  1/5, and report 1 - SS_res/SS_tot on those held-out trials. R^2 <= 0 means
  the decoder is worse than predicting the mean lick time.
* Shuffled-label null: identical pipeline with lick times randomly permuted
  across trials. Anything the pipeline reports here is what it invents from
  noise.
* Split-half stability of the axis: fit the decoder twice on disjoint halves
  of the trials and measure the angle between the two decoding axes. Small
  angle = the axis is a real, reproducible direction.

FIXED SETTINGS (stated, not swept unless noted)
-----------------------------------------------
  session bins 20 ms, causal 60 ms boxcar, qm_pass + rate >= 0.5 Hz units
  epochs  A = post-cue pre-lick 0-0.2 s ; B = full window -0.5 to 1.5 s
  reducer DEFAULT PCA k=10; also swept: full space, PCA 3/20, random 10, LDA 10
  decoder 1 hidden layer, 32 tanh units (a linear decoder is run alongside)

Run:  python stage3_decoder_pullback.py [SESSION_ID]
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
from neuralgeom.data.loader import load_session                          # noqa: E402
from neuralgeom.data.reduce import (Identity, PCA, RandomProjection)      # noqa: E402
from neuralgeom.geometry.jacobian import batch_jacobian                      # noqa: E402

SESSION = sys.argv[1] if len(sys.argv) > 1 else "SM239_20230302_g0"
WINDOW, KDEF, NFOLD = 0.20, 10, 5
# Compute control: EPOCH B spans 100 bins at 20 ms. The decoder is fitted
# every TSTRIDE bins (100 ms for epoch B) purely to bound runtime; no data
# is discarded, only the temporal resolution of the decoding CURVE.
STEPS = 200
FIG = fig_dir("neural_stage3")
torch.set_default_dtype(torch.float64)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 145, "font.size": 8,
                     "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.fontsize": 7})


def save(fig, name):
    p = FIG / name
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p.name}")


# --------------------------------------------------------------------------- #
def make_decoder(k, hidden=32, linear=False):
    if linear:
        return nn.Sequential(nn.Linear(k, 1))
    return nn.Sequential(nn.Linear(k, hidden), nn.Tanh(), nn.Linear(hidden, 1))


def fit_decoder(H, y, steps=STEPS, lr=5e-3, wd=1e-3, hidden=32, linear=False,
                seed=0):
    """H (n, k) states -> y (n,) lick times. Returns the fitted module."""
    torch.manual_seed(seed)
    net = make_decoder(H.shape[1], hidden, linear)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Ht, yt = torch.as_tensor(H), torch.as_tensor(y).reshape(-1, 1)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            loss = ((net(Ht) - yt) ** 2).mean()
            loss.backward()
            opt.step()
    net.eval()
    return net


def cv_decode(H, y, linear=False, nfold=NFOLD, seed=0):
    """Cross-validated R^2 and the out-of-fold gradients (pullback Jacobians).

    Returns (r2, J_oof) where J_oof is (n, k): each trial's decoder gradient
    from a model that never saw that trial.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, nfold)
    pred = np.zeros(n)
    J = np.zeros((n, H.shape[1]))
    for f in range(nfold):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(nfold) if g != f])
        net = fit_decoder(H[tr], y[tr], linear=linear, seed=seed + f)
        with torch.no_grad():
            pred[te] = net(torch.as_tensor(H[te])).numpy().ravel()
        Jt = batch_jacobian(net, torch.as_tensor(H[te]))    # (n_te, 1, k)
        J[te] = Jt.squeeze(1).detach().numpy()
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot, J, pred


def axis_angle(J1, J2):
    """Angle (deg) between two mean decoding axes."""
    a = J1.mean(0); b = J2.mean(0)
    c = abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, 0, 1))))


# --------------------------------------------------------------------------- #
print("=" * 74)
s_full = load_session(DATA_DIR / f"{SESSION}.h5")
s_pre = s_full.prelick(WINDOW)
print(f"STAGE 3 — {SESSION}  group={s_full.group}  day={s_full.training_day}")
print(f"  EPOCH A prelick {s_pre.X.shape}, EPOCH B full {s_full.X.shape}")
print(f"  default reducer PCA k={KDEF}; decoder = 32-unit tanh MLP; "
      f"{NFOLD}-fold CV")
print("=" * 74)

EPOCHS_ALL = [("A_prelick", s_pre, f"EPOCH A: post-cue pre-lick 0-{WINDOW}s"),
          ("B_full", s_full, "EPOCH B: full window -0.5 to 1.5s")]
which = [a for a in sys.argv if a in ("A", "B")]
EPOCHS = ([e for e in EPOCHS_ALL if e[0][0] in which] if which
          else EPOCHS_ALL)
store = {}

for tag, s, title in EPOCHS:
    Z = s.zscored(baseline=(s.t[0], s.t[0] + 0.1)
                  if s.align == "prelick_fixed" else (-0.5, -0.1))
    red = PCA(KDEF).fit(Z)
    P = red.transform(Z)                                    # (trials, T, k)
    y = s.lick.astype(float)
    stride = 1 if s.align == "prelick_fixed" else 5
    tidx = np.arange(0, P.shape[1], stride)
    T = len(tidx)
    print(f"  [{tag}] evaluating the decoder at {T} time points "
          f"(every {stride*s.bin_s*1000:.0f} ms)")

    r2_t, sens_t, J_t = [], [], []
    r2_lin, r2_shuf = [], []
    for ti in tidx:
        H = P[:, ti, :]
        r2, J, _ = cv_decode(H, y)
        r2_t.append(r2)
        sens_t.append(np.linalg.norm(J, axis=1))
        J_t.append(J)
        r2_lin.append(cv_decode(H, y, linear=True)[0])
        ysh = np.random.default_rng(int(ti)).permutation(y)
        r2_shuf.append(cv_decode(H, ysh, nfold=3)[0])
    r2_t = np.array(r2_t); r2_lin = np.array(r2_lin)
    r2_shuf = np.array(r2_shuf)
    sens_t = np.array(sens_t)                                # (T, trials)
    J_t = np.array(J_t)                                      # (T, trials, k)

    ti_best = int(np.argmax(r2_t))
    print(f"\n### {title}")
    print(f"  Q1 decoding: best CV R^2 = {r2_t[ti_best]:+.3f} at t = "
          f"{s.t[ti_best]:+.3f}s (linear {r2_lin[ti_best]:+.3f}, "
          f"shuffled {r2_shuf[ti_best]:+.3f})")
    print(f"     R^2 over time: {np.round(r2_t, 3)}")
    print(f"  Q2 sensitivity |grad f| at best bin: median "
          f"{np.median(sens_t[ti_best]):.3f} s per unit state, "
          f"IQR [{np.percentile(sens_t[ti_best],25):.3f}, "
          f"{np.percentile(sens_t[ti_best],75):.3f}]")

    # Q3 axis rotation over time
    ang = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            ang[i, j] = axis_angle(J_t[i], J_t[j])
    print(f"  Q3 decoding-axis rotation across the epoch: "
          f"first-vs-last bin = {ang[0, -1]:.1f} deg")

    # split-half stability of the axis at the best bin
    H = P[:, tidx[ti_best], :]
    rng = np.random.default_rng(0); pm = rng.permutation(len(y))
    h1, h2 = pm[:len(pm) // 2], pm[len(pm) // 2:]
    n1 = fit_decoder(H[h1], y[h1]); n2 = fit_decoder(H[h2], y[h2])
    J1 = batch_jacobian(n1, torch.as_tensor(H)).squeeze(1).detach().numpy()
    J2 = batch_jacobian(n2, torch.as_tensor(H)).squeeze(1).detach().numpy()
    ang_half = axis_angle(J1, J2)
    print(f"  Q5 split-half stability of the decoding axis: {ang_half:.1f} deg "
          f"(90 deg = unrelated)")

    store[tag] = dict(r2=r2_t, r2_lin=r2_lin, r2_shuf=r2_shuf, sens=sens_t,
                      J=J_t, ang=ang, t=s.t[tidx], tidx=tidx, y=y, P=P, red=red, s=s,
                      ti_best=ti_best, ang_half=ang_half, title=title)

# --------------------------------------------------------------------------- #
# Q4 — reducer comparison at the best bin of EPOCH A
print("\n### Q4 reducer comparison (EPOCH A, best time bin)")
A = store["A_prelick"]
sA = A["s"]
ZA = sA.zscored(baseline=(sA.t[0], sA.t[0] + 0.1))
specs = [("full space", Identity()), ("PCA 3", PCA(3)), ("PCA 10", PCA(10)),
         ("PCA 20", PCA(20))] + [(f"random 10 (s{sd})", RandomProjection(10, sd))
                                 for sd in range(5)]
red_rows = []
for name, red in specs:
    red.fit(ZA)
    Pr = red.transform(ZA)[:, A["tidx"][A["ti_best"]], :]
    r2, J, _ = cv_decode(Pr, A["y"])
    red_rows.append((name, red.k, r2, float(np.median(np.linalg.norm(J, axis=1)))))
    print(f"  {name:>18s}  k={red.k:3d}  CV R^2 = {r2:+.3f}  "
          f"median |grad f| = {red_rows[-1][3]:.3f}")
rand_r2 = [r[2] for r in red_rows if r[0].startswith("random")]
print(f"  -> random-projection null at k=10: R^2 = {np.mean(rand_r2):+.3f} "
      f"+- {np.std(rand_r2):.3f}   vs PCA-10 "
      f"{[r[2] for r in red_rows if r[0]=='PCA 10'][0]:+.3f}")

# --------------------------------------------------------------------------- #
for tag in store:
    d = store[tag]
    T = len(d["t"])
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.6), constrained_layout=True)

    a = ax[0, 0]
    a.plot(d["t"], d["r2"], "o-", label="MLP decoder")
    a.plot(d["t"], d["r2_lin"], "s--", label="linear decoder")
    a.plot(d["t"], d["r2_shuf"], "^:", color="0.6", label="shuffled lick times")
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("time in epoch (s)")
    a.set_ylabel("cross-validated $R^2$ for predicting lick time")
    a.legend()
    a.set_title("Q1 — CAN the state predict lick time?\n(negative = worse "
                "than predicting the mean)")

    a = ax[0, 1]
    med = np.median(d["sens"], axis=1)
    q1 = np.percentile(d["sens"], 25, axis=1)
    q3 = np.percentile(d["sens"], 75, axis=1)
    a.plot(d["t"], med, "o-", color="C3")
    a.fill_between(d["t"], q1, q3, color="C3", alpha=0.25)
    a.set_xlabel("time in epoch (s)")
    a.set_ylabel("$|\\nabla f|$  (s of lick time per unit state)")
    a.set_title("Q2 — SENSITIVITY of the pullback metric\n"
                "$\\sqrt{pseudo\\!-\\!\\det\\, g}=|\\nabla f|$ (median, IQR "
                "over trials)")

    a = ax[0, 2]
    im = a.imshow(d["ang"], cmap="magma_r", vmin=0, vmax=90,
                  extent=[d["t"][0], d["t"][-1], d["t"][-1], d["t"][0]])
    fig.colorbar(im, ax=a, label="angle between decoding axes (deg)")
    a.set_xlabel("time (s)"); a.set_ylabel("time (s)")
    a.set_title("Q3 — does the DECODING AXIS rotate?\n(0 = same direction, "
                "90 = unrelated)")

    a = ax[1, 0]
    ti = d["ti_best"]
    _, _, pred = cv_decode(d["P"][:, d["tidx"][ti], :], d["y"])
    a.scatter(d["y"], pred, s=6, alpha=0.4)
    lo, hi = d["y"].min(), np.percentile(d["y"], 99)
    a.plot([lo, hi], [lo, hi], "k--", lw=1)
    a.set_xlim(lo, hi)
    a.set_xlabel("actual lick time (s)")
    a.set_ylabel("predicted lick time (s), held out")
    a.set_title(f"Single-trial prediction at t = {d['t'][ti]:+.3f}s\n"
                f"CV $R^2$ = {d['r2'][ti]:+.3f}")

    a = ax[1, 1]
    a.hist(d["sens"][ti], bins=40, color="C3")
    a.set_xlabel("$|\\nabla f|$ per trial")
    a.set_ylabel("trials")
    a.set_title("Distribution of sensitivity across trials\n"
                "(a constant would mean an effectively LINEAR decoder)")

    a = ax[1, 2]
    axis_now = d["J"][ti].mean(0)
    axis_now = axis_now / np.linalg.norm(axis_now)
    a.bar(np.arange(1, len(axis_now) + 1), axis_now, color="C0")
    a.set_xlabel("principal component of neural state")
    a.set_ylabel("weight in the decoding axis")
    a.set_title(f"Which PCs carry the behaviour?\nsplit-half axis stability = "
                f"{d['ang_half']:.0f} deg (90 = unrelated)")
    fig.suptitle(f"Stage 3 — PULLBACK METRIC of the decoder "
                 f"f: neural state -> lick time.   {d['title']}   |   "
                 f"{SESSION} ({s_full.group}), PCA k={KDEF}", fontsize=10)
    save(fig, f"s3_{tag}_pullback.png")

# reducer-comparison figure
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
names = [r[0] for r in red_rows]; r2s = [r[2] for r in red_rows]
cols = ["C0" if not n.startswith("random") else "C3" for n in names]
ax[0].barh(range(len(names)), r2s, color=cols)
ax[0].set_yticks(range(len(names)), names, fontsize=6.5)
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set_xlabel("cross-validated $R^2$ (lick time)")
ax[0].set_title("Q4 — does the pullback need PCA's directions?\n"
                "red = random-projection null at matched k=10")
ax[1].barh(range(len(names)), [r[3] for r in red_rows], color=cols)
ax[1].set_yticks(range(len(names)), names, fontsize=6.5)
ax[1].set_xlabel("median $|\\nabla f|$")
ax[1].set_title("Sensitivity of the metric by reducer")
fig.suptitle(f"Stage 3 — reducer comparison at the best pre-lick time bin "
             f"({SESSION})", fontsize=10)
save(fig, "s3_reducer_comparison.png")
print("\nDone.")
