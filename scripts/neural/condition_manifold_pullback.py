"""
condition_manifold_pullback.py — pullback metric of the (time, lick-time) condition manifold.
=============================================================================================

The map (note the direction: a low-dimensional domain, so the metric is full rank)

        f : (t, tau)  ->  population state in R^n

    t   = time within the epoch (s)
    tau = that trial's lick time (s)

A 2-D DOMAIN mapped into the high-dimensional neural space. Because
rank(g) = dim(domain), the metric is now 2 x 2 and full rank, so all three
observables exist:

    volume element sqrt(det g)   area magnification of the (t, tau) plane
    anisotropy      lam1/lam2    is the code more sensitive to t or to tau?
    Gaussian curvature K         is the represented surface WARPED, or merely
                                 stretched? (K = 0 for a sheet that can be
                                 flattened without stretching, even if its
                                 volume element varies wildly — so K carries
                                 information the volume element cannot)

This mirrors Zavatone-Veth et al. (2023), who pull back onto a 2-D input
manifold and find that trained networks MAGNIFY AREA near decision
boundaries, and Cayco Gajic & Pellegrino (2026), who compare such metrics
across models with the spectral ratio.

The question
------------
Where in the (time, lick-time) plane does the population representation have
the highest resolution? If area is magnified at particular (t, tau)
combinations, the circuit devotes more coding capacity to distinguishing
those conditions. A ridge running along t = tau would mean resolution peaks
approaching the action; a ridge at small tau would mean the code is sharpest
for fast trials.

Codomain metric (both computed and compared)
--------------------------------------------
  Euclidean       g = J^T J. Every unit counts equally, so high-rate units
                  dominate purely by scale.
  Noise-weighted  g = J^T Sigma^-1 J, Sigma = trial-to-trial noise covariance
                  (residuals after removing the condition mean). Distances
                  become discriminability (d-prime) units. For Gaussian noise
                  this is the Fisher information metric of the population
                  code, so it answers "what could a downstream reader
                  actually resolve?" rather than "what varies most".

Epoch definitions (no trial is discarded)
-----------------------------------------
  EPOCH A  every (trial, bin) sample with t < that trial's lick time. Because
           we pool samples rather than build a rectangular tensor, ALL trials
           contribute — each supplies however many bins precede its own lick.
           Causal smoothing guarantees a bin at t < lick contains no post-lick
           spikes, so the mask needs no safety margin.
  EPOCH B  every sample in the fixed window; nothing masked.

Controls
--------
  N1 shuffled tau      permute lick times across trials. The map is refit,
                       so every geometric quantity gets its own null.
  N2 t-only map        fit f(t) alone (1-D domain). If the 2-D geometry adds
                       nothing over this, tau is not represented.
  CV                   5-fold BY TRIAL; the surface is fit on training trials
                       and the metric evaluated on a fixed grid, so the
                       geometry never sees the held-out trials.

Run:  python scripts/neural/condition_manifold_pullback.py [SESSION] [A|B]
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
from neuralgeom.data.loader import load_session                        # noqa: E402
from neuralgeom.data.reduce import PCA                                 # noqa: E402
from neuralgeom.geometry.manifold import (anisotropy, gaussian_curvature_2d,  # noqa: E402
                            metric_summary, noise_whitener,
                            pullback_metric_field, volume_element,
                            whiten_map)

SESSION = sys.argv[1] if len(sys.argv) > 1 else "SM239_20230302_g0"
EPOCH = "B" if "B" in sys.argv[1:] else "A"
WINDOW, KDEF, NFOLD, STEPS = 0.20, 10, 3, 200
NGRID, N_NULL = 18, 2
FIG = fig_dir("neural")
torch.set_default_dtype(torch.float64)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 145, "font.size": 8,
                     "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.fontsize": 7})


def save(fig, name):
    p = FIG / name
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p.name}")


MAXFIT = 6000   # cap on samples per surface fit (speed only; sampled at random)


def fit_surface(XY, Y, hidden=64, steps=STEPS, seed=0, d_in=2):
    """f: (t, tau) -> population state. Smooth 2-layer tanh surface."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(),
                        nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-4)
    if len(XY) > MAXFIT:
        sel = np.random.default_rng(seed).choice(len(XY), MAXFIT, replace=False)
        XY, Y = XY[sel], Y[sel]
    Xt, Yt = torch.as_tensor(XY), torch.as_tensor(Y)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(); ((net(Xt) - Yt) ** 2).mean().backward(); opt.step()
    net.eval()
    return net


# --------------------------------------------------------------------------- #
print("=" * 74)
s_full = load_session(DATA_DIR / f"{SESSION}.h5")
s = s_full                                   # ALL trials, always
Z = s.zscored(baseline=(-0.5, -0.1))
if EPOCH == "B":
    Z = Z[:, ::5, :]; tg = s.t[::5]
else:
    Z = Z[:, s.t >= 0, :]; tg = s.t[s.t >= 0]     # post-cue onwards
y = s.lick.astype(float)
ntr, T, nun = Z.shape
# EPOCH A: keep only samples strictly before each trial's own lick.
if EPOCH == "A":
    keep_mask = (tg[None, :] < y[:, None]).ravel()
else:
    keep_mask = np.ones(ntr * T, dtype=bool)
print(f"condition-manifold pullback — {SESSION} ({s_full.group}, day {s_full.training_day}) "
      f"EPOCH {EPOCH}")
print(f"  map: (t, lick time) -> population state in R^{KDEF} (PCA)")
print(f"  {ntr} trials x {T} bins = {ntr*T} candidate samples; "
      f"{keep_mask.sum()} kept ({100*keep_mask.mean():.0f}%) after the "
      f"pre-lick mask")
print(f"  trials contributing >=1 sample: "
      f"{int((keep_mask.reshape(ntr, T).sum(1) > 0).sum())}/{ntr} "
      f"(NO trial thresholding)")
print(f"  bins per trial: median "
      f"{np.median(keep_mask.reshape(ntr, T).sum(1)):.0f}, range "
      f"[{keep_mask.reshape(ntr, T).sum(1).min()}, "
      f"{keep_mask.reshape(ntr, T).sum(1).max()}]")
print(f"  domain is 2-D so rank(g) = 2")
print("=" * 74)

red = PCA(KDEF).fit(Z)
P = red.transform(Z)                                   # (trials, T, k)
XY_all = np.stack([np.tile(tg, ntr), np.repeat(y, T)], 1)   # (samples, 2)
Y_all = P.reshape(-1, KDEF)
tr_id_all = np.repeat(np.arange(ntr), T)
XY, Y, tr_id = XY_all[keep_mask], Y_all[keep_mask], tr_id_all[keep_mask]

# noise covariance = residuals after removing the (t, tau) conditional mean,
# approximated by the fitted surface on training data
net0 = fit_surface(XY, Y, seed=0)
with torch.no_grad():
    resid = Y - net0(torch.as_tensor(XY)).numpy()
W, Sigma = noise_whitener(resid, alpha=0.10)
print(f"  noise covariance: condition number "
      f"{np.linalg.cond(Sigma):.1f} after 10% shrinkage")

# evaluation grid over the domain the data actually covers
# SUPPORT MASK. For epoch A the domain is triangular: a sample can only
# exist where t < tau. Evaluating the metric outside that region is pure
# extrapolation and produces enormous, meaningless volume/anisotropy values.
# We additionally require a minimum number of real samples nearby, so the
# geometry is only reported where the surface was actually constrained.
tau_lo, tau_hi = np.percentile(y, [5, 95])
t_lo = max(0.0, tg.min())
t_hi = tau_hi if EPOCH == "A" else tg.max()
gt, gtau = np.meshgrid(np.linspace(t_lo, t_hi, NGRID),
                       np.linspace(tau_lo, tau_hi, NGRID), indexing="xy")
GRID = torch.as_tensor(np.stack([gt.ravel(), gtau.ravel()], 1))

dt = (t_hi - t_lo) / NGRID
dtau = (tau_hi - tau_lo) / NGRID
dens = np.array([
    np.sum((np.abs(XY[:, 0] - a) < 1.5 * dt) & (np.abs(XY[:, 1] - b) < 1.5 * dtau))
    for a, b in zip(gt.ravel(), gtau.ravel())])
SUPPORT = dens >= 20
if EPOCH == "A":
    SUPPORT &= (gt.ravel() < gtau.ravel())
print(f"  grid: t in [{t_lo:.3f}, {t_hi:.3f}] s, lick time in "
      f"[{tau_lo:.3f}, {tau_hi:.3f}] s ({NGRID}x{NGRID})")
print(f"  SUPPORT MASK: {SUPPORT.sum()}/{len(SUPPORT)} grid points have "
      f">=20 nearby samples" + (" and satisfy t < lick time" if EPOCH == "A"
                                else ""))
print("    (metric reported ONLY there; elsewhere it would be extrapolation)")


def geometry_of(XYtr, Ytr, seed=0, Wc=None):
    """Fit the surface on given samples, return geometry on the fixed grid."""
    net = fit_surface(XYtr, Ytr, seed=seed)
    f = whiten_map(lambda x: net(x), Wc)
    g = pullback_metric_field(f, GRID)
    K, det_g = gaussian_curvature_2d(f, GRID)
    return g.detach(), K.detach(), net


# ---- cross-validated: fit on 4/5 of trials, evaluate geometry on the grid -- #
rng = np.random.default_rng(0)
folds = np.array_split(rng.permutation(ntr), NFOLD)
res = {}
for label, Wc in [("euclidean", None), ("noise-weighted", W)]:
    vols, anis, Ks = [], [], []
    for f_ in range(NFOLD):
        tr = np.concatenate([folds[g_] for g_ in range(NFOLD) if g_ != f_])
        m = np.isin(tr_id, tr)
        g, K, _ = geometry_of(XY[m], Y[m], seed=f_, Wc=Wc)
        vols.append(volume_element(g).numpy())
        anis.append(anisotropy(g).numpy())
        Ks.append(K.numpy())
    res[label] = dict(vol=np.mean(vols, 0), ani=np.mean(anis, 0),
                      K=np.mean(Ks, 0), vol_sd=np.std(vols, 0))
    v = res[label]["vol"][SUPPORT]
    a = res[label]["ani"][SUPPORT]
    k_ = res[label]["K"][SUPPORT]
    print(f"\n#### CODOMAIN METRIC: {label}")
    print(f"  volume element sqrt(det g): median {np.median(v):.3f}, "
          f"p95/p5 = {np.percentile(v,95)/max(np.percentile(v,5),1e-12):.1f}x")
    print(f"  anisotropy lam1/lam2:       median {np.median(a):.1f}, "
          f"max {a.max():.1f}")
    print(f"  Gaussian curvature K:       median {np.median(k_):+.3f}, "
          f"IQR [{np.percentile(k_,25):+.3f}, {np.percentile(k_,75):+.3f}], "
          f"{100*np.mean(k_<0):.0f}% negative (saddle-like)")

# ---- nulls ---------------------------------------------------------------- #
print(f"\n#### CONTROLS ({N_NULL} draws)")
null_vol_spread, null_ani, null_K = [], [], []
for i in range(N_NULL):
    ysh = np.random.default_rng(100 + i).permutation(y)
    XYs = np.stack([np.tile(tg, ntr), np.repeat(ysh, T)], 1)[keep_mask]
    g, K, _ = geometry_of(XYs, Y, seed=i)
    v = volume_element(g).numpy()[SUPPORT]; a = anisotropy(g).numpy()[SUPPORT]
    null_vol_spread.append(np.percentile(v, 95) / max(np.percentile(v, 5), 1e-12))
    null_ani.append(np.median(a)); null_K.append(np.median(K.numpy()[SUPPORT]))
real = res["euclidean"]
rvv = real["vol"][SUPPORT]
rv = np.percentile(rvv, 95) / max(np.percentile(rvv, 5), 1e-12)
print(f"  N1 shuffled lick time:")
print(f"    volume spread  real {rv:.1f}x   null "
      f"{np.mean(null_vol_spread):.1f}+-{np.std(null_vol_spread):.1f}")
print(f"    anisotropy     real {np.median(real['ani']):.1f}    null "
      f"{np.mean(null_ani):.1f}+-{np.std(null_ani):.1f}")
print(f"    curvature      real {np.median(real['K']):+.3f}  null "
      f"{np.mean(null_K):+.3f}+-{np.std(null_K):.3f}")

# t-only control: does tau add anything?
netT = fit_surface(XY[:, :1], Y, seed=0, d_in=1)
with torch.no_grad():
    r2_2d = 1 - ((Y - net0(torch.as_tensor(XY)).numpy()) ** 2).sum() \
        / ((Y - Y.mean(0)) ** 2).sum()
    r2_1d = 1 - ((Y - netT(torch.as_tensor(XY[:, :1])).numpy()) ** 2).sum() \
        / ((Y - Y.mean(0)) ** 2).sum()
print(f"  N2 t-only map: variance explained {r2_1d:.3f} vs 2-D map "
      f"{r2_2d:.3f}  (gain from including lick time: {r2_2d-r2_1d:+.3f})")

# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.2), constrained_layout=True)
ext = [t_lo, t_hi, tau_lo, tau_hi]
for r_, (label, d) in enumerate(res.items()):
    for c_, (M, nm, cm) in enumerate([
            (d["vol"], "volume element $\\sqrt{\\det g}$\n(area magnification)",
             "magma"),
            (np.log10(d["ani"]), "$\\log_{10}$ anisotropy $\\lambda_1/\\lambda_2$\n"
             "(t vs lick-time sensitivity)", "viridis"),
            (d["K"], "Gaussian curvature $K$\n(warped vs merely stretched)",
             "RdBu_r")]):
        a = axes[r_, c_]
        M = np.where(SUPPORT, M, np.nan)
        vmax = np.nanpercentile(np.abs(M), 98)
        im = a.imshow(M.reshape(NGRID, NGRID), origin="lower", aspect="auto",
                      extent=ext, cmap=cm,
                      vmin=-vmax if cm == "RdBu_r" else None,
                      vmax=vmax if cm == "RdBu_r" else np.nanpercentile(M, 98))
        fig.colorbar(im, ax=a, fraction=0.046)
        a.plot([t_lo, t_hi], [t_lo, t_hi], "w--", lw=1, alpha=0.6)
        a.set_xlabel("time in epoch t (s)")
        a.set_ylabel("lick time $\\tau$ (s)")
        a.set_title(f"[{label}] {nm}", fontsize=7.5)
    a = axes[r_, 3]
    a.hist(d["vol"][SUPPORT], bins=40, color="C3" if r_ else "C0")
    a.set_xlabel("$\\sqrt{\\det g}$ over the grid")
    a.set_ylabel("grid points")
    a.set_title(f"[{label}] distribution of\narea magnification", fontsize=7.5)
fig.suptitle(
    f"Condition-manifold pullback.  map f: (t, lick time) -> "
    f"population state, so rank(g) = 2 and volume/anisotropy/curvature all "
    f"exist.\n{SESSION} ({s_full.group}), EPOCH {EPOCH}, PCA k={KDEF}, "
    f"{NFOLD}-fold CV by trial.  Top row: Euclidean codomain metric.  Bottom: "
    f"noise-weighted (g = J^T Sigma^-1 J, distances in d' units).\n"
    f"White dashed line is t = lick time (the moment of action).", fontsize=9)
save(fig, f"condition_manifold_{EPOCH}_geometry.png")

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5), constrained_layout=True)
ax[0].scatter(res["euclidean"]["vol"][SUPPORT],
              res["noise-weighted"]["vol"][SUPPORT], s=6, alpha=0.5)
ax[0].set_xlabel("volume element, Euclidean")
ax[0].set_ylabel("volume element, noise-weighted")
rr = np.corrcoef(res["euclidean"]["vol"][SUPPORT],
                 res["noise-weighted"]["vol"][SUPPORT])[0, 1]
ax[0].set_title(f"Does noise weighting change the geometry?\nr = {rr:+.2f}")
ax[1].bar([0, 1], [rv, np.mean(null_vol_spread)],
          yerr=[0, np.std(null_vol_spread)], color=["C0", "0.6"])
ax[1].set_xticks([0, 1], ["real", "shuffled $\\tau$"])
ax[1].set_ylabel("volume-element spread (p95/p5)")
ax[1].set_title("N1 control: is the magnification\npattern above chance?")
ax[2].bar([0, 1], [r2_1d, r2_2d], color=["0.6", "C2"])
ax[2].set_xticks([0, 1], ["f(t) only", "f(t, lick time)"])
ax[2].set_ylabel("variance of population state explained")
ax[2].set_title(f"N2 control: does lick time add\nanything? gain "
                f"{r2_2d-r2_1d:+.3f}")
fig.suptitle(f"Controls — {SESSION}, epoch {EPOCH}", fontsize=9.5)
save(fig, f"condition_manifold_{EPOCH}_controls.png")
print("\nDone.")
