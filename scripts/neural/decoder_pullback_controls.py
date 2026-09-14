"""
decoder_pullback_controls.py — permutation controls for the decoder pullback field.
===================================================================================

A near-linear decoder fitted to noisy data still has gradients that vary
across state space, so the variation of the pullback metric field cannot be
read as structure without a null. Every quantity reported from the field in
decoder_pullback_field.py therefore gets its own null distribution here, not
only the decoding R^2.

The controls
------------
C1  SHUFFLED TARGET (trial-level permutation)
      Permute lick_time across trials; each trial keeps its own activity and
      its internal temporal structure. Breaks only the state -> behaviour
      link. Applied to EVERY reported quantity, not just R^2.

C2  GRADIENT-VARIATION NULL
      Fit the identical model to C1-shuffled targets and measure how much its
      gradient rotates across state space. If the real model's rotation is no
      larger, the "curvature" is fitting noise. This is the decisive test for
      whether the metric FIELD has structure or is merely a noisy constant.

C3  PHASE / CIRCULAR SHIFT (within-trial)
      Roll each trial's activity in time by a random offset. Destroys the
      state->time relationship while preserving each unit's autocorrelation
      and the population covariance exactly. A stricter null than C1 for any
      claim involving temporal structure.

C4  RANDOM PROJECTION (matched k)
      k random orthonormal directions instead of PCs. Tests whether results
      need the high-variance directions or only k dimensions.

C5  TRIAL-IDENTITY SHUFFLE WITHIN TIME BIN
      For each time bin independently, permute which trial each state vector
      belongs to. Destroys single-trial state->behaviour correspondence while
      preserving the mean population trajectory and each bin's covariance.
      Isolates single-trial information from condition-average structure.

Quantities tested against each null
-----------------------------------
  R^2                     decoding performance
  median |grad f|         overall sensitivity scale
  spread of |grad f|      p95/p5 — does sensitivity vary across the field?
  axis rotation           median angle to the field-mean axis
  sensitivity-time corr   does sensitivity track time in the epoch?

All nulls are run with the identical pipeline (trial-wise CV, PCA inside
folds) and repeated N_NULL times to give a null distribution, from which an
empirical two-sided p-value is computed for each statistic.

Run:  python scripts/neural/decoder_pullback_controls.py [SESSION] [A|B] [N_NULL]
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
from neuralgeom.paths import CACHE_DIR, DATA_DIR, fig_dir  # noqa: E402
from neuralgeom.data.loader import load_session                       # noqa: E402
from neuralgeom.data.reduce import PCA, RandomProjection              # noqa: E402
from neuralgeom.geometry.jacobian import batch_jacobian                  # noqa: E402

SESSION = sys.argv[1] if len(sys.argv) > 1 else "SM239_20230302_g0"
EPOCH = "B" if "B" in sys.argv[1:] else "A"
N_NULL = int([a for a in sys.argv[1:] if a.isdigit()][0]) \
    if any(a.isdigit() for a in sys.argv[1:]) else 12
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
    torch.manual_seed(seed)
    k = H.shape[1]
    net = (nn.Sequential(nn.Linear(k, 1)) if linear else
           nn.Sequential(nn.Linear(k, hidden), nn.Tanh(), nn.Linear(hidden, 1)))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Ht, yt = torch.as_tensor(H), torch.as_tensor(y).reshape(-1, 1)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(); ((net(Ht) - yt) ** 2).mean().backward(); opt.step()
    net.eval()
    return net


def run_pipeline(Z, y_trial, t_grid, k=KDEF, reducer="pca", linear=False,
                 nfold=NFOLD, seed=0):
    """Identical to decoder_pullback_field.py: pooled samples, CV by trial, PCA inside folds."""
    ntr, T, _ = Z.shape
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(ntr), nfold)
    tt = np.tile(t_grid, ntr)
    tr_id = np.repeat(np.arange(ntr), T)
    target = np.repeat(y_trial, T) - tt
    pred = np.zeros_like(target); J = np.zeros((len(target), k))
    for f in range(nfold):
        te = np.sort(folds[f])
        tr = np.concatenate([folds[g] for g in range(nfold) if g != f])
        red = PCA(k) if reducer == "pca" else RandomProjection(k, seed=seed + f)
        red.fit(Z[tr])
        Ptr = red.transform(Z[tr]).reshape(-1, k)
        Pte = red.transform(Z[te]).reshape(-1, k)
        ytr = np.repeat(y_trial[tr], T) - np.tile(t_grid, len(tr))
        net = fit_mlp(Ptr, ytr, seed=seed + f, linear=linear)
        with torch.no_grad():
            pte = net(torch.as_tensor(Pte)).numpy().ravel()
        Jte = batch_jacobian(net, torch.as_tensor(Pte)).squeeze(1).detach().numpy()
        m = np.isin(tr_id, te)
        pred[m] = pte; J[m] = Jte
    return summarize(pred, target, J, tt)


def summarize(pred, target, J, tt):
    """The five statistics we test. All computed identically for real/null."""
    sens = np.linalg.norm(J, axis=1)
    Jn = J / np.maximum(np.linalg.norm(J, axis=1, keepdims=True), 1e-12)
    mean_axis = Jn.mean(0); mean_axis /= max(np.linalg.norm(mean_axis), 1e-12)
    ang = np.degrees(np.arccos(np.clip(np.abs(Jn @ mean_axis), 0, 1)))
    return dict(
        r2=1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum(),
        sens_med=float(np.median(sens)),
        sens_spread=float(np.percentile(sens, 95) / max(np.percentile(sens, 5), 1e-12)),
        axis_rot=float(np.median(ang)),
        sens_time_r=float(np.corrcoef(tt, sens)[0, 1]),
        sens=sens, ang=ang)


STATS = [("r2", "out-of-fold $R^2$", "higher"),
         ("sens_med", "median $|\\nabla f|$", "either"),
         ("sens_spread", "$|\\nabla f|$ spread (p95/p5)", "higher"),
         ("axis_rot", "axis rotation (deg, median)", "either"),
         ("sens_time_r", "corr($|\\nabla f|$, time)", "either")]

# --------------------------------------------------------------------------- #
print("=" * 74)
s_full = load_session(DATA_DIR / f"{SESSION}.h5")
s = s_full.prelick(WINDOW) if EPOCH == "A" else s_full
Z = s.zscored(baseline=(s.t[0], s.t[0] + 0.1) if EPOCH == "A" else (-0.5, -0.1))
if EPOCH == "B":
    Z = Z[:, ::5, :]; tg = s.t[::5]
else:
    tg = s.t
y = s.lick.astype(float)
ntr, T, _ = Z.shape
print(f"decoder pullback controls — {SESSION} ({s_full.group}) EPOCH {EPOCH}")
print(f"  {ntr} trials x {T} bins; {N_NULL} draws per null")
print("=" * 74)

# Cache: each null is expensive, so results accumulate across invocations.
# Delete the .npz to force a recompute.
CACHE = CACHE_DIR / f"decoder_controls_{SESSION}_{EPOCH}.npz"
_cache = dict(np.load(CACHE, allow_pickle=True)["d"].item()) if CACHE.exists() else {}


def cached(key, fn):
    if key in _cache:
        print(f"  [cached] {key}")
        return _cache[key]
    val = fn()
    _cache[key] = val
    np.savez(CACHE, d=_cache)
    print(f"  [computed] {key}")
    return val


real = cached("real", lambda: run_pipeline(Z, y, tg))
print(f"\nREAL: R2={real['r2']:+.3f}  |grad f| med={real['sens_med']:.4f}  "
      f"spread={real['sens_spread']:.2f}x  axis_rot={real['axis_rot']:.1f} deg  "
      f"corr(sens,t)={real['sens_time_r']:+.3f}")

nulls = {}

# C1 shuffled target
nulls["C1 shuffled target"] = [
    cached(f"C1_{i}", lambda i=i: run_pipeline(
        Z, np.random.default_rng(100 + i).permutation(y), tg, seed=i))
    for i in range(N_NULL)]

# C3 circular time shift within trial
def circshift(Zin, seed):
    rng = np.random.default_rng(seed)
    out = np.empty_like(Zin)
    for i in range(Zin.shape[0]):
        out[i] = np.roll(Zin[i], int(rng.integers(1, Zin.shape[1])), axis=0)
    return out


nulls["C3 circular time shift"] = [
    cached(f"C3_{i}", lambda i=i: run_pipeline(circshift(Z, 200 + i), y, tg,
                                               seed=i))
    for i in range(N_NULL)]

# C4 random projection
nulls["C4 random projection"] = [
    cached(f"C4_{i}", lambda i=i: run_pipeline(Z, y, tg, reducer="rand",
                                               seed=300 + i))
    for i in range(N_NULL)]

# C5 trial-identity shuffle within each time bin
def binshuffle(Zin, seed):
    rng = np.random.default_rng(seed)
    out = np.empty_like(Zin)
    for t in range(Zin.shape[1]):
        out[:, t, :] = Zin[rng.permutation(Zin.shape[0]), t, :]
    return out


nulls["C5 within-bin trial shuffle"] = [
    cached(f"C5_{i}", lambda i=i: run_pipeline(binshuffle(Z, 400 + i), y, tg,
                                               seed=i))
    for i in range(N_NULL)]

# --------------------------------------------------------------------------- #
print(f"\n{'statistic':>28s} {'real':>9s} | " +
      " | ".join(f"{n.split()[0]:>18s}" for n in nulls))
rows = {}
for key, lab, _ in STATS:
    line = f"{lab:>28s} {real[key]:+9.3f} | "
    cells = []
    for nm, draws in nulls.items():
        v = np.array([d[key] for d in draws])
        p = (np.sum(np.abs(v - v.mean()) >= abs(real[key] - v.mean())) + 1) / (len(v) + 1)
        cells.append(f"{v.mean():+7.3f}+-{v.std():.3f} p={p:.2f}")
        rows.setdefault(key, {})[nm] = (v, p)
    print(line + " | ".join(cells))

print("\nINTERPRETATION KEY")
print("  axis rotation: if the real value is NOT above the C1 null, the")
print("  apparent curvature of the metric field is gradient estimation noise")
print("  rather than genuine nonlinearity of the state->behaviour map.")

# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(14, 7.0), constrained_layout=True)
for ax, (key, lab, _) in zip(axes.ravel(), STATS):
    for i, (nm, (v, p)) in enumerate(rows[key].items()):
        ax.errorbar([i], [v.mean()], yerr=[v.std()], fmt="s", color="0.5",
                    capsize=3)
        ax.scatter(np.full(len(v), i) + np.random.uniform(-.08, .08, len(v)),
                   v, s=10, color="0.7", zorder=1)
        ax.text(i, v.mean() + v.std() * 1.3, f"p={p:.2f}", ha="center",
                fontsize=6.5)
    ax.axhline(real[key], color="C3", lw=2, label="real data")
    ax.set_xticks(range(len(rows[key])),
                  [n.split()[0] for n in rows[key]], fontsize=7)
    ax.set_ylabel(lab); ax.legend(fontsize=6.5)
    ax.set_title(lab)

a = axes.ravel()[5]
a.clear()
a.hist(real["ang"], bins=50, color="C3", alpha=0.75, density=True,
       label="real data")
a.hist(np.concatenate([d["ang"] for d in nulls["C1 shuffled target"]]),
       bins=50, color="0.6", alpha=0.6, density=True,
       label="C1 shuffled target")
a.set_xlabel("angle between local decoding axis and field mean (deg)")
a.set_ylabel("density"); a.legend(fontsize=7)
a.set_title("THE DECISIVE PANEL — is the field's curvature real?\n"
            "if the two distributions overlap, the rotation is noise")
fig.suptitle(f"Permutation controls for every reported quantity.  "
             f"{SESSION} ({s_full.group}), EPOCH {EPOCH}, {N_NULL} draws per "
             f"null.  Red line = real data; grey = null draws.", fontsize=9.5)
save(fig, f"decoder_controls_{EPOCH}.png")
print("\nDone.")
