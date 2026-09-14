"""
population_structure.py — descriptive population structure on two epochs.
==========================================================================

Two analyses, never mixed:

  EPOCH A  "prelick"  fixed post-cue window (0 -> W s), keeping ONLY trials
                      whose first lick comes after W. Raw traces, no warping.
                      Contains the timing computation and no outcome signal.
  EPOCH B  "full"     fixed -0.5 -> 1.5 s window: pre-cue baseline, cue
                      response, lick and post-lick outcome all included.

Everything here is descriptive. No metric, no model. The point is to know
what the population does before asking what its geometry is.

Run:  python scripts/neural/population_structure.py [SESSION_ID] [WINDOW_S]
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

ROOT = Path(__file__).resolve().parents[2]
from neuralgeom.paths import DATA_DIR, fig_dir  # noqa: E402
from neuralgeom.data.loader import load_session                       # noqa: E402

SESSION = sys.argv[1] if len(sys.argv) > 1 else "SM259_20230417_g0"
WINDOW = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
FIG = fig_dir("neural")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 145, "font.size": 8,
                     "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.grid": True, "grid.alpha": 0.2,
                     "legend.fontsize": 7})


def save(fig, name):
    p = FIG / name
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p.name}")
    return p


def note(ax, txt, right=True):
    ax.text(0.98 if right else 0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=6.3, va="top", ha="right" if right else "left",
            bbox=dict(fc="white", alpha=0.85, lw=0.4))


# --------------------------------------------------------------------------- #
def pca_of(X):
    """X (trials, time, units) -> mean, components (units,k), var ratio, PR."""
    F = X.reshape(-1, X.shape[-1])
    mu = F.mean(0, keepdims=True)
    U, S, Vh = np.linalg.svd(F - mu, full_matrices=False)
    var = S ** 2
    pr = float(var.sum() ** 2 / (var ** 2).sum())
    return mu, Vh.T, var / var.sum(), pr


def cv_dimensionality(X, n_max=25, n_rep=8, seed=0):
    """Cross-validated PCA: fit components on half the trials, measure
    variance explained on the held-out half. Guards against reading
    noise dimensions as structure."""
    rng = np.random.default_rng(seed)
    B = X.shape[0]
    out = np.zeros((n_rep, n_max))
    for r in range(n_rep):
        perm = rng.permutation(B)
        A, Bx = perm[:B // 2], perm[B // 2:]
        FA = X[A].reshape(-1, X.shape[-1]); FB = X[Bx].reshape(-1, X.shape[-1])
        muA = FA.mean(0, keepdims=True)
        V = np.linalg.svd(FA - muA, full_matrices=False)[2].T
        R = FB - FB.mean(0, keepdims=True)
        tot = (R ** 2).sum()
        for k in range(1, n_max + 1):
            P = V[:, :k]
            out[r, k - 1] = ((R @ P) ** 2).sum() / tot
    return out.mean(0), out.std(0)


def split_half_subspace_angle(X, k=5, n_rep=8, seed=0):
    """Principal angles between PCA subspaces fit on disjoint trial halves.
    Small angles = the subspace is a real, reproducible feature."""
    rng = np.random.default_rng(seed)
    B = X.shape[0]
    ang = []
    for r in range(n_rep):
        perm = rng.permutation(B)
        A, Bx = perm[:B // 2], perm[B // 2:]
        VA = np.linalg.svd(X[A].reshape(-1, X.shape[-1])
                           - X[A].reshape(-1, X.shape[-1]).mean(0),
                           full_matrices=False)[2][:k].T
        VB = np.linalg.svd(X[Bx].reshape(-1, X.shape[-1])
                           - X[Bx].reshape(-1, X.shape[-1]).mean(0),
                           full_matrices=False)[2][:k].T
        s = np.linalg.svd(VA.T @ VB, compute_uv=False)
        ang.append(np.degrees(np.arccos(np.clip(s, 0, 1))))
    return np.array(ang).mean(0)


# --------------------------------------------------------------------------- #
print(f"=== population structure: {SESSION} ===")
s_full = load_session(DATA_DIR / f"{SESSION}.h5")
s_pre = s_full.prelick(WINDOW)
print(f"  EPOCH B full   : {s_full.X.shape}  t {s_full.t[0]:.2f}..{s_full.t[-1]:.2f}s")
print(f"  EPOCH A prelick: {s_pre.X.shape}  t {s_pre.t[0]:.2f}..{s_pre.t[-1]:.2f}s"
      f"  (window {WINDOW}s)")
print(f"    kept {s_pre.meta['prelick_kept']}/{s_full.n_trials} trials; "
      f"frac rewarded {s_pre.meta['prelick_frac_rewarded']:.2f} "
      f"vs {s_pre.meta['prelick_frac_rewarded_all']:.2f} overall")

EPOCHS = [("A_prelick", s_pre, f"EPOCH A — post-cue pre-lick "
                               f"(0-{WINDOW}s, {s_pre.n_trials} trials)"),
          ("B_full", s_full, f"EPOCH B — full fixed window "
                             f"({s_full.t[0]:.1f} to {s_full.t[-1]:.1f}s, "
                             f"{s_full.n_trials} trials)")]

# --------------------------------------------------------------------------- #
# Figure 1: the window choice itself (diagnostic for EPOCH A)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2), constrained_layout=True)
Ws = np.arange(0.10, 0.55, 0.025)
ntr = [int((s_full.lick > w).sum()) for w in Ws]
frw = [float(s_full.cond["rewarded"][s_full.lick > w].mean())
       if (s_full.lick > w).sum() > 5 else np.nan for w in Ws]
ax[0].plot(Ws, ntr, "o-")
ax[0].axvline(WINDOW, color="C3", ls="--", label=f"chosen {WINDOW}s")
ax[0].set_xlabel("window length W (s)"); ax[0].set_ylabel("trials with lick > W")
ax[0].legend(); ax[0].set_title("Cost of a longer pre-lick window")
ax[1].plot(Ws, frw, "o-", color="C2")
ax[1].axhline(s_full.cond["rewarded"].mean(), color="k", ls=":",
              label="all trials")
ax[1].axvline(WINDOW, color="C3", ls="--")
ax[1].set_xlabel("window length W (s)"); ax[1].set_ylabel("fraction rewarded")
ax[1].legend()
ax[1].set_title("SELECTION BIAS: long windows keep only slow\n(mostly "
                "rewarded) trials")
ax[2].hist(s_full.lick, bins=50, color="C4")
ax[2].axvline(WINDOW, color="C3", ls="--", label=f"window end {WINDOW}s")
ax[2].set_xlabel("first lick time (s)"); ax[2].set_ylabel("trials")
ax[2].legend()
ax[2].set_title("Lick times; trials right of the line are kept")
fig.suptitle("Choosing the pre-lick window (epoch A)", fontsize=10)
save(fig, "population_window_choice.png")

# --------------------------------------------------------------------------- #
for tag, s, title in EPOCHS:
    Z = s.zscored(baseline=(s.t[0], s.t[0] + 0.1) if s.align == "prelick_fixed"
                  else (-0.5, -0.1))
    mu, comps, evr, pr = pca_of(Z)
    cvm, cvs = cv_dimensionality(Z)
    ang = split_half_subspace_angle(Z, k=5)
    lb = s.lick_bins(5)
    print(f"\n  [{tag}] PR={pr:.1f}  PC1-3={evr[:3].sum():.1%}  "
          f"CV var@5PC={cvm[4]:.1%}  split-half angles(5)="
          f"{np.round(ang,1)}")

    fig, ax = plt.subplots(2, 3, figsize=(13.2, 6.2), constrained_layout=True)
    ax[0, 0].plot(np.arange(1, 21), np.cumsum(evr)[:20], "o-", label="in-sample")
    ax[0, 0].errorbar(np.arange(1, len(cvm) + 1), cvm, yerr=cvs, fmt="s-",
                      color="C3", ms=3, label="cross-validated")
    ax[0, 0].set_xlabel("principal component")
    ax[0, 0].set_ylabel("cumulative variance explained")
    ax[0, 0].set_ylim(0, 1); ax[0, 0].legend()
    ax[0, 0].set_title(f"Dimensionality (PR = {pr:.1f} of {s.n_units} units)")
    note(ax[0, 0], "gap between the curves =\nover-fit noise dimensions",
         right=True)

    ax[0, 1].bar(range(1, len(ang) + 1), ang, color="C0")
    ax[0, 1].axhline(90, color="r", ls="--", label="orthogonal (chance)")
    ax[0, 1].set_xlabel("principal angle index")
    ax[0, 1].set_ylabel("angle between split-half subspaces (deg)")
    ax[0, 1].set_ylim(0, 95); ax[0, 1].legend()
    ax[0, 1].set_title("Is the top-5 subspace reproducible?\n(small = yes)")

    for b in range(lb["n_bins"]):
        m = lb["label"] == b
        if m.sum() < 10:
            continue
        ax[0, 2].plot(s.t, Z[m].mean(axis=(0, 2)),
                      color=plt.cm.viridis(b / max(1, lb["n_bins"] - 1)),
                      label=f"{lb['median'][b]*1000:.0f} ms")
    ax[0, 2].set_xlabel("time (s)"); ax[0, 2].set_ylabel("z-scored rate")
    ax[0, 2].legend(title="median lick", fontsize=6, title_fontsize=6)
    ax[0, 2].set_title("Population mean by lick-time bin")

    P = (Z - mu.reshape(1, 1, -1)) @ comps[:, :3]
    for b in range(lb["n_bins"]):
        m = lb["label"] == b
        if m.sum() < 10:
            continue
        tr = P[m].mean(0)
        c = plt.cm.viridis(b / max(1, lb["n_bins"] - 1))
        ax[1, 0].plot(tr[:, 0], tr[:, 1], "-o", ms=2.5, color=c,
                      label=f"{lb['median'][b]*1000:.0f} ms")
        ax[1, 0].scatter(tr[0, 0], tr[0, 1], marker="s", s=30, color=c,
                         zorder=5)
    ax[1, 0].set_xlabel("PC1"); ax[1, 0].set_ylabel("PC2")
    ax[1, 0].legend(fontsize=6)
    ax[1, 0].set_title("Condition-averaged trajectories\n(square = window "
                       "start)")

    prt = [float(np.linalg.eigvalsh(np.cov(Z[:, i, :].T)).sum() ** 2
                 / (np.linalg.eigvalsh(np.cov(Z[:, i, :].T)) ** 2).sum())
           for i in range(s.n_time)]
    ax[1, 1].plot(s.t, prt, color="C2")
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_ylabel("participation ratio (across trials)")
    ax[1, 1].set_title("Dimensionality over time\n(single time-bin, "
                       "across-trial covariance)")

    r_lick = np.array([np.corrcoef(s.lick, Z[:, :, u].mean(1))[0, 1]
                       for u in range(s.n_units)])
    r_pc = np.array([np.corrcoef(s.lick, P[:, :, k].mean(1))[0, 1]
                     for k in range(3)])
    ax[1, 2].hist(r_lick[np.isfinite(r_lick)], bins=25, color="C4",
                  label="single units")
    for k, rr in enumerate(r_pc):
        ax[1, 2].axvline(rr, color=f"C{k}", lw=1.6, label=f"PC{k+1} r={rr:+.2f}")
    ax[1, 2].set_xlabel("corr with lick time"); ax[1, 2].set_ylabel("units")
    ax[1, 2].legend(fontsize=6)
    ax[1, 2].set_title("Encoding of lick TIME\n(epoch-mean activity)")
    note(ax[1, 2], f"|r|>0.1: {int((np.abs(r_lick)>0.1).sum())}/{s.n_units}",
         right=False)
    fig.suptitle(title, fontsize=10)
    save(fig, f"population_{tag}_structure.png")

# --------------------------------------------------------------------------- #
# Figure: direct side-by-side of the two epochs
fig, ax = plt.subplots(1, 3, figsize=(13, 3.3), constrained_layout=True)
for (tag, s, _), c in zip(EPOCHS, ["C0", "C3"]):
    Z = s.zscored(baseline=(s.t[0], s.t[0] + 0.1) if s.align == "prelick_fixed"
                  else (-0.5, -0.1))
    _, _, evr, pr = pca_of(Z)
    cvm, _ = cv_dimensionality(Z)
    ax[0].plot(np.arange(1, 21), np.cumsum(evr)[:20], "-o", ms=3, color=c,
               label=f"{tag} (PR={pr:.1f})")
    ax[1].plot(np.arange(1, len(cvm) + 1), cvm, "-s", ms=3, color=c, label=tag)
    r = np.array([np.corrcoef(s.lick, Z[:, :, u].mean(1))[0, 1]
                  for u in range(s.n_units)])
    ax[2].hist(r[np.isfinite(r)], bins=25, alpha=0.55, color=c, label=tag)
ax[0].set_xlabel("PC"); ax[0].set_ylabel("cumulative variance")
ax[0].set_ylim(0, 1); ax[0].legend(); ax[0].set_title("In-sample dimensionality")
ax[1].set_xlabel("PC"); ax[1].set_ylabel("cross-validated variance")
ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].set_title("Cross-validated")
ax[2].set_xlabel("corr(unit, lick time)"); ax[2].set_ylabel("units")
ax[2].legend(); ax[2].set_title("Lick-time encoding")
fig.suptitle("The two epochs side by side (never pooled)",
             fontsize=10)
save(fig, "population_compare_epochs.png")
print("\nDone.")
