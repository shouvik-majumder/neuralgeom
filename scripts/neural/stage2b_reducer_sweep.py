"""
STAGE 2b — Where is the sweet spot? Sweeping the arbitrary choices.
===================================================================

MOTIVATION
----------
Stage 2 produced a number (geometry tracks lick time at r ~ 0.7) that depended
on three choices I made by hand: which reducer, how many dimensions, and how
many lick-time bins. None of those is principled. Rather than defend one
setting, this script maps the whole surface and asks where — if anywhere —
the structure is both STRONG and RELIABLE. That trade-off is the real object
of interest: too few dimensions and real geometry is discarded; too many and
covariance estimates are dominated by sampling noise.

TERMS DEFINED BEFORE USE
------------------------
* Covariance matrix of a condition: for a group of trials, the matrix whose
  (i,j) entry is how much dimension i and dimension j co-vary across the
  population states in that group. It is the "shape" of the cloud of states.
  It is a point on the SPD manifold (symmetric positive definite matrices).
* Affine-invariant distance: the natural Riemannian distance between two such
  matrices, unchanged if you linearly re-mix the dimensions. Formally
  ||log(A^-1/2 B A^-1/2)||_F.
* Mantel test: the significance test for "do two distance matrices agree?".
  Ordinary correlation is invalid here because pairwise distances share
  conditions and are not independent. Mantel permutes the CONDITION LABELS
  (not the pairs), which preserves that dependency exactly, and compares the
  observed correlation to that null.
* Split-half reliability: fit the whole distance matrix twice on disjoint
  halves of the trials and correlate them. Answers "is this matrix even
  measurable?", separately from "does it relate to behaviour?".
* Null reducer (random projection): k random directions instead of the top-k
  PCs. If an effect appears equally under random directions, it is a
  consequence of working in k dimensions, not of the structure PCA found.

WHAT IS SWEPT
-------------
  reducer      full space | PCA k in {2,3,5,8,10,15,20} | PCA at 80%/90%
               variance | random projection at matched k (null)
  n_bins       {4, 6, 8, 10, 12}
  sessions     several, run separately and never pooled

NOT SWEPT (held fixed, stated): 20 ms bins, causal 60 ms boxcar, qm_pass +
rate >= 0.5 Hz units, shrinkage alpha = 0.1, epoch = post-cue pre-lick 0-0.2 s
and full window, z-scoring per unit.

Run:  python stage2b_reducer_sweep.py [SESSION_ID ...]
"""
from __future__ import annotations

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuralgeom.stats import mantel, partial_mantel  # noqa: E402
from neuralgeom.fitting import shrink_cov  # noqa: E402

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from neuralgeom.paths import DATA_DIR, fig_dir  # noqa: E402
from neuralgeom.data.loader import load_session                          # noqa: E402
from neuralgeom.data.reduce import (Identity, PCA, PCAVariance,           # noqa: E402
                          RandomProjection)
from neuralgeom.geometry.spd import affine_invariant_distance              # noqa: E402

SESSIONS = sys.argv[1:] or ["SM259_20230417_g0", "SM239_20230302_g0",
                            "SM318_20230904_g0"]
WINDOW, ALPHA, N_PERM = 0.20, 0.10, 300
FIG = fig_dir("neural_stage2b")
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 145, "font.size": 8,
                     "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.fontsize": 7})

KS = [2, 3, 5, 8, 10, 15, 20]
NBINS = [4, 6, 8, 10, 12]


def save(fig, name):
    p = FIG / name
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p.name}")


def dmat_from(P, labels, keep, n_per, rng):
    covs = []
    for b in keep:
        idx = rng.choice(np.where(labels == b)[0], n_per, replace=False)
        F = P[idx].reshape(-1, P.shape[-1])
        covs.append(shrink_cov(F - F.mean(0)))
    covs = torch.as_tensor(np.stack(covs))
    n = len(keep)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(affine_invariant_distance(covs[i], covs[j]))
    return D


def split_half_rel(P, labels, keep, n_per, seed=0):
    rng = np.random.default_rng(seed)
    Ds = []
    for h in range(2):
        covs = []
        for b in keep:
            idx = rng.permutation(np.where(labels == b)[0])
            half = idx[h::2][:max(5, n_per // 2)]
            F = P[half].reshape(-1, P.shape[-1])
            covs.append(shrink_cov(F - F.mean(0)))
        covs = torch.as_tensor(np.stack(covs))
        n = len(keep); D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = float(
                    affine_invariant_distance(covs[i], covs[j]))
        Ds.append(D[np.triu_indices(n, 1)])
    return float(np.corrcoef(Ds[0], Ds[1])[0, 1])



def _mantel_rp(D1, D2, n_perm=None, seed=0):
    """(r, p) from neuralgeom.stats.mantel — kept as a local alias so call sites
    read the same as before the refactor."""
    r, p, _ = mantel(D1, D2, n_perm=n_perm or N_PERM, seed=seed)
    return r, p


# --------------------------------------------------------------------------- #
results = {}
for sid in SESSIONS:
    print("\n" + "=" * 70)
    s_full = load_session(DATA_DIR / f"{sid}.h5")
    s_pre = s_full.prelick(WINDOW)
    print(f"{sid}  group={s_full.group}  day={s_full.training_day}")
    print(f"  EPOCH A prelick: {s_pre.X.shape}  "
          f"(kept {s_pre.meta['prelick_kept']}/{s_full.n_trials})")
    for tag, s in [("A_prelick", s_pre), ("B_full", s_full)]:
        Z = s.zscored(baseline=(s.t[0], s.t[0] + 0.1)
                      if s.align == "prelick_fixed" else (-0.5, -0.1))
        grid_r = np.full((len(KS) + 3, len(NBINS)), np.nan)
        grid_p = np.full_like(grid_r, np.nan)
        grid_rel = np.full_like(grid_r, np.nan)
        grid_k = np.full_like(grid_r, np.nan)
        reducers = ([("PCA", PCA(k)) for k in KS]
                    + [("PCAvar", PCAVariance(0.80)),
                       ("PCAvar", PCAVariance(0.90)),
                       ("full", Identity())])
        rows = [f"PCA {k}" for k in KS] + ["PCA 80%", "PCA 90%", "full space"]
        for ri, (_, red) in enumerate(reducers):
            red.fit(Z)
            P = red.transform(Z)
            for ci, nb in enumerate(NBINS):
                lb = s.lick_bins(nb)
                keep = [b for b in range(lb["n_bins"])
                        if lb["counts"][b] >= 15]
                if len(keep) < 4:
                    continue
                n_per = int(min(lb["counts"][b] for b in keep))
                med = np.array([lb["median"][b] for b in keep])
                D_beh = np.abs(med[:, None] - med[None, :])
                D = dmat_from(P, lb["label"], keep, n_per,
                              np.random.default_rng(0))
                r, p = _mantel_rp(D_beh, D, seed=1)
                grid_r[ri, ci] = r
                grid_p[ri, ci] = p
                grid_k[ri, ci] = red.k
                grid_rel[ri, ci] = split_half_rel(P, lb["label"], keep, n_per)
        results[(sid, tag)] = dict(r=grid_r, p=grid_p, rel=grid_rel,
                                   kk=grid_k, rows=rows, group=s_full.group,
                                   day=s_full.training_day)
        best = np.nanargmax(np.where(np.isnan(grid_r), -9, grid_r))
        bi, bj = np.unravel_index(best, grid_r.shape)
        print(f"  [{tag}] best r = {grid_r[bi,bj]:+.3f} at {rows[bi]}, "
              f"{NBINS[bj]} bins (p={grid_p[bi,bj]:.3f}, "
              f"reliability={grid_rel[bi,bj]:.2f})")

# --------------------------------------------------------------------------- #
# FIGURE 1 — the (reducer x n_bins) surface, per session and epoch
for tag in ["A_prelick", "B_full"]:
    keys = [k for k in results if k[1] == tag]
    if not keys:
        continue
    fig, axes = plt.subplots(3, len(keys), figsize=(4.6 * len(keys), 10.5),
                             constrained_layout=True, squeeze=False)
    for c, key in enumerate(keys):
        R = results[key]
        for r_, (M, lab, cm, vlim) in enumerate([
                (R["r"], "MANTEL r = corr( |$\\Delta$ median lick time| ,\n"
                         "affine-inv. distance between bin covariances )",
                 "RdBu_r", (-1, 1)),
                (R["rel"], "split-half reliability of the geometric\n"
                           "distance matrix (is it measurable at all?)",
                 "viridis", (0, 1)),
                (np.log10(np.maximum(R["p"], 1e-3)),
                 "log10 Mantel p (permuting bin labels)\n"
                 "NB: floor is ~1/n_bins! — 4 bins cannot go below 0.04",
                 "magma_r", (-3, 0))]):
            a = axes[r_, c]
            im = a.imshow(M, aspect="auto", cmap=cm, vmin=vlim[0], vmax=vlim[1])
            fig.colorbar(im, ax=a, fraction=0.046)
            a.set_xticks(range(len(NBINS)), NBINS)
            a.set_yticks(range(len(R["rows"])), R["rows"], fontsize=6.5)
            a.set_xlabel("number of lick-time bins")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        a.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                               fontsize=5.5,
                               color="k" if abs(M[i, j]) < 0.6 * max(abs(vlim[0]), abs(vlim[1])) else "w")
            if r_ == 0:
                a.set_title(f"{key[0][:5]} ({R['group']}, d{R['day']})\n{lab}",
                            fontsize=8)
            else:
                a.set_title(lab, fontsize=8)
    fig.suptitle(
        "Stage 2b — sweeping reducer (rows) x number of lick-time bins "
        "(columns).   EPOCH "
        f"{'A: post-cue pre-lick 0-0.2 s' if tag=='A_prelick' else 'B: full window -0.5 to 1.5 s'}\n"
        "QUANTITY PLOTTED (top row): trials are grouped into lick-time bins; "
        "each bin gives one covariance matrix of population states.\n"
        "Mantel r correlates, over all bin PAIRS, the behavioural distance "
        "|difference in median lick time| against the\n"
        "geometric distance between those bins' covariance matrices. "
        "r = +1 means the two distance matrices agree perfectly.",
        fontsize=9)
    save(fig, f"s2b_surface_{tag}.png")

# --------------------------------------------------------------------------- #
# FIGURE 2 — the strength/reliability trade-off, all configs as a point cloud
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
mk = {"A_prelick": "o", "B_full": "s"}
for key, R in results.items():
    sid, tag = key
    kk = R["kk"].ravel(); rr = R["r"].ravel(); rel = R["rel"].ravel()
    pp = R["p"].ravel()
    ok = np.isfinite(rr)
    sc = axes[0].scatter(rel[ok], rr[ok], c=np.log10(kk[ok]), cmap="viridis",
                         marker=mk[tag], s=26, alpha=0.85)
    axes[1].scatter(kk[ok], rr[ok], marker=mk[tag], s=26, alpha=0.7,
                    label=f"{sid[:5]} {tag[0]}")
    axes[2].scatter(kk[ok], pp[ok], marker=mk[tag], s=26, alpha=0.7)
fig.colorbar(sc, ax=axes[0], label="log10 dimensionality k")
axes[0].set_xlabel("split-half reliability of the geometric distance matrix")
axes[0].set_ylabel("Mantel r:  corr( |$\\Delta$ median lick time| ,\n"
                   "distance between bin covariances )")
axes[0].axhline(0, color="k", lw=0.7)
axes[0].set_title("THE TRADE-OFF\nreliable AND strong = upper right")
axes[1].set_xscale("log"); axes[1].set_xlabel("dimensionality k")
axes[1].set_ylabel("Mantel r (see left panel)"); axes[1].axhline(0, color="k", lw=0.7)
axes[1].legend(fontsize=5.5, ncol=2)
axes[1].set_title("Effect size vs dimensionality\n(circle = pre-lick, square = full)")
axes[2].set_xscale("log"); axes[2].set_yscale("log")
axes[2].axhline(0.05, color="r", ls="--", label="p = 0.05")
axes[2].set_xlabel("dimensionality k"); axes[2].set_ylabel("Mantel p")
axes[2].legend(fontsize=6)
axes[2].set_title("Significance vs dimensionality")
fig.suptitle("Stage 2b — each point is ONE (reducer, n_bins) configuration.  "
             "Mantel r = agreement between the behavioural distance matrix\n"
             "(|difference in median lick time| between bin pairs) and the "
             "geometric distance matrix (affine-invariant distance between "
             "those bins' population covariances).", fontsize=9)
save(fig, "s2b_tradeoff.png")

# --------------------------------------------------------------------------- #
# FIGURE 3 — null reducer: does the effect need PCA's directions at all?
print("\n=== NULL REDUCER CHECK (random projections at matched k) ===")
fig, axes = plt.subplots(1, len(SESSIONS), figsize=(4.5 * len(SESSIONS), 3.6),
                         constrained_layout=True, squeeze=False)
for c, sid in enumerate(SESSIONS):
    s_full = load_session(DATA_DIR / f"{sid}.h5")
    s = s_full.prelick(WINDOW)
    Z = s.zscored(baseline=(s.t[0], s.t[0] + 0.1))
    lb = s.lick_bins(8)
    keep = [b for b in range(lb["n_bins"]) if lb["counts"][b] >= 15]
    n_per = int(min(lb["counts"][b] for b in keep))
    med = np.array([lb["median"][b] for b in keep])
    D_beh = np.abs(med[:, None] - med[None, :])
    rp, rr_pca = [], []
    for k in KS:
        red = PCA(k).fit(Z)
        D = dmat_from(red.transform(Z), lb["label"], keep, n_per,
                      np.random.default_rng(0))
        rr_pca.append(_mantel_rp(D_beh, D, n_perm=200)[0])
        rs = []
        for seed in range(8):
            redr = RandomProjection(k, seed=seed).fit(Z)
            Dr = dmat_from(redr.transform(Z), lb["label"], keep, n_per,
                           np.random.default_rng(0))
            rs.append(_mantel_rp(D_beh, Dr, n_perm=100)[0])
        rp.append(rs)
    rp = np.array(rp)
    a = axes[0, c]
    a.plot(KS, rr_pca, "o-", color="C0", label="PCA (top-k)")
    a.errorbar(KS, rp.mean(1), yerr=rp.std(1), fmt="s--", color="C3",
               label="random k directions (null)")
    a.set_xscale("log"); a.set_xlabel("dimensionality k")
    a.set_ylabel("Mantel r:  corr( |$\\Delta$ median lick|,\ndist. between bin covariances )")
    a.axhline(0, color="k", lw=0.7)
    a.legend(fontsize=6)
    a.set_title(f"{sid[:5]} ({s_full.group})\nDoes the effect need PCA's "
                f"directions?")
    print(f"  {sid[:5]}: PCA r = {np.round(rr_pca,2)}")
    print(f"         random r = {np.round(rp.mean(1),2)} "
          f"(+-{np.round(rp.std(1),2)})")
fig.suptitle("Stage 2b — NULL REDUCER.  Same Mantel r as elsewhere (behavioural "
             "vs geometric distance matrices, 8 lick-time bins),\ncomputed in "
             "the top-k PC space (blue) versus k RANDOM orthonormal directions "
             "(red, mean+-sd over 8 seeds).\nA gap means the effect depends on "
             "WHICH directions, not merely on how many.  EPOCH A, 8 bins.",
             fontsize=9)
save(fig, "s2b_null_reducer.png")
print("\nDone.")
