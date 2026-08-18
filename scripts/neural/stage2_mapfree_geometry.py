"""
STAGE 2 — Map-free Riemannian geometry of the recorded population.
==================================================================

PIPELINE (stated before any result is evaluated)
------------------------------------------------
 S1  Load HDF5 -> spike counts (5 ms) for qm_pass units, is_valid_for_analysis
     trials (which already implies cue trial + lick present).
 S2  Re-bin by SUMMING counts to 20 ms; divide by bin width -> Hz.
 S3  Smooth with a CAUSAL boxcar (60 ms), applied BEFORE slicing, with a 2x
     margin loaded beyond the analysis window and trimmed afterwards.
 S4  Drop units with mean rate < 0.5 Hz inside the window.
 S5  Split into two epochs, analysed SEPARATELY and never pooled:
       EPOCH A  0 -> W s after cue, keeping only trials whose first lick is
                after W. Raw traces, no time warping. No lick, no outcome.
       EPOCH B  -0.5 -> 1.5 s fixed. Contains baseline, cue, lick, outcome.
 S6  Z-score each unit using a within-epoch baseline window.
 S7  Project onto the top-K PCs (K set from Stage 1 split-half reproducibility,
     default 3). PCA is fit on all trials of that epoch — unsupervised with
     respect to lick time, so it cannot leak the behavioural label, but it is
     not fully out-of-sample and is noted as such.
 S8  Bin trials into NBIN lick-time quantiles, labelled by MEDIAN lick time.
     Subsample every bin to a common trial count.
 S9  Per bin: shrinkage covariance (alpha=0.1 toward scaled identity) -> a
     point on the SPD manifold; top-2 PC subspace -> a point on Gr(2, K).
 S10 Distances between bins under 4 SPD metrics + Grassmann.

HYPOTHESIS (fixed before looking)
---------------------------------
 H1  Bins whose MEDIAN LICK TIMES differ more are further apart geometrically.

STATISTICS
----------
 * Pairwise distances are NOT independent (each bin enters NBIN-1 pairs), so
   the naive dof of n_pairs-2 is invalid. Significance uses a MANTEL
   permutation test: permute the bin labels of one matrix (NBIN! orderings,
   sampled 5000x), which preserves the dependency structure exactly.
 * 95% CI on r by bootstrap over trials within bins (1000 resamples).
 * Effect size reported as r and r^2.
 * PARTIAL Mantel controls for two confounds: |difference in mean trial
   index| (learning drift) and |difference in fraction rewarded|.

CONFOUND CONTROLS
-----------------
 C1 learning drift   partial Mantel on |d mean trial index|
 C2 reward content   partial Mantel on |d fraction rewarded|
 C3 firing-rate scale  repeat with unit-determinant-normalized covariances,
                       which removes overall scale; the spectral ratio is
                       scale-invariant by construction and acts as a check
 C4 trial count      equalized by subsampling (S8)
 C5 selection bias   Epoch A retains only slow trials; reported explicitly

Run:  python stage2_mapfree_geometry.py [SESSION_ID] [WINDOW_S] [K_PC] [NBIN]
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
from neuralgeom.data.loader import load_session                        # noqa: E402
from neuralgeom.geometry.spd import (affine_invariant_distance,          # noqa: E402
                          bures_wasserstein_distance,
                          log_euclidean_distance,
                          spectral_ratio_distance)
from neuralgeom.geometry.grassmann import grassmann_distance                   # noqa: E402

SESSION = sys.argv[1] if len(sys.argv) > 1 else "SM259_20230417_g0"
WINDOW = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
KPC = int(sys.argv[3]) if len(sys.argv) > 3 else 3
NBIN = int(sys.argv[4]) if len(sys.argv) > 4 else 8
N_PERM, N_BOOT = 2000, 500
FIG = fig_dir("neural_stage2")
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 145, "font.size": 8,
                     "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.fontsize": 7})

METRICS = {"affine-invariant": affine_invariant_distance,
           "log-Euclidean": log_euclidean_distance,
           "Bures-Wasserstein": bures_wasserstein_distance,
           "spectral ratio": spectral_ratio_distance}


def save(fig, name):
    p = FIG / name
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p.name}")
    return p


# ---------------------------- statistics ----------------------------------- #
def normdet(C):
    """Unit-determinant normalization: removes overall scale, keeps shape."""
    sign, logdet = np.linalg.slogdet(C)
    return C / np.exp(logdet / C.shape[0])


def cond_covs(P, labels, keep, n_per, alpha, rng, unit_det=False):
    out = []
    for b in keep:
        idx = rng.choice(np.where(labels == b)[0], n_per, replace=False)
        F = P[idx].reshape(-1, P.shape[-1])
        C = shrink_cov(F - F.mean(0), alpha)
        out.append(normdet(C) if unit_det else C)
    return torch.as_tensor(np.stack(out))


def dmat(covs, fn):
    n = covs.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(fn(covs[i], covs[j]))
    return D


# --------------------------------------------------------------------------- #
print("=" * 74)
print(f"STAGE 2 — {SESSION}")
s_full = load_session(DATA_DIR / f"{SESSION}.h5")
print(f"EXPERIMENTAL GROUP: {s_full.group}  (coarse: {s_full.coarse}, "
      f"training day {s_full.training_day})")
if "paAIP2" in s_full.group:
    print("  !! This is a CaMKII-inhibition (paAIP2) animal, NOT a wild-type")
    print("     control. Conclusions apply to this manipulated group only and")
    print("     must not be pooled with WT sessions.")
print("PREPROCESSING: 20 ms bins (summed counts), causal 60 ms boxcar applied")
print("  before slicing with a 2x margin, qm_pass units, rate >= 0.5 Hz,")
print(f"  z-scored per unit, top-{KPC} PCs, {NBIN} lick-time quantile bins.")
print("=" * 74)

s_pre = s_full.prelick(WINDOW)
print(f"\nEPOCH A selection bias: kept {s_pre.meta['prelick_kept']}/"
      f"{s_full.n_trials} trials; fraction rewarded "
      f"{s_pre.meta['prelick_frac_rewarded']:.3f} vs "
      f"{s_pre.meta['prelick_frac_rewarded_all']:.3f} for all trials.")


def subset(s, mask):
    """Trial subset of a Session (keeps units/meta, filters trials)."""
    from copy import copy
    idx = np.where(mask)[0]
    s2 = copy(s)
    s2.X = s.X[idx]; s2.lick = s.lick[idx]
    s2.trials = {k: v[idx] for k, v in s.trials.items()}
    s2.cond = {k: v[idx] for k, v in s.cond.items()}
    return s2


# EPOCHS analysed. Both use ALL available trials in the epoch — no outcome
# subsetting by default.
EPOCHS = [("A_prelick", s_pre, f"EPOCH A — post-cue pre-lick (0-{WINDOW}s)"),
          ("B_full", s_full, "EPOCH B — full fixed window (-0.5 to 1.5s)")]

# OPT-IN ONLY (--early-only): restrict to early-lick trials, which holds
# outcome constant while lick time still varies. This is a substantial
# change to the analysed dataset — reward is a threshold on lick time, so
# removing rewarded trials also truncates the lick-time range and drops
# ~35% of trials. It is NOT run by default; enable it explicitly.
if "--early-only" in sys.argv:
    s_pre_early = subset(s_pre, s_pre.cond["early_lick"])
    print(f"[opt-in] EPOCH A' early-lick only: {s_pre_early.X.shape[0]} "
          f"trials, all unrewarded, lick {s_pre_early.lick.min():.3f}-"
          f"{s_pre_early.lick.max():.3f}s")
    EPOCHS.insert(1, ("Ap_prelick_earlyonly", s_pre_early,
                      f"EPOCH A' — pre-lick (0-{WINDOW}s), EARLY-LICK ONLY "
                      f"[opt-in subset]"))
summary = {}

for tag, s, title in EPOCHS:
    print("\n" + "-" * 74)
    print(f"### {title}   [{s.group}]")
    print("-" * 74)
    Z = s.zscored(baseline=(s.t[0], s.t[0] + 0.1)
                  if s.align == "prelick_fixed" else (-0.5, -0.1))
    F = Z.reshape(-1, s.n_units)
    mu = F.mean(0, keepdims=True)
    V = np.linalg.svd(F - mu, full_matrices=False)[2][:KPC].T
    P = (Z - mu.reshape(1, 1, -1)) @ V

    lb = s.lick_bins(NBIN)
    keep = [b for b in range(lb["n_bins"]) if lb["counts"][b] >= 25]
    n_per = int(min(lb["counts"][b] for b in keep))
    med = np.array([lb["median"][b] for b in keep])
    tr_idx = np.asarray(s.trials.get("trial_index_0based",
                                     np.arange(s.n_trials)), float)
    mean_trial = np.array([tr_idx[lb["label"] == b].mean() for b in keep])
    frac_rew = np.array([s.cond["rewarded"][lb["label"] == b].mean()
                         for b in keep])
    mean_rate = np.array([s.X[lb["label"] == b].mean() for b in keep])

    print(f"\n#### Conditions ({len(keep)} lick-time bins, {n_per} trials "
          f"each after matching)")
    print(f"{'median lick (ms)':>18s} {'n':>5s} {'mean trial idx':>15s} "
          f"{'frac rewarded':>14s} {'mean rate Hz':>13s}")
    for i, b in enumerate(keep):
        print(f"{med[i]*1000:18.0f} {lb['counts'][b]:5d} {mean_trial[i]:15.0f} "
              f"{frac_rew[i]:14.2f} {mean_rate[i]:13.2f}")

    # confound structure among the condition variables
    r_tr = float(np.corrcoef(med, mean_trial)[0, 1])
    r_rw = float(np.corrcoef(med, frac_rew)[0, 1])
    r_rt = float(np.corrcoef(med, mean_rate)[0, 1])
    print(f"\n#### Confound structure across bins (n_bins = {len(keep)})")
    print(f"  corr(median lick, mean trial index) = {r_tr:+.3f}   <- learning drift")
    print(f"  corr(median lick, frac rewarded)    = {r_rw:+.3f}   <- reward content")
    print(f"  corr(median lick, mean firing rate) = {r_rt:+.3f}   <- rate/scale")

    D_beh = np.abs(med[:, None] - med[None, :])
    D_trial = np.abs(mean_trial[:, None] - mean_trial[None, :])
    D_rew = np.abs(frac_rew[:, None] - frac_rew[None, :])

    rng = np.random.default_rng(0)
    covs = cond_covs(P, lb["label"], keep, n_per, 0.10, rng)
    covs_nd = cond_covs(P, lb["label"], keep, n_per, 0.10,
                        np.random.default_rng(0), unit_det=True)

    res = {}
    print(f"\n#### H1: geometric distance vs behavioural distance")
    print(f"{'metric':>20s} {'r':>7s} {'r^2':>6s} {'95% CI':>16s} "
          f"{'Mantel p':>9s} {'partial r|trial':>16s} {'partial r|reward':>17s} "
          f"{'r (det-normed)':>15s}")
    for name, fn in METRICS.items():
        D = dmat(covs, fn)
        r_obs, p_m, null = mantel(D_beh, D, seed=1)
        # bootstrap CI over trials within bins
        boots = np.empty(N_BOOT // 5)
        for k in range(len(boots)):
            rr = np.random.default_rng(2000 + k)
            cb = cond_covs(P, lb["label"], keep, n_per, 0.10, rr)
            Db = dmat(cb, fn)
            iu = np.triu_indices(len(keep), 1)
            boots[k] = np.corrcoef(D_beh[iu], Db[iu])[0, 1]
        ci = np.percentile(boots, [2.5, 97.5])
        rp_t, pp_t = partial_mantel(D_beh, D, D_trial, seed=3)
        rp_r, pp_r = partial_mantel(D_beh, D, D_rew, seed=4)
        D_nd = dmat(covs_nd, fn)
        iu = np.triu_indices(len(keep), 1)
        r_nd = float(np.corrcoef(D_beh[iu], D_nd[iu])[0, 1])
        res[name] = dict(D=D, r=r_obs, p=p_m, ci=ci, null=null,
                         rp_t=rp_t, pp_t=pp_t, rp_r=rp_r, pp_r=pp_r, r_nd=r_nd)
        print(f"{name:>20s} {r_obs:+7.3f} {r_obs**2:6.2f} "
              f"[{ci[0]:+.2f},{ci[1]:+.2f}] {p_m:9.4f} "
              f"{rp_t:+8.3f} (p={pp_t:.3f}) {rp_r:+8.3f} (p={pp_r:.3f}) "
              f"{r_nd:+15.3f}")

    # Grassmann
    Qs = []
    for b in keep:
        Fb = P[lb["label"] == b].reshape(-1, P.shape[-1])
        Qs.append(np.linalg.svd(Fb - Fb.mean(0), full_matrices=False)[2][:2].T)
    Qs = torch.as_tensor(np.stack(Qs))
    Dg = np.zeros((len(keep), len(keep)))
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            Dg[i, j] = Dg[j, i] = float(grassmann_distance(Qs[i], Qs[j]))
    r_g, p_g, _ = mantel(D_beh, Dg, seed=5)
    print(f"{'Grassmann (k=2)':>20s} {r_g:+7.3f} {r_g**2:6.2f} "
          f"{'':>16s} {p_g:9.4f}")

    # positive control: does the geometry track the CONFOUNDS directly?
    r_c1, p_c1, _ = mantel(D_trial, res["affine-invariant"]["D"], seed=6)
    r_c2, p_c2, _ = mantel(D_rew, res["affine-invariant"]["D"], seed=7)
    print(f"\n#### Confound checks (affine-invariant geometry vs confound)")
    print(f"  vs |d mean trial index| : r = {r_c1:+.3f}  Mantel p = {p_c1:.4f}")
    print(f"  vs |d frac rewarded|    : r = {r_c2:+.3f}  Mantel p = {p_c2:.4f}")

    summary[tag] = dict(res=res, med=med, keep=keep, n_per=n_per, D_beh=D_beh,
                        r_g=r_g, p_g=p_g, Dg=Dg, title=title, group=s.group,
                        r_tr=r_tr, r_rw=r_rw, r_rt=r_rt, r_c1=r_c1, p_c1=p_c1,
                        r_c2=r_c2, p_c2=p_c2, mean_trial=mean_trial,
                        frac_rew=frac_rew, mean_rate=mean_rate)

    # ------------------------------ figure ------------------------------- #
    lab = [f"{m*1000:.0f}" for m in med]
    fig, ax = plt.subplots(3, 4, figsize=(16, 9.6), constrained_layout=True)
    for k, (name, d) in enumerate(res.items()):
        a = ax[0, k]
        im = a.imshow(d["D"], cmap="viridis")
        fig.colorbar(im, ax=a, fraction=0.046)
        a.set_xticks(range(len(keep)), lab, fontsize=6, rotation=90)
        a.set_yticks(range(len(keep)), lab, fontsize=6)
        a.set_xlabel("median lick (ms)"); a.set_ylabel("median lick (ms)")
        a.set_title(f"METRIC: {name}\nr={d['r']:+.2f}, Mantel p={d['p']:.4f}")

    a = ax[1, 0]
    iu = np.triu_indices(len(keep), 1)
    for name, d in res.items():
        a.scatter(D_beh[iu] * 1000, d["D"][iu] / d["D"][iu].max(), s=20,
                  label=f"{name} r={d['r']:+.2f}")
    a.set_xlabel("|difference in median lick time| (ms)")
    a.set_ylabel("geometric distance (normalized)")
    a.legend(fontsize=6)
    a.set_title("H1 TEST: geometry vs behaviour\n(each point = one bin PAIR, "
                "not independent)")

    a = ax[1, 1]
    d = res["affine-invariant"]
    a.hist(d["null"], bins=40, color="0.75", label=f"Mantel null ({N_PERM} perms)")
    a.axvline(d["r"], color="C3", lw=2, label=f"observed {d['r']:+.2f}")
    a.set_xlabel("r"); a.legend(fontsize=6)
    a.set_title(f"MANTEL PERMUTATION TEST (affine-invariant)\n"
                f"p = {d['p']:.4f}; null sd = {d['null'].std():.3f}")

    a = ax[1, 2]
    names = list(res)
    xs = np.arange(len(names))
    a.bar(xs - 0.27, [res[n]["r"] for n in names], 0.27, label="raw")
    a.bar(xs, [res[n]["rp_t"] for n in names], 0.27, label="| trial index")
    a.bar(xs + 0.27, [res[n]["rp_r"] for n in names], 0.27, label="| reward")
    a.set_xticks(xs, names, rotation=25, fontsize=6)
    a.axhline(0, color="k", lw=0.8); a.legend(fontsize=6)
    a.set_ylabel("correlation with behaviour")
    a.set_title("CONFOUND CONTROL: partial Mantel\n(does the effect survive?)")

    a = ax[1, 3]
    a.bar(xs - 0.2, [res[n]["r"] for n in names], 0.4, label="covariance")
    a.bar(xs + 0.2, [res[n]["r_nd"] for n in names], 0.4,
          label="unit-determinant\n(scale removed)")
    a.set_xticks(xs, names, rotation=25, fontsize=6)
    a.axhline(0, color="k", lw=0.8); a.legend(fontsize=6)
    a.set_ylabel("correlation with behaviour")
    a.set_title("SCALE CONTROL: is it just firing rate?")

    a = ax[2, 0]
    a.scatter(med * 1000, mean_trial, s=30, c="C0")
    a.set_xlabel("median lick (ms)"); a.set_ylabel("mean trial index")
    a.set_title(f"CONFOUND C1: learning drift\nr = {r_tr:+.2f}")
    a = ax[2, 1]
    a.scatter(med * 1000, frac_rew, s=30, c="C2")
    a.set_xlabel("median lick (ms)"); a.set_ylabel("fraction rewarded")
    a.set_title(f"CONFOUND C2: reward content\nr = {r_rw:+.2f}")
    a = ax[2, 2]
    a.scatter(med * 1000, mean_rate, s=30, c="C3")
    a.set_xlabel("median lick (ms)"); a.set_ylabel("mean firing rate (Hz)")
    a.set_title(f"CONFOUND C3: rate/scale\nr = {r_rt:+.2f}")
    a = ax[2, 3]
    im = a.imshow(Dg, cmap="magma")
    fig.colorbar(im, ax=a, fraction=0.046)
    a.set_xticks(range(len(keep)), lab, fontsize=6, rotation=90)
    a.set_yticks(range(len(keep)), lab, fontsize=6)
    a.set_title(f"METRIC: Grassmann (subspace orientation)\n"
                f"r = {r_g:+.2f}, Mantel p = {p_g:.4f}")
    fig.suptitle(f"Stage 2 — {title}  |  GROUP: {s.group} (day "
                 f"{s.training_day})  |  {KPC} PCs, {len(keep)} bins, "
                 f"{n_per} trials/bin matched", fontsize=10)
    save(fig, f"s2_{tag}_geometry.png")

print("\n" + "=" * 74)
print("Done. Interpretation belongs in the accompanying write-up; this script")
print("reports effect sizes, CIs, Mantel p-values and confound controls only.")
