"""Fit one linear flow to synthetic data with known dynamics, describe it, and check the
recovery against the generating dynamics.

Same simple recipe as the real demo -- one vector field dz/dt = A z + b fit to all post-cue
trials, a within-trial time-shuffle null as the significance test, no binning, no instrumental variable,
no metric choices -- but here we also know the truth, so every panel carries a ground-truth
reference.

Two figures:
  synthetic_data.png : the known latent field + single-trial latent paths, the embedded
                       'neural' state trajectories, PC time courses, and the measured velocity
                       field.
  synthetic_fit.png  : the fitted flow's streamlines + fixed point (compare to the known lick
                       attractor), the recovered cue input over time (compare to the known cue
                       pulse), the potential landscape, and the fit quality + decomposition vs
                       the shuffle null.

Run:  python scripts/attractor/dynamics_synthetic.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# --- make the neuralgeom package importable without installing it ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import neuralgeom.synth as syn
import neuralgeom.dynamics as dyn
from neuralgeom.data import from_synthetic, state_pca
from neuralgeom.synth.attractor import attractor_field, REST_ATTRACTOR, LICK_ATTRACTOR, T_CUE, CUE_DUR
from neuralgeom.paths import fig_dir

FIGDIR = str(fig_dir("demos"))
NPC = 3
BIN = 0.05
BASE = -0.6


def imag_eigenvalue_fraction(A):
    """Coordinate-invariant rotation measure: sum|Im(eig)| / sum|eig|.  0 = pure
    gradient/contraction (all-real spectrum), higher = more rotation."""
    ev = np.linalg.eigvals(np.asarray(A))
    return float(np.sum(np.abs(ev.imag)) / (np.sum(np.abs(ev)) + 1e-12))


def _roll(V, group, rng):
    Vs = V.copy()
    for g in np.unique(group):
        m = np.where(group == g)[0]
        if len(m) > 2:
            Vs[m] = np.roll(V[m], int(rng.integers(1, len(m))), axis=0)
    return Vs


def _r2(Y, Yh):
    return 1 - np.sum((Y - Yh) ** 2) / np.sum((Y - Y.mean(0)) ** 2)



def figure_data(d, sess, Z, t):
    lick = sess.lick; lat = d["latent"]
    ok = np.where(np.isfinite(lick))[0]
    norm = plt.Normalize(np.nanpercentile(lick[ok], 5), np.nanpercentile(lick[ok], 95))
    col = plt.cm.viridis(norm(lick))
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # 1. KNOWN latent field + single-trial latent paths
    a = ax[0, 0]
    gx, gy = np.meshgrid(np.linspace(-2, 20, 20), np.linspace(-2, 20, 20))
    U, Vv = attractor_field(gx.ravel(), gy.ravel())
    a.streamplot(gx, gy, U.reshape(gx.shape), Vv.reshape(gx.shape), color="#bbb", density=1.0,
                 linewidth=.5, arrowsize=.6)
    for i in ok[:60]:
        m = (t >= 0) & (t < lick[i] + 0.1)
        a.plot(lat[i][m, 0], lat[i][m, 1], color=col[i], alpha=.4, lw=.7)
    a.scatter(*REST_ATTRACTOR, c="k", s=70, marker="o", label="baseline attractor")
    a.scatter(*LICK_ATTRACTOR, c="r", s=120, marker="*", label="lick attractor")
    a.set_xlabel("cue mode X"); a.set_ylabel("ramping mode Y")
    a.set_title("KNOWN latent dynamics + single-trial paths", fontsize=10); a.legend(fontsize=8)

    # 2. embedded neural state (PCA) trajectories
    a = ax[0, 1]
    for i in ok[:60]:
        m = (t >= BASE) & (t < lick[i] + 0.1)
        zz = gaussian_filter1d(Z[i][m], 1.2, axis=0)
        a.plot(zz[:, 0], zz[:, 1], color=col[i], alpha=.4, lw=.8)
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.set_title("Embedded 'neural' state trajectories\n(PCA of the noisy embedding)", fontsize=10)

    # 3. ramping / PC time courses
    a = ax[1, 0]
    tm = (t >= BASE) & (t < 1.2)
    for i in ok[:50]:
        a.plot(t[tm], lat[i][tm, 1], color=col[i], alpha=.25, lw=.5)
    a.plot(t[tm], lat[ok][:, tm, 1].mean(0), "k", lw=2.5, label="mean ramping mode")
    a.axhline(15, color="r", ls="--", lw=1, label="lick threshold"); a.axvline(0, color="k", lw=1)
    a.set_xlabel("time from cue (s)"); a.set_ylabel("ramping mode Y")
    a.set_title("Ramp to threshold (the timed action)", fontsize=10); a.legend(fontsize=8)

    # 4. measured velocity field of the embedded state
    a = ax[1, 1]
    trials = [Z[i][(t >= BASE) & (t < lick[i] + 0.05)] for i in ok
              if ((t >= BASE) & (t < lick[i] + 0.05)).sum() >= 6]
    dv = dyn.trial_velocities(trials, BIN, gap=2, smooth_bins=1.0, causal=True)
    idx = np.random.default_rng(0).choice(len(dv["Z"]), min(400, len(dv["Z"])), replace=False)
    a.quiver(dv["Z"][idx, 0], dv["Z"][idx, 1], dv["V"][idx, 0], dv["V"][idx, 1],
             np.linalg.norm(dv["V"][idx], axis=1), cmap="viridis", alpha=.7, width=.004)
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.set_title("Measured velocity field dz/dt (embedded state)", fontsize=10)

    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    fig.colorbar(sm, ax=ax[0, 1], label="lick time (s)", shrink=.7)
    fig.suptitle("neuralgeom.dynamics, SYNTHETIC: the data (dynamics known)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGDIR, "dynamics_synthetic_data.png")
    fig.savefig(out, dpi=105); plt.close(fig); print("Saved", out)
    return dv


def figure_fit(d, sess, Z, t, dv):
    lick = sess.lick
    ok = np.where(np.isfinite(lick))[0]
    # ONE model: shared internal field A + onset-only cue input I(t); the input panel and the
    # field decomposition below both come from THIS fit.  The true synthetic field is a genuine
    # potential flow (little rotation), so the rotation measure should sit at/below its null.
    trials, keep = [], []
    for i in ok:
        mm = (t >= BASE) & (t < lick[i] + 0.1)
        if mm.sum() >= 8:
            trials.append(Z[i]); keep.append(mm)
    dI = dyn.trial_velocities(trials, BIN, gap=2, smooth_bins=3, causal=True, time=t, keep=keep)
    cond = np.array(["cue"] * dI["Z"].shape[0])
    free = dyn.fit_shared_input_free(dI["Z"], dI["V"], cond, dI["tsec"], groups=dI["group"],
                                     input_bin_s=0.05)
    A, b = free["field"]["A"], free["field"]["b"]
    ev = np.linalg.eigvals(A); zfix = np.linalg.solve(A, -b)

    rng = np.random.default_rng(0)
    null_cv, null_rot, null_ev = [], [], []
    for _ in range(20):
        fr = dyn.fit_shared_input_free(dI["Z"], _roll(dI["V"], dI["group"], rng), cond,
                                       dI["tsec"], groups=dI["group"], input_bin_s=0.05)
        null_cv.append(fr["cv_r2"]); null_rot.append(imag_eigenvalue_fraction(fr["field"]["A"]))
        null_ev.append(np.linalg.eigvals(fr["field"]["A"]))
    null_cv = np.array(null_cv); null_rot = np.array(null_rot)
    null_ev = np.concatenate(null_ev)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # 1. internal field streamlines + fixed point
    a = ax[0, 0]
    g0 = np.linspace(*np.percentile(dv["Z"][:, 0], [2, 98]), 22)
    g1 = np.linspace(*np.percentile(dv["Z"][:, 1], [2, 98]), 22)
    GX, GY = np.meshgrid(g0, g1)
    grid = np.tile(dv["Z"].mean(0), (GX.size, 1)); grid[:, 0] = GX.ravel(); grid[:, 1] = GY.ravel()
    F = grid @ A.T + b
    a.streamplot(GX, GY, F[:, 0].reshape(GX.shape), F[:, 1].reshape(GX.shape), color="#2a7",
                 density=1.0, linewidth=.7, arrowsize=.8)
    idx = np.random.default_rng(0).choice(len(dv["Z"]), min(250, len(dv["Z"])), replace=False)
    a.quiver(dv["Z"][idx, 0], dv["Z"][idx, 1], dv["V"][idx, 0], dv["V"][idx, 1], color="#999",
             alpha=.25, width=.003)
    if np.all(np.isfinite(zfix)) and g0.min() < zfix[0] < g0.max():
        a.scatter([zfix[0]], [zfix[1]], c="r", marker="*", s=180, zorder=6, label="fitted fixed point")
        a.legend(fontsize=8)
    a.set_xlim(g0.min(), g0.max()); a.set_ylim(g1.min(), g1.max())
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.set_title(f"Internal field  A z + b  (cue input removed)\n(model CV R2 = {free['cv_r2']:.2f})",
                fontsize=10)

    # 2. recovered input vs the KNOWN cue pulse (SAME fit)
    a = ax[0, 1]
    support = np.zeros(free["I"].shape[1]); kb = dyn.lds._input_bins(dI["tsec"], 0.05)
    for k in kb[kb >= 0]:
        if k < len(support):
            support[k] += 1
    y = np.where(support >= 5, free["input_norm"][0], np.nan)
    a.axvspan(0, CUE_DUR, color="#fd8", alpha=.5, label="TRUE cue window")
    a.plot(free["tau"], y, "-", lw=2.5, color="#c60", label="recovered |I(t)|")
    a.axvline(0, color="k", ls=":"); a.set_xlim(BASE, 1.2)
    a.set_xlabel("time from cue (s)"); a.set_ylabel("recovered input |I(t)|")
    a.set_title("Inferred cue input vs the known pulse\n(transient at onset, ~0 before)",
                fontsize=10)
    a.legend(fontsize=8)

    # 3. eigenvalue spectrum of the internal field (INVARIANT)
    a = ax[1, 0]
    a.scatter(null_ev.real, null_ev.imag, s=10, color="#ccc", alpha=.5, label="shuffle null")
    a.scatter(ev.real, ev.imag, s=90, color="#c33", zorder=5, label="data eigenvalues")
    a.axvline(0, color="k", lw=1); a.axhline(0, color="k", lw=.8)
    a.set_xlabel("Re(eig)  <0 contracting | >0 expanding"); a.set_ylabel("Im(eig)  = rotation rate")
    a.set_title("Eigenvalues of the internal field (coordinate-INVARIANT)\n"
                "true field is a potential flow -> little rotation", fontsize=9)
    a.legend(fontsize=8)

    # 4. testable claims vs null
    a = ax[1, 1]
    names = ["is there a flow?\n(model CV R2)", "is there rotation?\n(sum|Im|/sum|eig|)"]
    dat = [free["cv_r2"], imag_eigenvalue_fraction(A)]
    nullm = [null_cv.mean(), null_rot.mean()]; nulls = [null_cv.std(), null_rot.std()]
    x = np.arange(2)
    a.bar(x, dat, 0.45, color=["#37a", "#c39"], label="data")
    a.errorbar(x, nullm, yerr=np.array(nulls) * 2, fmt="_k", ms=30, lw=2, capsize=6,
               label="shuffle null (mean +/- 2 SD)")
    a.axhline(0, color="k", lw=.6); a.set_xticks(x); a.set_xticklabels(names, fontsize=9)
    a.set_ylim(min(-0.05, min(dat) - .05), max(1.0, max(dat) + .1))
    a.set_title("Data vs time-shuffle null", fontsize=10); a.legend(fontsize=8)

    fig.suptitle("neuralgeom.dynamics, SYNTHETIC: one fit -> cue input + internal-field "
                 "geometry", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIGDIR, "dynamics_synthetic_fit.png")
    fig.savefig(out, dpi=105); plt.close(fig); print("Saved", out)
    print(f"synthetic: model CV R2 = {free['cv_r2']:.2f} (null {null_cv.mean():+.2f}); "
          f"imaginary-eigenvalue fraction = {imag_eigenvalue_fraction(A):.2f} (null {null_rot.mean():.2f}); "
          f"{int(np.sum(np.abs(ev.imag) > 1e-6) // 2)} complex pairs")


def figure_regression_group(rows, title, shuffle_txt, target_unit, out):
    """Table + a row of scatters per regression: [target vs regressor (r, pre/post) |
    predicted-vs-actual PRE-cue | predicted-vs-actual POST-cue].  Same as the real demo."""
    n = len(rows)
    fig = plt.figure(figsize=(15.5, 2.6 + 3.0 * n))
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[0.55 + 0.09 * n] + [1] * n)
    axt = fig.add_subplot(gs[0, :]); axt.axis("off")
    header = ["regressor X", "target Y", "shuffle control", "CV R2 pre", "CV R2 post",
              "shuffle pre", "shuffle post", "r pre", "r post"]
    cells = [[r["regressor"], r["target"], shuffle_txt, f"{r['cv_pre']:+.2f}", f"{r['cv_post']:+.2f}",
              f"{r['sh_pre']:+.2f}", f"{r['sh_post']:+.2f}", f"{r['corr_pre']:+.2f}",
              f"{r['corr_post']:+.2f}"] for r in rows]
    tab = axt.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.7)
    for j in range(len(header)):
        tab[0, j].set_facecolor("#dde"); tab[0, j].set_text_props(weight="bold")
    axt.set_title(title, fontsize=12, pad=12)

    def pva(a, xy, cv, sh, unit, ep):
        if xy is None or len(xy[0]) < 3:
            a.axis("off"); a.set_title(f"{ep}: (too few samples)", fontsize=9); return
        x, y = np.asarray(xy[0]), np.asarray(xy[1])
        a.scatter(x, y, s=3, alpha=.12, color="#347")
        lo, hi = np.percentile(x, [1, 99])
        a.plot([lo, hi], [lo, hi], "r-", lw=1, label="identity")
        sl, ic = np.polyfit(x, y, 1); a.plot([lo, hi], [sl*lo+ic, sl*hi+ic], "k--", lw=1.1, label="fit")
        a.set_xlabel(f"actual {unit}"); a.set_ylabel("predicted")
        a.set_title(f"{ep}-cue:  CV R2 = {cv:+.2f}   shuffle = {sh:+.2f}", fontsize=9)
        a.legend(fontsize=7)

    for i, r in enumerate(rows):
        a = fig.add_subplot(gs[i + 1, 0])
        x, y, ep = r["tvr"]
        a.scatter(np.asarray(x)[ep == 0], np.asarray(y)[ep == 0], s=3, alpha=.12, color="#999",
                  label=f"pre-cue  r={r['corr_pre']:+.2f}")
        a.scatter(np.asarray(x)[ep == 1], np.asarray(y)[ep == 1], s=3, alpha=.12, color="#28a",
                  label=f"post-cue r={r['corr_post']:+.2f}")
        a.set_xlabel(f"{r['regressor']} (1st comp.)"); a.set_ylabel(r["target"])
        a.set_title(f"{r['name']}\nTARGET vs REGRESSOR", fontsize=9); a.legend(fontsize=7)
        pva(fig.add_subplot(gs[i + 1, 1]), r["scat_pre"], r["cv_pre"], r["sh_pre"], target_unit, "PRE")
        pva(fig.add_subplot(gs[i + 1, 2]), r["scat_post"], r["cv_post"], r["sh_post"], target_unit, "POST")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=100); plt.close(fig); print("Saved", out)


def figure_regressions(d, sess, Z, t):
    lick = sess.lick
    use = np.array([i for i in range(len(Z)) if np.isfinite(lick[i]) and lick[i] > 0.2])
    R = dyn.run_regression_suite(Z, lick, t, use, bin_s=BIN, base=BASE)
    figure_regression_group(
        R["velocity"], "neuralgeom.dynamics, SYNTHETIC: PREDICT VELOCITY (dz/dt) from "
        "neural state, pre- vs post-cue", "velocity rolled in time (within trial)",
        "velocity (a.u.)", os.path.join(FIGDIR, "dynamics_synthetic_regr_velocity.png"))
    figure_regression_group(
        R["time_to_lick"], "neuralgeom.dynamics, SYNTHETIC: PREDICT TIME-TO-LICK, pre- vs post-cue  "
        "(velocity regressors are MODEL velocities A z + b)",
        "regressor rolled in time (within trial)", "time-to-lick (s)",
        os.path.join(FIGDIR, "dynamics_synthetic_regr_timetolick.png"))
    print(f"synthetic: time-to-lick decoding  CV R2 pre {R['time_to_lick'][0]['cv_pre']:+.2f}  "
          f"post {R['time_to_lick'][0]['cv_post']:+.2f}")


def main():
    d = syn.generate("input", syn.CUE_AMPLITUDE_LEVELS, n_trials=150, seed=1, bin_s=BIN, n_neurons=80,
                     poisson=True, mean_count=0.24)
    sess = from_synthetic(d)
    t = sess.t
    fitmask = np.zeros(sess.X.shape[:2], bool)
    for i in range(sess.X.shape[0]):
        if np.isfinite(sess.lick[i]):
            fitmask[i] = (t >= BASE) & (t < sess.lick[i] + 0.1)
    Z = state_pca(sess, NPC, fit_mask=fitmask)
    dv = figure_data(d, sess, Z, t)
    figure_fit(d, sess, Z, t, dv)
    figure_regressions(d, sess, Z, t)


if __name__ == "__main__":
    main()
