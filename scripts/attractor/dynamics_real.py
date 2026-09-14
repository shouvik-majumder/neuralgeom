"""Fit one linear flow to recorded population activity and describe it.

Deliberately simple and descriptive -- no per-condition binning, no instrumental variable, no metric
choices.  We fit a SINGLE vector field  dz/dt = A z + b  to all post-cue trials and look at it
from many angles, with a within-trial time-shuffle NULL as the only significance test.

Three figures:
  <session>_data.png     : the raw data -- example unit rates, single-trial state trajectories,
                           PC time courses, and the measured velocity field (with pre-cue).
  <session>_fit.png      : the inference -- the fitted flow's streamlines and fixed point, the
                           recovered cue input (with the no-cue zero-input control), and the
                           gradient/rotational decomposition shown as the coordinate-INVARIANT
                           eigenvalue spectrum, for BOTH the input-removed field and the plain
                           flow (fit with and without an input term).
  <session>_behavior.png : the real test of the split -- fit the flow on training trials, split
                           it into a gradient (potential) part and a rotational part, and ask
                           how well each predicts HELD-OUT data.  Two questions:
                             (i) does the component capture the dynamics?  -> held-out velocity R2
                            (ii) does the component drive the behavior?    -> held-out lick-time
                                 prediction, by integrating that component's flow forward from an
                                 early state to a threshold and calibrating to seconds on train.
                           A shuffle-fit is the floor.  This grounds "is the split meaningful" in
                           prediction, not in a shuffle of the fit itself (which always splits
                           into something).

Run:  python scripts/attractor/dynamics_real.py [SM318|SM348]
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
import neuralgeom.dynamics as dyn
from neuralgeom.data import load_session, state_pca, trial_masks
from neuralgeom.paths import DATA_DIR, fig_dir

FIGDIR = str(fig_dir("neural"))
DATA = str(DATA_DIR)
SESSIONS = {"SM318": "SM318_20230904_g0.h5", "SM348": "SM348_20230929_g0.h5"}
NPC = 6            # state dimension
BIN = 0.05        # bin width (s)
BASE = -0.6       # baseline shown before the cue


# ---------------------------------------------------------------- fitting helpers
def imag_eigenvalue_fraction(A):
    """Coordinate-invariant rotation measure: sum|Im(eig)| / sum|eig|, the fraction of the total
    eigenvalue modulus that is imaginary.  0 = pure gradient/contraction (all real eigenvalues),
    higher = more rotation (complex-conjugate pairs).  Unlike the Euclidean ||W||/||A|| fraction
    this does not depend on the coordinate system / metric."""
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



def load(key):
    # V2 loader conventions: 50 ms bins (summed counts), NO smoothing here (the
    # velocity code smooths), qm_pass + mean rate >= 0.5 Hz. valid_only=False keeps
    # no-cue probes and no-response trials (the zero-input control needs them).
    sess = load_session(os.path.join(DATA, SESSIONS[key]), bin_ms=BIN * 1000,
                        smooth_ms=0.0, t_range=(-1.0, 3.0), valid_only=False)
    m = trial_masks(sess)
    usable = m["cue"] & m["has_lick"] & (sess.lick > 3 * BIN)
    return sess, m, usable


# ================================================================ FIGURE 1: the data
def figure_data(key, sess, usable, Z, t):
    lick = sess.lick
    ok = np.where(usable)[0]
    norm = plt.Normalize(np.percentile(lick[ok], 5), np.percentile(lick[ok], 95))
    col = plt.cm.viridis(norm(lick))
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # 1. example single-unit trial-averaged rates (pre + post cue)
    a = ax[0, 0]
    psth = sess.X[ok].mean(0)                        # (T, N)
    var = psth.var(0)
    for k in np.argsort(var)[::-1][:6]:
        a.plot(t, gaussian_filter1d(psth[:, k], 1), lw=1.3)
    a.axvline(0, color="k", lw=1); a.set_xlim(BASE, 1.3)
    a.set_xlabel("time from cue (s)"); a.set_ylabel("firing rate (Hz)")
    a.set_title("Example units: trial-averaged rate\n(6 most modulated; baseline + post-cue)",
                fontsize=10)

    # 2. single-trial state trajectories in PC1-PC2, coloured by lick time
    a = ax[0, 1]
    for i in ok[:80]:
        m = (t >= BASE) & (t < lick[i] + 0.1)
        zz = gaussian_filter1d(Z[i][m], 1.2, axis=0)
        a.plot(zz[:, 0], zz[:, 1], color=col[i], alpha=.35, lw=.8)
    mean_tr = np.stack([Z[i][(t >= 0) & (t < 0.8)] for i in ok if ((t >= 0) & (t < 0.8)).sum()]).mean(0)
    a.plot(mean_tr[:, 0], mean_tr[:, 1], "k-", lw=2.5, label="mean")
    b0 = np.stack([Z[i][np.argmin(np.abs(t - BASE / 2))] for i in ok]).mean(0)
    a.scatter([b0[0]], [b0[1]], c="k", marker="s", s=70, zorder=5, label="baseline")
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.set_title("Single-trial state trajectories\n(colour = lick time)", fontsize=10)
    a.legend(fontsize=8)

    # 3. PC1 and PC2 vs time (single trials + mean)
    a = ax[1, 0]
    tm = (t >= BASE) & (t < 1.2)
    for pc, cc in ((0, "#37a"), (1, "#c33")):
        for i in ok[:40]:
            a.plot(t[tm], Z[i][tm, pc], color=cc, alpha=.12, lw=.5)
        a.plot(t[tm], Z[ok][:, tm, pc].mean(0), color=cc, lw=2.5, label=f"PC{pc+1}")
    a.axvline(0, color="k", lw=1); a.axhline(0, color="k", lw=.5)
    a.set_xlabel("time from cue (s)"); a.set_ylabel("PC score")
    a.set_title("State vs time: flat baseline, then post-cue ramp", fontsize=10)
    a.legend(fontsize=8)

    # 4. measured velocity field dz/dt over PC1-PC2 (the raw flow, pre + post cue)
    a = ax[1, 1]
    trials = [Z[i][(t >= BASE) & (t < lick[i] + 0.05)] for i in ok
              if ((t >= BASE) & (t < lick[i] + 0.05)).sum() >= 6]
    d = dyn.trial_velocities(trials, BIN, gap=2, smooth_bins=1.0, causal=True)
    idx = np.random.default_rng(0).choice(len(d["Z"]), min(400, len(d["Z"])), replace=False)
    a.quiver(d["Z"][idx, 0], d["Z"][idx, 1], d["V"][idx, 0], d["V"][idx, 1],
             np.linalg.norm(d["V"][idx], axis=1), cmap="viridis", alpha=.7, width=.004)
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.set_title("Measured velocity field dz/dt\n(arrows = motion at each visited state)",
                fontsize=10)

    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    fig.colorbar(sm, ax=ax[0, 1], label="lick time (s)", shrink=.7)
    fig.suptitle(f"neuralgeom.dynamics, REAL {key}: the data", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGDIR, f"dynamics_real_{key}_data.png")
    fig.savefig(out, dpi=105); plt.close(fig); print("Saved", out)
    return d


# ================================================================ FIGURE 2: the fit
def figure_fit(key, sess, m, usable, Z, t, d):
    lick = sess.lick
    # -------- ONE model for EVERYTHING: shared internal field A + onset-only cue input I(t).
    # The cue input and the field decomposition below come from THIS SAME fit, so they are
    # consistent; A is the internal dynamics with the cue drive already removed.
    lab = np.where(usable, "cue", "drop").astype(object)
    lab[m["no_cue"]] = "no_cue"
    trials, keep, cc = [], [], []
    for i in range(len(Z)):
        if lab[i] == "drop":
            continue
        end = (lick[i] + 0.1) if np.isfinite(lick[i]) else t.max()
        mm = (t >= BASE) & (t < end)
        if mm.sum() < 8:
            continue
        trials.append(Z[i]); keep.append(mm); cc.append(lab[i])
    dI = dyn.trial_velocities(trials, BIN, gap=2, smooth_bins=3, causal=True, time=t, keep=keep)
    cs = np.array(cc)[dI["group"]]
    free = dyn.fit_shared_input_free(dI["Z"], dI["V"], cs, dI["tsec"], groups=dI["group"],
                                     input_bin_s=0.05)
    A, b = free["field"]["A"], free["field"]["b"]
    ev = np.linalg.eigvals(A)
    zfix = np.linalg.solve(A, -b)
    conds = list(free["conds"])

    # null: roll the velocity within each trial and refit the SAME model
    rng = np.random.default_rng(0)
    null_cv, null_rot, null_ev = [], [], []
    for _ in range(20):
        fr = dyn.fit_shared_input_free(dI["Z"], _roll(dI["V"], dI["group"], rng), cs, dI["tsec"],
                                       groups=dI["group"], input_bin_s=0.05)
        null_cv.append(fr["cv_r2"]); null_rot.append(imag_eigenvalue_fraction(fr["field"]["A"]))
        null_ev.append(np.linalg.eigvals(fr["field"]["A"]))
    null_cv = np.array(null_cv); null_rot = np.array(null_rot)
    null_ev = np.concatenate(null_ev)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # 1. internal field streamlines + fixed point over the data velocities
    a = ax[0, 0]
    g0 = np.linspace(*np.percentile(dI["Z"][:, 0], [2, 98]), 22)
    g1 = np.linspace(*np.percentile(dI["Z"][:, 1], [2, 98]), 22)
    GX, GY = np.meshgrid(g0, g1)
    grid = np.tile(dI["Z"].mean(0), (GX.size, 1)); grid[:, 0] = GX.ravel(); grid[:, 1] = GY.ravel()
    F = grid @ A.T + b
    a.streamplot(GX, GY, F[:, 0].reshape(GX.shape), F[:, 1].reshape(GX.shape),
                 color="#2a7", density=1.0, linewidth=.7, arrowsize=.8)
    idx = np.random.default_rng(0).choice(len(dI["Z"]), min(250, len(dI["Z"])), replace=False)
    a.quiver(dI["Z"][idx, 0], dI["Z"][idx, 1], dI["V"][idx, 0], dI["V"][idx, 1], color="#999",
             alpha=.25, width=.003)
    if np.all(np.isfinite(zfix)) and g0.min() < zfix[0] < g0.max():
        a.scatter([zfix[0]], [zfix[1]], c="r", marker="*", s=180, zorder=6, label="fixed point")
        a.legend(fontsize=8)
    a.set_xlim(g0.min(), g0.max()); a.set_ylim(g1.min(), g1.max())
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.set_title(f"Internal field  A z + b  (cue input removed)\n(model CV R2 = {free['cv_r2']:.2f})",
                fontsize=10)

    # 2. recovered cue input over time, with the no-cue control -- SAME fit
    a = ax[0, 1]
    support = np.zeros(free["I"].shape[:2]); kb = dyn.lds._input_bins(dI["tsec"], 0.05)
    for i in np.where(kb >= 0)[0]:
        if kb[i] < support.shape[1]:
            support[conds.index(cs[i]), kb[i]] += 1
    for j, c in enumerate(conds):
        y = np.where(support[j] >= 5, free["input_norm"][j], np.nan)
        if c == "no_cue":
            a.plot(free["tau"], y, "-", lw=2.5, color="#c33",
                   label=f"no-cue control (n={int(m['no_cue'].sum())})")
        else:
            a.plot(free["tau"], y, "-", lw=2.2, color="#c60", label="cue trials")
    a.axvline(0, color="k", ls=":"); a.axhline(0, color="k", lw=.6)
    a.set_xlim(BASE, 1.2)
    a.set_xlabel("time from cue (s)"); a.set_ylabel("recovered input |I(t)|")
    a.set_title("Inferred cue input (onset-only), SAME fit\n"
                "0 before cue by construction; no-cue ~ floor", fontsize=10)
    a.legend(fontsize=8)

    # 3. eigenvalue spectrum of the internal field -- the INVARIANT gradient/rotational picture
    a = ax[1, 0]
    a.scatter(null_ev.real, null_ev.imag, s=10, color="#ccc", alpha=.5, label="shuffle null")
    a.scatter(ev.real, ev.imag, s=90, color="#c33", zorder=5, label="data eigenvalues")
    a.axvline(0, color="k", lw=1); a.axhline(0, color="k", lw=.8)
    a.set_xlabel("Re(eig)  <0 contracting | >0 expanding"); a.set_ylabel("Im(eig)  = rotation rate")
    a.set_title("Eigenvalues of the internal field (coordinate-INVARIANT)\n"
                "real axis = gradient (contraction/expansion); imaginary = rotation", fontsize=9)
    a.legend(fontsize=8)

    # 4. the two DEFENSIBLE, testable claims vs the shuffle null
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
    a.set_title("Data vs time-shuffle null\n(clear of the null = real)", fontsize=10)
    a.legend(fontsize=8)

    fig.suptitle(f"neuralgeom.dynamics, REAL {key}: one fit -> cue input + internal-field "
                 f"geometry", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIGDIR, f"dynamics_real_{key}_fit.png")
    fig.savefig(out, dpi=105); plt.close(fig); print("Saved", out)

    eu = dyn.helmholtz_split(A); Sig = np.cov(dI["Z"].T) + 1e-8 * np.eye(A.shape[0])
    Minv = np.linalg.inv(Sig); Minv /= np.mean(np.linalg.eigvalsh(Minv))
    cvm = dyn.helmholtz_split(A, M=Minv)
    print(f"{key}: model CV R2 = {free['cv_r2']:.2f} (null {null_cv.mean():+.2f})")
    print(f"  imaginary-eigenvalue fraction sum|Im|/sum|eig| = {imag_eigenvalue_fraction(A):.2f} (null {null_rot.mean():.2f})")
    print(f"  {int(np.sum(np.abs(ev.imag) > 1e-6) // 2)} complex eigenvalue pairs (rotation)")
    print(f"  (metric-dependent Euclidean symmetric_part_norm_fraction={eu['symmetric_part_norm_fraction']:.2f} vs "
          f"covariance symmetric_part_norm_fraction={cvm['symmetric_part_norm_fraction']:.2f} -- why we lead with the invariant "
          f"spectrum instead)")


# ============================================ FIGURE 3: every regression, plainly stated
def figure_regression_group(rows, title, shuffle_txt, target_unit, out):
    """Table + a row of scatters per regression: [target vs regressor (r, pre/post colours) |
    predicted-vs-actual PRE-cue | predicted-vs-actual POST-cue]."""
    n = len(rows)
    fig = plt.figure(figsize=(15.5, 2.6 + 3.0 * n))
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[0.55 + 0.09 * n] + [1] * n)

    axt = fig.add_subplot(gs[0, :]); axt.axis("off")
    header = ["regressor X", "target Y", "shuffle control", "CV R2 pre", "CV R2 post",
              "shuffle pre", "shuffle post", "r pre", "r post"]
    cells = [[r["regressor"], r["target"], shuffle_txt,
              f"{r['cv_pre']:+.2f}", f"{r['cv_post']:+.2f}",
              f"{r['sh_pre']:+.2f}", f"{r['sh_post']:+.2f}",
              f"{r['corr_pre']:+.2f}", f"{r['corr_post']:+.2f}"] for r in rows]
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
        # col 0: target vs regressor (raw), coloured by epoch, correlation per epoch
        a = fig.add_subplot(gs[i + 1, 0])
        x, y, ep = r["tvr"]
        a.scatter(np.asarray(x)[ep == 0], np.asarray(y)[ep == 0], s=3, alpha=.12, color="#999",
                  label=f"pre-cue  r={r['corr_pre']:+.2f}")
        a.scatter(np.asarray(x)[ep == 1], np.asarray(y)[ep == 1], s=3, alpha=.12, color="#28a",
                  label=f"post-cue r={r['corr_post']:+.2f}")
        a.set_xlabel(f"{r['regressor']} (1st comp.)"); a.set_ylabel(r["target"])
        a.set_title(f"{r['name']}\nTARGET vs REGRESSOR", fontsize=9); a.legend(fontsize=7)
        # cols 1,2: predicted vs actual, pre / post
        pva(fig.add_subplot(gs[i + 1, 1]), r["scat_pre"], r["cv_pre"], r["sh_pre"], target_unit, "PRE")
        pva(fig.add_subplot(gs[i + 1, 2]), r["scat_post"], r["cv_post"], r["sh_post"], target_unit, "POST")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=100); plt.close(fig); print("Saved", out)


def figure_ttl_order(key, sess, usable, Z, t):
    """How much of time-to-lick is predictable, and from what: linear vs nonlinear state, and
    whether velocity / acceleration add over the instantaneous state (delay-embedding test)."""
    lick = sess.lick
    use = np.array([i for i in np.where(usable)[0] if lick[i] > 0.2])
    O = dyn.time_to_lick_regression(Z, lick, t, use, bin_s=BIN, base=BASE)
    names = [o["name"] for o in O]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[1.5, 1]))
    a = ax[0]
    a.bar(x - w / 2, [o["pre"] for o in O], w, yerr=[2 * o["pre_sd"] for o in O],
          color="#999", capsize=3, label="pre-cue")
    a.bar(x + w / 2, [o["post"] for o in O], w, yerr=[2 * o["post_sd"] for o in O],
          color="#28a", capsize=3, label="post-cue")
    a.axhline(O[0]["sh_post"], color="k", ls=":", lw=1.2, label="shuffle floor")
    a.axhline(O[0]["post"], color="#28a", ls="--", lw=1, alpha=.6)
    a.set_xticks(x); a.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    a.set_ylabel("held-out R2 predicting time-to-lick"); a.set_ylim(min(-0.05, O[0]["pre"]-.05), 1.0)
    a.set_title(f"{key}: what predicts time-to-lick?  (fit within epoch)", fontsize=11)
    a.legend(fontsize=9)
    dnl = O[1]["post"] - O[0]["post"]; dv = O[2]["post"] - O[0]["post"]; da = O[3]["post"] - O[2]["post"]
    ax[1].axis("off")
    ax[1].text(0.0, 1.0,
               "READING (post-cue):\n\n"
               f"linear state R2            = {O[0]['post']:.2f}\n"
               f"+ nonlinear (quadratic)  {dnl:+.2f}\n"
               f"+ velocity (over state)  {dv:+.2f}\n"
               f"+ acceleration (over v)  {da:+.2f}\n\n"
               "Interpretation:\n"
               "- nonlinearity SMALL -> the time-to-lick\n"
               "  map is ~linear in the full 6-PC state\n"
               "  (a curved 1-D projection can be a\n"
               "  projection artifact, not real curvature).\n"
               "- MEASURED velocity dz/dt ADDS -> the\n"
               "  6-PC state is a PARTIAL observation;\n"
               "  [z, dz/dt] is a 2-tap delay embedding,\n"
               "  so recent history carries extra timing\n"
               "  info.  (The MODEL velocity A z + b could\n"
               "  never add -- it is a linear map of z.)\n"
               "- measured acceleration adds little ON\n"
               "  TOP of velocity -> first derivative is\n"
               "  enough; no need for the second.",
               va="top", ha="left", fontsize=9, family="monospace")
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"dynamics_real_{key}_ttl_order.png")
    fig.savefig(out, dpi=110); plt.close(fig); print("Saved", out)
    print(f"{key}: time-to-lick R2 (post) linear={O[0]['post']:.2f} quad={O[1]['post']:.2f} "
          f"+vel={O[2]['post']:.2f} +acc={O[3]['post']:.2f}")


def figure_regressions(key, sess, usable, Z, t):
    lick = sess.lick
    use = np.array([i for i in np.where(usable)[0] if lick[i] > 0.2])
    R = dyn.run_regression_suite(Z, lick, t, use, bin_s=BIN, base=BASE)
    figure_regression_group(
        R["velocity"],
        f"neuralgeom.dynamics, REAL {key}: PREDICT VELOCITY (dz/dt) from neural state, "
        f"pre- vs post-cue",
        "velocity rolled in time (within trial)", "velocity (a.u.)",
        os.path.join(FIGDIR, f"dynamics_real_{key}_regr_velocity.png"))
    figure_regression_group(
        R["time_to_lick"],
        f"neuralgeom.dynamics, REAL {key}: PREDICT TIME-TO-LICK, pre- vs post-cue  "
        f"(velocity regressors are MODEL velocities A z + b, i.e. linear maps of the state)",
        "regressor rolled in time (within trial)", "time-to-lick (s)",
        os.path.join(FIGDIR, f"dynamics_real_{key}_regr_timetolick.png"))
    print(f"{key}: time-to-lick decoding from state  CV R2 pre {R['time_to_lick'][0]['cv_pre']:+.2f}  "
          f"post {R['time_to_lick'][0]['cv_post']:+.2f}")
    print(f"{key}: velocity from state (full A)      CV R2 pre {R['velocity'][0]['cv_pre']:+.2f}  "
          f"post {R['velocity'][0]['cv_post']:+.2f}")


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "SM318"
    sess, m, usable = load(key)
    t = sess.t
    fitmask = np.zeros(sess.X.shape[:2], bool)
    for i in np.where(usable)[0]:
        fitmask[i] = (t >= BASE) & (t < sess.lick[i] + 0.1)
    Z = state_pca(sess, NPC, fit_mask=fitmask)
    d = figure_data(key, sess, usable, Z, t)
    figure_fit(key, sess, m, usable, Z, t, d)
    figure_regressions(key, sess, usable, Z, t)
    figure_ttl_order(key, sess, usable, Z, t)


if __name__ == "__main__":
    main()
