"""Module 2 demo (REAL data): which population-activity direction controls lick timing?

state x = PCA of post-cue firing rates
readout f: x -> predicted time-to-lick   (ridge; linear so the Jacobian is exactly the weights)
output metric g_Y = 1/sigma^2  (Gaussian; sigma = readout residual std)
pullback g(x) = J^T g_Y J = (1/sigma^2) w w^T   -> rank 1; its top eigenvector is THE direction
population activity must move along to change the predicted lick time, and sqrt(lambda_max) is
how many seconds-worth (in sigma units) of timing change one unit of motion buys.

Run:  python scripts/attractor/pullback_real.py [SM318|SM348]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- make the neuralgeom package importable without installing it ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import neuralgeom.geometry as pb
from neuralgeom.data import load_session, state_pca
from neuralgeom.paths import DATA_DIR, fig_dir
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

FIGDIR = str(fig_dir("neural"))
DATA = str(DATA_DIR)
SESSIONS = {"SM318": "SM318_20230904_g0.h5", "SM348": "SM348_20230929_g0.h5"}
NPC = 6
BIN = 0.05


def main():
    key = (sys.argv[1] if len(sys.argv) > 1 else "SM318")
    # V2 loader conventions: 50 ms bins (summed counts), NO smoothing here (the
    # velocity code smooths), qm_pass + mean rate >= 0.5 Hz, valid trials only.
    sess = load_session(os.path.join(DATA, SESSIONS[key]), bin_ms=BIN * 1000,
                        smooth_ms=0.0, t_range=(-0.5, 3.0), valid_only=True)
    t = sess.t; lick = sess.lick

    # build (state, time-to-lick) samples over the post-cue, pre-lick window
    fitmask = np.zeros(sess.X.shape[:2], bool)
    for i in range(sess.X.shape[0]):
        if np.isfinite(lick[i]) and lick[i] > 3 * BIN:
            fitmask[i] = (t >= 0) & (t < lick[i])
    Z = state_pca(sess, NPC, fit_mask=fitmask)
    Xs, y, grp = [], [], []
    for i in range(sess.X.shape[0]):
        m = fitmask[i]
        if m.sum() >= 4:
            Xs.append(Z[i][m]); y.append(lick[i] - t[m]); grp.append(np.full(m.sum(), i))
    Xs = np.vstack(Xs); y = np.concatenate(y); grp = np.concatenate(grp)

    r2 = []
    for tr, te in GroupKFold(5).split(Xs, y, groups=grp):
        mdl = Ridge(1.0).fit(Xs[tr], y[tr])
        r2.append(1 - np.sum((y[te] - mdl.predict(Xs[te])) ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
    r2 = float(np.mean(r2))
    ridge = Ridge(1.0).fit(Xs, y)
    w = ridge.coef_.ravel()
    sigma = float(np.std(y - ridge.predict(Xs)))

    # pullback via a linear readout: g = (1/sigma^2) w w^T
    readout = pb.LinearReadout(w.reshape(1, -1), b=np.array([ridge.intercept_]))
    pm = pb.PullbackMetric(readout, pb.ScaledGaussian(sigma=sigma))
    g = pm.metric_matrix(Xs.mean(0))
    lam, Vv = pm.spectrum(Xs.mean(0))
    axis = Vv[:, 0]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    a = ax[0]
    a.bar(range(1, NPC + 1), axis, color="#37a")
    a.set_xlabel("PC"); a.set_ylabel("loading")
    a.set_title(f"{key}: behavior-relevant direction in state space\n"
                f"(top eigenvector of g; readout CV R2 = {r2:+.2f})", fontsize=10)

    a = ax[1]
    lamn = lam / lam.max()
    a.bar(range(1, NPC + 1), lamn, color=["#c33"] + ["#bbb"] * (NPC - 1))
    a.set_xlabel("eigenvalue index"); a.set_ylabel("eigenvalue / max")
    a.set_title(f"Metric spectrum: rank 1\nsqrt(lambda_max) = {np.sqrt(lam.max()):.1f} "
                f"(sensitivity)", fontsize=10)

    # trajectories in the plane of (behavior axis, PC of most variance) coloured by time-to-lick
    a = ax[2]
    proj = Xs @ axis
    other = Xs[:, 0]
    sccol = a.scatter(proj, other, c=y, s=4, cmap="viridis", alpha=.4)
    plt.colorbar(sccol, ax=a, label="time to lick (s)")
    a.set_xlabel("projection on behavior axis"); a.set_ylabel("PC1")
    a.set_title("States along the behavior axis track timing", fontsize=10)

    fig.suptitle(f"neuralgeom pullback, REAL {key}: population direction that controls "
                 f"lick timing", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGDIR, f"pullback_real_{key}.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    print("Saved", out)
    print(f"{key}: readout CV R2 = {r2:+.2f}; sigma = {sigma:.3f} s; "
          f"metric rank ~1, sqrt(lambda_max) = {np.sqrt(lam.max()):.2f}")


if __name__ == "__main__":
    main()
