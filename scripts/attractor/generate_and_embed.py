"""Generate synthetic neural data from the two-attractor model and embed it.

Shows the whole output of the generator in one figure:
  - the 2-D latent vector field (two Gaussian-well attractors + the brief cue pulse),
  - single-trial latent trajectories coloured by lick time,
  - the ramping mode vs time (what crosses threshold to trigger the 'lick'),
  - the embedded 'neural' activity (a few example units) and its spike statistics,
  - the noiseless latent recovered by PCA of the embedding (a sanity check that the embedding
    is invertible up to rotation).

Run:  python scripts/attractor/generate_and_embed.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- make the neuralgeom package importable without installing it ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import neuralgeom.synth as syn
from neuralgeom.synth.attractor import attractor_field, REST_ATTRACTOR, LICK_ATTRACTOR
from neuralgeom.paths import fig_dir

FIGDIR = str(fig_dir("demos"))


def main():
    # Default embedding: additive Gaussian observation noise -> high signal-to-noise, so the
    # latent structure is clearly visible (this demo is about showing what the generator makes).
    # The realistic, very sparse Poisson regime (~0.24 spikes/50 ms, ~90% empty bins) is shown
    # for contrast in the last panel; that regime is where the instrumental-variable correction matters.
    d = syn.generate("input", syn.CUE_AMPLITUDE_LEVELS, n_trials=120, seed=1, bin_s=0.02,
                     n_neurons=80, poisson=False, obs_noise=0.15)
    t, lick, Z = d["time"], d["lick"], d["latent"]
    ok = np.isfinite(lick)
    norm = plt.Normalize(np.nanpercentile(lick[ok], 5), np.nanpercentile(lick[ok], 95))
    col = plt.cm.viridis(norm(lick))

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # 1. latent vector field + attractors
    a = ax[0, 0]
    gx, gy = np.meshgrid(np.linspace(-2, 20, 22), np.linspace(-2, 20, 22))
    U, V = attractor_field(gx.ravel(), gy.ravel())
    a.streamplot(gx, gy, U.reshape(gx.shape), V.reshape(gx.shape), color="#aab", density=1.1,
                 linewidth=.5, arrowsize=.7)
    a.scatter(*REST_ATTRACTOR, c="k", s=80, marker="o", label="baseline attractor")
    a.scatter(*LICK_ATTRACTOR, c="r", s=120, marker="*", label="lick attractor")
    a.set_xlabel("cue mode X"); a.set_ylabel("ramping mode Y")
    a.set_title("Known dynamics: 2-attractor vector field", fontsize=10); a.legend(fontsize=8)

    # 2. single-trial latent trajectories
    a = ax[0, 1]
    for i in np.where(ok)[0][:60]:
        m = (t >= 0) & (t < lick[i] + 0.1)
        a.plot(Z[i][m, 0], Z[i][m, 1], color=col[i], alpha=.5, lw=.8)
    a.scatter(*REST_ATTRACTOR, c="k", s=60, marker="o"); a.scatter(*LICK_ATTRACTOR, c="r", s=100, marker="*")
    a.set_xlabel("cue mode X"); a.set_ylabel("ramping mode Y")
    a.set_title("Single-trial latent paths (colour = lick time)", fontsize=10)

    # 3. ramping mode vs time
    a = ax[0, 2]
    for i in np.where(ok)[0][:60]:
        a.plot(t, Z[i][:, 1], color=col[i], alpha=.4, lw=.6)
    a.axhline(15.0, color="k", ls="--", lw=1, label="lick threshold")
    a.axvline(0, color="k", lw=1)
    a.set_xlim(-0.6, 1.3); a.set_xlabel("time from cue (s)"); a.set_ylabel("ramping mode Y")
    a.set_title("Ramp to threshold = the timed 'lick'", fontsize=10); a.legend(fontsize=8)

    # 4. example embedded units
    # 4. example embedded units -- the embedding is rate_i(t) = softplus(L_i . z_norm(t) + b_i),
    #    L_i random per neuron, so ~half of units RISE and ~half FALL as the state ramps; show a
    #    few of each so the population isn't misread as 'all going down'.
    a = ax[1, 0]
    X = d["X"]
    psth = X[ok].mean(0)                            # (T, N) trial-averaged rate
    ramp = psth[(t > 0.2) & (t < 0.8)].mean(0) - psth[(t > -0.5) & (t < 0)].mean(0)
    up = np.argsort(ramp)[::-1][:3]; down = np.argsort(ramp)[:3]
    for k in up:
        a.plot(t, psth[:, k], lw=1.3, color="#c33")
    for k in down:
        a.plot(t, psth[:, k], lw=1.3, color="#37a")
    a.plot([], [], color="#c33", label="units that rise with the ramp")
    a.plot([], [], color="#37a", label="units that fall")
    a.axvline(0, color="k", lw=1)
    a.set_xlim(-0.6, 1.3); a.set_xlabel("time from cue (s)")
    a.set_ylabel("trial-averaged rate (a.u.)")
    a.set_title("Embedded 'neurons': random loadings ->\nsome rise, some fall", fontsize=10)
    a.legend(fontsize=7.5)

    # 5. PCA of the CLEAN embedding recovers the latent (up to rotation): PC1 tracks lick time
    a = ax[1, 1]
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    fm = (t >= 0) & (t < 1.2)
    Xf = X[ok][:, fm].reshape(-1, X.shape[2])
    sc = StandardScaler().fit(Xf); pca = PCA(2, random_state=0).fit(sc.transform(Xf))
    pc1 = []
    for i in np.where(ok)[0]:
        p = pca.transform(sc.transform(X[i][(t >= 0) & (t < lick[i] + 0.1)]))
        pc1.append(p[:, 0].mean())
        if i < 60:
            a.plot(p[:, 0], p[:, 1], color=col[i], alpha=.5, lw=.8)
    r_pc1 = np.corrcoef(pc1, lick[ok])[0, 1]
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.set_title(f"PCA of the embedding: PC1 tracks lick time\n(PC1-2 = "
                f"{100*pca.explained_variance_ratio_.sum():.0f}% of variance, "
                f"corr(PC1,lick) = {r_pc1:+.2f})", fontsize=10)

    # 6. CONTRAST: the realistic sparse-Poisson regime (~0.24 spikes/50 ms) buries PC1 in noise
    a = ax[1, 2]
    dp = syn.generate("input", syn.CUE_AMPLITUDE_LEVELS, n_trials=120, seed=1, bin_s=0.02, n_neurons=80,
                      poisson=True, mean_count=0.24 * (0.02 / 0.05))
    Xp, lp = dp["X"], dp["lick"]; op = np.isfinite(lp)
    Xpf = Xp[op][:, fm].reshape(-1, Xp.shape[2])
    scp = StandardScaler().fit(Xpf); pcp = PCA(2, random_state=0).fit(scp.transform(Xpf))
    pc1p = [pcp.transform(scp.transform(Xp[i][(t >= 0) & (t < lp[i] + 0.1)]))[:, 0].mean()
            for i in np.where(op)[0]]
    r_p = np.corrcoef(pc1p, lp[op])[0, 1]
    a.hist(Xp.ravel(), bins=np.arange(0, Xp.max() + 2) - 0.5, color="#a55")
    a.set_yscale("log"); a.set_xlabel("spikes per unit per bin")
    a.set_ylabel("# (unit, time, trial)")
    a.set_title(f"Realistic Poisson (0.24/50 ms): {100*(Xp==0).mean():.0f}% empty bins\n"
                f"PC1-2 = {100*pcp.explained_variance_ratio_.sum():.0f}% variance, "
                f"corr(PC1,lick) = {r_p:+.2f}\n(the sparse regime the instrumental-variable correction addresses)",
                fontsize=9)

    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    fig.colorbar(sm, ax=ax[0, :].tolist(), label="lick time (s)", shrink=.6)
    fig.suptitle("neuralgeom.synth: synthetic neural data from known 2-attractor dynamics",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGDIR, "generate_and_embed.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    print("Saved", out)
    print(f"generated {X.shape[0]} trials x {X.shape[1]} bins x {X.shape[2]} units; "
          f"median lick {np.nanmedian(lick):.3f} s; {int((~ok).sum())} no-lick trials")
    print(f"clean Gaussian embedding: PC1-2 = {100*pca.explained_variance_ratio_.sum():.0f}% var, "
          f"corr(PC1,lick) = {r_pc1:+.2f}")


if __name__ == "__main__":
    main()
