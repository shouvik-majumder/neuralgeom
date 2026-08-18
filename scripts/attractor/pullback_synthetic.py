"""Module 2 demo (SYNTHETIC, ground truth known): does the pullback metric recover the state
direction that actually controls behavior?

We build a synthetic readout whose behavior depends on the latent state in a KNOWN way, so the
correct 'most behavior-relevant direction' is known in closed form, then check that the pullback
metric's top eigenvector matches it.

Pipeline (the module's core objects):
    state x  ->  readout f: x -> behavior y  ->  output metric g_Y  ->  pullback g = J^T g_Y J
The top eigenvector of g is the state direction the behavior is most sensitive to; its kernel is
the directions behavior ignores.

Run:  python scripts/attractor/pullback_synthetic.py
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
from neuralgeom.paths import fig_dir

FIGDIR = str(fig_dir("demos"))


def main():
    rng = np.random.default_rng(0)
    D = 6

    # KNOWN ground truth: behavior = a nonlinear function of ONE direction w_true in state space.
    w_true = rng.standard_normal(D); w_true /= np.linalg.norm(w_true)
    # y = softplus-type readout along w_true only -> the behavior-relevant direction is w_true
    def f(x):
        return np.array([0.15 + np.log1p(np.exp(1.5 - x @ w_true))])

    readout = pb.FunctionReadout(f)                       # finite-difference Jacobian
    pm = pb.PullbackMetric(readout, pb.ScaledGaussian(sigma=0.1))

    # sample states, compute the pullback metric's top eigenvector at each, compare to w_true
    X = rng.standard_normal((400, D))
    cos = []
    for x in X:
        _, Vv = pm.spectrum(x)
        top = Vv[:, 0]
        cos.append(abs(top @ w_true))
    cos = np.array(cos)
    ranks = np.array([pm.effective_rank(x) for x in X[:100]])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    a = ax[0]
    a.hist(cos, bins=np.linspace(0, 1, 31), color="#3a7")
    a.axvline(1.0, color="k", ls="--")
    a.set_xlabel("|cos(top eigenvector of g, true direction)|")
    a.set_ylabel("# states")
    a.set_title(f"Pullback recovers the behavior direction\nmedian |cos| = {np.median(cos):.3f}",
                fontsize=10)

    a = ax[1]
    a.hist(ranks, bins=np.arange(0.5, D + 1.5), color="#37a")
    a.set_xlabel("effective rank of g(x)")
    a.set_ylabel("# states")
    a.set_title("Scalar readout -> rank-1 metric\n(behavior sensitive to ONE direction)",
                fontsize=10)

    # sensitivity field: sqrt(lambda_max) along a 2D slice spanned by w_true and an orthogonal dir
    a = ax[2]
    u = w_true[:2] if D >= 2 else w_true
    e2 = rng.standard_normal(D); e2 -= (e2 @ w_true) * w_true; e2 /= np.linalg.norm(e2)
    g0 = np.linspace(-2, 3, 40); g1 = np.linspace(-2, 2, 40)
    GX, GY = np.meshgrid(g0, g1)
    S = np.zeros_like(GX)
    for ii in range(GX.shape[0]):
        for jj in range(GX.shape[1]):
            x = GX[ii, jj] * w_true + GY[ii, jj] * e2
            lam, _ = pm.spectrum(x)
            S[ii, jj] = np.sqrt(lam.max())
    cf = a.contourf(GX, GY, S, 20, cmap="magma")
    plt.colorbar(cf, ax=a, label="sqrt(lambda_max) = behavioral sensitivity")
    a.set_xlabel("along the TRUE direction"); a.set_ylabel("orthogonal direction")
    a.set_title("Sensitivity field\n(varies along w_true only)", fontsize=10)

    fig.suptitle("neuralgeom pullback, SYNTHETIC ground truth: the metric finds the "
                 "behavior-relevant state direction", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGDIR, "pullback_synthetic.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    print("Saved", out)
    print(f"median |cos(recovered, true)| = {np.median(cos):.3f}; "
          f"median effective rank = {np.median(ranks):.2f} (expected 1)")


if __name__ == "__main__":
    main()
