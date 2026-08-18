"""
Demo 5 — mixed populations: overlapping classes, uncertainty, memorization.
===========================================================================

The earlier classification demos used cleanly separable moons. Real
populations overlap. This demo trains the same architecture on two-moons
data at three noise levels (mild / moderate / heavy overlap) and asks what
the geometry does when the classes genuinely mix.

Key conceptual point: the pullback geometry depends on WHICH map you pull
back through.

  * logit map  f(x) = network logits     — geometry of the learned FEATURES;
    it does not know about probability and stays sharp even where the model
    is uncertain.
  * probability map  p(x) = softmax(f)_1 — geometry of the learned BELIEF;
    |grad p| = p(1-p)|grad logit-margin| is largest exactly where the model
    is uncertain, so this pullback turns the toolbox into an uncertainty
    detector.

Pipeline:
  Step 0  Three datasets (increasing overlap) + fitted boundaries
  Step 1  Logit-map volume fields: the ridge broadens and WEAKENS with overlap
  Step 2  Probability-map pullback vs predictive entropy (they nearly agree,
          and their correlation is quantified; the logit map's is weak)
  Step 3  Same data, two fits: regularized vs overfit — memorization is
          VISIBLE in the geometry
  Step 4  Summary: uncertainty bandwidth grows with class overlap
  Step 5  What this CANNOT do (aleatoric vs epistemic, calibration, ...)

Run:  python demo_mixed_populations.py
"""

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from demo_common import (banner, decision_background, input_grid,
                         make_classifier, savefig, scatter_data,
                         train_classifier, two_moons)
from neuralgeom.geometry.jacobian import batch_jacobian, log_volume_element
from neuralgeom.geometry.grassmann import grassmann_frechet_mean, principal_angles, \
    tangent_subspaces

torch.manual_seed(0)

NOISES = [0.08, 0.25, 0.45]
LABELS = ["mild overlap", "moderate overlap", "heavy overlap"]


def prob_fn(model):
    """Scalar probability map x -> P(class 1 | x). Pullback through THIS map
    (not the logits) makes the geometry track uncertainty."""
    def p(x):
        return torch.softmax(model(x), dim=-1)[..., 1:2]
    return p


def entropy(model, P):
    with torch.no_grad():
        pr = torch.softmax(model(P), dim=-1).clamp_min(1e-12)
    return -(pr * pr.log()).sum(-1)


# --------------------------------------------------------------------------- #
banner("Step 0", "Three versions of the same problem, increasing class overlap")
data, models = [], []
for nz in NOISES:
    torch.manual_seed(3)
    Xn, yn = two_moons(500, noise=nz)
    m = make_classifier()
    train_classifier(m, Xn, yn, steps=500, weight_decay=1e-3)
    data.append((Xn, yn))
    models.append(m)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)
for ax, (Xn, yn), m, nz, lab in zip(axes, data, models, NOISES, LABELS):
    decision_background(ax, m, Xn)
    scatter_data(ax, Xn, yn, s=5)
    ax.set_aspect("equal")
    ax.set_title(f"noise = {nz}  ({lab})")
savefig(fig, "mx_step0_datasets.png")

# --------------------------------------------------------------------------- #
banner("Step 1", "Logit-map volume fields: overlap broadens and WEAKENS "
       "the expansion ridge")
res = 70
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), constrained_layout=True)
peak_s1 = []
for ax, (Xn, yn), m, lab in zip(axes, data, models, LABELS):
    P, GX, GY = input_grid(Xn, res)
    lv = log_volume_element(m, P, chunk_size=1024).reshape(res, res)
    J = batch_jacobian(m, P, chunk_size=1024)
    peak_s1.append(float(torch.linalg.svdvals(J)[:, 0].max()))
    im = ax.pcolormesh(GX, GY, lv.numpy(), cmap="magma", shading="auto")
    fig.colorbar(im, ax=ax, label="$\\log\\sqrt{\\det g}$")
    decision_background(ax, m, Xn, alpha=0.0)
    ax.set_aspect("equal")
    ax.set_title(lab)
savefig(fig, "mx_step1_logit_volume.png")
for lab, s1 in zip(LABELS, peak_s1):
    print(f"  {lab:18s}: peak stretch s1 = {s1:7.1f}")
print("  -> with overlap the network cannot afford a steep logit cliff: the")
print("     data itself punishes overconfidence, and the geometry records it.")

# --------------------------------------------------------------------------- #
banner("Step 2", "Pull back through the PROBABILITY map and you get an "
       "uncertainty detector")
Xn, yn = data[1]                       # moderate overlap
m = models[1]
P, GX, GY = input_grid(Xn, res)
H = entropy(m, P)
Jp = batch_jacobian(prob_fn(m), P, chunk_size=1024).squeeze(1)
s1_prob = Jp.norm(dim=1)               # |grad p| = pseudo-volume of the p-map
Jl = batch_jacobian(m, P, chunk_size=1024)
s1_logit = torch.linalg.svdvals(Jl)[:, 0]

fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.7), constrained_layout=True)
im = axes[0].pcolormesh(GX, GY, H.reshape(res, res).numpy(), cmap="viridis",
                        shading="auto")
fig.colorbar(im, ax=axes[0], label="entropy [nats]")
axes[0].set_title("Predictive entropy (model uncertainty)")
im = axes[1].pcolormesh(GX, GY, s1_prob.reshape(res, res).numpy(),
                        cmap="viridis", shading="auto")
fig.colorbar(im, ax=axes[1], label="$|\\nabla p|$")
axes[1].set_title("Probability-map pseudo-volume $|\\nabla p|$\n"
                  "(same structure — geometry finds the uncertain set)")
axes[2].scatter(H.numpy(), s1_prob.numpy(), s=2, alpha=0.15, label="prob map")
axes[2].scatter(H.numpy(), (s1_logit / s1_logit.max() * s1_prob.max()).numpy(),
                s=2, alpha=0.15, color="C3", label="logit map (rescaled)")
axes[2].set_xlabel("predictive entropy [nats]")
axes[2].set_ylabel("max stretch $s_1$")
axes[2].legend(markerscale=6, fontsize=8)
axes[2].set_title("$s_1$ vs entropy, per grid point")
for ax in axes[:2]:
    ax.set_aspect("equal")
savefig(fig, "mx_step2_uncertainty_pullback.png")

corr_p = float(np.corrcoef(H.numpy(), s1_prob.numpy())[0, 1])
corr_l = float(np.corrcoef(H.numpy(), s1_logit.numpy())[0, 1])
print(f"  corr(entropy, s1 of probability map) = {corr_p:+.3f}   <- strong")
print(f"  corr(entropy, s1 of logit map)       = {corr_l:+.3f}   <- weak/unreliable")
print("  -> choose the codomain to match the question: logits for feature")
print("     geometry, probabilities for uncertainty geometry.")

# --------------------------------------------------------------------------- #
banner("Step 3", "Same overlapping data, two fits: regularization is VISIBLE "
       "in the geometry")
Xh, yh = data[2]                       # heavy overlap
m_reg = models[2]                      # weight decay + early stop (500 steps)
torch.manual_seed(7)
m_ofit = make_classifier(hidden=64)
train_classifier(m_ofit, Xh, yh, steps=2500, weight_decay=0.0)

P, GX, GY = input_grid(Xh, res)
fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), constrained_layout=True)
for col, (mm, name) in enumerate([(m_reg, "regularized (wd=1e-3, 500 steps)"),
                                  (m_ofit, "overfit (no wd, 2500 steps)")]):
    decision_background(axes[0, col], mm, Xh)
    scatter_data(axes[0, col], Xh, yh, s=5)
    axes[0, col].set_title(f"{name}\ndecision function")
    lv = log_volume_element(mm, P, chunk_size=1024).reshape(res, res)
    im = axes[1, col].pcolormesh(GX, GY, lv.numpy(), cmap="magma",
                                 shading="auto")
    fig.colorbar(im, ax=axes[1, col], label="$\\log\\sqrt{\\det g}$")
    decision_background(axes[1, col], mm, Xh, alpha=0.0)
    axes[1, col].set_title("log-volume field")
    for r in range(2):
        axes[r, col].set_aspect("equal")
savefig(fig, "mx_step3_overfit_geometry.png")

# Grassmann coherence of the sensitive-direction field at the data points
angs = {}
for mm, name in [(m_reg, "regularized"), (m_ofit, "overfit")]:
    Q, _, _ = tangent_subspaces(mm, Xh, which="row", k=1)
    mu = grassmann_frechet_mean(Q, method="projection")
    angs[name] = principal_angles(Q, mu.expand(Xh.shape[0], 2, 1)).squeeze(-1)
fig, ax = plt.subplots(figsize=(5.6, 3.2), constrained_layout=True)
for name, c in [("regularized", "C0"), ("overfit", "C3")]:
    ax.hist(angs[name].numpy(), bins=40, alpha=0.6, color=c, density=True,
            label=f"{name} (mean {float(angs[name].mean()):.2f} rad)")
ax.set_xlabel("angle of each point's sensitive direction to the field mean [rad]")
ax.legend(fontsize=8)
ax.set_title("Direction-field coherence on the data:\noverfitting scatters "
             "the row-space field")
savefig(fig, "mx_step3b_direction_coherence.png")
tr_acc_reg = float((m_reg(Xh).argmax(1) == yh).double().mean())
tr_acc_ofit = float((m_ofit(Xh).argmax(1) == yh).double().mean())
print(f"  train acc: regularized {tr_acc_reg:.2f} vs overfit {tr_acc_ofit:.2f}"
      f"  (Bayes-optimal here is ~0.85-0.9)")
print("  -> the overfit model's extra 'accuracy' is memorization, and its")
print("     geometry shows it: fragmented ridges snaking between individual")
print("     points, and a less coherent sensitive-direction field.")

# --------------------------------------------------------------------------- #
banner("Step 4", "Summary: the geometric uncertainty band widens with overlap")
band_frac_H, band_frac_g = [], []
for (Xn, yn), m in zip(data, models):
    P, _, _ = input_grid(Xn, res)
    H = entropy(m, P)
    Jp = batch_jacobian(prob_fn(m), P, chunk_size=1024).squeeze(1)
    s1p = Jp.norm(dim=1)
    band_frac_H.append(float((H > 0.35).double().mean()))
    band_frac_g.append(float((s1p > 0.25 * s1p.max()).double().mean()))
fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
w = 0.35
xs = np.arange(3)
ax.bar(xs - w / 2, band_frac_H, w, label="entropy > 0.35 nats", color="C0")
ax.bar(xs + w / 2, band_frac_g, w,
       label="$|\\nabla p| > 25\\%$ of max", color="C2")
ax.set_xticks(xs, [f"noise {nz}\n{lab}" for nz, lab in zip(NOISES, LABELS)],
              fontsize=8)
ax.set_ylabel("fraction of input plane")
ax.set_title("Both the probabilistic and the geometric\nuncertainty band "
             "grow with class overlap")
ax.legend(fontsize=8)
savefig(fig, "mx_step4_bandwidth.png")
for lab, fh, fg in zip(LABELS, band_frac_H, band_frac_g):
    print(f"  {lab:18s}: entropy band {fh:5.1%}   geometric band {fg:5.1%}")

# --------------------------------------------------------------------------- #
banner("Step 5", "LIMITATIONS — what geometry does NOT give you here")
print("""
(5a) Aleatoric vs epistemic: the geometry describes the FITTED map only.
     It cannot tell 'the classes truly overlap here' (aleatoric) apart from
     'the model happens to be shallow-sloped here' (epistemic/underfit).
     The entropy agreement in Step 2 is a property of the model, not of
     the true populations.

(5b) No calibration guarantee: |grad p| flags where the MODEL is uncertain.
     If the model is miscalibrated (see the overfit model: near-0/1
     probabilities inside genuinely mixed regions), the geometric band is
     confidently wrong in exactly the same places.

(5c) The logit-map geometry is NOT an uncertainty measure (Step 2 scatter,
     red): a model can carve steep features in perfectly certain regions.
     Uncertainty statements require pulling back through softmax.

(5d) Everything is derivative-based and local: two distant uncertain
     islands are not connected or counted by these tools; use them with a
     density/clustering analysis for population-level statements.

(5e) With heavy overlap the geometry is only as meaningful as the fit is
     regularized (Step 3): geometry can EXPOSE memorization, but it cannot
     prevent it.
""")
print("Done. Figures in outputs/figures/")
