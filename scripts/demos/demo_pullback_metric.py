"""
Demo 1 — the pullback metric tensor field of a trained classifier.
===========================================================================

Pipeline (each step saves a figure to outputs/figures/):

  Step 0  Raw data + trained decision function
  Step 1  Raw geometric objects: per-point Jacobians and metric tensors
  Step 2  Fields: log-volume element and anisotropy over the input plane
  Step 3  Tissot ellipses: how the network stretches each neighborhood
  Step 4  Summaries: spectra along a transect, per-class distributions
  Step 5  What these tools CANNOT do (live failure demos)

CAN:  pointwise Jacobians/metrics/volumes for any smooth model at any batch,
      rank diagnostics, stable log-volumes in high dimension.
CANNOT: finite det-volumes for rank-deficient maps (pseudo-det instead),
        plain volume_element in high dimension (overflow — use log),
        meaningful smooth fields for ReLU nets (piecewise-constant Jacobian),
        geodesics/curvature by itself (bridge to geomstats for that).

Run:  python demo_pullback_metric.py       (figures land in figures/)
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
                         make_classifier, metric_ellipse, savefig,
                         scatter_data, train_classifier, two_moons)
from neuralgeom.geometry.jacobian import (ModelPullbackGeometry, batch_jacobian,
                             log_volume_element, metric_spectrum,
                             euclidean_pullback_metric, volume_element)

# --------------------------------------------------------------------------- #
banner("Step 0", "Raw data and trained model (two moons, 2->32->32->2 Tanh net)")
X, y = two_moons(400)
model = train_classifier(make_classifier(), X, y)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
scatter_data(axes[0], X, y)
axes[0].set_title("Raw data: two moons in input space $\\mathbb{R}^2$")
axes[0].legend(loc="lower left")
decision_background(axes[1], model, X)
scatter_data(axes[1], X, y)
axes[1].set_title("Trained decision function $f:\\mathbb{R}^2\\to\\mathbb{R}^2$ (logits)")
for ax in axes:
    ax.set_aspect("equal")
savefig(fig, "pb_step0_data_and_model.png")

# --------------------------------------------------------------------------- #
banner("Step 1", "Raw objects: Jacobians J_x (2x2 here) and metrics g = J^T J")
geom = ModelPullbackGeometry(model)
J = geom.jacobian(X[:5])
g = geom.metric(X[:5])
print("  J shape:", tuple(J.shape), " g shape:", tuple(g.shape))
print("  J at first data point:\n", np.array_str(J[0].numpy(), precision=3))
print("  g at first data point:\n", np.array_str(g[0].numpy(), precision=3))
print("  g is symmetric PSD: eigvals =", torch.linalg.eigvalsh(g[0]).numpy())

# --------------------------------------------------------------------------- #
banner("Step 2", "Fields over the input plane: log sqrt(det g) and anisotropy")
res = 80
P, GX, GY = input_grid(X, res)
logvol = log_volume_element(model, P, chunk_size=512).reshape(res, res)
eig, rank = metric_spectrum(model, P, chunk_size=512)
aniso = (eig[:, 0] / eig[:, 1].clamp_min(1e-300)).log10().reshape(res, res)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
im = axes[0].pcolormesh(GX, GY, logvol.numpy(), cmap="magma", shading="auto")
fig.colorbar(im, ax=axes[0], label="$\\log\\sqrt{\\det g_x}$")
decision_background(axes[0], model, X, alpha=0.0)  # boundary contour only
axes[0].set_title("Volume expansion: network magnifies\nspace near the decision boundary")
im = axes[1].pcolormesh(GX, GY, aniso.numpy(), cmap="viridis", shading="auto")
fig.colorbar(im, ax=axes[1], label="$\\log_{10}(\\lambda_1/\\lambda_2)$")
decision_background(axes[1], model, X, alpha=0.0)
axes[1].set_title("Anisotropy: stretching is strongly\ndirection-dependent (rank ~ collapse off-boundary)")
for ax in axes:
    ax.set_aspect("equal")
savefig(fig, "pb_step2_fields.png")
print(f"  log-vol range: [{logvol.min():.1f}, {logvol.max():.1f}]  "
      f"anisotropy up to 10^{aniso.max():.1f}")

# --------------------------------------------------------------------------- #
banner("Step 3", "Tissot ellipses: image of a unit circle under J at grid points")
Pc, GXc, GYc = input_grid(X, 14)
gc = euclidean_pullback_metric(model, Pc)
smax = torch.linalg.eigvalsh(gc)[:, -1].sqrt()

fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
decision_background(ax, model, X, alpha=0.25)
norm = plt.Normalize(float(smax.min()), float(smax.max()))
for i in range(Pc.shape[0]):
    color = plt.cm.plasma(norm(float(smax[i])))
    # normalize major axis to a fixed length: shape shows anisotropy,
    # color shows magnitude
    metric_ellipse(ax, Pc[i].numpy(), gc[i].numpy(),
                   scale=0.11 / float(smax[i]),
                   facecolor=color, edgecolor=color, linewidth=1.4, alpha=0.95)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=ax,
             label="max stretch $s_1(J_x)$")
ax.set_aspect("equal")
ax.set_title("Tissot glyphs (unit-circle images, major axis normalized).\n"
             "Ellipses degenerate to needles: the net collapses one input\n"
             "direction ~$10^6$:1 — orientation = the direction that survives")
savefig(fig, "pb_step3_tissot.png")

# --------------------------------------------------------------------------- #
banner("Step 4", "Summary statistics: transect across the boundary + distributions")
# transect: straight line crossing both moons and the boundary
t = torch.linspace(0, 1, 200).unsqueeze(1)
a, b = torch.tensor([-0.5, 1.2]), torch.tensor([1.5, -0.7])
line = a + t * (b - a)
eig_l, _ = metric_spectrum(model, line)
lv_l = log_volume_element(model, line)
with torch.no_grad():
    prob_l = torch.softmax(model(line), 1)[:, 1]

lv_data = log_volume_element(model, X)
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), constrained_layout=True)
s = t.squeeze().numpy()
axes[0].plot(s, eig_l[:, 0].sqrt().numpy(), label="$s_1$ (max stretch)")
axes[0].plot(s, eig_l[:, 1].sqrt().numpy(), label="$s_2$ (min stretch)")
axes[0].set_yscale("log")
axes[0].set_xlabel("position along transect")
axes[0].set_title("Singular values of $J$ along a transect")
axes[0].legend()
ax2 = axes[1]
ax2.plot(s, lv_l.numpy(), color="C3", label="$\\log\\sqrt{\\det g}$")
ax2b = ax2.twinx()
ax2b.plot(s, prob_l.numpy(), color="0.5", ls="--", label="P(class 1)")
ax2b.set_ylabel("P(class 1)")
ax2.set_xlabel("position along transect")
ax2.set_title("Volume element peaks exactly at the\ndecision boundary (dashed = class prob.)")
axes[2].hist(lv_data[y == 0].numpy(), bins=30, alpha=0.6, color="#b40426",
             label="class 0", density=True)
axes[2].hist(lv_data[y == 1].numpy(), bins=30, alpha=0.6, color="#3b4cc0",
             label="class 1", density=True)
axes[2].set_xlabel("$\\log\\sqrt{\\det g_x}$ at data points")
axes[2].set_title("Per-class distribution of log-volume\n(most data sits in contracted regions)")
axes[2].legend()
savefig(fig, "pb_step4_summary.png")

# --------------------------------------------------------------------------- #
banner("Step 5", "LIMITATIONS — live demonstrations of what this CANNOT do")

print("\n(5a) Rank-deficient maps: det g = 0, only a pseudo-volume exists.")
squeeze = nn.Sequential(nn.Linear(2, 1))  # m < n: g is 2x2 of rank 1
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    v = volume_element(squeeze, X[:4])
    print(f"   pseudo-volume: {v.numpy().round(3)}  "
          f"(warning raised: {w[0].category.__name__}: {str(w[0].message)[:60]}...)")
lv_neginf = log_volume_element(squeeze, X[:4], degenerate="neginf")
print(f"   degenerate='neginf' gives the honest answer: {lv_neginf.numpy()}")

print("\n(5b) volume_element overflows in high dimension; log_volume_element does not.")
dims = [2, 50, 200, 500, 1000, 1500]
vols, logs = [], []
for d in dims:
    f = lambda z: 2.0 * z  # J = 2I in R^d -> det g = 4^d
    xd = torch.randn(1, d)
    vols.append(float(volume_element(f, xd)))
    logs.append(float(log_volume_element(f, xd)))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
axes[0].plot(dims, vols, "o-")
axes[0].set_yscale("log")
axes[0].set_title("volume_element = $2^d$: overflows to inf\nbeyond float64 range (d ~ 1024)")
axes[0].set_xlabel("input dimension d")
axes[1].plot(dims, logs, "o-", color="C2")
axes[1].set_title("log_volume_element = $d\\,\\log 2$: exact at any d")
axes[1].set_xlabel("input dimension d")
for d, v in zip(dims, vols):
    print(f"   d={d:5d}: volume={v:12.4g}   log-volume={logs[dims.index(d)]:10.2f}")
savefig(fig, "pb_step5_overflow.png")

print("\n(5c) ReLU nets: Jacobian is piecewise constant -> fields are blocky,")
print("     boundaries of linear regions produce artificial discontinuities.")
relu_model = train_classifier(make_classifier(act=nn.ReLU), X, y)
lv_relu = log_volume_element(relu_model, P, chunk_size=512).reshape(res, res)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
for ax, lv, name in [(axes[0], logvol, "Tanh (smooth)"),
                     (axes[1], lv_relu, "ReLU (piecewise linear)")]:
    im = ax.pcolormesh(GX, GY, lv.numpy(), cmap="magma", shading="auto")
    fig.colorbar(im, ax=ax, label="$\\log\\sqrt{\\det g}$")
    ax.set_title(f"{name}: log-volume field")
    ax.set_aspect("equal")
savefig(fig, "pb_step5_relu_vs_tanh.png")
print("   -> ReLU field is constant on polytopes; derivatives at kinks are")
print("      undefined, so curvature-style quantities are meaningless there.")

print("\n(5d) This module is POINTWISE. It does not compute geodesics, exp/log")
print("     maps, or curvature of the input manifold — bridge to geomstats via")
print("     ModelPullbackGeometry.as_geomstats_metric(dim) for that machinery.")

print("\nDone. Figures in", "outputs/figures/")
