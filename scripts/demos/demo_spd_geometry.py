"""
Demo 3 — spd_geometry: comparing metric tensors as points on SPD/PSD.
=====================================================================

Pipeline (figures land in outputs/figures/):

  Step 0  Raw data + model; the metric tensors g_x we will compare
  Step 1  Raw objects: metric ellipses at points along a transect
  Step 2  Distances: affine vs log-Euclidean vs Bures — do they agree?
  Step 3  Interpolation & the SWELLING EFFECT: why Euclidean averaging of
          metrics is wrong, and what each geometry does instead
  Step 4  Summary: Fréchet means of the boundary metric cloud, per geometry
  Step 5  What these tools CANNOT do (degenerate metrics, live failures)

CAN:  affine-invariant / log-Euclidean / Bures-Wasserstein distances and
      Fréchet means between pullback metrics at different points, layers,
      or models; fixed-rank PSD distance for degenerate metrics with a
      Grassmann-part / SPD-part decomposition.
CANNOT: affine & log-Euclidean on singular metrics (results depend entirely
        on the regularization you choose — demonstrated below),
        Fréchet mean on the fixed-rank manifold (NotImplementedError),
        treat fixed_rank as a true geodesic distance (it is a structure
        metric à la Bonnabel-Sepulchre, not a length metric).

Run:  python demo_spd_geometry.py
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
from neuralgeom.geometry.jacobian import pullback_metric
from neuralgeom.geometry.spd import (MetricFieldGeometry, affine_invariant_distance,
                          bures_wasserstein_distance, log_euclidean_distance,
                          pairwise_spd_distance, psd_decompose,
                          psd_fixed_rank_distance, regularize,
                          spd_expm, spd_frechet_mean, spd_invsqrtm,
                          spd_logm, spd_sqrtm, sym)

# --------------------------------------------------------------------------- #
banner("Step 0", "Data, model, and the raw objects: metric tensors g_x")
X, y = two_moons(400)
model = train_classifier(make_classifier(), X, y)
geo = MetricFieldGeometry(model)

# transect crossing the decision boundary twice
t = torch.linspace(0, 1, 9).unsqueeze(1)
a, b = torch.tensor([-0.5, 1.2]), torch.tensor([1.5, -0.7])
line = a + t * (b - a)
G_line = geo.metrics(line)
print("  9 metric tensors along a transect, shape:", tuple(G_line.shape))
print("  eigenvalue ranges per point (min..max):")
for i, g in enumerate(G_line):
    w = torch.linalg.eigvalsh(g)
    print(f"   t={float(t[i]):.2f}:  [{float(w[0]):.2e}, {float(w[1]):.2e}]"
          f"   (condition number {float(w[1] / w[0].clamp_min(1e-300)):.1e})")

# --------------------------------------------------------------------------- #
banner("Step 1", "The raw objects, visually: metric ellipses along the transect")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), constrained_layout=True)
decision_background(axes[0], model, X, alpha=0.3)
axes[0].plot(line[:, 0], line[:, 1], "k--", lw=0.8)
smax_line = torch.linalg.eigvalsh(G_line)[:, -1].sqrt()
for i in range(line.shape[0]):
    metric_ellipse(axes[0], line[i].numpy(), G_line[i].numpy(),
                   scale=0.12 / float(smax_line[i]),
                   facecolor="none", edgecolor="C3", linewidth=1.6)
axes[0].set_aspect("equal")
axes[0].set_title("Metric ellipses along a transect\n(shape-normalized: "
                  "needle = extreme anisotropy)")
axes[1].plot(t.squeeze().numpy(), smax_line.numpy(), "o-", label="$s_1$")
axes[1].plot(t.squeeze().numpy(),
             torch.linalg.eigvalsh(G_line)[:, 0].sqrt().numpy(), "o-",
             label="$s_2$")
axes[1].set_yscale("log")
axes[1].set_xlabel("t along transect")
axes[1].set_title("Their spectra: each $g_x$ is a point on the\nSPD manifold "
                  "(here nearly on its PSD boundary)")
axes[1].legend()
savefig(fig, "spd_step1_metric_ellipses.png")

# --------------------------------------------------------------------------- #
banner("Step 2", "Three geometries, one question: how far apart are two metrics?")
tt = torch.linspace(0, 1, 60).unsqueeze(1)
path = a + tt * (b - a)
G_path = regularize(geo.metrics(path), 1e-9)  # ridge for the SPD-only metrics
names = ["affine", "log_euclidean", "bures"]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), constrained_layout=True)
Ds = {}
for ax, nm in zip(axes[:3], names):
    D = pairwise_spd_distance(G_path, metric=nm)
    Ds[nm] = D
    im = ax.pcolormesh(tt.squeeze().numpy(), tt.squeeze().numpy(), D.numpy(),
                       cmap="inferno", shading="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{nm}\npairwise distances along transect")
    ax.set_xlabel("t")
iu = torch.triu_indices(60, 60, offset=1)
axes[3].scatter(Ds["affine"][iu[0], iu[1]].numpy(),
                Ds["bures"][iu[0], iu[1]].numpy(), s=3, alpha=0.3)
axes[3].set_xlabel("affine-invariant distance")
axes[3].set_ylabel("Bures-Wasserstein distance")
axes[3].set_title("The geometries DISAGREE nonlinearly:\naffine sees "
                  "conditioning, Bures sees scale")
savefig(fig, "spd_step2_distance_comparison.png")
print("  Spearman-like check: the two distances are not monotone transforms "
      "of each other\n  (scatter has thickness) — choose the geometry to "
      "match your invariance needs.")

# --------------------------------------------------------------------------- #
banner("Step 3", "Interpolation between two metrics + the swelling effect")
A = G_line[1]                      # anisotropic, before boundary
B_ = G_line[4]                     # boundary metric, differently oriented
A, B_ = regularize(A.unsqueeze(0), 1e-6)[0], regularize(B_.unsqueeze(0), 1e-6)[0]

def interp(A, B_, s, how):
    if how == "euclidean":
        return (1 - s) * A + s * B_
    if how == "log_euclidean":
        return spd_expm((1 - s) * spd_logm(A) + s * spd_logm(B_))
    if how == "affine":                       # geodesic: A^(1/2)(A^-1/2 B A^-1/2)^s A^(1/2)
        sA, isA = spd_sqrtm(A), spd_invsqrtm(A)
        W = sym(isA @ B_ @ isA)
        w, V = torch.linalg.eigh(W)
        Ws = (V * w.clamp_min(1e-300).pow(s).unsqueeze(-2)) @ V.transpose(-1, -2)
        return sym(sA @ Ws @ sA)
    if how == "bures":                        # McCann interpolation
        sA, isA = spd_sqrtm(A), spd_invsqrtm(A)
        C = sym(sA @ B_ @ sA)
        T = sym(isA @ spd_sqrtm(C) @ isA)     # optimal transport map A -> B
        M = (1 - s) * torch.eye(2) + s * T
        return sym(M @ A @ M.transpose(-1, -2))

hows = ["euclidean", "log_euclidean", "affine", "bures"]
ss = np.linspace(0, 1, 7)
fig, axes = plt.subplots(len(hows), 1, figsize=(8.5, 7.5), constrained_layout=True)
for ax, how in zip(axes, hows):
    dets = []
    scale = 0.035 if how == "euclidean" else 0.12   # naive mean balloons
    for j, s in enumerate(ss):
        Gi = interp(A, B_, float(s), how)
        dets.append(float(torch.det(Gi)))
        metric_ellipse(ax, (j * 1.0, 0.0), Gi.numpy(), scale=scale,
                       facecolor="C0" if how != "euclidean" else "C3",
                       alpha=0.6, edgecolor="k", linewidth=0.6)
    ax.set_xlim(-0.6, len(ss) - 0.4)
    ax.set_ylim(-0.75, 0.75)
    ax.set_aspect("equal")
    ax.set_yticks([])
    label = how + ("\n(drawn 3.4x smaller!)" if how == "euclidean" else "")
    ax.set_ylabel(label, rotation=0, ha="right", va="center")
    for j, d in enumerate(dets):
        ax.text(j, 0.58, f"det={d:.2g}", ha="center", fontsize=7)
axes[0].set_title("Interpolating between two pullback metrics "
                  "($t=0\\ldots1$, det printed above each)\n"
                  "Euclidean averaging SWELLS the determinant — "
                  "the Riemannian geodesics do not")
axes[-1].set_xlabel("interpolation step")
savefig(fig, "spd_step3_interpolation_swelling.png")
det_e = float(torch.det(interp(A, B_, 0.5, "euclidean")))
det_a = float(torch.det(interp(A, B_, 0.5, "affine")))
print(f"  det at midpoint: euclidean={det_e:.3g}  affine={det_a:.3g}  "
      f"endpoints: {float(torch.det(A)):.3g}, {float(torch.det(B_)):.3g}")
print("  -> arithmetic averaging of metric tensors inflates volume; use a "
      "Fréchet mean.")

# --------------------------------------------------------------------------- #
banner("Step 4", "Summary: Fréchet means of the 'boundary metric cloud'")
with torch.no_grad():
    prob = torch.softmax(model(X), 1)[:, 1]
near = X[torch.topk(-(prob - 0.5).abs(), 60).indices]  # 60 most boundary-ish
print(f"  {near.shape[0]} points nearest the boundary")
G_near = regularize(geo.metrics(near), 1e-8)
means = {"euclidean (naive)": G_near.mean(0)}
for nm in ["log_euclidean", "affine", "bures"]:
    means[nm] = spd_frechet_mean(G_near, metric=nm)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
colors = {"euclidean (naive)": "C3", "log_euclidean": "C0",
          "affine": "C2", "bures": "C4"}
smax_mean = max(float(torch.linalg.eigvalsh(M)[-1].sqrt()) for M in means.values())
for nm, M in means.items():
    metric_ellipse(axes[0], (0, 0), M.numpy(), scale=1.0 / smax_mean,
                   facecolor="none", edgecolor=colors[nm], linewidth=2,
                   label=nm)
axes[0].legend(handles=[plt.Line2D([], [], color=c, label=n)
                        for n, c in colors.items()], loc="upper right",
               fontsize=7)
axes[0].set_xlim(-1.6, 1.6); axes[0].set_ylim(-1.3, 1.3)
axes[0].set_aspect("equal")
axes[0].set_title("Mean metric tensor, 4 ways (same scale):\nthe naive mean "
                  "is fatter (swelling) and differently oriented")
dets = {nm: float(torch.det(M)) for nm, M in means.items()}
gm_det = float(torch.det(G_near).log().mean().exp())   # geometric mean of dets
bars = axes[1].bar(range(len(dets)), list(dets.values()),
                   color=[colors[n] for n in dets])
axes[1].axhline(gm_det, color="k", ls="--",
                label="geometric mean of individual dets")
axes[1].set_xticks(range(len(dets)), list(dets.keys()), rotation=15, fontsize=7)
axes[1].set_yscale("log")
axes[1].set_title("det of the mean: affine/LE means preserve the\ngeometric "
                  "mean determinant; the naive mean inflates it")
axes[1].legend(fontsize=7)
savefig(fig, "spd_step4_frechet_means.png")
for nm, d in dets.items():
    print(f"   det[{nm:18s}] = {d:.4g}")
print(f"   geometric mean of dets   = {gm_det:.4g}   <- affine/LE match this")

# --------------------------------------------------------------------------- #
banner("Step 5", "LIMITATIONS — degenerate metrics and live failures")

print("\n(5a) Bottleneck models (2 -> 1 -> 2) have rank-1 pullback metrics "
      "EVERYWHERE.\n     Comparing the metrics of TWO such models (misaligned "
      "ranges): the\n     affine / log-Euclidean answer depends entirely on "
      "your ridge eps:")
bott = nn.Sequential(nn.Linear(2, 1), nn.Tanh(), nn.Linear(1, 2))
bott2 = nn.Sequential(nn.Linear(2, 1), nn.Tanh(), nn.Linear(1, 2))
Xs = X[:20]
G_deg = pullback_metric(bott, Xs)     # rank-1, one range direction
G_deg2 = pullback_metric(bott2, Xs)   # rank-1, a different range direction
print(f"   eigenvalues of g at one point: "
      f"{torch.linalg.eigvalsh(G_deg[0]).numpy()}")
eps_range = [1e-12, 1e-9, 1e-6, 1e-3]
d_aff, d_bur = [], []
for eps in eps_range:
    A_, B2_ = regularize(G_deg[0], eps), regularize(G_deg2[0], eps)
    d_aff.append(float(affine_invariant_distance(A_, B2_)))
    d_bur.append(float(bures_wasserstein_distance(G_deg[0], G_deg2[0])))
    print(f"   eps={eps:.0e}:  d_affine = {d_aff[-1]:8.3f}   "
          f"d_bures = {d_bur[-1]:.6f}  (eps-independent)")
print("   -> affine distance DIVERGES like log(1/eps): it is not defined on "
      "PSD.\n      The number you get is an artifact of the ridge, not of "
      "the models.")

fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
ax.plot(eps_range, d_aff, "o-", label="affine-invariant (needs ridge)")
ax.plot(eps_range, d_bur, "s-", label="Bures-Wasserstein (PSD-native)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("regularization eps")
ax.set_ylabel("distance between $g_{x_1}, g_{x_2}$")
ax.set_title("Degenerate metrics: the affine answer is an artifact\nof the "
             "ridge; Bures and fixed-rank are the honest tools")
ax.legend(fontsize=8)
savefig(fig, "spd_step5_degenerate_eps.png")

print("\n(5b) The fixed-rank PSD metric handles degeneracy natively and "
      "DECOMPOSES:\n     total^2 = (range misalignment)^2 + alpha^2 "
      "(spectral mismatch)^2")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    d, d_g, d_s = psd_fixed_rank_distance(G_deg[:4], G_deg2[:4], k=1,
                                          return_parts=True)
for i in range(4):
    print(f"   pair {i}: total={float(d[i]):.3f}  "
          f"grassmann={float(d_g[i]):.3f} (range misalignment)  "
          f"spd={float(d_s[i]):.3f} (spectral)")
print("   BUT it is a structure metric (Bonnabel-Sepulchre), NOT a geodesic "
      "distance:\n   triangle inequality is not guaranteed; don't use it "
      "where that matters.")

print("\n(5c) No Fréchet mean on the fixed-rank manifold (open problem-ish):")
sgeo = MetricFieldGeometry(bott)
try:
    sgeo.frechet_mean(Xs, metric="fixed_rank")
except NotImplementedError as e:
    print(f"   NotImplementedError (as intended): {str(e)[:70]}...")

print("\n(5d) These are distances between METRIC TENSORS at points — not "
      "distances\n     between the points themselves. For lengths of curves "
      "in input space,\n     integrate sqrt(v^T g v) or bridge to geomstats.")

print("\nDone. Figures in outputs/figures/")
