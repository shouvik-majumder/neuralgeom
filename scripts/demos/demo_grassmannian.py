"""
Demo 2 — grassmannian: subspace fields of a network, at any layer.
==================================================================

Pipeline (figures land in outputs/figures/):

  Step 0  Raw data + models (a 2->3 embedding to *see* column spaces in 3D,
          and the two-moons classifier for everything else)
  Step 1  Raw objects: column spaces as tangent planes on the embedded surface
  Step 2  Row-space field: which input direction each layer listens to
  Step 3  Distances: pairwise Grassmann matrices along a loop, all metrics
  Step 4  Cross-layer + summary: layer agreement map, Fréchet mean, angles
  Step 5  What these tools CANNOT do (live failure demos)

CAN:  tangent subspaces (column/row) at any layer via FeatureExtractor,
      principal angles, six distances, log/exp maps, Fréchet means,
      cross-layer comparison of *row* spaces (shared input space).
CANNOT: compare subspaces of different ambient spaces (column spaces of
        different layers), compare subspaces of different dimension k,
        take log maps at/near the cut locus (angle pi/2),
        use the martin distance on orthogonal subspaces (infinite),
        make k-independent statements (distances depend on k).

Run:  python demo_grassmannian.py
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
from neuralgeom.geometry.grassmann import (FeatureExtractor, ModelSubspaceGeometry,
                          grassmann_distance, grassmann_log,
                          grassmann_frechet_mean, pairwise_grassmann_distance,
                          principal_angles, tangent_subspaces)

# --------------------------------------------------------------------------- #
banner("Step 0", "Data + two models: a 2->3 embedding (to visualize column "
       "spaces)\nand the two-moons classifier (row spaces, distances, means)")
X, y = two_moons(400)
clf = train_classifier(make_classifier(), X, y)
# a fixed smooth embedding R^2 -> R^3: input space rendered as a surface
embed = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3))
for p in embed.parameters():  # squash for a gentle, visible curvature
    p.data *= 1.6
print("  embedding: random smooth immersion f: R^2 -> R^3 (fixed seed)")

# --------------------------------------------------------------------------- #
banner("Step 1", "Column spaces = tangent planes of the embedded surface, "
       "as points on Gr(2, 3)")
res = 30
P, GX, GY = input_grid(X, res, pad=0.1)
with torch.no_grad():
    S = embed(P).reshape(res, res, 3)
# column-space bases at a few sample points
idx = torch.randint(0, P.shape[0], (8,))
Q_col, s_col, _ = tangent_subspaces(embed, P[idx], which="column", k=2)
print("  bases:", tuple(Q_col.shape), " orthonormal:",
      bool(torch.allclose(Q_col[0].T @ Q_col[0], torch.eye(2), atol=1e-8)))

fig = plt.figure(figsize=(10, 4.2), constrained_layout=True)
ax = fig.add_subplot(1, 2, 1)
scatter_data(ax, X, y)
ax.scatter(*P[idx].T.numpy(), c="k", s=40, marker="x", label="sample points")
ax.set_aspect("equal")
ax.set_title("Input space with sample points")
ax.legend(loc="lower left")
ax3 = fig.add_subplot(1, 2, 2, projection="3d")
ax3.plot_surface(S[..., 0].numpy(), S[..., 1].numpy(), S[..., 2].numpy(),
                 alpha=0.45, cmap="viridis", linewidth=0)
with torch.no_grad():
    base = embed(P[idx])
for i in range(len(idx)):
    for j, col in enumerate(["r", "b"]):
        v = 0.18 * Q_col[i, :, j]
        ax3.quiver(*base[i].numpy(), *v.numpy(), color=col, linewidth=1.8)
ax3.set_title("Image surface $f(\\mathbb{R}^2)\\subset\\mathbb{R}^3$; "
              "arrows span the\ncolumn space of $J_x$ = tangent plane "
              "(a point on Gr(2,3))")
ax3.grid(False)
ax3.view_init(elev=28, azim=-55)
lims = torch.stack([S.reshape(-1, 3).min(0).values,
                    S.reshape(-1, 3).max(0).values])
ax3.set_xlim(*lims[:, 0]); ax3.set_ylim(*lims[:, 1]); ax3.set_zlim(*lims[:, 2])
ax3.set_box_aspect((float(lims[1, 0] - lims[0, 0]),
                    float(lims[1, 1] - lims[0, 1]),
                    float(lims[1, 2] - lims[0, 2])))
savefig(fig, "gr_step1_tangent_planes.png")

# --------------------------------------------------------------------------- #
banner("Step 2", "Row spaces (k=1): the single input direction each map "
       "is most sensitive to")
res = 24
Pg, GXg, GYg = input_grid(X, res)
layers = [(1, "after 1st Tanh"), (3, "after 2nd Tanh"), (None, "output logits")]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
for ax, (layer, name) in zip(axes, layers):
    Q, s, _ = tangent_subspaces(clf, Pg, which="row", k=1, layer=layer)
    d = Q[:, :, 0].numpy()            # unit direction per grid point
    s1 = s[:, 0].numpy()
    decision_background(ax, clf, X, alpha=0.2)
    ax.quiver(Pg[:, 0].numpy(), Pg[:, 1].numpy(), d[:, 0], d[:, 1], s1,
              cmap="magma", angles="xy", pivot="mid",
              headwidth=1, headlength=0.1, headaxislength=0.1, width=4e-3)
    ax.set_aspect("equal")
    ax.set_title(f"layer = {name}\n(bars: top row-space direction, color: $s_1$)")
print("  note: a 1-dim subspace has no sign — bars, not arrows.")
fig.suptitle("Row-space fields live in the SAME input plane for every layer "
             "-> directly comparable", y=1.04)
savefig(fig, "gr_step2_rowspace_fields.png")

# --------------------------------------------------------------------------- #
banner("Step 3", "Pairwise Grassmann distances along a loop around one moon")
# a loop in input space: circle around the left moon
t = torch.linspace(0, 2 * math.pi, 80)
loop = torch.stack([0.15 + 0.9 * t.cos(), 0.25 + 0.9 * t.sin()], dim=1)
Q_loop, _, _ = tangent_subspaces(clf, loop, which="row", k=1)
metrics = ["geodesic", "projection", "chordal", "binet_cauchy"]
fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(16, 3.2),
                         constrained_layout=True)
decision_background(axes[0], clf, X, alpha=0.3)
pts = axes[0].scatter(loop[:, 0], loop[:, 1], c=t.numpy(), cmap="twilight", s=14)
fig.colorbar(pts, ax=axes[0], label="loop parameter t")
axes[0].set_aspect("equal")
axes[0].set_title("The loop (colored by t)")
for ax, mname in zip(axes[1:], metrics):
    D = pairwise_grassmann_distance(Q_loop, metric=mname)
    im = ax.pcolormesh(t.numpy(), t.numpy(), D.numpy(), cmap="inferno",
                       shading="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"metric = {mname}")
    ax.set_xlabel("t")
print("  all metrics agree on the qualitative structure; scales differ.")
savefig(fig, "gr_step3_distance_matrices.png")

# --------------------------------------------------------------------------- #
banner("Step 4", "Cross-layer agreement + Fréchet mean summary")
res = 60
Pf, GXf, GYf = input_grid(X, res)
Q_hid, _, _ = tangent_subspaces(clf, Pf, which="row", k=1, layer=1,
                                chunk_size=512)
Q_out, _, _ = tangent_subspaces(clf, Pf, which="row", k=1, layer=None,
                                chunk_size=512)
d_layers = grassmann_distance(Q_hid, Q_out).reshape(res, res)

Q_data, _, _ = tangent_subspaces(clf, X, which="row", k=1)
mu = grassmann_frechet_mean(Q_data, method="karcher")
ang_to_mean = principal_angles(Q_data, mu.expand(X.shape[0], 2, 1)).squeeze(-1)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
im = axes[0].pcolormesh(GXf, GYf, d_layers.numpy(), cmap="cividis",
                        shading="auto")
fig.colorbar(im, ax=axes[0], label="Grassmann distance")
decision_background(axes[0], clf, X, alpha=0.0)
axes[0].set_aspect("equal")
axes[0].set_title("Where hidden layer and output DISAGREE\non the sensitive "
                  "direction (row spaces, k=1)")
sc = axes[1].scatter(X[:, 0], X[:, 1], c=ang_to_mean.numpy(), cmap="plasma",
                     s=10)
fig.colorbar(sc, ax=axes[1], label="principal angle to mean [rad]")
axes[1].arrow(0.6, 0.25, 0.45 * float(mu[0, 0]), 0.45 * float(mu[1, 0]),
              color="k", width=0.015, zorder=5)
axes[1].set_aspect("equal")
axes[1].set_title("Karcher mean direction (arrow) and each\npoint's angle "
                  "to it — dispersion on Gr(1,2)")
axes[2].hist(ang_to_mean.numpy(), bins=40, color="C4")
axes[2].set_xlabel("principal angle to Fréchet mean [rad]")
axes[2].set_title("Summary: distribution of subspace dispersion\n"
                  "(concentrated field vs. uniform would be flat)")
savefig(fig, "gr_step4_crosslayer_and_mean.png")
print(f"  mean direction: {mu.squeeze().numpy().round(3)}, "
      f"dispersion: mean angle = {float(ang_to_mean.mean()):.3f} rad")

# --------------------------------------------------------------------------- #
banner("Step 5", "LIMITATIONS — live demonstrations of what this CANNOT do")

print("\n(5a) Column spaces of DIFFERENT layers live in different ambient "
      "spaces\n     (32-dim hidden vs 2-dim output) — comparison must fail:")
Q_h, _, _ = tangent_subspaces(clf, X[:4], which="column", k=1, layer=1)
Q_o, _, _ = tangent_subspaces(clf, X[:4], which="column", k=1, layer=None)
try:
    grassmann_distance(Q_h, Q_o)
except ValueError as e:
    print(f"   ValueError (as intended): {str(e)[:96]}...")

print("\n(5b) Subspaces of different dimension k are not directly comparable:")
Qa, _, _ = tangent_subspaces(clf, X[:4], which="row", k=1)
Qb, _, _ = tangent_subspaces(clf, X[:4], which="row", k=2)
try:
    grassmann_distance(Qa, Qb)
except ValueError as e:
    print(f"   ValueError (as intended): {str(e)[:96]}...")

print("\n(5c) The log map is UNDEFINED at the cut locus (principal angle pi/2):")
print("     ||log|| = angle stays finite, but at exactly pi/2 the geodesic is")
print("     non-unique and the linear solve is singular:")
angles = torch.tensor([0.3, 1.0, 1.5, 1.5701])
for a in angles:
    q0 = torch.tensor([[1.0], [0.0]])
    q1 = torch.tensor([[math.cos(a)], [math.sin(a)]])
    n = float(torch.linalg.matrix_norm(grassmann_log(q0, q1)))
    print(f"   angle {float(a):7.4f} rad: ||log|| = {n:10.4f}")
try:
    q_orth = torch.tensor([[0.0], [1.0]])
    out = grassmann_log(torch.tensor([[1.0], [0.0]]), q_orth)
    print(f"   angle exactly pi/2: log = {out.squeeze().numpy()}  "
          "(singular solve -> inf/nan, NOT a usable tangent)")
except RuntimeError as e:
    print(f"   angle exactly pi/2: RuntimeError (singular system): {str(e)[:60]}")

print("\n(5d) The martin distance is +inf for orthogonal subspaces; the")
print("     implementation clamps cos^2 at 1e-12, so it SATURATES instead:")
q0 = torch.tensor([[1.0], [0.0]])
q1 = torch.tensor([[0.0], [1.0]])
d_mart = float(grassmann_distance(q0, q1, metric='martin'))
print(f"   d_martin(e1, e2) = {d_mart:.3f}  (= sqrt(-log 1e-12), the cap —")
print("   mathematically this is infinite; don't average martin distances")
print("   over near-orthogonal pairs.)")

print("\n(5e) Distances DEPEND on the choice of k — there is no canonical "
      "'subspace\n     distance' without fixing the subspace dimension:")
x1, x2 = X[:1], X[100:101]
Jfe = FeatureExtractor(clf, 1)
ks, ds = [], []
for k in [1, 2]:
    Qa, _, _ = tangent_subspaces(clf, x1, which="row", k=k, layer=1)
    Qb, _, _ = tangent_subspaces(clf, x2, which="row", k=k, layer=1)
    ks.append(k)
    ds.append(float(grassmann_distance(Qa, Qb)))
    print(f"   k={k}: d_geodesic = {ds[-1]:.4f}")
print("   (k=2 in a 2-dim input space is the whole plane -> distance 0. "
      "Interpret k relative to ambient dim!)")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
aa = torch.linspace(0.05, math.pi / 2 - 1e-4, 200)
nn_ = [float(torch.linalg.matrix_norm(grassmann_log(
    torch.tensor([[1.0], [0.0]]),
    torch.tensor([[math.cos(a)], [math.sin(a)]])))) for a in aa]
axes[0].plot(aa.numpy(), nn_)
axes[0].axvline(math.pi / 2, color="r", ls="--", label="cut locus $\\pi/2$")
axes[0].set_xlabel("principal angle [rad]")
axes[0].set_ylabel("$\\|\\log_{Q_0}(Q_1)\\|_F$")
axes[0].set_title("Log map is exact ($\\|\\log\\| =$ angle) but\nUNDEFINED at the cut locus")
axes[0].legend()
mart = [float(grassmann_distance(
    torch.tensor([[1.0], [0.0]]),
    torch.tensor([[math.cos(a)], [math.sin(a)]]), metric="martin")) for a in aa]
axes[1].plot(aa.numpy(), mart, color="C3")
axes[1].axvline(math.pi / 2, color="r", ls="--")
axes[1].axhline(math.sqrt(-math.log(1e-12)), color="0.5", ls=":",
                label="implementation cap")
axes[1].set_yscale("log")
axes[1].set_xlabel("principal angle [rad]")
axes[1].set_title("Martin distance diverges toward\northogonality (clamped at the cap)")
axes[1].legend()
savefig(fig, "gr_step5_cut_locus.png")

print("\nDone. Figures in outputs/figures/")
