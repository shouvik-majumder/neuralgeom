"""
Demo 4 — regression: the geometry of a scalar-output network.
=============================================================

A regression network f: R^2 -> R is geometrically EXTREME: the output is
1-dimensional, so the pullback metric g = J^T J = grad(f) grad(f)^T has rank
1 at every single point. The "degenerate metric" machinery is not an edge
case here — it is the whole story:

    pseudo-volume element  = |grad f|      (the only stretch there is)
    row space (k=1)        = gradient direction (perpendicular to level sets)
    kernel of g            = the level-set direction f is blind to

Because the regression target is a known analytic function, this demo can do
something the classification demos could not: VALIDATE the learned geometry
against exact ground truth.

Pipeline:
  Step 0  Raw data + fit: target surface, samples, learned surface, residuals
  Step 1  Rank collapse by construction; pseudo-volume |grad f|, learned vs true
  Step 2  Row-space field = gradient directions vs level sets
  Step 3  Validation: angle error against the ANALYTIC gradient +
          breakdown outside the training support (extrapolation warning map)
  Step 4  Fixed-rank PSD tools on a transect (the honest metric comparison)
  Step 5  What regression geometry CANNOT tell you

Run:  python demo_regression.py
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

from demo_common import banner, savefig
from neuralgeom.geometry.jacobian import (batch_jacobian, log_volume_element,
                             metric_spectrum, pullback_metric, volume_element)
from neuralgeom.geometry.grassmann import grassmann_distance, tangent_subspaces
from neuralgeom.geometry.spd import (MetricFieldGeometry, affine_invariant_distance,
                          psd_fixed_rank_distance, regularize)

torch.manual_seed(1)


# --------------------------------------------------------------------------- #
def target(P):
    """Ground-truth regression surface h(x, y) = sin(2x) cos(2y)."""
    return torch.sin(2 * P[:, 0]) * torch.cos(2 * P[:, 1])


def target_grad(P):
    """Analytic gradient of the target (for validating learned geometry)."""
    gx = 2 * torch.cos(2 * P[:, 0]) * torch.cos(2 * P[:, 1])
    gy = -2 * torch.sin(2 * P[:, 0]) * torch.sin(2 * P[:, 1])
    return torch.stack([gx, gy], dim=1)


def grid(lo, hi, res):
    g1 = torch.linspace(lo, hi, res)
    GX, GY = torch.meshgrid(g1, g1, indexing="xy")
    return torch.stack([GX.reshape(-1), GY.reshape(-1)], 1), GX.numpy(), GY.numpy()


banner("Step 0", "Data + model: 600 noisy samples of sin(2x)cos(2y) on "
       "[-1.5, 1.5]^2,\nMLP 2 -> 48 -> 48 -> 1 (Tanh), MSE")
Xtr = (torch.rand(600, 2) * 3.0) - 1.5          # training support: [-1.5, 1.5]^2
ytr = target(Xtr) + 0.05 * torch.randn(600)

model = nn.Sequential(nn.Linear(2, 48), nn.Tanh(), nn.Linear(48, 48),
                      nn.Tanh(), nn.Linear(48, 1))
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
with torch.enable_grad():
    for i in range(1500):
        opt.zero_grad()
        loss = ((model(Xtr).squeeze(-1) - ytr) ** 2).mean()
        loss.backward()
        opt.step()
model.eval()
print(f"  final train MSE: {float(loss):.5f} (noise floor 0.0025)")

res = 70
P_in, GX, GY = grid(-1.5, 1.5, res)             # inside the data support
with torch.no_grad():
    f_in = model(P_in).squeeze(-1).reshape(res, res)
h_in = target(P_in).reshape(res, res)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), constrained_layout=True)
for ax, Z, name in [(axes[0], h_in, "TARGET  $h = \\sin 2x\\,\\cos 2y$"),
                    (axes[1], f_in, "LEARNED  $f$ (MLP)"),
                    (axes[2], (f_in - h_in), "residual  $f - h$")]:
    im = ax.pcolormesh(GX, GY, Z.numpy(), cmap="RdBu_r" if "resid" not in name
                       else "PuOr", shading="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(name)
    ax.set_aspect("equal")
axes[0].scatter(Xtr[:, 0], Xtr[:, 1], s=1.5, c="k", alpha=0.25)
savefig(fig, "rg_step0_fit.png")

# --------------------------------------------------------------------------- #
banner("Step 1", "Regression geometry is rank-1 BY CONSTRUCTION: "
       "g = grad f grad f^T")
g5 = pullback_metric(model, P_in[:3])
w = torch.linalg.eigvalsh(g5)
print("  eigenvalues of g at 3 points (one is always ~0):")
for i in range(3):
    print(f"   {w[i].numpy()}")
with warnings.catch_warnings(record=True) as wrec:
    warnings.simplefilter("always")
    pv = volume_element(model, P_in[:3])
print(f"  volume_element warns ({wrec[0].category.__name__}) and returns the "
      f"pseudo-volume = |grad f|: {pv.numpy().round(3)}")

J = batch_jacobian(model, P_in, chunk_size=1024)      # (B, 1, 2)
gradnorm = J.squeeze(1).norm(dim=1).reshape(res, res)
truenorm = target_grad(P_in).norm(dim=1).reshape(res, res)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
vmax = float(truenorm.max())
for ax, Z, name in [(axes[0], gradnorm, "LEARNED pseudo-volume $|\\nabla f|$"),
                    (axes[1], truenorm, "TRUE $|\\nabla h|$ (analytic)")]:
    im = ax.pcolormesh(GX, GY, Z.numpy(), cmap="magma", vmin=0, vmax=vmax,
                       shading="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(name)
    ax.set_aspect("equal")
fig.suptitle("For scalar outputs the volume element IS the gradient "
             "magnitude — and it matches the analytic truth", y=1.05)
savefig(fig, "rg_step1_pseudo_volume.png")

# --------------------------------------------------------------------------- #
banner("Step 2", "Row space (k=1) = gradient direction; the kernel of g is "
       "the level-set direction")
res_c = 20
Pc, GXc, GYc = grid(-1.5, 1.5, res_c)
Q, s, _ = tangent_subspaces(model, Pc, which="row", k=1)
d = Q[:, :, 0]
fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
cs = ax.contour(GX, GY, f_in.numpy(), levels=12, cmap="RdBu_r", linewidths=1.0)
ax.quiver(Pc[:, 0].numpy(), Pc[:, 1].numpy(), d[:, 0].numpy(), d[:, 1].numpy(),
          s[:, 0].numpy(), cmap="magma", angles="xy", pivot="mid",
          headwidth=1, headlength=0.1, headaxislength=0.1, width=4e-3)
ax.set_aspect("equal")
ax.set_title("Row-space bars (color: $|\\nabla f|$) over level sets of $f$.\n"
             "Bars are perpendicular to level curves: the network can only\n"
             "'see' motion across level sets — the along-level direction is "
             "its kernel")
savefig(fig, "rg_step2_gradient_field.png")

print("  column space check: the output is 1-dim, so ALL column spaces are "
      "the same\n  point on Gr(1,1) — column-space analysis is vacuous for "
      "regression:")
Qc1, _, _ = tangent_subspaces(model, Pc[:2], which="column", k=1)
print(f"   d(col(x1), col(x2)) = "
      f"{float(grassmann_distance(Qc1[0], Qc1[1])):.2e}  (always 0)")

# --------------------------------------------------------------------------- #
banner("Step 3", "VALIDATION vs analytic truth + extrapolation breakdown")
res_b = 90
P_big, GXb, GYb = grid(-2.5, 2.5, res_b)              # bigger than the data!
J_big = batch_jacobian(model, P_big, chunk_size=1024).squeeze(1)
tg = target_grad(P_big)
# angle between learned gradient direction and true gradient direction
cosang = (J_big * tg).sum(1).abs() / (J_big.norm(dim=1) * tg.norm(dim=1))
ang = torch.rad2deg(torch.acos(cosang.clamp(0, 1))).reshape(res_b, res_b)
mask = (tg.norm(dim=1) < 0.15).reshape(res_b, res_b)  # direction undefined
ang_ma = np.ma.masked_where(mask.numpy(), ang.numpy())

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
im = axes[0].pcolormesh(GXb, GYb, ang_ma, cmap="inferno", vmin=0, vmax=90,
                        shading="auto")
fig.colorbar(im, ax=axes[0], label="angle error [deg]")
axes[0].add_patch(plt.Rectangle((-1.5, -1.5), 3, 3, fill=False, ec="cyan",
                                lw=2))
axes[0].set_title("Angle between learned and TRUE gradient direction\n"
                  "(cyan box = training support; white = critical points)")
axes[0].set_aspect("equal")
inside = (P_big.abs().max(dim=1).values <= 1.5)
ok = ~mask.reshape(-1)
axes[1].hist(ang.reshape(-1)[inside & ok].numpy(), bins=45, alpha=0.65,
             density=True, label="inside data support", color="C0")
axes[1].hist(ang.reshape(-1)[~inside & ok].numpy(), bins=45, alpha=0.65,
             density=True, label="outside (extrapolation)", color="C3")
axes[1].set_xlabel("gradient direction error [deg]")
axes[1].legend()
axes[1].set_title("The geometry is accurate ON the data manifold\nand "
                  "unreliable off it — geometry inherits the fit's limits")
savefig(fig, "rg_step3_validation.png")
print(f"  median angle error inside support: "
      f"{float(ang.reshape(-1)[inside & ok].median()):.2f} deg")
print(f"  median angle error outside:        "
      f"{float(ang.reshape(-1)[~inside & ok].median()):.2f} deg")

# --------------------------------------------------------------------------- #
banner("Step 4", "Comparing rank-1 metrics honestly: fixed-rank PSD transect")
tt = torch.linspace(0, 1, 60).unsqueeze(1)
a, b = torch.tensor([-1.3, -1.3]), torch.tensor([1.3, 1.3])
path = a + tt * (b - a)
geo = MetricFieldGeometry(model)
G_path = geo.metrics(path)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    d_tot, d_gr, d_sp = psd_fixed_rank_distance(G_path[:-1], G_path[1:], k=1,
                                                return_parts=True)
    Dfr = geo.distance_matrix(path, metric="fixed_rank")

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)
mid = (tt[:-1] + tt[1:]).squeeze() / 2
axes[0].plot(mid.numpy(), d_gr.numpy(), label="Grassmann part (direction turns)")
axes[0].plot(mid.numpy(), d_sp.numpy(), label="SPD part (magnitude changes)")
axes[0].plot(mid.numpy(), d_tot.numpy(), "k--", lw=1, label="total")
axes[0].set_xlabel("t along diagonal transect")
axes[0].set_title("Consecutive fixed-rank distances DECOMPOSE:\nwhen the "
                  "gradient rotates vs. when it grows/shrinks")
axes[0].legend(fontsize=8)
im = axes[1].pcolormesh(tt.squeeze().numpy(), tt.squeeze().numpy(),
                        Dfr.numpy(), cmap="inferno", shading="auto")
fig.colorbar(im, ax=axes[1])
axes[1].set_title("Pairwise fixed-rank distance matrix along the transect")
axes[1].set_xlabel("t")
savefig(fig, "rg_step4_fixed_rank.png")

eps_check = [1e-10, 1e-4]
d_eps = [float(affine_invariant_distance(regularize(G_path[0], e),
                                         regularize(G_path[30], e)))
         for e in eps_check]
print(f"  (affine-invariant on these rank-1 metrics is still an eps artifact:"
      f" d={d_eps[0]:.1f} at eps=1e-10 vs d={d_eps[1]:.1f} at eps=1e-4)")

# --------------------------------------------------------------------------- #
banner("Step 5", "LIMITATIONS — what regression geometry cannot tell you")
print("""
(5a) g is blind ALONG level sets by construction: any input change that
     keeps f constant is invisible. rank(g)=1 is not a bug of the fit — it
     is the geometry of scalar prediction. (Demonstrated: eigenvalues, Step 1.)

(5b) The geometry describes the FITTED surface, not the noise: the 0.05
     target noise appears nowhere in g. Two networks fit on different noise
     draws have nearly identical geometry. No error bars are implied.

(5c) Off the training support the geometry is exactly as wrong as the fit
     (Step 3, right histogram) — it cannot flag its own extrapolation.
     Combine with the data density before trusting any field.

(5d) Column-space (Grassmannian) analysis is vacuous for scalar outputs:
     Gr(1,1) is a single point (Step 2 check printed above). Use row spaces.

(5e) Affine-invariant / log-Euclidean SPD distances remain ill-defined for
     these rank-1 metrics (eps artifact printed in Step 4); use
     bures or fixed_rank.
""")
print("Done. Figures in outputs/figures/")
