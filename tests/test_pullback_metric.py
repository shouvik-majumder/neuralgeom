"""Numerical checks for pullback_metric.py — run with: python test_pullback_metric.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import warnings

import torch
import torch.nn as nn

from neuralgeom.geometry.jacobian import (
    PullbackGeometry,
    batch_jacobian,
    log_volume_element,
    metric_spectrum,
    pullback_metric,
    volume_element,
)

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def check(name, cond):
    assert cond, f"FAILED: {name}"
    print(f"  ok: {name}")


# 1. Linear map: J must equal the weight matrix, g = W^T W, vol = sqrt(det W^T W)
print("[1] Linear map ground truth")
n, m, B = 3, 5, 7
lin = nn.Linear(n, m, bias=True).double()
X = torch.randn(B, n)
J = batch_jacobian(lin, X)
check("J shape (B, m, n)", J.shape == (B, m, n))
check("J == W for all samples", torch.allclose(J, lin.weight.expand(B, m, n)))
g = pullback_metric(lin, X)
g_true = lin.weight.T @ lin.weight
check("g == W^T W", torch.allclose(g, g_true.expand(B, n, n)))
check("g symmetric", torch.allclose(g, g.transpose(-1, -2)))
vol = volume_element(lin, X)
vol_true = torch.sqrt(torch.det(g_true))
check("vol == sqrt(det W^T W)", torch.allclose(vol, vol_true.expand(B)))
lv = log_volume_element(lin, X)
check("logvol == log vol", torch.allclose(lv, vol.log()))

# 2. Nonlinear map with analytic Jacobian: f(x) = (sin x1, x1*x2)
print("[2] Nonlinear analytic Jacobian")
def f(x):  # (B, 2) -> (B, 2)
    return torch.stack([torch.sin(x[:, 0]), x[:, 0] * x[:, 1]], dim=1)

X2 = torch.randn(10, 2)
J2 = batch_jacobian(f, X2)
J2_true = torch.zeros(10, 2, 2)
J2_true[:, 0, 0] = torch.cos(X2[:, 0])
J2_true[:, 1, 0] = X2[:, 1]
J2_true[:, 1, 1] = X2[:, 0]
check("autodiff matches analytic J", torch.allclose(J2, J2_true))
g2 = pullback_metric(f, X2)
check("det g == det(J)^2", torch.allclose(torch.det(g2), torch.det(J2_true) ** 2, atol=1e-10))
check("vol == |det J|", torch.allclose(volume_element(f, X2), torch.det(J2_true).abs()))

# 3. Degenerate case m < n: rank <= m, pseudo-det vs -inf policies
print("[3] Degenerate metric (m < n)")
lin_deg = nn.Linear(4, 2, bias=False).double()  # rank(g) <= 2 < 4
X3 = torch.randn(6, 4)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    lv3 = log_volume_element(lin_deg, X3, degenerate="pseudo")
    check("pseudo warns on deficiency", any("rank" in str(x.message) for x in w))
s = torch.linalg.svdvals(lin_deg.weight)
check("pseudo logvol == sum log s_i", torch.allclose(lv3, s.log().sum().expand(6)))
lv_inf = log_volume_element(lin_deg, X3, degenerate="neginf")
check("neginf policy -> -inf", torch.isinf(lv_inf).all() and (lv_inf < 0).all())
try:
    log_volume_element(lin_deg, X3, degenerate="strict")
    raise AssertionError("strict should raise")
except ValueError:
    print("  ok: strict raises on deficiency")
eig, rank = metric_spectrum(lin_deg, X3)
check("rank == 2 everywhere", (rank == 2).all())
check("eig zero-padded to n", eig.shape == (6, 4) and (eig[:, 2:] == 0).all())

# 4. MLP, both jac modes agree; wrapper API; non-flat I/O shapes
print("[4] MLP + modes + shapes")
mlp = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 8)).double()
X4 = torch.randn(5, 3)
Jr = batch_jacobian(mlp, X4, mode="rev")
Jf = batch_jacobian(mlp, X4, mode="fwd")
check("rev == fwd", torch.allclose(Jr, Jf, atol=1e-12))
geom = PullbackGeometry(mlp)
check("wrapper metric == J^T J", torch.allclose(geom.metric(X4), Jr.transpose(1, 2) @ Jr))
check("wrapper logvol finite (m>n, generic full rank)", torch.isfinite(geom.log_volume_element(X4)).all())

conv = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.Flatten()).double()
X5 = torch.randn(4, 1, 5, 5)  # n = 25, m = 50
J5 = batch_jacobian(conv, X5)
check("conv input flattened: J is (4, 50, 25)", J5.shape == (4, 50, 25))

# 5. Volume element measures expansion: scaling map x -> c*x has vol = c^n
print("[5] Expansion/contraction sanity")
c = 2.5
scale = lambda x: c * x
X6 = torch.randn(3, 4)
check("vol(c*x) == c^n", torch.allclose(volume_element(scale, X6), torch.tensor(c ** 4).expand(3)))

print("\nAll checks passed.")
