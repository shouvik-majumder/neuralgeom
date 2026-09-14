"""Numerical checks for grassmannian.py — run with: python test_grassmannian.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import warnings

import torch
import torch.nn as nn

from neuralgeom.geometry.grassmann import (
    FeatureExtractor,
    ModelSubspaceGeometry,
    grassmann_distance,
    grassmann_exp,
    grassmann_frechet_mean,
    grassmann_log,
    pairwise_grassmann_distance,
    principal_angles,
    projector,
    tangent_subspaces,
)
from neuralgeom.geometry.jacobian import batch_jacobian

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def check(name, cond):
    assert cond, f"FAILED: {name}"
    print(f"  ok: {name}")


def rand_subspace(D, k, B=None):
    shape = (B, D, k) if B else (D, k)
    Q, _ = torch.linalg.qr(torch.randn(*shape))
    return Q


# 1. Principal angles: analytic ground truth
print("[1] Principal angles ground truth")
t = 0.3
e1 = torch.tensor([[1.0], [0.0], [0.0]])
v = torch.tensor([[math.cos(t)], [math.sin(t)], [0.0]])
theta = principal_angles(e1, v)
check("angle(e1, rot(e1, t)) == t", torch.allclose(theta, torch.tensor([t])))

Q1 = torch.eye(4)[:, :2]  # span(e1, e2)
Q2 = torch.eye(4)[:, 2:]  # span(e3, e4)
theta = principal_angles(Q1, Q2)
check("orthogonal planes: both angles pi/2",
      torch.allclose(theta, torch.full((2,), math.pi / 2)))
check("geodesic distance == pi/sqrt(2) * sqrt(k)... = pi/sqrt(2)",
      torch.allclose(grassmann_distance(Q1, Q2),
                     torch.tensor(math.pi / 2 * math.sqrt(2.0))))

# 2. Basis invariance
print("[2] Invariance to basis choice")
Q = rand_subspace(6, 3)
O = torch.linalg.qr(torch.randn(3, 3))[0]
R = rand_subspace(6, 3)
for metric in ["geodesic", "projection", "chordal", "max_angle", "binet_cauchy"]:
    d1 = grassmann_distance(Q, R, metric=metric)
    d2 = grassmann_distance(Q @ O, R, metric=metric)
    check(f"{metric}: d(QO, R) == d(Q, R)", torch.allclose(d1, d2, atol=1e-10))
check("d(Q, Q) == 0",
      grassmann_distance(Q, Q).abs() < 1e-6)
check("projector basis-invariant",
      torch.allclose(projector(Q), projector(Q @ O)))

# 3. Subspaces of a linear model: constant across points, equal to spaces of W
print("[3] Linear model: column/row spaces == spaces of W")
n, m, B = 3, 6, 5
lin = nn.Linear(n, m, bias=True).double()
X = torch.randn(B, n)
Qc, s, rank = tangent_subspaces(lin, X, which="column", k=n)
check("column bases shape (B, m, n)", Qc.shape == (B, m, n))
check("rank == n everywhere", bool((rank == n).all()))
Uw, sw, Vwh = torch.linalg.svd(lin.weight, full_matrices=False)
check("singular values == svd(W)", torch.allclose(s, sw.expand(B, n)))
check("col space == col space of W",
      pairwise_grassmann_distance(Qc, Uw.unsqueeze(0)).max() < 1e-6)
Qr, _, _ = tangent_subspaces(lin, X, which="row", k=n)
check("row space == row space of W (= all of R^n here)",
      pairwise_grassmann_distance(Qr, Vwh.transpose(-1, -2).unsqueeze(0)).max() < 1e-6)
check("constant field: pairwise distances all ~0",
      pairwise_grassmann_distance(Qc).max() < 1e-6)

# 4. FeatureExtractor: hidden-layer Jacobians match manual slicing
print("[4] FeatureExtractor hidden layers")
model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 5), nn.Tanh(),
                      nn.Linear(5, 4)).double()
X = torch.randn(6, 3)
J_hook = batch_jacobian(FeatureExtractor(model, 1), X)   # after first Tanh
J_slice = batch_jacobian(model[:2], X)
check("layer=int: J matches sliced Sequential", torch.allclose(J_hook, J_slice))
J_name = batch_jacobian(FeatureExtractor(model, "2"), X)  # named submodule
check("layer=str: J matches model[:3]",
      torch.allclose(J_name, batch_jacobian(model[:3], X)))
J_mod = batch_jacobian(FeatureExtractor(model, model[2]), X)
check("layer=Module: same result", torch.allclose(J_mod, J_name))
J_none = batch_jacobian(FeatureExtractor(model, None), X)
check("layer=None: full model", torch.allclose(J_none, batch_jacobian(model, X)))

# 5. Row spaces of different layers are comparable (same ambient space)
print("[5] Cross-layer comparison in input space")
g1 = ModelSubspaceGeometry(model, layer=1, which="row", k=2)
g2 = ModelSubspaceGeometry(model, layer=3, which="row", k=2)
Q1_, _, _ = g1.subspaces(X)
Q2_, _, _ = g2.subspaces(X)
d = grassmann_distance(Q1_, Q2_)
check("finite distances, shape (B,)", d.shape == (6,) and bool(torch.isfinite(d).all()))

# 6. log/exp are mutually inverse
print("[6] Riemannian log/exp")
Q0 = rand_subspace(7, 3)
Q1_ = rand_subspace(7, 3)
Delta = grassmann_log(Q0, Q1_)
check("log is horizontal: Q0^T Delta == 0",
      (Q0.transpose(-1, -2) @ Delta).abs().max() < 1e-9)
Q1_rec = grassmann_exp(Q0, Delta)
check("exp(log(Q1)) spans Q1", grassmann_distance(Q1_rec, Q1_) < 1e-6)
check("||log|| == geodesic distance",
      torch.allclose(torch.linalg.svdvals(Delta).square().sum().sqrt(),
                     grassmann_distance(Q0, Q1_), atol=1e-8))

# 7. Fréchet means
print("[7] Fréchet means")
Qb = rand_subspace(5, 2).expand(4, 5, 2)
mu = grassmann_frechet_mean(Qb, method="projection")
check("projection mean of identical subspaces == subspace",
      grassmann_distance(mu, Qb[0]) < 1e-6)
mu = grassmann_frechet_mean(Qb, method="karcher")
check("karcher mean of identical subspaces == subspace",
      grassmann_distance(mu, Qb[0]) < 1e-6)
# two lines at +/- t around e1 in the (e1,e2) plane -> mean is e1
t = 0.4
lines = torch.stack([
    torch.tensor([[math.cos(t)], [math.sin(t)], [0.0]]),
    torch.tensor([[math.cos(t)], [-math.sin(t)], [0.0]]),
])
for method in ["projection", "karcher"]:
    mu = grassmann_frechet_mean(lines, method=method)
    check(f"{method} mean of symmetric pair is bisector e1",
          grassmann_distance(mu, torch.tensor([[1.0], [0.0], [0.0]])) < 1e-6)

# 8. Wrapper: distance matrix properties on a nonlinear model
print("[8] ModelSubspaceGeometry wrapper")
geo = ModelSubspaceGeometry(model, which="column", k=2)
D = geo.distance_matrix(X)
check("distance matrix (B, B)", D.shape == (6, 6))
check("zero diagonal", D.diagonal().abs().max() < 1e-6)
check("symmetric", torch.allclose(D, D.T, atol=1e-9))
mu = geo.frechet_mean(X)
check("mean subspace shape (m, k)", mu.shape == (4, 2))
check("mean orthonormal",
      torch.allclose(mu.T @ mu, torch.eye(2), atol=1e-9))

# 9. Automatic k selection
print("[9] Automatic k")
# rank-2 map: R^4 -> R^4 through a rank-2 bottleneck
bottleneck = nn.Sequential(nn.Linear(4, 2, bias=False),
                           nn.Linear(2, 4, bias=False)).double()
Xb = torch.randn(5, 4)
Qa, _, rank = tangent_subspaces(bottleneck, Xb, which="column")
check("auto k == numerical rank == 2", Qa.shape[-1] == 2 and bool((rank == 2).all()))
Qe, _, _ = tangent_subspaces(lin, X, which="column", variance_fraction=1.0)
check("variance_fraction=1.0 keeps all directions", Qe.shape[-1] == n)

print("\nAll grassmannian.py checks passed.")
