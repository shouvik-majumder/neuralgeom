"""Numerical checks for spd_geometry.py — run with: python test_spd_geometry.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import warnings

import torch
import torch.nn as nn

from neuralgeom.geometry.spd import (
    MetricFieldGeometry,
    affine_invariant_distance,
    bures_wasserstein_distance,
    log_euclidean_distance,
    pairwise_spd_distance,
    psd_decompose,
    psd_fixed_rank_distance,
    regularize,
    spd_distance,
    spd_expm,
    spd_frechet_mean,
    spd_invsqrtm,
    spd_logm,
    spd_sqrtm,
    sym,
)

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def check(name, cond):
    assert cond, f"FAILED: {name}"
    print(f"  ok: {name}")


def rand_spd(n, B=None, scale=1.0):
    shape = (B, n, n) if B else (n, n)
    A = torch.randn(*shape) * scale
    return A @ A.transpose(-1, -2) + 0.5 * torch.eye(n)


# 1. Matrix functions
print("[1] Matrix functions")
A = rand_spd(5)
check("expm(logm(A)) == A", torch.allclose(spd_expm(spd_logm(A)), A, atol=1e-10))
S = spd_sqrtm(A)
check("sqrtm(A)^2 == A", torch.allclose(S @ S, A, atol=1e-10))
iS = spd_invsqrtm(A)
check("invsqrtm(A) == inv(sqrtm(A))",
      torch.allclose(iS, torch.linalg.inv(S), atol=1e-9))

# 2. Scalar (1x1) closed forms
print("[2] 1x1 closed forms")
a, b = 4.0, 9.0
Ta = torch.tensor([[[a]]])
Tb = torch.tensor([[[b]]])
check("affine: |log(a/b)|",
      torch.allclose(affine_invariant_distance(Ta, Tb),
                     torch.tensor([abs(math.log(a / b))])))
check("log-euclidean: |log a - log b|",
      torch.allclose(log_euclidean_distance(Ta, Tb),
                     torch.tensor([abs(math.log(a) - math.log(b))])))
check("bures: |sqrt a - sqrt b|",
      torch.allclose(bures_wasserstein_distance(Ta, Tb),
                     torch.tensor([abs(math.sqrt(a) - math.sqrt(b))])))

# 3. Commuting (diagonal) matrices: affine == log-euclidean
print("[3] Commuting case")
D1 = torch.diag(torch.tensor([1.0, 4.0, 0.5]))
D2 = torch.diag(torch.tensor([2.0, 1.0, 3.0]))
check("diag: d_AI == d_LE",
      torch.allclose(affine_invariant_distance(D1, D2),
                     log_euclidean_distance(D1, D2), atol=1e-10))

# 4. Invariances
print("[4] Invariances")
A, B_ = rand_spd(4), rand_spd(4)
G = torch.randn(4, 4)
G = G + 4 * torch.eye(4)  # well-conditioned GL(4) element
check("affine: congruence invariance d(GAG^T, GBG^T) == d(A, B)",
      torch.allclose(affine_invariant_distance(sym(G @ A @ G.T), sym(G @ B_ @ G.T)),
                     affine_invariant_distance(A, B_), atol=1e-8))
O = torch.linalg.qr(torch.randn(4, 4))[0]
for name, fn in [("log_euclidean", log_euclidean_distance),
                 ("bures", bures_wasserstein_distance)]:
    check(f"{name}: orthogonal invariance",
          torch.allclose(fn(sym(O @ A @ O.T), sym(O @ B_ @ O.T)), fn(A, B_),
                         atol=1e-8))
check("affine: d(A, A) == 0", affine_invariant_distance(A, A).abs() < 1e-7)
check("affine: symmetry", torch.allclose(affine_invariant_distance(A, B_),
                                         affine_invariant_distance(B_, A)))

# 5. Bures on rank-deficient PSD
print("[5] PSD (rank-deficient) support")
u = torch.randn(4, 2)
P1 = u @ u.T                     # rank 2
P2 = rand_spd(4)
d = bures_wasserstein_distance(P1.unsqueeze(0), P2.unsqueeze(0))
check("bures finite on rank-deficient input", bool(torch.isfinite(d).all()))
check("bures d(P, P) == 0",
      bures_wasserstein_distance(P1.unsqueeze(0), P1.unsqueeze(0)).abs() < 1e-6)

# 6. Fréchet means
print("[6] Fréchet means")
Gb = rand_spd(3, B=4)
for metric in ["affine", "log_euclidean", "bures"]:
    M = spd_frechet_mean(Gb[:1].expand(4, 3, 3), metric=metric)
    check(f"{metric}: mean of identical == input",
          torch.allclose(M, Gb[0], atol=1e-8))
# scalar: AI mean of {a, 1/a} is 1 (geometric mean)
pair = torch.tensor([[[4.0]], [[0.25]]])
M = spd_frechet_mean(pair, metric="affine")
check("affine scalar mean of {a, 1/a} == 1", torch.allclose(M, torch.ones(1, 1)))
M = spd_frechet_mean(pair, metric="log_euclidean")
check("LE scalar mean of {a, 1/a} == 1", torch.allclose(M, torch.ones(1, 1)))
# Karcher mean minimizes sum of squared distances (compare to perturbations)
M = spd_frechet_mean(Gb, metric="affine")
f_at = lambda C: affine_invariant_distance(Gb, C.expand(4, 3, 3)).square().sum()
f0 = f_at(M)
worse = all(f_at(sym(M + 0.05 * sym(torch.randn(3, 3)))) > f0 for _ in range(5))
check("affine Karcher mean is a local minimum", worse)

# 7. Pairwise
print("[7] Pairwise distance matrices")
for metric in ["affine", "log_euclidean", "bures"]:
    Dm = pairwise_spd_distance(Gb, metric=metric)
    check(f"{metric}: (B,B), zero diag, symmetric",
          Dm.shape == (4, 4)
          and Dm.diagonal().abs().max() < 1e-6
          and torch.allclose(Dm, Dm.T, atol=1e-8))

# 8. Fixed-rank PSD (Bonnabel–Sepulchre)
print("[8] Fixed-rank PSD")
U, R, rank = psd_decompose(P1.unsqueeze(0), 2)
check("decompose: rank detected == 2", bool((rank == 2).all()))
check("reconstruction U R U^T == A",
      torch.allclose(U @ R @ U.transpose(-1, -2), P1.unsqueeze(0), atol=1e-9))
check("fixed-rank d(A, A) == 0",
      psd_fixed_rank_distance(P1.unsqueeze(0), P1.unsqueeze(0), k=2).abs() < 1e-6)
# pure rotation of the range with identical spectrum -> distance == ||theta||
lam = torch.diag(torch.tensor([3.0, 3.0]))   # isotropic spectrum on the subspace
Ua = torch.eye(4)[:, :2]
t = 0.3
c, s = math.cos(t), math.sin(t)
Rot = torch.tensor([[c, 0.0], [0.0, 1.0], [s, 0.0], [0.0, 0.0]])  # e1 -> rot in (e1,e3)
Aa = (Ua @ lam @ Ua.T).unsqueeze(0)
Bb = (Rot @ lam @ Rot.T).unsqueeze(0)
d, d_g, d_s = psd_fixed_rank_distance(Aa, Bb, k=2, return_parts=True)
check("pure range rotation: grassmann part == t", abs(float(d_g) - t) < 1e-8)
check("pure range rotation: spd part == 0", float(d_s) < 1e-8)
check("total == sqrt(t^2)", abs(float(d) - t) < 1e-8)

# 9. MetricFieldGeometry end-to-end
print("[9] MetricFieldGeometry")
model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 5)).double()
X = torch.randn(6, 3)
geo = MetricFieldGeometry(model)
Gx = geo.metrics(X)
check("metrics (B, n, n) symmetric", Gx.shape == (6, 3, 3)
      and torch.allclose(Gx, Gx.transpose(-1, -2)))
Dm = geo.distance_matrix(X, metric="affine")
check("affine distance matrix finite/symmetric",
      bool(torch.isfinite(Dm).all()) and torch.allclose(Dm, Dm.T, atol=1e-8))
M = geo.frechet_mean(X, metric="log_euclidean")
check("mean metric (n, n) SPD", M.shape == (3, 3)
      and bool((torch.linalg.eigvalsh(M) > 0).all()))
# linear model -> constant metric field -> all distances 0
lingeo = MetricFieldGeometry(nn.Linear(3, 5).double())
check("linear model: constant field, distances ~0",
      lingeo.distance_matrix(X, metric="affine").max() < 1e-6)
# degenerate case: m < n makes g rank-deficient; bures + fixed_rank still work
squeeze = nn.Linear(4, 2).double()
Xs = torch.randn(5, 4)
sgeo = MetricFieldGeometry(squeeze)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Db = sgeo.distance_matrix(Xs, metric="bures")
    Df = sgeo.distance_matrix(Xs, metric="fixed_rank")
check("degenerate metrics: bures matrix finite", bool(torch.isfinite(Db).all()))
check("degenerate metrics: fixed_rank matrix finite, zero diag",
      bool(torch.isfinite(Df).all()) and Df.diagonal().abs().max() < 1e-6)
# hidden layer
hgeo = MetricFieldGeometry(model, layer=1)
check("hidden-layer metric field works",
      hgeo.metrics(X).shape == (6, 3, 3))

print("\nAll spd_geometry.py checks passed.")
