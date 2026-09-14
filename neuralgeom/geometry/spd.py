"""
neuralgeom.geometry.spd
=======================

Geometry of the metric-tensor field: each pullback metric g_x = J_x^T J_x is
a point on the SPD manifold (or, when rank-deficient, on the fixed-rank PSD
manifold). This module compares metric tensors across points/layers/models.

(You may see this called the "SPD manifold" — symmetric positive definite;
"PSD" usually refers to the rank-deficient boundary, handled here by the
fixed-rank tools.)

Distances (all batched, broadcasting):
  * ``affine_invariant_distance`` — ||logm(A^{-1/2} B A^{-1/2})||_F, the
    canonical GL(n)-congruence-invariant geometry. SPD only.
  * ``log_euclidean_distance``    — ||logm A - logm B||_F. SPD only; less
    costly than affine-invariant, to which it is a first-order approximation.
  * ``bures_wasserstein_distance``— Wasserstein-2 between centered Gaussians;
    well-defined on all PSD matrices (rank-deficiency OK).
  * ``psd_fixed_rank_distance``   — Bonnabel–Sepulchre structure metric on
    PSD matrices of fixed rank k: Grassmannian term (range subspace) +
    affine-invariant term (k x k SPD factor).

Means:
  * ``spd_frechet_mean`` — closed-form (log-Euclidean), Karcher iteration
    (affine-invariant), fixed-point iteration (Bures–Wasserstein).

Utilities: ``spd_logm/expm/sqrtm/invsqrtm``, ``regularize``,
``pairwise_spd_distance``, ``psd_decompose``; ``ModelMetricFieldGeometry`` bundles
everything for one (model, layer): pullback metrics at sample points and
their pairwise geometry.

Example
-------
>>> geo = ModelMetricFieldGeometry(model)              # or layer="encoder.2"
>>> G = geo.metrics(X)                            # (B, n, n) pullback metrics
>>> D = geo.distance_matrix(X, metric="affine")   # (B, B)
>>> M = geo.frechet_mean(X, metric="log_euclidean")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .jacobian import ModelLike, euclidean_pullback_metric
from .grassmann import LayerLike, _resolve_fn

__all__ = [
    "sym",
    "spd_logm",
    "spd_expm",
    "spd_sqrtm",
    "spd_invsqrtm",
    "regularize",
    "affine_invariant_distance",
    "log_euclidean_distance",
    "bures_wasserstein_distance",
    "spectral_ratio_distance",
    "spd_distance",
    "pairwise_spd_distance",
    "spd_frechet_mean",
    "psd_decompose",
    "psd_fixed_rank_distance",
    "ModelMetricFieldGeometry",
]


# --------------------------------------------------------------------------- #
# Symmetric matrix functions (batched, via eigh)
# --------------------------------------------------------------------------- #
def sym(A: Tensor) -> Tensor:
    """Exact symmetrization (A + A^T)/2."""
    return 0.5 * (A + A.transpose(-1, -2))


def _eigh_clamped(A: Tensor, min_eig: float = 0.0) -> Tuple[Tensor, Tensor]:
    w, V = torch.linalg.eigh(sym(A))
    return w.clamp_min(min_eig) if min_eig is not None else w, V


def _funcm(A: Tensor, fn, *, min_eig: float = 0.0) -> Tensor:
    """V fn(w) V^T for symmetric A (eigenvalues clamped at min_eig first)."""
    w, V = _eigh_clamped(A, min_eig)
    return (V * fn(w).unsqueeze(-2)) @ V.transpose(-1, -2)


_TINY = 1e-30  # guards log/inv of clamped-to-zero eigenvalues


def spd_logm(A: Tensor) -> Tensor:
    """Matrix log of an SPD matrix. Raises eigenvalues floor to avoid -inf."""
    return _funcm(A, lambda w: w.clamp_min(_TINY).log())


def spd_expm(S: Tensor) -> Tensor:
    """Matrix exp of a symmetric matrix (result is SPD)."""
    w, V = torch.linalg.eigh(sym(S))
    return (V * w.exp().unsqueeze(-2)) @ V.transpose(-1, -2)


def spd_sqrtm(A: Tensor) -> Tensor:
    """PSD square root."""
    return _funcm(A, torch.sqrt)


def spd_invsqrtm(A: Tensor) -> Tensor:
    """A^{-1/2} for SPD A."""
    return _funcm(A, lambda w: w.clamp_min(_TINY).rsqrt())


def regularize(G: Tensor, eps: Optional[float] = None) -> Tensor:
    """G + eps * I. Default eps: n * dtype-eps * max eigenvalue (per matrix).

    Use to push rank-deficient pullback metrics into SPD before applying
    affine-invariant / log-Euclidean tools. For principled handling of
    degeneracy prefer ``bures_wasserstein_distance`` or
    ``psd_fixed_rank_distance``.
    """
    n = G.shape[-1]
    I = torch.eye(n, dtype=G.dtype, device=G.device)
    if eps is None:
        lam_max = torch.linalg.eigvalsh(sym(G))[..., -1:].clamp_min(_TINY)
        return sym(G) + (n * torch.finfo(G.dtype).eps) * lam_max.unsqueeze(-1) * I
    return sym(G) + eps * I


# --------------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------------- #
def affine_invariant_distance(A: Tensor, B: Tensor) -> Tensor:
    """d(A,B) = ||logm(A^{-1/2} B A^{-1/2})||_F  (canonical SPD geometry).

    Invariant under congruence A -> G A G^T. Requires strictly positive
    definite inputs; ``regularize`` degenerate metrics first, or use a
    PSD-compatible metric.
    """
    iSa = spd_invsqrtm(A)
    W = sym(iSa @ B @ iSa)
    w = torch.linalg.eigvalsh(W).clamp_min(_TINY)
    return w.log().square().sum(-1).sqrt()


def log_euclidean_distance(A: Tensor, B: Tensor) -> Tensor:
    """d(A,B) = ||logm A - logm B||_F."""
    return (spd_logm(A) - spd_logm(B)).flatten(-2).norm(dim=-1)


def bures_wasserstein_distance(A: Tensor, B: Tensor) -> Tensor:
    """d(A,B)^2 = tr A + tr B - 2 tr[(A^{1/2} B A^{1/2})^{1/2}].

    The Wasserstein-2 distance between N(0, A) and N(0, B). Defined for all
    PSD matrices — the right choice when pullback metrics are rank-deficient.
    """
    sA = spd_sqrtm(A)
    M = sym(sA @ B @ sA)
    tr_cross = torch.linalg.eigvalsh(M).clamp_min(0.0).sqrt().sum(-1)
    d2 = A.diagonal(dim1=-2, dim2=-1).sum(-1) \
        + B.diagonal(dim1=-2, dim2=-1).sum(-1) - 2.0 * tr_cross
    return d2.clamp_min(0.0).sqrt()


def spectral_ratio_distance(A: Tensor, B: Tensor) -> Tensor:
    """d_SR(A,B) = 1 - sqrt(lambda_min / lambda_max) in [0, 1].

    The lambdas are the generalized eigenvalues of A v = lambda B v (i.e. the
    eigenvalues of B^{-1/2} A B^{-1/2}). Introduced for metric similarity
    analysis by Cayco Gajic & Pellegrino (arXiv:2603.28764) as a bounded
    alternative to the affine-invariant distance for comparing pullback
    metrics: it is affine-invariant, satisfies the triangle inequality, and
    because it is bounded it yields a similarity 1 - d_SR in [0, 1].

    Where the affine-invariant distance uses the full L2 norm of the log
    generalized eigenvalues, the spectral ratio uses only their SPREAD, so it
    ignores overall scale differences and reports 1 (maximal distance) when
    the two matrices differ in rank.
    """
    iSb = spd_invsqrtm(B)
    w = torch.linalg.eigvalsh(sym(iSb @ A @ iSb)).clamp_min(0.0)
    lo, hi = w[..., 0], w[..., -1]
    return 1.0 - (lo / hi.clamp_min(_TINY)).clamp(0.0, 1.0).sqrt()


_SPD_METRICS = {
    "affine": affine_invariant_distance,
    "affine_invariant": affine_invariant_distance,
    "log_euclidean": log_euclidean_distance,
    "bures": bures_wasserstein_distance,
    "bures_wasserstein": bures_wasserstein_distance,
    "spectral_ratio": spectral_ratio_distance,
}


def spd_distance(A: Tensor, B: Tensor, *, metric: str = "affine") -> Tensor:
    """Dispatch to one of: affine(_invariant), log_euclidean, bures(_wasserstein)."""
    try:
        return _SPD_METRICS[metric](A, B)
    except KeyError:
        raise ValueError(
            f"Unknown metric {metric!r}. Choose from {sorted(_SPD_METRICS)}."
        ) from None


def pairwise_spd_distance(
    G: Tensor, G2: Optional[Tensor] = None, *, metric: str = "affine"
) -> Tensor:
    """(B1, B2) all-pairs distances between batches of SPD/PSD matrices."""
    Gb = G if G2 is None else G2
    return spd_distance(G.unsqueeze(1), Gb.unsqueeze(0), metric=metric)


# --------------------------------------------------------------------------- #
# Fréchet means
# --------------------------------------------------------------------------- #
def spd_frechet_mean(
    G: Tensor,
    *,
    metric: str = "affine",
    weights: Optional[Tensor] = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Tensor:
    """Fréchet (Karcher) mean of a batch G : (B, n, n). Returns (n, n).

    * log_euclidean : closed form expm(mean logm G_i).
    * affine        : Karcher iteration
                      M <- M^{1/2} expm(sum_i w_i logm(M^{-1/2} G_i M^{-1/2})) M^{1/2},
                      initialized at the log-Euclidean mean.
    * bures         : fixed-point iteration (Álvarez-Esteban et al.)
                      M <- M^{-1/2} (sum_i w_i (M^{1/2} G_i M^{1/2})^{1/2})^2 M^{-1/2}.
                      Works on PSD inputs (mean of rank-deficient matrices).
    """
    B = G.shape[0]
    w = (G.new_full((B,), 1.0 / B) if weights is None
         else weights / weights.sum()).view(B, 1, 1)

    if metric == "log_euclidean":
        return spd_expm((w * spd_logm(G)).sum(0))

    if metric in ("affine", "affine_invariant"):
        M = spd_expm((w * spd_logm(regularize(G))).sum(0))
        for _ in range(max_iter):
            iS = spd_invsqrtm(M)
            S = spd_sqrtm(M)
            T = (w * spd_logm(sym(iS @ G @ iS))).sum(0)
            M = sym(S @ spd_expm(T) @ S)
            step = T.norm()
            if step < tol:
                return M
        warnings.warn(
            f"Affine-invariant Karcher mean: no convergence in {max_iter} "
            f"iterations (last step {step:.2e}).", RuntimeWarning, stacklevel=2,
        )
        return M

    if metric in ("bures", "bures_wasserstein"):
        M = (w * G).sum(0)  # Euclidean mean as init
        for _ in range(max_iter):
            S = spd_sqrtm(M)
            iS = spd_invsqrtm(M)
            R = (w * spd_sqrtm(sym(S @ G @ S))).sum(0)
            M_new = sym(iS @ R @ R @ iS)
            step = (M_new - M).norm()
            M = M_new
            if step < tol:
                return M
        warnings.warn(
            f"Bures–Wasserstein mean: no convergence in {max_iter} iterations "
            f"(last step {step:.2e}).", RuntimeWarning, stacklevel=2,
        )
        return M

    raise ValueError(f"Unknown metric {metric!r} for the Fréchet mean.")


# --------------------------------------------------------------------------- #
# Fixed-rank PSD manifold  S+(k, n)  (Bonnabel–Sepulchre)
# --------------------------------------------------------------------------- #
def psd_decompose(
    A: Tensor, k: Optional[int] = None, *, rtol: Optional[float] = None
) -> Tuple[Tensor, Tensor, Tensor]:
    """Factor PSD A ≈ U R U^T with U (…, n, k) orthonormal (range basis, a
    point on Gr(k, n)) and R (…, k, k) SPD (the metric on that subspace).

    k=None: common numerical rank (min over batch), LAPACK-style cutoff.
    Returns (U, R, rank).
    """
    w, V = torch.linalg.eigh(sym(A))            # ascending
    w = w.clamp_min(0.0)
    n = A.shape[-1]
    if rtol is None:
        rtol = n * torch.finfo(A.dtype).eps
    rank = (w > rtol * w[..., -1:]).sum(-1)
    if k is None:
        k = int(rank.min())
        if bool((rank != rank.reshape(-1)[0]).any()):
            warnings.warn(
                f"Heterogeneous ranks (min={int(rank.min())}, "
                f"max={int(rank.max())}); using common k={k}.",
                RuntimeWarning, stacklevel=2,
            )
    if k < 1:
        raise ValueError("Zero-rank matrix encountered; cannot decompose.")
    U = V[..., -k:].flip(-1)                     # top-k eigenvectors
    lam = w[..., -k:].flip(-1)                   # top-k eigenvalues
    R = torch.diag_embed(lam)
    return U, R, rank


def psd_fixed_rank_distance(
    A: Tensor,
    B: Tensor,
    *,
    k: Optional[int] = None,
    alpha: float = 1.0,
    return_parts: bool = False,
):
    """Structure distance on the fixed-rank PSD manifold S+(k, n)
    (Bonnabel & Sepulchre, SIAM J. Matrix Anal. 2009):

        d^2(A, B) = ||Theta||_2^2 + alpha^2 * d_AI(R_A', R_B')^2

    where Theta are the principal angles between range(A) and range(B)
    (Grassmannian part), and R', after aligning the two bases by the
    Procrustes rotations from the SVD of U_A^T U_B, are the k x k SPD factors
    compared in the affine-invariant metric. ``alpha`` weighs subspace
    misalignment vs. spectral mismatch.

    This is exactly the "Grassmannian x SPD" picture of a degenerate pullback
    metric: *which* directions survive + *how strongly* they are stretched.
    Not a true geodesic distance on S+(k, n), but symmetric, O(n)-invariant,
    and the standard practical choice.

    A, B : (..., n, n) PSD, broadcast-compatible. If A or B is a tuple
    (U, R) from ``psd_decompose``, the decomposition is reused.
    """
    UA, RA, _ = A if isinstance(A, tuple) else psd_decompose(A, k)
    UB, RB, _ = B if isinstance(B, tuple) else psd_decompose(B, k)
    if UA.shape[-1] != UB.shape[-1]:
        kk = min(UA.shape[-1], UB.shape[-1])
        warnings.warn(
            f"Rank mismatch ({UA.shape[-1]} vs {UB.shape[-1]}); truncating "
            f"both to k={kk}.", RuntimeWarning, stacklevel=2,
        )
        UA, RA = UA[..., :kk], RA[..., :kk, :kk]
        UB, RB = UB[..., :kk], RB[..., :kk, :kk]

    OA, s, OBh = torch.linalg.svd(UA.transpose(-1, -2) @ UB)
    theta = torch.acos(s.clamp(0.0, 1.0))
    OB = OBh.transpose(-1, -2)
    RAp = sym(OA.transpose(-1, -2) @ RA @ OA)
    RBp = sym(OB.transpose(-1, -2) @ RB @ OB)

    d_grass = theta.square().sum(-1)
    d_spd = affine_invariant_distance(RAp, RBp).square()
    d = (d_grass + alpha**2 * d_spd).sqrt()
    if return_parts:
        return d, d_grass.sqrt(), d_spd.sqrt()
    return d


# --------------------------------------------------------------------------- #
# Convenience wrapper: the metric field of one (model, layer)
# --------------------------------------------------------------------------- #
@dataclass
class ModelMetricFieldGeometry:
    """Pullback metrics g_x of one (model, layer) as points on SPD/PSD.

    Parameters
    ----------
    model : nn.Module or callable
    layer : None | str | int | nn.Module — compute the pullback metric of the
            map input -> that layer's activation (see FeatureExtractor).
    metric : default distance ("affine", "log_euclidean", "bures",
             or "fixed_rank")
    reg : optional ridge added to g before affine/log-Euclidean ops
          (None = auto-regularize only if needed... explicit is safer)
    """

    model: ModelLike
    layer: LayerLike = None
    metric: str = "affine"
    reg: Optional[float] = None
    mode: str = "auto"
    chunk_size: Optional[int] = None

    def metrics(self, X: Tensor) -> Tensor:
        """Pullback metric tensors (B, n, n) at the given points."""
        g = euclidean_pullback_metric(
            _resolve_fn(self.model, self.layer), X,
            mode=self.mode, chunk_size=self.chunk_size,
        )
        if self.reg is not None:
            g = regularize(g, self.reg)
        return g

    def distance(self, X1: Tensor, X2: Tensor, *, metric: Optional[str] = None,
                 **kw) -> Tensor:
        """Distances between the metric tensors at paired points (B,)."""
        m = metric or self.metric
        G1, G2 = self.metrics(X1), self.metrics(X2)
        if m == "fixed_rank":
            return psd_fixed_rank_distance(G1, G2, **kw)
        return spd_distance(G1, G2, metric=m)

    def distance_matrix(self, X: Tensor, *, metric: Optional[str] = None,
                        **kw) -> Tensor:
        """(B, B) pairwise distances between g_x across the batch."""
        m = metric or self.metric
        G = self.metrics(X)
        if m == "fixed_rank":
            U, R, _ = psd_decompose(G, kw.pop("k", None))
            return psd_fixed_rank_distance(
                (U.unsqueeze(1), R.unsqueeze(1), None),
                (U.unsqueeze(0), R.unsqueeze(0), None), **kw,
            )
        return pairwise_spd_distance(G, metric=m)

    def frechet_mean(self, X: Tensor, *, metric: Optional[str] = None,
                     **kw) -> Tensor:
        """Mean metric tensor (n, n) over the batch."""
        m = metric or self.metric
        if m == "fixed_rank":
            raise NotImplementedError(
                "Fréchet mean on S+(k,n) is not implemented; use "
                "metric='bures' (PSD-safe) or regularize + 'affine'."
            )
        return spd_frechet_mean(self.metrics(X), metric=m, **kw)
