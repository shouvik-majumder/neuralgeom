"""
neuro.geometry — pullback geometry of a map from a LOW-dimensional domain
into neural state space.
=========================================================================

WHY THE DIRECTION MATTERS
-------------------------
For a smooth map f : M -> R^n, the pullback metric is g = J^T M_cod J, and

        rank(g) = dim(M)     (when f is an immersion)

Decoding behaviour puts a 1-D quantity in the CODOMAIN, giving rank-1 g:
one direction, one number, no volume, no curvature. Both reference works
(Zavatone-Veth et al. 2023; Cayco Gajic & Pellegrino 2026) go the other way —
a low-dimensional DOMAIN mapped into a high-dimensional representation — which
is what makes volume element, anisotropy and curvature meaningful.

This module implements that construction for arbitrary domain dimension.

THE CODOMAIN METRIC
-------------------
g = J^T M J where M is an inner product on neural state space.

  M = I            Euclidean. Treats every unit as equally important, so a
                   50 Hz unit dominates a 2 Hz unit purely by scale.
  M = Sigma^-1     Noise-weighted (Mahalanobis). Sigma is the trial-to-trial
                   noise covariance. Distances become "how many noise standard
                   deviations apart", i.e. DISCRIMINABILITY (d-prime) units,
                   which is what a downstream reader of the population could
                   actually resolve. For Gaussian noise this IS the Fisher
                   information metric of the population code.

Implementation trick: choosing M is equivalent to whitening the codomain,
because J^T M J = (W J)^T (W J) with W = M^(1/2). So every routine below just
takes an already-whitened map and uses Euclidean formulas.

QUANTITIES (all defined before use)
-----------------------------------
* metric g (d x d)       inner product induced on the domain. g_ab tells you
                         the neural-space length of a step in domain
                         coordinate a and b.
* volume element         sqrt(det g). The factor by which the map magnifies
                         d-dimensional volume. Large = a small change in the
                         condition produces a large change in population
                         state, i.e. locally high resolution.
* anisotropy             ratio of largest to smallest eigenvalue of g. 1 =
                         the map stretches all domain directions equally;
                         large = it is sensitive to some combinations and
                         nearly blind to others.
* Gaussian curvature K   (2-D domains only) the intrinsic curvature of the
                         image surface. Computed with the Gauss equation
                         K = (<h_uu, h_vv> - |h_uv|^2) / det g, where h_ab is
                         the component of the second derivative d^2f/da db
                         ORTHOGONAL to the tangent plane. K distinguishes a
                         genuinely warped sheet (K != 0) from a flat sheet
                         that is merely stretched unevenly (K = 0) — the
                         latter has a non-constant volume element but zero
                         curvature, so the two observables are independent.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .jacobian import batch_jacobian                    # noqa: E402

__all__ = ["noise_whitener", "whiten_map", "pullback_metric_field",
           "volume_element", "anisotropy", "gaussian_curvature_2d",
           "metric_summary"]


# --------------------------------------------------------------------------- #
def noise_whitener(residuals: np.ndarray, alpha: float = 0.1
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """W = Sigma^(-1/2) from trial-to-trial residuals, with shrinkage.

    residuals : (samples, n_units) deviations from the condition mean, i.e.
        the NOISE, not the signal. Pass X - (condition mean of X).
    alpha : shrinkage toward a scaled identity; see spd_geometry.regularize.
        Needed because Sigma is rank-deficient whenever samples < units, and
        its inverse would otherwise be unbounded.

    Returns (W, Sigma). Use W to whiten the codomain: f_white = W @ f.
    """
    S = np.cov(residuals.T)
    S = (1 - alpha) * S + alpha * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-10, None)
    W = V @ np.diag(w ** -0.5) @ V.T
    return W, S


def whiten_map(f: Callable[[Tensor], Tensor], W: Optional[np.ndarray]
               ) -> Callable[[Tensor], Tensor]:
    """Compose a map with a codomain whitener. W=None returns f unchanged."""
    if W is None:
        return f
    Wt = torch.as_tensor(W)

    def fw(x: Tensor) -> Tensor:
        return f(x) @ Wt.T
    return fw


# --------------------------------------------------------------------------- #
def pullback_metric_field(f: Callable[[Tensor], Tensor], X: Tensor,
                          chunk_size: Optional[int] = None) -> Tensor:
    """g(x) = J^T J at every point of X. X: (B, d) -> returns (B, d, d)."""
    J = batch_jacobian(f, X, chunk_size=chunk_size)        # (B, n, d)
    g = torch.einsum("bni,bnj->bij", J, J)
    return 0.5 * (g + g.transpose(-1, -2))


def volume_element(g: Tensor) -> Tensor:
    """sqrt(det g) — the local volume magnification factor. (B,) """
    return torch.linalg.eigvalsh(g).clamp_min(0).sqrt().prod(-1)


def anisotropy(g: Tensor) -> Tensor:
    """Largest / smallest eigenvalue of g. (B,)"""
    w = torch.linalg.eigvalsh(g).clamp_min(1e-300)
    return w[..., -1] / w[..., 0]


# --------------------------------------------------------------------------- #
def gaussian_curvature_2d(f: Callable[[Tensor], Tensor], X: Tensor
                          ) -> Tuple[Tensor, Tensor]:
    """Gaussian curvature of the image surface, for a 2-D domain.

    Uses the Gauss equation for a surface immersed in R^n:

        K = ( <h_uu, h_vv> - <h_uv, h_uv> ) / det(g)

    where h_ab is the NORMAL part of d^2 f / da db, i.e. the second derivative
    with its tangential component removed. K > 0 is locally sphere-like,
    K < 0 saddle-like, K = 0 developable (a sheet that can be flattened
    without stretching, even if the volume element varies).

    X : (B, 2). Returns (K, det_g), each (B,).
    """
    B = X.shape[0]
    Xr = X.detach().clone().requires_grad_(False)

    def single(x):
        return f(x.unsqueeze(0)).reshape(-1)                # (n,)

    from torch.func import hessian, jacrev, vmap
    J = vmap(jacrev(single))(Xr)                            # (B, n, 2)
    H = vmap(hessian(single))(Xr)                           # (B, n, 2, 2)

    g = torch.einsum("bni,bnj->bij", J, J)
    det_g = torch.linalg.det(g)
    ginv = torch.linalg.pinv(g)

    # normal component of each second derivative: h_ab = f_ab - tangential part
    def normal_part(fab):                                   # (B, n)
        coef = torch.einsum("bij,bnj,bn->bi", ginv, J, fab)  # (B, 2)
        return fab - torch.einsum("bni,bi->bn", J, coef)

    h_uu = normal_part(H[:, :, 0, 0])
    h_uv = normal_part(H[:, :, 0, 1])
    h_vv = normal_part(H[:, :, 1, 1])
    num = (h_uu * h_vv).sum(-1) - (h_uv * h_uv).sum(-1)
    return num / det_g.clamp_min(1e-300), det_g


# --------------------------------------------------------------------------- #
def metric_summary(g: Tensor) -> dict:
    """Scalar descriptors of a field of metrics, for reporting/nulls."""
    vol = volume_element(g)
    ani = anisotropy(g)
    return dict(
        vol_med=float(vol.median()),
        vol_spread=float(torch.quantile(vol, 0.95)
                         / torch.quantile(vol, 0.05).clamp_min(1e-30)),
        ani_med=float(ani.median()),
        ani_max=float(ani.max()),
        vol=vol.detach().numpy(), ani=ani.detach().numpy(),
    )
