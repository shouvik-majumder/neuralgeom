"""
pullback_metric.py
==================

Riemannian geometry of neural representations via the pullback metric.

Given a (trained) network  f : R^n -> R^m  viewed as a smooth immersion of the
input space into representation space, the Euclidean metric on R^m pulls back
to a (possibly degenerate) metric on R^n:

    g_x = J_x^T J_x,      J_x = df/dx  in R^{m x n}.

This module provides:

  * ``batch_jacobian``       — per-sample Jacobians J_x for a batch, via
                               ``torch.func`` (vmap + jacrev/jacfwd).
  * ``pullback_metric``      — g_x = J_x^T J_x for each point in the batch.
  * ``volume_element``       — sqrt(det g_x), with a pseudo-determinant
                               (product of nonzero singular values) fallback
                               for rank-deficient metrics (e.g. m < n or
                               collapsed representations).
  * ``log_volume_element``   — numerically stable log sqrt(det g_x) via SVD
                               of J (avoids forming/decomposing g explicitly,
                               and avoids overflow of det for large n).
  * ``metric_spectrum``      — eigenvalues of g_x (= squared singular values
                               of J_x): local expansion factors along
                               principal input directions.
  * ``PullbackGeometry``     — thin convenience wrapper; optionally exposes
                               the metric as a ``geomstats`` RiemannianMetric
                               for downstream geodesic/curvature machinery.

Conventions
-----------
* Model maps (B, n) -> (B, m); non-flat shapes are flattened internally.
* All functions are pure w.r.t. model parameters (uses ``torch.func``
  functional calls), so BatchNorm should be in eval mode.
* Jacobians of stochastic layers (dropout) are only meaningful in eval mode;
  we set ``model.eval()`` defensively where a Module is passed.

Example
-------
>>> model = torch.nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 10))
>>> X = torch.randn(128, 2)
>>> geom = PullbackGeometry(model)
>>> g = geom.metric(X)                 # (128, 2, 2)
>>> vol = geom.volume_element(X)       # (128,)  sqrt(det g)
>>> logvol = geom.log_volume_element(X)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

# Jacobian backend, in order of preference:
#   "torch.func" (torch >= 2.0) > "functorch" (torch 1.12/1.13) > None (loop)
try:
    from torch.func import functional_call, jacfwd, jacrev, vmap
    _JAC_BACKEND = "torch.func"
except ImportError:
    try:
        from functorch import jacfwd, jacrev, vmap  # type: ignore[no-redef]
        functional_call = None  # type: ignore[assignment]
        _JAC_BACKEND = "functorch"
    except ImportError:  # pragma: no cover — fall back to autograd loop
        _JAC_BACKEND = None

__all__ = [
    "batch_jacobian",
    "pullback_metric",
    "volume_element",
    "log_volume_element",
    "metric_spectrum",
    "PullbackGeometry",
]

ModelLike = Union[torch.nn.Module, Callable[[Tensor], Tensor]]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _as_single_input_fn(model: ModelLike) -> Callable[[Tensor], Tensor]:
    """Wrap ``model`` into a function acting on ONE unbatched input x in R^n.

    ``torch.func.vmap`` maps this single-sample function over the batch dim.
    Most nn.Modules expect a batch dimension, so we unsqueeze/squeeze.
    Outputs of arbitrary shape are flattened to R^m.
    """
    if isinstance(model, torch.nn.Module):
        model.eval()  # freeze dropout / batchnorm statistics
        if _JAC_BACKEND == "torch.func":
            params = dict(model.named_parameters())
            buffers = dict(model.named_buffers())

            def fn(x: Tensor) -> Tensor:
                out = functional_call(model, (params, buffers), (x.unsqueeze(0),))
                return out.reshape(-1)  # (m,)
        else:  # functorch / loop fallback: direct stateless forward in eval mode

            def fn(x: Tensor) -> Tensor:
                return model(x.unsqueeze(0)).reshape(-1)

        return fn

    def fn(x: Tensor) -> Tensor:
        out = model(x.unsqueeze(0))
        return out.reshape(-1)

    return fn


def _flatten_batch(X: Tensor) -> Tuple[Tensor, int]:
    """(B, *input_shape) -> (B, n); returns flattened X and n."""
    if X.dim() < 2:
        raise ValueError(f"Expected batched input (B, ...), got shape {tuple(X.shape)}")
    B = X.shape[0]
    return X.reshape(B, -1), X[0].numel()


# --------------------------------------------------------------------------- #
# 1–2. Batched Jacobian
# --------------------------------------------------------------------------- #
def batch_jacobian(
    model: ModelLike,
    X: Tensor,
    *,
    mode: str = "auto",
    chunk_size: Optional[int] = None,
) -> Tensor:
    """Per-sample Jacobians  J_x = df/dx  for a batch.

    Parameters
    ----------
    model : nn.Module or callable
        Map f : (B, *in_shape) -> (B, *out_shape). Output is flattened to R^m.
    X : Tensor, shape (B, *in_shape)
        Input batch. Flattened internally to (B, n); returned Jacobians are
        w.r.t. the flattened input.
    mode : {"auto", "rev", "fwd"}
        Autodiff mode. "rev" (jacrev) costs O(m) VJPs — cheap when the output
        dim m is small (e.g. classifiers). "fwd" (jacfwd) costs O(n) JVPs —
        cheap when the input dim n is small. "auto" picks by comparing m, n.
    chunk_size : int, optional
        Passed to ``vmap`` to bound peak memory on large batches.

    Returns
    -------
    Tensor, shape (B, m, n)
    """
    if _JAC_BACKEND is None:
        return _batch_jacobian_fallback(model, X)

    Xf, n = _flatten_batch(X)
    # reshape flattened rows back to the model's expected per-sample shape
    in_shape = X.shape[1:]
    fn_flat_in = _as_single_input_fn(model)

    def fn(x_flat: Tensor) -> Tensor:
        return fn_flat_in(x_flat.reshape(in_shape))

    if mode == "auto":
        with torch.no_grad():
            m = fn(Xf[0]).numel()
        mode = "rev" if m <= n else "fwd"

    jac_fn = jacrev(fn) if mode == "rev" else jacfwd(fn)
    if _JAC_BACKEND == "torch.func":
        J = vmap(jac_fn, chunk_size=chunk_size)(Xf)  # (B, m, n)
    else:  # functorch has no chunk_size kwarg
        if chunk_size is not None:
            warnings.warn("chunk_size ignored (functorch backend).", RuntimeWarning)
        J = vmap(jac_fn)(Xf)
    return J


def _batch_jacobian_fallback(model: ModelLike, X: Tensor) -> Tensor:
    """Loop-based Jacobian via autograd.functional for torch < 2.0."""
    from torch.autograd.functional import jacobian as _jac

    fn = _as_single_input_fn(model)
    Xf, _ = _flatten_batch(X)
    in_shape = X.shape[1:]
    rows = [
        _jac(lambda x: fn(x.reshape(in_shape)), xi, vectorize=True)
        for xi in Xf
    ]
    return torch.stack(rows, dim=0)


# --------------------------------------------------------------------------- #
# 3. Pullback metric tensor
# --------------------------------------------------------------------------- #
def pullback_metric(
    model: ModelLike,
    X: Tensor,
    *,
    mode: str = "auto",
    chunk_size: Optional[int] = None,
    return_jacobian: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Pullback metric  g_x = J_x^T J_x  at each point of the batch.

    g_x is symmetric positive semi-definite; positive definite iff J_x has
    full column rank (requires m >= n and f immersive at x).

    Returns
    -------
    g : Tensor, shape (B, n, n)
    J : Tensor, shape (B, m, n)   (only if ``return_jacobian=True``)
    """
    J = batch_jacobian(model, X, mode=mode, chunk_size=chunk_size)
    g = torch.einsum("bki,bkj->bij", J, J)  # J^T J, batched
    g = 0.5 * (g + g.transpose(-1, -2))     # enforce exact symmetry
    return (g, J) if return_jacobian else g


# --------------------------------------------------------------------------- #
# 4. Volume element & spectra
# --------------------------------------------------------------------------- #
def log_volume_element(
    model: ModelLike,
    X: Tensor,
    *,
    mode: str = "auto",
    chunk_size: Optional[int] = None,
    rtol: Optional[float] = None,
    degenerate: str = "pseudo",
    jacobian: Optional[Tensor] = None,
) -> Tensor:
    """log sqrt(det g_x) — the local log volume-expansion factor of f.

    Computed from the singular values s_i of J_x (eigenvalues of g_x are
    s_i^2), which is far more stable than det/logdet of g:

        log sqrt(det g_x) = sum_i log s_i .

    Degenerate (rank-deficient) handling, controlled by ``degenerate``:
      * "pseudo" : pseudo-determinant — sum over singular values above the
                   rank cutoff only. Volume element of the metric restricted
                   to the non-collapsed subspace (rank reported separately by
                   ``metric_spectrum``). This matches the Moore–Penrose /
                   pseudo-inverse convention.
      * "neginf" : return -inf where rank(J) < n (true det is 0).
      * "strict" : raise if any point is rank-deficient.

    Parameters
    ----------
    rtol : float, optional
        Relative rank cutoff: singular values below rtol * s_max are treated
        as zero. Default: max(m, n) * eps of the dtype (NumPy/LAPACK style).
    jacobian : Tensor, optional
        Precomputed (B, m, n) Jacobians to reuse (skips autodiff).

    Returns
    -------
    Tensor, shape (B,)
    """
    J = jacobian if jacobian is not None else batch_jacobian(
        model, X, mode=mode, chunk_size=chunk_size
    )
    B, m, n = J.shape
    s = torch.linalg.svdvals(J)  # (B, min(m, n)), descending

    if rtol is None:
        rtol = max(m, n) * torch.finfo(J.dtype).eps
    cutoff = rtol * s[:, :1]  # (B, 1), relative to largest sv per sample
    nonzero = s > cutoff
    rank = nonzero.sum(dim=1)
    deficient = rank < n  # note: automatic when m < n

    if degenerate == "strict" and bool(deficient.any()):
        raise ValueError(
            f"Rank-deficient Jacobian at {int(deficient.sum())}/{B} points "
            f"(n={n}, min rank={int(rank.min())}). Use degenerate='pseudo' "
            "for the pseudo-determinant volume on the non-collapsed subspace."
        )

    # sum of log s_i over retained singular values (masked)
    log_s = torch.where(nonzero, s, torch.ones_like(s)).log()
    logvol = log_s.sum(dim=1)

    if degenerate == "neginf":
        logvol = torch.where(deficient, torch.full_like(logvol, -math.inf), logvol)
    elif degenerate == "pseudo":
        if bool(deficient.any()):
            warnings.warn(
                f"{int(deficient.sum())}/{B} points have rank(J) < n; "
                "returning pseudo-determinant volume (nonzero spectrum only).",
                RuntimeWarning,
                stacklevel=2,
            )
    elif degenerate != "strict":
        raise ValueError(f"Unknown degenerate policy: {degenerate!r}")

    return logvol


def volume_element(
    model: ModelLike,
    X: Tensor,
    *,
    mode: str = "auto",
    chunk_size: Optional[int] = None,
    rtol: Optional[float] = None,
    degenerate: str = "pseudo",
    jacobian: Optional[Tensor] = None,
) -> Tensor:
    """sqrt(det g_x) (or pseudo-det variant). See ``log_volume_element``.

    Prefer the log version for high-dimensional inputs: exp() here can
    overflow/underflow even when the log-volume is perfectly finite.
    """
    return log_volume_element(
        model, X, mode=mode, chunk_size=chunk_size, rtol=rtol,
        degenerate=degenerate, jacobian=jacobian,
    ).exp()


def metric_spectrum(
    model: ModelLike,
    X: Tensor,
    *,
    mode: str = "auto",
    chunk_size: Optional[int] = None,
    jacobian: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Eigenvalues of g_x and numerical rank of J_x per point.

    Eigenvalues (squared singular values of J, descending, zero-padded to n)
    are the squared local stretch factors along principal input directions;
    their heterogeneity measures anisotropy of the learned representation.

    Returns
    -------
    eigvals : Tensor, shape (B, n)
    rank    : LongTensor, shape (B,)
    """
    J = jacobian if jacobian is not None else batch_jacobian(
        model, X, mode=mode, chunk_size=chunk_size
    )
    B, m, n = J.shape
    s = torch.linalg.svdvals(J)
    eig = torch.zeros(B, n, dtype=J.dtype, device=J.device)
    eig[:, : s.shape[1]] = s.square()
    cutoff = max(m, n) * torch.finfo(J.dtype).eps * s[:, :1]
    rank = (s > cutoff).sum(dim=1)
    return eig, rank


# --------------------------------------------------------------------------- #
# Convenience wrapper (+ optional geomstats bridge)
# --------------------------------------------------------------------------- #
@dataclass
class PullbackGeometry:
    """Bundles the pullback-geometry operations for one model.

    Parameters
    ----------
    model : nn.Module or callable
    mode : Jacobian autodiff mode ("auto" | "rev" | "fwd")
    chunk_size : vmap chunk size (memory control)
    """

    model: ModelLike
    mode: str = "auto"
    chunk_size: Optional[int] = None

    def jacobian(self, X: Tensor) -> Tensor:
        return batch_jacobian(self.model, X, mode=self.mode, chunk_size=self.chunk_size)

    def metric(self, X: Tensor) -> Tensor:
        return pullback_metric(self.model, X, mode=self.mode, chunk_size=self.chunk_size)

    def volume_element(self, X: Tensor, **kw) -> Tensor:
        return volume_element(self.model, X, mode=self.mode, chunk_size=self.chunk_size, **kw)

    def log_volume_element(self, X: Tensor, **kw) -> Tensor:
        return log_volume_element(self.model, X, mode=self.mode, chunk_size=self.chunk_size, **kw)

    def spectrum(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        return metric_spectrum(self.model, X, mode=self.mode, chunk_size=self.chunk_size)

    # ---- geomstats interoperability ------------------------------------- #
    def as_geomstats_metric(self, dim: int):
        """Return a ``geomstats`` RiemannianMetric backed by this pullback
        metric, enabling geomstats' geodesic / exp / log / curvature solvers.

        Requires ``geomstats`` with the pytorch backend
        (``GEOMSTATS_BACKEND=pytorch``). Builds a minimal RiemannianMetric
        subclass whose ``metric_matrix`` is this module's pullback metric.
        """
        try:
            import geomstats.backend as gs  # noqa: F401
            from geomstats.geometry.euclidean import Euclidean
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "geomstats is required: pip install geomstats "
                "(and set GEOMSTATS_BACKEND=pytorch)"
            ) from e

        from geomstats.geometry.riemannian_metric import RiemannianMetric

        parent = self

        class _NeuralPullbackMetric(RiemannianMetric):
            def __init__(self, space):
                super().__init__(space=space)

            def metric_matrix(self, base_point):
                bp = torch.as_tensor(base_point, dtype=torch.get_default_dtype())
                squeeze = bp.dim() == 1
                if squeeze:
                    bp = bp.unsqueeze(0)
                g = parent.metric(bp)
                return g[0] if squeeze else g

        space = Euclidean(dim=dim)
        space.metric = _NeuralPullbackMetric(space)
        return space.metric
