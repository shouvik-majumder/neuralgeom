"""
grassmannian.py
===============

Grassmannian geometry of neural representations.

At each input point x, the Jacobian J_x of any (sub)network f : R^n -> R^D
distinguishes two natural subspaces:

  * **column space** of J_x — the image of the local tangent map inside the
    output/hidden space R^D: where the data manifold's tangent plane sits in
    representation space. A point on Gr(k, D).
  * **row space** of J_x — the input directions the map is locally sensitive
    to (orthogonal complement of the local kernel). A point on Gr(k, n).

This module extracts those subspaces for a batch of points — for the **full
model, any hidden layer, or the input space** (via ``FeatureExtractor``) —
and provides the standard Grassmannian machinery to compare them:

  * ``tangent_subspaces``            — orthonormal bases (B, D, k) via SVD of J
  * ``principal_angles``             — angles between two subspaces
  * ``grassmann_distance``           — geodesic / projection / chordal /
                                       Binet–Cauchy / Martin / max-angle
  * ``pairwise_grassmann_distance``  — (B, B) distance matrices
  * ``grassmann_log`` / ``grassmann_exp`` — Riemannian log/exp maps
  * ``grassmann_frechet_mean``       — projection ("flag") mean and iterative
                                       Karcher mean
  * ``ModelSubspaceGeometry``            — convenience wrapper per (model, layer)

Comparability notes
-------------------
* Two subspaces are comparable only if they live in the same ambient space.
  Column spaces of *different layers* have different ambient dimensions in
  general; **row spaces always live in the input space R^n**, so row-space
  fields of different layers are directly comparable.
* Fixed ``k`` puts every point on the same Grassmannian Gr(k, D). With
  ``k=None`` the common k is chosen automatically (min numerical rank across
  the batch, or a variance-fraction threshold), with a warning if ranks vary.

Example
-------
>>> model = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 10))
>>> geo = ModelSubspaceGeometry(model, which="column", k=3)     # full model
>>> hid = ModelSubspaceGeometry(model, layer=1, which="row")    # after Tanh
>>> Q, s, rank = geo.subspaces(X)          # (B, 10, 3) orthonormal bases
>>> D = geo.distance_matrix(X)             # (B, B) geodesic distances
>>> mu = geo.frechet_mean(X)               # (10, 3) mean subspace
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from .jacobian import ModelLike, batch_jacobian

__all__ = [
    "FeatureExtractor",
    "tangent_subspaces",
    "principal_angles",
    "grassmann_distance",
    "pairwise_grassmann_distance",
    "projector",
    "grassmann_log",
    "grassmann_exp",
    "grassmann_frechet_mean",
    "ModelSubspaceGeometry",
]

LayerLike = Union[None, str, int, torch.nn.Module]


# --------------------------------------------------------------------------- #
# Layer access: view any hidden layer as the map's codomain
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    """Callable ``x -> activation at layer`` for an arbitrary ``nn.Module``.

    Turns any hidden layer into "the output space", so every function in this
    package (Jacobians, pullback metrics, subspaces) applies unchanged to
    input space, hidden layers, or the final output.

    Parameters
    ----------
    model : nn.Module
    layer : None | str | int | nn.Module
        * ``None``      — the model's own output (identity wrapper).
        * ``str``       — a name from ``model.named_modules()``,
                          e.g. ``"encoder.3"``.
        * ``int``       — index into ``model.children()`` (Sequential-style);
                          negative indices allowed.
        * ``nn.Module`` — the submodule object itself.

    The activation is captured with a forward hook, so this works for
    arbitrary architectures (not just ``nn.Sequential``) and composes with
    ``torch.func`` Jacobians (parameters are treated as constants, which is
    correct for input-space Jacobians).
    """

    def __init__(self, model: torch.nn.Module, layer: LayerLike = None):
        if not isinstance(model, torch.nn.Module):
            if layer is not None:
                raise TypeError("layer selection requires an nn.Module model")
            self.model, self.layer = model, None
            return
        model.eval()
        self.model = model
        if layer is None:
            self.layer = None
        elif isinstance(layer, str):
            named = dict(model.named_modules())
            if layer not in named:
                raise KeyError(
                    f"No submodule named {layer!r}. Available: "
                    f"{[k for k in named if k][:20]}"
                )
            self.layer = named[layer]
        elif isinstance(layer, int):
            self.layer = list(model.children())[layer]
        elif isinstance(layer, torch.nn.Module):
            self.layer = layer
        else:
            raise TypeError(f"Unsupported layer spec: {type(layer)}")

    def __call__(self, x: Tensor) -> Tensor:
        if self.layer is None:
            return self.model(x)
        captured = []
        handle = self.layer.register_forward_hook(
            lambda _mod, _inp, out: captured.append(out)
        )
        try:
            self.model(x)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Selected layer was never executed in forward().")
        return captured[-1]


def _resolve_fn(model: ModelLike, layer: LayerLike) -> ModelLike:
    return model if layer is None else FeatureExtractor(model, layer)


# --------------------------------------------------------------------------- #
# Subspace extraction
# --------------------------------------------------------------------------- #
def tangent_subspaces(
    model: ModelLike,
    X: Tensor,
    *,
    which: str = "column",
    k: Optional[int] = None,
    variance_fraction: Optional[float] = None,
    rtol: Optional[float] = None,
    layer: LayerLike = None,
    mode: str = "auto",
    chunk_size: Optional[int] = None,
    jacobian: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Orthonormal bases of the local column/row space of J_x per point.

    Parameters
    ----------
    which : {"column", "row"}
        "column" (aliases: "col", "image", "output") — image of J_x in the
        codomain, a point on Gr(k, D). "row" (aliases: "input", "domain") —
        sensitive input directions, a point on Gr(k, n).
    k : int, optional
        Subspace dimension. If given, top-k singular directions are used at
        every point (all points on the same Grassmannian). If None, k is
        chosen automatically:
          * with ``variance_fraction`` in (0, 1]: smallest k whose singular values carry
            that fraction of the total squared spectrum at every point
            (max over batch, so no point is truncated below the threshold);
          * otherwise: minimum numerical rank across the batch (cutoff
            ``rtol * s_max``, LAPACK-style default), so bases stay comparable.
        A warning is emitted if per-point ranks are heterogeneous.
    layer : see ``FeatureExtractor`` — compute at any hidden layer.
    jacobian : optional precomputed (B, m, n) Jacobians.

    Returns
    -------
    Q : Tensor, (B, D, k) — orthonormal bases (D = m for columns, n for rows)
    s : Tensor, (B, min(m, n)) — singular values of J_x (descending)
    rank : LongTensor, (B,) — numerical rank of J_x per point
    """
    J = jacobian if jacobian is not None else batch_jacobian(
        _resolve_fn(model, layer), X, mode=mode, chunk_size=chunk_size
    )
    B, m, n = J.shape
    U, s, Vh = torch.linalg.svd(J, full_matrices=False)

    if rtol is None:
        rtol = max(m, n) * torch.finfo(J.dtype).eps
    cutoff = rtol * s[:, :1]
    rank = (s > cutoff).sum(dim=1)

    which = which.lower()
    if which in ("column", "col", "image", "output"):
        basis = U                       # (B, m, min(m, n))
    elif which in ("row", "input", "domain"):
        basis = Vh.transpose(-1, -2)    # (B, n, min(m, n))
    else:
        raise ValueError(f"which must be 'column' or 'row', got {which!r}")

    if k is None:
        if variance_fraction is not None:
            if not 0.0 < variance_fraction <= 1.0:
                raise ValueError("variance_fraction must be in (0, 1]")
            frac = s.square().cumsum(dim=1) / s.square().sum(dim=1, keepdim=True)
            k_pt = (frac < variance_fraction).sum(dim=1) + 1          # per-point k
            k = int(k_pt.max())
        else:
            k = int(rank.min())
            if bool((rank != rank[0]).any()):
                warnings.warn(
                    f"Heterogeneous numerical ranks across batch "
                    f"(min={int(rank.min())}, max={int(rank.max())}); using "
                    f"common k={k}. Pass k explicitly to override.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    if not 1 <= k <= basis.shape[-1]:
        raise ValueError(f"k={k} out of range [1, {basis.shape[-1]}]")
    if bool((rank < k).any()):
        warnings.warn(
            f"{int((rank < k).sum())}/{B} points have rank(J) < k={k}; their "
            "trailing basis vectors are numerically arbitrary.",
            RuntimeWarning,
            stacklevel=2,
        )
    return basis[..., :k], s, rank


# --------------------------------------------------------------------------- #
# Grassmannian core (operates on orthonormal bases, batch-broadcasting)
# --------------------------------------------------------------------------- #
def principal_angles(Q1: Tensor, Q2: Tensor) -> Tensor:
    """Principal angles between subspaces span(Q1), span(Q2).

    Q1, Q2 : (..., D, k) with orthonormal columns (broadcast-compatible).

    Returns (..., k) angles in [0, pi/2], ascending. Invariant to the choice
    of orthonormal basis within each subspace.
    """
    if Q1.shape[-1] != Q2.shape[-1]:
        raise ValueError(
            f"Subspace dims differ: k1={Q1.shape[-1]}, k2={Q2.shape[-1]}. "
            "Principal angles between unequal-dimension subspaces are defined "
            "for the min(k1,k2) smallest angles; truncate bases explicitly."
        )
    if Q1.shape[-2] != Q2.shape[-2]:
        raise ValueError(
            f"Ambient dims differ ({Q1.shape[-2]} vs {Q2.shape[-2]}): "
            "subspaces live in different spaces and are not comparable. "
            "(Hint: row spaces of different layers share the input space.)"
        )
    cos = torch.linalg.svdvals(Q1.transpose(-1, -2) @ Q2)   # descending
    return torch.acos(cos.clamp(0.0, 1.0))                  # ascending angles


_EPS_LOG = 1e-12


def _grassmann_dist_from_angles(theta: Tensor, metric: str) -> Tensor:
    if metric == "geodesic":            # arc length on Gr(k, D)
        return theta.square().sum(-1).sqrt()
    if metric == "projection":          # (1/sqrt 2)||P1 - P2||_F
        return theta.sin().square().sum(-1).sqrt()
    if metric == "chordal":             # Procrustes: 2||sin(theta/2)||_2
        return 2.0 * (theta / 2).sin().square().sum(-1).sqrt()
    if metric == "max_angle":           # Asimov
        return theta.max(-1).values
    if metric == "binet_cauchy":
        return (1.0 - theta.cos().square().prod(-1)).clamp_min(0.0).sqrt()
    if metric == "martin":
        c2 = theta.cos().square().clamp_min(_EPS_LOG)
        return (-c2.log()).sum(-1).sqrt()
    raise ValueError(
        f"Unknown metric {metric!r}. Choose from: geodesic, projection, "
        "chordal, max_angle, binet_cauchy, martin."
    )


def grassmann_distance(Q1: Tensor, Q2: Tensor, *, metric: str = "geodesic") -> Tensor:
    """Distance between subspaces, from their principal angles theta_i.

    metric :
      * "geodesic"     — ||theta||_2, the Riemannian arc length (canonical).
      * "projection"   — ||sin theta||_2 = (1/sqrt2)||P1 - P2||_F.
      * "chordal"      — 2||sin(theta/2)||_2 (Procrustes / embedded chordal).
      * "max_angle"    — max_i theta_i (Asimov).
      * "binet_cauchy" — sqrt(1 - prod_i cos^2 theta_i).
      * "martin"       — sqrt(sum_i log(1 / cos^2 theta_i)); infinite for
                         orthogonal directions.
    """
    return _grassmann_dist_from_angles(principal_angles(Q1, Q2), metric)


def pairwise_grassmann_distance(
    Q: Tensor,
    Q2: Optional[Tensor] = None,
    *,
    metric: str = "geodesic",
) -> Tensor:
    """All-pairs distance matrix between batches of subspaces.

    Q : (B1, D, k); Q2 : (B2, D, k) or None (defaults to Q).
    Returns (B1, B2). Memory is O(B1*B2*k^2); chunk the batch if huge.
    """
    Qb = Q if Q2 is None else Q2
    return grassmann_distance(
        Q.unsqueeze(1), Qb.unsqueeze(0), metric=metric
    )


def projector(Q: Tensor) -> Tensor:
    """Orthogonal projector P = Q Q^T (basis-invariant subspace representation)."""
    return Q @ Q.transpose(-1, -2)


# --------------------------------------------------------------------------- #
# Riemannian log / exp maps (canonical metric)
# --------------------------------------------------------------------------- #
def grassmann_log(Q0: Tensor, Q1: Tensor) -> Tensor:
    """Log map: horizontal tangent Delta at span(Q0) with exp_{Q0}(Delta) = span(Q1).

    Standard formula (Edelman–Arias–Smith): with M = Q0^T Q1,
        T = (Q1 - Q0 M) M^{-1} = U S V^T  (thin SVD)
        Delta = U atan(S) V^T,   Q0^T Delta = 0,  ||Delta||-singular values
        are the principal angles.
    Defined when no principal angle equals pi/2 (M invertible).
    """
    M = Q0.transpose(-1, -2) @ Q1
    # T = (I - Q0 Q0^T) Q1 M^{-1}, computed as solve with M^T on the right
    T = torch.linalg.solve(
        M.transpose(-1, -2), (Q1 - Q0 @ M).transpose(-1, -2)
    ).transpose(-1, -2)
    U, S, Vh = torch.linalg.svd(T, full_matrices=False)
    return (U * torch.atan(S).unsqueeze(-2)) @ Vh


def grassmann_exp(Q0: Tensor, Delta: Tensor) -> Tensor:
    """Exp map along horizontal tangent Delta (thin SVD Delta = U S V^T):

        exp_{Q0}(Delta) = Q0 V cos(S) V^T + U sin(S) V^T,

    re-orthonormalized for numerical safety. Returns an orthonormal basis of
    the endpoint subspace.
    """
    U, S, Vh = torch.linalg.svd(Delta, full_matrices=False)
    V = Vh.transpose(-1, -2)
    Q = Q0 @ (V * S.cos().unsqueeze(-2)) @ Vh + (U * S.sin().unsqueeze(-2)) @ Vh
    Qn, _ = torch.linalg.qr(Q)
    return Qn


# --------------------------------------------------------------------------- #
# Fréchet means
# --------------------------------------------------------------------------- #
def grassmann_frechet_mean(
    Q: Tensor,
    *,
    method: str = "karcher",
    weights: Optional[Tensor] = None,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Tensor:
    """Fréchet mean of a batch of subspaces Q : (B, D, k). Returns (D, k).

    method :
      * "projection" — top-k eigenvectors of the average projector
        mean_i(Q_i Q_i^T): the exact Fréchet mean w.r.t. the projection
        metric. Closed form, always converges; also used to initialize
        "karcher".
      * "karcher" — Riemannian center of mass for the geodesic metric via
        exp/log iteration, initialized at the projection mean. Requires all
        principal angles to the mean < pi/2 (holds for reasonably clustered
        subspaces).
    """
    B, D, k = Q.shape
    if weights is None:
        w = Q.new_full((B,), 1.0 / B)
    else:
        w = weights / weights.sum()

    Pbar = torch.einsum("b,bij->ij", w, Q @ Q.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(Pbar)
    mu = evecs[:, -k:].flip(-1)          # top-k eigenvectors
    if method == "projection":
        return mu
    if method != "karcher":
        raise ValueError(f"Unknown method {method!r}")

    for _ in range(max_iter):
        Delta = torch.einsum("b,bij->ij", w, grassmann_log(mu.unsqueeze(0), Q))
        step = Delta.norm()
        mu = grassmann_exp(mu, Delta)
        if step < tol:
            break
    else:
        warnings.warn(
            f"Karcher mean did not converge in {max_iter} iterations "
            f"(last step {step:.2e}).",
            RuntimeWarning,
            stacklevel=2,
        )
    return mu


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #
@dataclass
class ModelSubspaceGeometry:
    """Grassmannian analysis of one (model, layer) pair.

    Parameters
    ----------
    model : nn.Module or callable
    layer : None | str | int | nn.Module — hidden layer (see FeatureExtractor)
    which : "column" (image in codomain) or "row" (input directions)
    k : subspace dimension (None = auto, see ``tangent_subspaces``)
    variance_fraction : variance-fraction threshold used when k is None
    mode, chunk_size : forwarded to ``batch_jacobian``

    Row-space geometries of *different layers* of the same model live on the
    same Grassmannian Gr(k, n_input) and can be compared directly, e.g.::

        g1 = ModelSubspaceGeometry(model, layer=1, which="row", k=2)
        g2 = ModelSubspaceGeometry(model, layer=3, which="row", k=2)
        d = grassmann_distance(g1.subspaces(X)[0], g2.subspaces(X)[0])
    """

    model: ModelLike
    layer: LayerLike = None
    which: str = "column"
    k: Optional[int] = None
    variance_fraction: Optional[float] = None
    mode: str = "auto"
    chunk_size: Optional[int] = None

    def jacobian(self, X: Tensor) -> Tensor:
        return batch_jacobian(
            _resolve_fn(self.model, self.layer), X,
            mode=self.mode, chunk_size=self.chunk_size,
        )

    def subspaces(self, X: Tensor, *, jacobian: Optional[Tensor] = None
                  ) -> Tuple[Tensor, Tensor, Tensor]:
        """(Q, singular_values, ranks) — see ``tangent_subspaces``."""
        out = tangent_subspaces(
            self.model, X, which=self.which, k=self.k, variance_fraction=self.variance_fraction,
            layer=self.layer, mode=self.mode, chunk_size=self.chunk_size,
            jacobian=jacobian,
        )
        self._ambient_dim = out[0].shape[-2]
        return out

    def distance(self, X1: Tensor, X2: Tensor, *, metric: str = "geodesic") -> Tensor:
        """Elementwise distances between subspaces at paired points (B,)."""
        Q1, _, _ = self.subspaces(X1)
        Q2, _, _ = self.subspaces(X2)
        kk = min(Q1.shape[-1], Q2.shape[-1])
        return grassmann_distance(Q1[..., :kk], Q2[..., :kk], metric=metric)

    def distance_matrix(self, X: Tensor, *, metric: str = "geodesic") -> Tensor:
        """(B, B) pairwise subspace distances across the batch."""
        Q, _, _ = self.subspaces(X)
        return pairwise_grassmann_distance(Q, metric=metric)

    def frechet_mean(self, X: Tensor, *, method: str = "karcher", **kw) -> Tensor:
        """Mean subspace (D, k) of the field over the batch."""
        Q, _, _ = self.subspaces(X)
        return grassmann_frechet_mean(Q, method=method, **kw)

    # ---- geomstats interoperability ------------------------------------- #
    def as_geomstats_grassmannian(self):
        """The matching ``geomstats`` Grassmannian(D, k) manifold (points are
        projectors P = QQ^T; use ``projector()`` to convert bases)."""
        try:
            from geomstats.geometry.grassmannian import Grassmannian
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "geomstats is required: pip install geomstats "
                "(and set GEOMSTATS_BACKEND=pytorch)"
            ) from e
        if self.k is None:
            raise ValueError("Set an explicit k to build a geomstats manifold.")
        dim = getattr(self, "_ambient_dim", None)
        if dim is None:
            raise ValueError(
                "Ambient dimension unknown — call subspaces(X) once first, or "
                "construct geomstats Grassmannian(D, k) directly."
            )
        return Grassmannian(dim, self.k)


# =========================================================================== #
#  NumPy / geomstats frame-based Grassmannian API                             #
# =========================================================================== #
#
# The torch API above is the *model-facing* layer: it takes an ``nn.Module``
# (or callable) and reads subspaces out of batched Jacobians. The NumPy layer
# below is the *array-facing / trajectory* layer used by ``neuralgeom.subspace``
# and ``neuralgeom.topology``: points on Gr(k, N) are handled either as
# orthonormal ``frames`` (N, k) or as N×N orthogonal projectors ``P = U Uᵀ``,
# and the closed-form geodesic geometry is delegated to ``geomstats`` (with a
# fast principal-angle shortcut for plain distances). Keeping both layers in
# one module is deliberate: they share the same mathematics (principal angles
# on Gr(k, N)); they differ only in whether the input is a torch model or a
# NumPy array of frames.
#
# Naming convention to avoid clashing with the torch API:
#   * torch (bases Q, batched):  ``principal_angles``, ``grassmann_distance`` …
#   * numpy (frames U / projectors P):  ``frame_principal_angles``,
#     ``frame_distance``, ``frame_distance_matrix``, ``GrassmannManifold`` …
#
# Metric conventions (fixed; downstream results depend on them):
#   For two k-frames with principal angles {θ_i}:
#       d_arc        = sqrt(Σ θ_i²)          (principal-angle / arc-length)
#       d_sqrt2      = sqrt(2) · d_arc       (the geomstats GrassmannianCanonicalMetric convention)
#   Both are valid Riemannian distances differing only by the global factor
#   √2, so persistent homology and curve *shapes* are invariant to the choice.

import numpy as _np

_SQRT2 = float(_np.sqrt(2.0))

__all__ += [
    "frame_to_projector",
    "projector_to_frame",
    "orthonormalize",
    "frame_principal_angles",
    "frame_distance",
    "frame_distance_matrix",
    "GrassmannManifold",
    "check_frame_distance_against_geomstats",
]


def frame_to_projector(U: "_np.ndarray") -> "_np.ndarray":
    """Orthonormal frame ``U`` (N, k) → orthogonal projector ``P = U Uᵀ`` (N, N).

    The projector is the basis-*invariant* representation of a point on
    Gr(k, N): ``P = Pᵀ``, ``P² = P``, ``tr P = k``. geomstats represents
    Grassmannian points as projectors, so this is the bridge to it.
    """
    U = _np.asarray(U, float)
    return U @ U.T


def projector_to_frame(P: "_np.ndarray", k: int) -> "_np.ndarray":
    """Projector ``P`` (N, N) → an orthonormal frame (N, k) via its top-k
    eigenvectors (the eigenvectors with eigenvalue ≈ 1)."""
    _w, V = _np.linalg.eigh(_np.asarray(P, float))
    return V[:, -k:]


def orthonormalize(A: "_np.ndarray") -> "_np.ndarray":
    """QR-orthonormalize the columns of ``A`` (N, k) → an orthonormal frame."""
    A = _np.asarray(A, float)
    Q, _ = _np.linalg.qr(A)
    return Q[:, : A.shape[1]]


def frame_principal_angles(A: "_np.ndarray", B: "_np.ndarray") -> "_np.ndarray":
    """Principal angles (ascending, in radians) between the subspaces spanned by
    two orthonormal frames ``A`` (N, k) and ``B`` (N, k).

    Computed from the singular values of ``Aᵀ B`` (the cosines of the angles).
    This is the NumPy analogue of :func:`principal_angles` (which acts on torch
    bases); both are invariant to the choice of basis within each subspace.
    """
    A = _np.asarray(A, float)
    B = _np.asarray(B, float)
    s = _np.linalg.svd(A.T @ B, compute_uv=False)
    s = _np.clip(s, -1.0, 1.0)
    return _np.arccos(s)


def frame_distance(A: "_np.ndarray", B: "_np.ndarray",
                   metric: str = "sqrt2_principal_angle") -> float:
    """Geodesic distance between two subspaces given as frames.

    ``metric="sqrt2_principal_angle"`` matches the geomstats canonical metric
    (``√2 × arc-length``); ``metric="principal_angle"`` returns the plain
    arc-length ``sqrt(Σ θ_i²)``. The fast path here uses only the k×k SVD of
    ``Aᵀ B`` (cost ``O(N k²)``) and is validated equal to geomstats'
    ``metric.dist`` to ~1e-15 by :func:`check_frame_distance_against_geomstats`.
    """
    theta = frame_principal_angles(A, B)
    d = float(_np.sqrt(_np.sum(theta ** 2)))
    return _SQRT2 * d if metric == "sqrt2_principal_angle" else d


def frame_distance_matrix(frames: "_np.ndarray",
                          metric: str = "sqrt2_principal_angle") -> "_np.ndarray":
    """Pairwise geodesic distance matrix (M, M) for a stack of frames ``(M, N, k)``.

    Uses the fast principal-angle path, so this is ``O(M² · N k²)`` and returns
    a symmetric matrix suitable for MDS, persistent homology (see
    :mod:`neuralgeom.topology`), or clustering.
    """
    frames = _np.asarray(frames, float)
    M = len(frames)
    D = _np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            d = frame_distance(frames[i], frames[j], metric)
            D[i, j] = D[j, i] = d
    return D


class GrassmannManifold:
    """Closed-form Grassmannian geometry on Gr(N, k), backed by geomstats.

    Points are handled as N×N orthogonal projectors (use
    :func:`frame_to_projector` to convert an orthonormal frame). This wrapper
    exposes exactly the operations the trajectory analyses in
    :mod:`neuralgeom.subspace` need — ``dist``/``log``/``exp``/``norm``/
    ``inner_product``/``parallel_transport``/``belongs`` — which is also the
    :ref:`Manifold interface` those analyses are written against, so the same
    kinematics/tangent-PCA code runs on any manifold implementing it (e.g. the
    SPD manifold; see :mod:`neuralgeom.geometry.spd`).

    ``geomstats`` is an optional dependency (install the ``[geom]`` extra); it
    is imported lazily on construction, with a clear error if it is missing.
    """

    def __init__(self, N: int, k: int):
        try:
            from geomstats.geometry.grassmannian import Grassmannian
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "GrassmannManifold needs the optional 'geom' extra: "
                "pip install 'neuralgeom[geom]' (geomstats)."
            ) from e
        self.N = int(N)
        self.k = int(k)
        self.space = Grassmannian(N, k)
        self.metric = self.space.metric

    # -- conversions -------------------------------------------------------- #
    def frame_projector(self, U):
        """Frame (N, k) → projector (N, N)."""
        return frame_to_projector(U)

    def frames_to_projectors(self, frames):
        """Stack of frames (M, N, k) → stack of projectors (M, N, N)."""
        return _np.stack([frame_to_projector(U) for U in frames])

    # -- geometry (projector arguments) ------------------------------------ #
    def dist(self, P, Q):
        """Canonical geodesic distance between two projectors."""
        return float(self.metric.dist(P, Q))

    def log(self, P, base):
        """Tangent vector at ``base`` pointing toward ``P`` (geomstats order:
        ``log(point, base_point)``)."""
        return self.metric.log(P, base)

    def exp(self, tangent, base):
        """Riemannian exponential of ``tangent`` at ``base``."""
        return self.metric.exp(tangent, base)

    def norm(self, tangent, base):
        """Riemannian norm of a tangent vector at ``base``."""
        return float(self.metric.norm(tangent, base))

    def inner_product(self, u, v, base):
        """Riemannian inner product of two tangents at ``base``."""
        return float(self.metric.inner_product(u, v, base))

    def parallel_transport(self, v, base, end_point):
        """Parallel-transport tangent ``v`` from ``base`` to ``end_point``."""
        return self.metric.parallel_transport(v, base, end_point=end_point)

    def belongs(self, P):
        """Whether ``P`` is a valid Gr(N, k) projector."""
        return bool(self.space.belongs(P))


def check_frame_distance_against_geomstats(N=None, k=None, n_pairs=8, seed=0,
                                    frames=None, tol=1e-8):
    """Assert the fast principal-angle distance equals geomstats' canonical
    ``metric.dist`` on random (or supplied) frame pairs; return the max error.

    This is the regression guard for the ``√2 × arc-length`` convention: if the
    fast path in :func:`frame_distance` ever drifts from geomstats, this fails.
    """
    rng = _np.random.default_rng(seed)
    if frames is None:
        A = [orthonormalize(rng.standard_normal((N, k))) for _ in range(n_pairs)]
        B = [orthonormalize(rng.standard_normal((N, k))) for _ in range(n_pairs)]
    else:
        idx = rng.integers(0, len(frames), size=(n_pairs, 2))
        A = [frames[i] for i, _ in idx]
        B = [frames[j] for _, j in idx]
        N, k = frames[0].shape
    man = GrassmannManifold(N, k)
    max_err = 0.0
    for a, b in zip(A, B):
        d_fast = frame_distance(a, b, "sqrt2_principal_angle")
        d_gs = man.dist(frame_to_projector(a), frame_to_projector(b))
        max_err = max(max_err, abs(d_fast - d_gs))
    assert max_err < tol, f"fast vs geomstats mismatch: {max_err:.2e}"
    return max_err
