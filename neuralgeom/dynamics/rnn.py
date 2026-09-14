"""
neuralgeom.dynamics.rnn — geometry of recurrent dynamics.
=========================================================

The feedforward tools in :mod:`neuralgeom.geometry` pull back a metric through
a map x -> f(x). For an RNN the relevant map is the **one-step update**

    h_{t+1} = F(h_t, x_t)

and there are two derivatives worth taking at every point of a trajectory:

    J^rec = dF/dh   (hidden x hidden)  — how the network's own state evolves
    J^inp = dF/dx   (hidden x input)   — how new evidence enters the state

``J^rec`` governs memory: an eigenvalue of modulus ~1 is a direction
along which activity neither decays nor explodes, i.e. an integrator (a
line attractor); |lambda| < 1 directions forget with time constant
tau = -dt / log|lambda|. ``J^inp`` pulled back gives the metric that says
how strongly, and along which state directions, a stimulus perturbs the
network — the input-to-state analogue of the feedforward pullback metric.

Contents
--------
recurrent_jacobian / input_jacobian    batched dF/dh, dF/dx along trajectories
jacobian_spectrum                      complex eigenvalues + time constants
recurrent_update_pullback_metric                  g = J^T J of the update map (SPD tools)
find_slow_points                       fixed / slow points (Sussillo & Barak)
participation_ratio                    effective dimensionality of activity
trajectory_subspaces                   per-condition subspaces for Grassmann
readout_subspace / input_subspace      the readout subspace and input directions
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor


from ..geometry.jacobian import batch_jacobian, euclidean_pullback_metric  # noqa: E402
from ..geometry.grassmann import (grassmann_distance, principal_angles,  # noqa: E402
                          tangent_subspaces)

__all__ = [
    "recurrent_jacobian",
    "input_jacobian",
    "jacobian_spectrum",
    "recurrent_update_pullback_metric",
    "SlowPoints",
    "find_slow_points",
    "participation_ratio",
    "trajectory_subspaces",
    "readout_subspace",
    "input_subspace",
    "subspace_alignment",
]


# --------------------------------------------------------------------------- #
# Jacobians of the update map
# --------------------------------------------------------------------------- #
def recurrent_jacobian(model, H: Tensor, X: Tensor, **kw) -> Tensor:
    """dF/dh at each (h, x) pair.

    H : (B, hidden) states.  X : (B, in) the input present at that step.
    Returns (B, hidden, hidden).

    Model must be in eval mode (no noise) — this is enforced defensively.
    """
    was_training = model.training
    model.eval()
    try:
        J = _paired_jacobian(model, H, X, wrt="h", **kw)
    finally:
        if was_training:
            model.train()
    return J


def input_jacobian(model, H: Tensor, X: Tensor, **kw) -> Tensor:
    """dF/dx at each (h, x) pair. Returns (B, hidden, in)."""
    was_training = model.training
    model.eval()
    try:
        J = _paired_jacobian(model, H, X, wrt="x", **kw)
    finally:
        if was_training:
            model.train()
    return J


def _paired_jacobian(model, H: Tensor, X: Tensor, wrt: str,
                     chunk_size: Optional[int] = None) -> Tensor:
    """Per-sample Jacobian of F(h, x) w.r.t. h or x, holding the other fixed.

    Implemented by concatenating [h, x] into one input vector and taking the
    Jacobian of the full map, then slicing — this reuses the package's
    batched ``batch_jacobian`` (vmap + jacrev/jacfwd) and keeps one code path.
    """
    B, nh = H.shape
    nx = X.shape[1]
    Z = torch.cat([H, X], dim=1)                     # (B, nh + nx)

    def f(z: Tensor) -> Tensor:                      # (B, nh+nx) -> (B, nh)
        return model.step(z[:, nh:], z[:, :nh])

    J = batch_jacobian(f, Z, chunk_size=chunk_size)  # (B, nh, nh+nx)
    J = J.detach()          # analysis tool: never carries a training graph
    return J[:, :, :nh] if wrt == "h" else J[:, :, nh:]


def jacobian_spectrum(J: Tensor, dt: float = 1.0
                      ) -> Tuple[Tensor, Tensor, Tensor]:
    """Eigenvalues of a batch of recurrent Jacobians + derived time constants.

    Returns
    -------
    eigvals : complex (B, n), sorted by decreasing modulus
    moduli  : (B, n) |lambda|
    tau     : (B, n) effective time constant in the same units as ``dt``:
              tau = -dt / log|lambda|. |lambda| -> 1 gives tau -> inf
              (a perfect integrator); |lambda| > 1 returns a negative tau
              (an unstable/expanding direction) — inspect ``moduli`` too.
    """
    ev = torch.linalg.eigvals(J)                     # (B, n) complex
    mod = ev.abs()
    order = mod.argsort(dim=1, descending=True)
    ev = torch.gather(ev, 1, order)
    mod = torch.gather(mod, 1, order)
    logm = mod.clamp_min(1e-300).log()
    tau = torch.where(logm.abs() < 1e-12,
                      torch.full_like(logm, float("inf")), -dt / logm)
    return ev, mod, tau


def recurrent_update_pullback_metric(model, H: Tensor, X: Tensor, wrt: str = "h",
                          **kw) -> Tensor:
    """g = J^T J for the update map — feeds directly into spd_geometry.

    wrt="h": (B, hidden, hidden) metric on state space (how state
    perturbations propagate one step).
    wrt="x": (B, in, in) metric on *input* space — how strongly each
    stimulus direction perturbs the state at this point of the trajectory.
    """
    J = _paired_jacobian(model, H, X, wrt=wrt, **kw)
    g = torch.einsum("bki,bkj->bij", J, J)
    return 0.5 * (g + g.transpose(-1, -2))


# --------------------------------------------------------------------------- #
# Fixed points / slow points  (Sussillo & Barak 2013)
# --------------------------------------------------------------------------- #
@dataclass
class SlowPoints:
    """Result of a slow-point search."""
    h: Tensor                    # (P, hidden) the points found
    q: Tensor                    # (P,) speed 0.5*||F(h,x)-h||^2 (lower = closer)
    jac: Optional[Tensor] = None       # (P, hidden, hidden)
    eigvals: Optional[Tensor] = None   # (P, hidden) complex
    n_unstable: Optional[Tensor] = None  # (P,) count of |lambda| > 1

    def filter(self, q_max: float) -> "SlowPoints":
        keep = self.q <= q_max
        return SlowPoints(
            self.h[keep], self.q[keep],
            None if self.jac is None else self.jac[keep],
            None if self.eigvals is None else self.eigvals[keep],
            None if self.n_unstable is None else self.n_unstable[keep],
        )

    def unique(self, tol: float = 1e-2) -> "SlowPoints":
        """Greedy de-duplication of points closer than ``tol`` (in state space)."""
        order = self.q.argsort()
        keep_idx = []
        for i in order.tolist():
            hi = self.h[i]
            if all(torch.norm(hi - self.h[j]) > tol for j in keep_idx):
                keep_idx.append(i)
        idx = torch.tensor(keep_idx, dtype=torch.long)
        return SlowPoints(
            self.h[idx], self.q[idx],
            None if self.jac is None else self.jac[idx],
            None if self.eigvals is None else self.eigvals[idx],
            None if self.n_unstable is None else self.n_unstable[idx],
        )

    def __len__(self) -> int:
        return int(self.h.shape[0])


def find_slow_points(model, x_const: Tensor, h_init: Tensor, *,
                     steps: int = 800, lr: float = 0.05,
                     q_tol: float = 1e-7, compute_jacobian: bool = True,
                     verbose: bool = False) -> SlowPoints:
    """Find fixed points and slow points of the autonomous dynamics.

    Minimizes the speed objective  q(h) = 0.5 * ||F(h, x_const) - h||^2  from
    many initial conditions (typically states sampled from real trajectories,
    which is what makes the search find the points the network actually uses).

    Parameters
    ----------
    x_const : (in,) or (1, in) — the static input to hold (e.g. the mean
        stimulus, or zeros for the delay period). The dynamics analyzed are
        those of the network *under this constant input*.
    h_init : (P, hidden) initial guesses.
    q_tol : points with q above this are still returned (as "slow points"),
        but ``SlowPoints.filter`` lets you keep only true fixed points.

    Returns
    -------
    SlowPoints with per-point q, Jacobian, eigenvalues, and the number of
    unstable directions (|lambda| > 1) — a saddle with one unstable direction
    is the classic signature of a decision boundary.
    """
    was_training = model.training
    model.eval()
    P = h_init.shape[0]
    x = x_const.reshape(1, -1).expand(P, -1).to(h_init.dtype)
    h = h_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=lr)

    with torch.enable_grad():
        for i in range(steps):
            opt.zero_grad()
            q = 0.5 * (model.step(x, h) - h).pow(2).sum(1)
            q.sum().backward()
            opt.step()
            if verbose and (i + 1) % max(1, steps // 5) == 0:
                print(f"      slow-point search {i+1}/{steps}: "
                      f"median q = {float(q.median()):.3e}")

    h = h.detach()
    with torch.no_grad():
        q = (0.5 * (model.step(x, h) - h).pow(2).sum(1)).detach()

    jac = eig = nun = None
    if compute_jacobian:
        jac = _paired_jacobian(model, h, x, wrt="h")
        eig = torch.linalg.eigvals(jac)
        nun = (eig.abs() > 1.0).sum(1)

    if was_training:
        model.train()
    return SlowPoints(h, q, jac, eig, nun)


# --------------------------------------------------------------------------- #
# Population-level descriptors
# --------------------------------------------------------------------------- #
def participation_ratio(H: Tensor) -> float:
    """Effective dimensionality of a set of states H: (N, hidden).

    PR = (sum_i lambda_i)^2 / sum_i lambda_i^2 over PCA eigenvalues.
    1 = all activity on one axis; hidden = isotropic.
    """
    Hc = H - H.mean(0, keepdim=True)
    lam = torch.linalg.svdvals(Hc).square()
    return float(lam.sum().square() / lam.square().sum().clamp_min(1e-300))


def trajectory_subspaces(H: Tensor, groups: Tensor, k: int = 2
                         ) -> Tuple[Tensor, Tensor]:
    """Per-condition activity subspaces, ready for Grassmann comparison.

    H : (B, T, hidden) trajectories; groups : (B,) integer condition labels.
    For each condition, the top-k principal directions of its (centered)
    state cloud — a point on Gr(k, hidden).

    Returns (Q, labels): Q is (n_groups, hidden, k) orthonormal bases.
    """
    labels = torch.unique(groups)
    Qs = []
    for g in labels:
        Hg = H[groups == g].reshape(-1, H.shape[-1])
        Hg = Hg - Hg.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Hg, full_matrices=False)
        Qs.append(Vh[:k].transpose(0, 1))            # (hidden, k)
    return torch.stack(Qs), labels


def readout_subspace(model, k: Optional[int] = None) -> Tensor:
    """The decision plane: column space of the readout weights (hidden, k).

    Directions of state space the output actually reads. Everything
    orthogonal to it is, for the decision, a null space.
    """
    W = model.out.weight                              # (out, hidden)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = k or W.shape[0]
    return Vh[:k].transpose(0, 1)


def input_subspace(model, channels: Optional[Sequence[int]] = None,
                   k: Optional[int] = None) -> Tensor:
    """Directions in state space that a set of input channels drives.

    channels : which input dimensions to include (default: all).
    """
    W = model.inp.weight if hasattr(model, "inp") else model.cell.weight_ih
    W = W[:, list(channels)] if channels is not None else W   # (hidden, n_ch)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = k or W.shape[1]
    return U[:, :k]


def subspace_alignment(Q1: Tensor, Q2: Tensor) -> Dict[str, float]:
    """Summary of how two state-space subspaces relate."""
    th = principal_angles(Q1, Q2)
    return {
        "principal_angles_deg": [float(a) for a in torch.rad2deg(th)],
        "geodesic_distance": float(grassmann_distance(Q1, Q2)),
        "mean_squared_cosine": float(th.cos().square().mean()),
    }
