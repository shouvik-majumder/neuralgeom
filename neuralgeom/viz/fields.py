"""Visualize a pulled-back metric on a 2-D slice of state space (e.g. two PCs).

Two views, matched to the output type:

- Scalar readout (rank-1 metric): `scalar_metric_field` returns the predicted-behavior
  surface (its contours are iso-metric level sets), the local sensitivity sqrt(lambda_max),
  and the top-eigenvector field (the single behaviorally-relevant direction). Draw with
  `plot_scalar_metric_field`.

- Vector readout (rank>=2 metric): `plot_ellipse_field` draws the metric ellipse
  {v : v^T g_2 v = c} at a grid of points, so anisotropy and orientation are visible. The
  ellipse's SHORT axis is the most behaviorally-sensitive direction.

Non-plane state dimensions are held at `fixed` (default 0 = the PCA-score mean).
"""

from __future__ import annotations
import numpy as np


def _plane_points(Z, dims, n, pad):
    d0, d1 = dims
    lo0, hi0 = np.percentile(Z[:, d0], [2, 98])
    lo1, hi1 = np.percentile(Z[:, d1], [2, 98])
    r0, r1 = hi0 - lo0, hi1 - lo1
    g0 = np.linspace(lo0 - pad * r0, hi0 + pad * r0, n)
    g1 = np.linspace(lo1 - pad * r1, hi1 + pad * r1, n)
    return g0, g1


def _lift(p0, p1, dims, dim_total, fixed):
    z = np.full(dim_total, fixed, float)
    z[dims[0]] = p0
    z[dims[1]] = p1
    return z


def scalar_metric_field(pm, Z, dims=(0, 1), n=28, pad=0.08, fixed=0.0):
    """Grids of predicted value, sqrt(lambda_max), and top-eigenvector (in-plane comps)."""
    dim_total = Z.shape[1]
    g0, g1 = _plane_points(Z, dims, n, pad)
    F = np.zeros((n, n)); S = np.zeros((n, n))
    U = np.zeros((n, n)); V = np.zeros((n, n))
    for i, a in enumerate(g0):
        for j, b in enumerate(g1):
            z = _lift(a, b, dims, dim_total, fixed)
            F[j, i] = float(pm.readout.forward(z)[0])
            w, Vec = pm.spectrum(z)
            S[j, i] = np.sqrt(max(w[0], 0.0))
            U[j, i] = Vec[dims[0], 0]
            V[j, i] = Vec[dims[1], 0]
    return g0, g1, F, S, U, V


def plot_scalar_metric_field(ax, pm, Z, y=None, dims=(0, 1), n=30, quiver_step=3,
                             fixed=0.0, vmin=None, vmax=None):
    """Background sqrt(lambda_max), predicted-value contours, top-eigenvector quiver.

    Pass vmin/vmax to share a color scale across panels (so a constant field reads flat).
    """
    g0, g1, F, S, U, V = scalar_metric_field(pm, Z, dims, n=n, fixed=fixed)
    X0, X1 = np.meshgrid(g0, g1)
    im = ax.pcolormesh(X0, X1, S, shading="auto", cmap="magma", alpha=.9,
                       vmin=vmin, vmax=vmax)
    cs = ax.contour(X0, X1, F, levels=10, colors="w", linewidths=.7, alpha=.8)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    # sign of the rank-1 eigenvector is arbitrary; orient to a coherent field (U >= 0)
    flip = np.where(U < 0, -1.0, 1.0)
    U, V = U * flip, V * flip
    sp = quiver_step
    ax.quiver(X0[::sp, ::sp], X1[::sp, ::sp],
              (U * S)[::sp, ::sp], (V * S)[::sp, ::sp],
              color="c", width=.004, alpha=.9)
    if y is not None:
        sc = ax.scatter(Z[:, dims[0]], Z[:, dims[1]], c=y, s=12, cmap="viridis",
                        edgecolor="k", linewidth=.2)
    else:
        sc = None
    ax.set_xlabel(f"PC{dims[0]+1}"); ax.set_ylabel(f"PC{dims[1]+1}")
    return im, sc


def metric_ellipse_polygon(g2, center=(0.0, 0.0), scale=1.0, n=64):
    """Bounded 'sensitivity ellipse': image of the unit circle under g2^{1/2}.

    Semi-axes are scale * sqrt(lambda_i) along the eigenvectors of g2, so the LONG axis is
    the most behavior-sensitive direction. Unlike the indicatrix {v: v^T g2 v = 1} this
    stays finite when the metric is (near) rank-deficient (it collapses to a line segment).
    """
    g2 = 0.5 * (np.asarray(g2, float) + np.asarray(g2, float).T)
    w, Vt = np.linalg.eigh(g2)
    w = np.clip(w, 0.0, None)
    theta = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(theta), np.sin(theta)], 0)          # (2, n)
    axes = Vt @ np.diag(np.sqrt(w))                               # semi-axes ~ sqrt(lambda)
    pts = scale * (axes @ circle)
    return pts[0] + center[0], pts[1] + center[1]


def plot_ellipse_field(ax, pm, Z, y=None, dims=(0, 1), n=9, scale=None, fixed=0.0):
    """Draw metric ellipses over a grid; short axis = most behavior-sensitive direction."""
    dim_total = Z.shape[1]
    g0, g1 = _plane_points(Z, dims, n, pad=0.0)
    # auto-scale so the largest ellipse axis ~ 0.45 * grid spacing
    if scale is None:
        gap = min(g0[1] - g0[0], g1[1] - g1[0])
        lam = []
        for a in g0:
            for b in g1:
                g = pm.metric_matrix(_lift(a, b, dims, dim_total, fixed))
                w, _ = np.linalg.eigh(g[np.ix_(dims, dims)])
                lam.append(max(w))
        scale = 0.45 * gap / (np.sqrt(np.median(lam)) + 1e-12)
    for a in g0:
        for b in g1:
            z = _lift(a, b, dims, dim_total, fixed)
            g = pm.metric_matrix(z)
            g2 = g[np.ix_(dims, dims)]
            px, py = metric_ellipse_polygon(g2, center=(a, b), scale=scale)
            ax.plot(px, py, color="#c33", lw=1.1, alpha=.9)
    if y is not None:
        ax.scatter(Z[:, dims[0]], Z[:, dims[1]], c=y, s=10, cmap="viridis",
                   edgecolor="k", linewidth=.2, zorder=3)
    ax.set_xlabel(f"PC{dims[0]+1}"); ax.set_ylabel(f"PC{dims[1]+1}")
    ax.set_aspect("equal", adjustable="datalim")
