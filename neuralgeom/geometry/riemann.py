"""Riemannian geometry of a metric field g(x) defined on a region of state space.

`RiemannianField` takes any callable metric_fn: x -> (n, n) SPD matrix (for example
`PullbackMetric.metric_matrix`) and provides Christoffel symbols, geodesics, geodesic
distance, and curvature (Riemann / Ricci / scalar) by numerical differentiation.

All derivatives are central finite differences of the (assumed smooth) metric field.
`h1` is the step for first derivatives of the metric (Christoffel symbols); `h2` is the
step for differentiating the Christoffel symbols (curvature). They are separated because
the optimal step for a second derivative is larger than for a first.

Curvature by nested finite differencing is noise-sensitive; it is intended for smooth
analytic / low-dimensional metric fields (validation, 2-3 D latent slices), not for
raw high-dimensional pullbacks. For those, use the spectral read-outs in PullbackMetric.
"""

from __future__ import annotations
from typing import Callable
import numpy as np


class RiemannianField:
    def __init__(self, metric_fn: Callable[[np.ndarray], np.ndarray], dim: int,
                 h1: float = 1e-4, h2: float = 1e-3):
        self.metric_fn = metric_fn
        self.dim = int(dim)
        self.h1 = h1
        self.h2 = h2

    # --- metric and its derivatives ----------------------------------------
    def metric(self, x) -> np.ndarray:
        g = np.atleast_2d(np.asarray(self.metric_fn(np.asarray(x, float)), float))
        return 0.5 * (g + g.T)

    def _metric_grad(self, x, h):
        """List dg[l] = d g / d x_l, each (n, n)."""
        x = np.asarray(x, float)
        out = []
        for l in range(self.dim):
            step = h * (1.0 + abs(x[l]))
            xp = x.copy(); xp[l] += step
            xm = x.copy(); xm[l] -= step
            out.append((self.metric(xp) - self.metric(xm)) / (2.0 * step))
        return out

    # --- connection ---------------------------------------------------------
    def christoffel(self, x, h=None) -> np.ndarray:
        """Christoffel symbols Gamma[k, i, j] = Gamma^k_{ij} (symmetric in i, j)."""
        h = self.h1 if h is None else h
        g = self.metric(x)
        ginv = np.linalg.inv(g)
        dg = self._metric_grad(x, h)
        n = self.dim
        Gamma = np.zeros((n, n, n))
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    s = 0.0
                    for l in range(n):
                        s += ginv[k, l] * (dg[i][j, l] + dg[j][i, l] - dg[l][i, j])
                    Gamma[k, i, j] = 0.5 * s
        return Gamma

    # --- curvature ----------------------------------------------------------
    def riemann(self, x) -> np.ndarray:
        """Riemann tensor R[rho, sig, mu, nu] = R^rho_{sig mu nu}."""
        n = self.dim
        x = np.asarray(x, float)
        Gamma = self.christoffel(x, self.h1)
        # dGamma[m][k, i, j] = d Gamma^k_{ij} / d x_m
        dG = []
        for m in range(n):
            step = self.h2 * (1.0 + abs(x[m]))
            xp = x.copy(); xp[m] += step
            xm = x.copy(); xm[m] -= step
            dG.append((self.christoffel(xp, self.h1) - self.christoffel(xm, self.h1))
                      / (2.0 * step))
        R = np.zeros((n, n, n, n))
        for rho in range(n):
            for sig in range(n):
                for mu in range(n):
                    for nu in range(n):
                        term = dG[mu][rho, nu, sig] - dG[nu][rho, mu, sig]
                        for lam in range(n):
                            term += (Gamma[rho, mu, lam] * Gamma[lam, nu, sig]
                                     - Gamma[rho, nu, lam] * Gamma[lam, mu, sig])
                        R[rho, sig, mu, nu] = term
        return R

    def ricci(self, x) -> np.ndarray:
        """Ricci tensor Ric[sig, nu] = R^rho_{sig rho nu}."""
        R = self.riemann(x)
        n = self.dim
        Ric = np.zeros((n, n))
        for sig in range(n):
            for nu in range(n):
                Ric[sig, nu] = sum(R[rho, sig, rho, nu] for rho in range(n))
        return Ric

    def scalar_curvature(self, x) -> float:
        """Scalar curvature R = g^{ij} Ric_{ij}."""
        g = self.metric(x)
        ginv = np.linalg.inv(g)
        Ric = self.ricci(x)
        return float(np.sum(ginv * Ric))

    # --- geodesics ----------------------------------------------------------
    def geodesic(self, x0, x1, n_nodes: int = 24, maxiter: int = 300):
        """Minimizing geodesic between x0 and x1 by discretized path-energy descent.

        Returns the path as an (n_nodes, n) array. Endpoints are fixed; interior nodes
        minimize sum_k (dgamma_k)^T g(midpoint_k) (dgamma_k).
        """
        from scipy.optimize import minimize
        x0 = np.atleast_1d(np.asarray(x0, float))
        x1 = np.atleast_1d(np.asarray(x1, float))
        n = self.dim
        ts = np.linspace(0.0, 1.0, n_nodes)
        init = x0[None, :] + (x1 - x0)[None, :] * ts[:, None]

        def assemble(interior):
            return np.vstack([x0, interior.reshape(n_nodes - 2, n), x1])

        def energy(interior):
            P = assemble(interior)
            E = 0.0
            for a in range(n_nodes - 1):
                d = P[a + 1] - P[a]
                mid = 0.5 * (P[a] + P[a + 1])
                g = self.metric(mid)
                E += d @ g @ d
            return E

        res = minimize(energy, init[1:-1].ravel(), method="L-BFGS-B",
                       options={"maxiter": maxiter})
        return assemble(res.x)

    def length(self, path) -> float:
        """Riemannian length of a polyline path (n_nodes, n)."""
        P = np.atleast_2d(np.asarray(path, float))
        L = 0.0
        for a in range(len(P) - 1):
            d = P[a + 1] - P[a]
            mid = 0.5 * (P[a] + P[a + 1])
            g = self.metric(mid)
            L += np.sqrt(max(d @ g @ d, 0.0))
        return float(L)

    def distance(self, x0, x1, n_nodes: int = 24, maxiter: int = 300) -> float:
        """Geodesic distance = length of the minimizing geodesic."""
        return self.length(self.geodesic(x0, x1, n_nodes, maxiter))
