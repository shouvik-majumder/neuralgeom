"""The pullback metric g(x) = J(x)^T g_Y(f(x)) J(x) and its local geometry read-outs."""

from __future__ import annotations
import numpy as np

from .output_metrics import OutputMetric, Euclidean
from .maps import ReadoutMap


class PullbackMetric:
    """Pullback of an output-space metric through a readout map onto state space.

    Parameters
    ----------
    readout : ReadoutMap        the map f: X -> Y (state -> output).
    output_metric : OutputMetric  metric g_Y on the output space (default Euclidean).

    The metric at a state x is the (n, n) PSD matrix g(x) = J^T g_Y(f(x)) J. Its
    eigenstructure is the primary interpretive object:
      - large eigenvalues / their eigenvectors  = state directions the output is most
        sensitive to (the directions the computation "uses");
      - the kernel (zero eigenvalues)            = directions the readout ignores
        (the equivalence classes the computation collapses);
      - sqrt(det g) / pseudo-volume              = local warping / magnification;
      - effective rank / participation ratio     = how many directions matter locally.
    """

    def __init__(self, readout: ReadoutMap, output_metric: OutputMetric | None = None):
        self.readout = readout
        self.output_metric = output_metric if output_metric is not None else Euclidean()

    # --- core ---------------------------------------------------------------
    def _jacobian_and_G(self, x):
        J = np.atleast_2d(self.readout.jacobian(x))
        y = self.readout.forward(x)
        G = np.atleast_2d(self.output_metric.matrix(y))
        return J, G

    def metric_matrix(self, x) -> np.ndarray:
        """Pullback metric g(x) = J^T G J, shape (n, n)."""
        J, G = self._jacobian_and_G(x)
        return J.T @ G @ J

    def metric_vector_product(self, x, v) -> np.ndarray:
        """g(x) v without forming g: J^T (G (J v)). Avoids the (n, n) matrix for
        high-dimensional states."""
        J, G = self._jacobian_and_G(x)
        v = np.atleast_1d(np.asarray(v, float))
        return J.T @ (G @ (J @ v))

    # --- spectral read-outs -------------------------------------------------
    def spectrum(self, x):
        """Eigenvalues (descending) and eigenvectors (columns) of g(x)."""
        g = self.metric_matrix(x)
        g = 0.5 * (g + g.T)  # symmetrize against round-off
        w, V = np.linalg.eigh(g)
        idx = np.argsort(w)[::-1]
        return w[idx], V[:, idx]

    def principal_directions(self, x, k: int = 1):
        """Top-k eigenvalues and eigenvectors: the most output-sensitive directions."""
        w, V = self.spectrum(x)
        return w[:k], V[:, :k]

    def volume(self, x, eps: float = 0.0) -> float:
        """Riemannian volume element sqrt(det g). Add eps*I to regularize a singular g."""
        g = self.metric_matrix(x)
        if eps > 0:
            g = g + eps * np.eye(g.shape[0])
        sign, logdet = np.linalg.slogdet(g)
        return float(np.exp(0.5 * logdet)) if sign > 0 else 0.0

    def pseudo_volume(self, x, tol: float = 1e-10) -> float:
        """Product of sqrt of the positive eigenvalues (volume on the non-null subspace)."""
        w, _ = self.spectrum(x)
        w = w[w > tol]
        return float(np.exp(0.5 * np.sum(np.log(w)))) if w.size else 0.0

    def effective_rank(self, x, tol: float = 1e-12) -> float:
        """Entropy-based effective rank exp(-sum p_i log p_i), p_i = lambda_i / sum."""
        w, _ = self.spectrum(x)
        w = w[w > tol]
        if w.size == 0:
            return 0.0
        p = w / w.sum()
        return float(np.exp(-np.sum(p * np.log(p))))

    def participation_ratio(self, x, tol: float = 1e-12) -> float:
        """(sum lambda)^2 / sum(lambda^2): a robust soft count of significant directions."""
        w, _ = self.spectrum(x)
        w = w[w > tol]
        return float((w.sum() ** 2) / np.sum(w ** 2)) if w.size else 0.0

    def rank(self, x, rtol: float = 1e-9) -> int:
        """Numerical rank of g(x) (count of eigenvalues above rtol * max eigenvalue)."""
        w, _ = self.spectrum(x)
        wmax = max(w.max(), 1e-300)
        return int(np.sum(w > rtol * wmax))

    # --- convenience over a set of states -----------------------------------
    def field(self, X):
        """Vectorized read-outs over an array of states X (n_points, n).

        Returns a dict of arrays: 'volume', 'effective_rank', 'participation_ratio',
        'top_eigenvalue', and 'top_direction' (n_points, n).
        """
        X = np.atleast_2d(np.asarray(X, float))
        out = {k: [] for k in
               ("volume", "effective_rank", "participation_ratio", "top_eigenvalue")}
        top_dir = []
        for x in X:
            w, V = self.spectrum(x)
            out["top_eigenvalue"].append(w[0])
            top_dir.append(V[:, 0])
            out["volume"].append(self.pseudo_volume(x))
            out["effective_rank"].append(self.effective_rank(x))
            out["participation_ratio"].append(self.participation_ratio(x))
        result = {k: np.asarray(v) for k, v in out.items()}
        result["top_direction"] = np.asarray(top_dir)
        return result
