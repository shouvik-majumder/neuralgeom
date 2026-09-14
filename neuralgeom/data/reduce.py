"""
neuralgeom.data.reduce — pluggable dimensionality reduction for population geometry.
====================================================================================

Motivation
----------
Choosing to run PCA before computing a geometry is itself an arbitrary
modelling decision, not a neutral preprocessing step. Different reducers keep
different things and throw different things away, so a geometric result that
only appears under one of them is a property of that reducer, not of the
brain. This module puts every option behind one interface so the geometry can
be recomputed identically across all of them and the dependence reported.

The interface
-------------
Every reducer implements:

    red.fit(X, y=None)      X: (trials, time, units); y optional labels
    red.transform(X)        -> (trials, time, k)
    red.name                short label for figures
    red.k                   output dimensionality after fitting

Reducers are fit on the *training* trials handed to them and applied to any
trials, so cross-validated / leakage-free use is possible where it matters.

Terminology
-----------
*Dimensionality reduction*: replacing each population state (a vector of N
firing rates, one per unit) with a shorter vector of k numbers that captures
most of what varies. Formally a linear map R^N -> R^k, i.e. a k x N matrix.

*Principal Component Analysis (PCA)*: finds the k directions in the
N-dimensional space along which the data varies most, ranked. It is
UNSUPERVISED: it never sees the behaviour, so using it cannot leak the
behavioural label into a later behaviour test. Its weakness is that "varies
most" and "matters most" are different things — a large, behaviourally
irrelevant signal (e.g. overall arousal) will dominate PC1.

*Variance retained*: instead of fixing k, keep however many components are
needed to explain a target fraction (say 80%) of the total variance. Makes
the choice adaptive to each dataset rather than fixed by hand, at the cost of
different k across conditions, which must be reported.

*Linear Discriminant Analysis (LDA)*: finds directions that best SEPARATE
labelled groups, maximizing between-group over within-group variance. It is
SUPERVISED. If it is fit on lick-time labels and the geometry is then tested
against lick time, the test is circular — the space was built to contain that
signal. This module therefore only exposes LDA with an explicit fit/apply
split so it can be fit on held-out trials.

*Identity / full space*: no reduction at all. The most faithful option and the
correct baseline, but with N ~ 100 units and ~50 trials per condition the
sample covariance is severely ill-conditioned (more unknowns than data), so
distances that invert it lean heavily on shrinkage.

*Random projection*: project onto k random orthonormal directions. A NULL
reducer: it preserves dimensionality-reduction-induced effects (same k, same
shrinkage regime) while destroying any claim that the chosen directions are
special. If a result survives PCA but also appears under random projection at
the same k, the result is about dimensionality, not about structure.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["Reducer", "Identity", "PCA", "PCAVariance", "RandomProjection",
           "LDA", "make_reducer", "REDUCERS"]


def _flat(X: np.ndarray) -> np.ndarray:
    """(trials, time, units) -> (trials*time, units)."""
    return X.reshape(-1, X.shape[-1])


class Reducer:
    """Base class. Subclasses set ``self.W`` (units, k) and ``self.mu``."""
    name = "base"

    def __init__(self):
        self.W: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None

    @property
    def k(self) -> int:
        return 0 if self.W is None else int(self.W.shape[1])

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Reducer":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError(f"{self.name}: call fit() first")
        return (X - self.mu.reshape(1, 1, -1)) @ self.W

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def __repr__(self):
        return f"{self.name}(k={self.k})"


class Identity(Reducer):
    """No reduction: keep all units. The faithful-but-ill-conditioned baseline."""
    name = "full"

    def fit(self, X, y=None):
        N = X.shape[-1]
        self.mu = _flat(X).mean(0)
        self.W = np.eye(N)
        return self


class PCA(Reducer):
    """Top-k principal components (unsupervised)."""

    def __init__(self, k: int = 3):
        super().__init__()
        self._k = int(k)
        self.name = f"PCA{k}"
        self.evr: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        F = _flat(X)
        self.mu = F.mean(0)
        U, S, Vh = np.linalg.svd(F - self.mu, full_matrices=False)
        var = S ** 2
        self.evr = var / var.sum()
        kk = min(self._k, Vh.shape[0])
        self.W = Vh[:kk].T
        self.var_retained = float(self.evr[:kk].sum())
        return self


class PCAVariance(Reducer):
    """Enough principal components to retain a target fraction of variance."""

    def __init__(self, target: float = 0.80):
        super().__init__()
        self.target = float(target)
        self.name = f"PCA{int(target*100)}%"

    def fit(self, X, y=None):
        F = _flat(X)
        self.mu = F.mean(0)
        U, S, Vh = np.linalg.svd(F - self.mu, full_matrices=False)
        var = S ** 2
        evr = var / var.sum()
        kk = int(np.searchsorted(np.cumsum(evr), self.target) + 1)
        kk = max(2, min(kk, Vh.shape[0]))
        self.W = Vh[:kk].T
        self.evr = evr
        self.var_retained = float(evr[:kk].sum())
        return self


class RandomProjection(Reducer):
    """k random orthonormal directions — the NULL reducer (see module doc)."""

    def __init__(self, k: int = 3, seed: int = 0):
        super().__init__()
        self._k = int(k)
        self.seed = seed
        self.name = f"rand{k}"

    def fit(self, X, y=None):
        N = X.shape[-1]
        self.mu = _flat(X).mean(0)
        rng = np.random.default_rng(self.seed)
        self.W = np.linalg.qr(rng.standard_normal((N, min(self._k, N))))[0]
        self.var_retained = np.nan
        return self


class LDA(Reducer):
    """Supervised: directions separating labelled groups.

    WARNING (circularity): if fit on the same behavioural labels the geometry
    is later tested against, the test is meaningless. Always fit on trials
    disjoint from those used for the test.
    """

    def __init__(self, k: int = 3, shrink: float = 0.1):
        super().__init__()
        self._k = int(k)
        self.shrink = shrink
        self.name = f"LDA{k}"

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("LDA requires labels y (one per trial)")
        F = _flat(X)
        self.mu = F.mean(0)
        Xc = X - self.mu.reshape(1, 1, -1)
        classes = np.unique(y[y >= 0])
        N = X.shape[-1]
        Sw = np.zeros((N, N))
        Sb = np.zeros((N, N))
        gm = Xc.reshape(-1, N).mean(0)
        for c in classes:
            Fc = Xc[y == c].reshape(-1, N)
            mc = Fc.mean(0)
            Sw += np.cov(Fc.T) * len(Fc)
            d = (mc - gm)[:, None]
            Sb += len(Fc) * (d @ d.T)
        Sw /= max(1, len(y))
        Sw = (1 - self.shrink) * Sw + self.shrink * np.trace(Sw) / N * np.eye(N)
        w, V = np.linalg.eigh(np.linalg.solve(Sw, Sb))
        self.W = V[:, ::-1][:, :min(self._k, N)]
        self.var_retained = np.nan
        return self


REDUCERS = {"full": Identity, "pca": PCA, "pca_var": PCAVariance,
            "random": RandomProjection, "lda": LDA}


def make_reducer(spec: str) -> Reducer:
    """Build a reducer from a short string.

    'full'        no reduction
    'pca:5'       top-5 PCs
    'pcavar:0.8'  enough PCs for 80% variance
    'rand:5'      5 random directions (null reducer)
    'lda:3'       3 LDA directions (needs labels, fit out-of-sample)
    """
    if spec == "full":
        return Identity()
    kind, _, arg = spec.partition(":")
    if kind == "pca":
        return PCA(int(arg))
    if kind == "pcavar":
        return PCAVariance(float(arg))
    if kind == "rand":
        return RandomProjection(int(arg))
    if kind == "lda":
        return LDA(int(arg))
    raise ValueError(f"unknown reducer spec {spec!r}")
