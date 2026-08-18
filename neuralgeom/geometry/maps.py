"""Readout maps f: X -> Y and their Jacobians.

A `ReadoutMap` is the differentiable map whose output metric we pull back. It only has
to implement `forward(x)`; if it does not provide an analytic `jacobian(x)`, a central
finite-difference Jacobian is used, so any black-box callable (sklearn model, trained
torch/JAX net wrapped as a function, hand-written decoder) works unchanged.

Conventions
-----------
- x is a 1-D state vector of arbitrary dimension n (a neural/latent state at one time).
- forward(x) returns a 1-D output of arbitrary dimension m (behavior, logits, ...).
- jacobian(x) returns the (m, n) matrix J_ij = d f_i / d x_j.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np


def numeric_jacobian(f: Callable[[np.ndarray], np.ndarray], x, h: float = 1e-5) -> np.ndarray:
    """Central finite-difference Jacobian of f at x, shape (m, n)."""
    x = np.atleast_1d(np.asarray(x, float))
    n = x.size
    y0 = np.atleast_1d(np.asarray(f(x), float))
    m = y0.size
    J = np.zeros((m, n))
    for i in range(n):
        step = h * (1.0 + abs(x[i]))
        xp = x.copy(); xp[i] += step
        xm = x.copy(); xm[i] -= step
        fp = np.atleast_1d(np.asarray(f(xp), float))
        fm = np.atleast_1d(np.asarray(f(xm), float))
        J[:, i] = (fp - fm) / (2.0 * step)
    return J


class ReadoutMap(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Map state x (n,) to output (m,)."""

    def jacobian(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        return numeric_jacobian(self.forward, x, h)

    def __call__(self, x):
        return self.forward(x)


class LinearReadout(ReadoutMap):
    """Affine readout f(x) = W x + b, with exact (constant) Jacobian W."""

    def __init__(self, W, b=None):
        self.W = np.atleast_2d(np.asarray(W, float))
        self.b = (np.zeros(self.W.shape[0]) if b is None
                  else np.atleast_1d(np.asarray(b, float)))

    def forward(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        return self.W @ x + self.b

    def jacobian(self, x, h: float = 1e-5):
        return self.W.copy()


def _activation(name, a):
    """Return (activation(a), activation'(a)) for sklearn MLP activation names."""
    if name == "tanh":
        t = np.tanh(a); return t, 1.0 - t ** 2
    if name == "relu":
        return np.maximum(a, 0.0), (a > 0).astype(float)
    if name == "logistic":
        s = 1.0 / (1.0 + np.exp(-a)); return s, s * (1.0 - s)
    if name == "identity":
        return a, np.ones_like(a)
    raise ValueError(f"unsupported activation {name!r}")


class MLPReadout(ReadoutMap):
    """Wrap a fitted scikit-learn MLPRegressor (optionally in a Pipeline with leading linear
    steps such as StandardScaler / PCA) and provide the EXACT analytic Jacobian.

    For layers out_l = act_l(out_{l-1} W_l + b_l) the Jacobian is the product of weight
    matrices interleaved with diagonal activation-derivative matrices, evaluated at the
    current pre-activations:  J = W_L^T D_{L-1} W_{L-1}^T ... D_1 W_1^T,  D_l = diag(act'(a_l)).
    Any leading linear pipeline steps contribute a constant matrix P by the chain rule.
    """

    def __init__(self, pipeline):
        from sklearn.pipeline import Pipeline
        if hasattr(pipeline, "steps"):
            self.pipeline = pipeline
            self.mlp = pipeline.steps[-1][1]
            self.pre = Pipeline(pipeline.steps[:-1]) if len(pipeline.steps) > 1 else None
        else:                                    # a bare MLPRegressor
            self.pipeline = pipeline
            self.mlp = pipeline
            self.pre = None
        self.k = int(self.pipeline.n_features_in_)
        self.P, self.o = self._pre_linear_map()

    def _pre_linear_map(self):
        """Constant Jacobian P and offset o of the (affine) pre-pipeline: s = P z + o."""
        if self.pre is None:
            return np.eye(self.k), np.zeros(self.k)
        o = self.pre.transform(np.zeros((1, self.k)))[0]
        P = np.zeros((o.size, self.k))
        for i in range(self.k):
            e = np.zeros((1, self.k)); e[0, i] = 1.0
            P[:, i] = self.pre.transform(e)[0] - o
        return P, o

    def forward(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        return np.atleast_1d(self.pipeline.predict(x[None])[0])

    def jacobian(self, x, h=1e-5):
        x = np.atleast_1d(np.asarray(x, float))
        s = self.P @ x + self.o                          # MLP input (pre-pipeline output)
        a = s
        J = np.eye(s.size)
        coefs, inters = self.mlp.coefs_, self.mlp.intercepts_
        L = len(coefs)
        for l, (W, b) in enumerate(zip(coefs, inters)):
            pre = a @ W + b
            Jpre = W.T @ J                               # linear part
            name = self.mlp.activation if l < L - 1 else self.mlp.out_activation_
            a, der = _activation(name, pre)
            J = der[:, None] * Jpre                      # diag(act'(pre)) @ Jpre
        return J @ self.P                                # (m, k)


class FunctionReadout(ReadoutMap):
    """Wrap any callable f(x)->y as a ReadoutMap, with an optional analytic Jacobian.

    Parameters
    ----------
    fn  : callable mapping (n,) -> (m,)
    jac : optional callable mapping (n,) -> (m, n); finite differences if omitted.
    """

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray],
                 jac: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.fn = fn
        self._jac = jac

    def forward(self, x):
        return np.atleast_1d(np.asarray(self.fn(np.asarray(x, float)), float))

    def jacobian(self, x, h: float = 1e-5):
        if self._jac is not None:
            return np.atleast_2d(np.asarray(self._jac(np.asarray(x, float)), float))
        return numeric_jacobian(self.forward, x, h)
