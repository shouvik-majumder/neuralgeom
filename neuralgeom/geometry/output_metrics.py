"""Output-space metrics g_Y to be pulled back through a readout map.

The choice of g_Y defines what "distance in output space" means:

- `Euclidean`            : g_Y = I. Pullback g = J^T J (Gram / sensitivity metric).
- `ScaledGaussian`       : Fisher metric of N(mean=y, fixed sigma) -> I / sigma^2.
- `BernoulliFisher`      : Fisher metric of independent Bernoulli(prob=y) ->
                           diag(1 / (y(1-y))). Use with a probability-valued readout.
- `GaussianMeanVarFisher`: Fisher metric of N(mu, var) in (mu, var) coords ->
                           diag(1/var, 1/(2 var^2)). Expects y = [mu, var].
- `CategoricalFisher`     : softmax Fisher of a k-way categorical in logit coords ->
                           diag(p) - p p^T, p = softmax(logits). Use with a
                           classification / choice readout that emits logits.

Each returns an (m, m) positive-(semi)definite matrix given the output y = f(x).

The task's output type therefore selects the metric: continuous scalar/vector ->
Euclidean/ScaledGaussian; binary -> BernoulliFisher; k-way choice -> CategoricalFisher;
predicted mean+variance -> GaussianMeanVarFisher. Inputs to the task are irrelevant to
this layer -- only the readout's output space matters.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class OutputMetric(ABC):
    @abstractmethod
    def matrix(self, y: np.ndarray) -> np.ndarray:
        """Return the (m, m) metric matrix at output point y."""

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self.matrix(y)


class Euclidean(OutputMetric):
    """Standard Euclidean metric; pullback becomes J^T J."""

    def matrix(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        return np.eye(y.size)


class ScaledGaussian(OutputMetric):
    """Fisher metric of an isotropic Gaussian readout N(mean=y, sigma^2 I).

    Equivalent to Euclidean scaled by 1/sigma^2. `sigma` is the observation noise
    of the behavioral readout (e.g. trial-to-trial std of first-lick latency).
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = float(sigma)

    def matrix(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        return np.eye(y.size) / self.sigma ** 2


class BernoulliFisher(OutputMetric):
    """Fisher metric of independent Bernoulli outputs parameterized by probability.

    y are probabilities in (0, 1); metric is diag(1 / (y (1 - y))).
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def matrix(self, y):
        p = np.clip(np.atleast_1d(np.asarray(y, float)), self.eps, 1 - self.eps)
        return np.diag(1.0 / (p * (1.0 - p)))


class GaussianMeanVarFisher(OutputMetric):
    """Fisher metric of N(mu, var) in (mu, var) coordinates: diag(1/var, 1/(2 var^2)).

    Use when the readout emits both a predicted mean and variance, y = [mu, var].
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def matrix(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        if y.size != 2:
            raise ValueError("GaussianMeanVarFisher expects y = [mu, var].")
        var = max(float(y[1]), self.eps)
        return np.array([[1.0 / var, 0.0], [0.0, 1.0 / (2.0 * var ** 2)]])


class LogScaleGaussian(OutputMetric):
    """Logarithmic (relative-scale) metric on positive outputs.

    The Fisher metric of a log-normal observation model, i.e. a Gaussian on log(y):
    distance is measured in log units, ds = d(log y) / sigma = dy / (sigma * y), so
    g_Y = diag(1 / (sigma^2 * y_i^2)). A fixed *percentage* change in the output is the
    same distance everywhere -- the Weber-law / scalar-timing scale appropriate for
    positive, right-skewed behavioral variables like reaction / lick times.

    Requires y > 0 (values are clipped at `eps`).
    """

    def __init__(self, sigma: float = 1.0, eps: float = 1e-6):
        self.sigma = float(sigma)
        self.eps = eps

    def matrix(self, y):
        y = np.clip(np.atleast_1d(np.asarray(y, float)), self.eps, None)
        return np.diag(1.0 / (self.sigma ** 2 * y ** 2))


class GaussianMuLogSigmaFisher(OutputMetric):
    """Fisher metric of N(mu, sigma^2) in coordinates (mu, s = log sigma).

    Expects y = [mu, s]. In these coordinates the Fisher information is diagonal and
    state-dependent through sigma:  g_Y = diag( exp(-2 s), 2 )  =  diag( 1/sigma^2, 2 ).
    (The mu-block is the usual 1/sigma^2; the log-sigma block is the constant 2.) Use with a
    readout that outputs a predicted mean and log-std, so the pulled-back metric is rank-2
    and encodes behavioral precision, not just the mean.
    """

    def matrix(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        if y.size != 2:
            raise ValueError("GaussianMuLogSigmaFisher expects y = [mu, log sigma].")
        s = y[1]
        return np.array([[np.exp(-2.0 * s), 0.0], [0.0, 2.0]])


class CategoricalFisher(OutputMetric):
    """Softmax Fisher metric of a k-way categorical distribution in logit coordinates.

    Given logits y (length k), p = softmax(y), the Fisher information (equivalently the
    covariance of the one-hot label under p) is diag(p) - p p^T. This is the canonical
    metric for a classification / choice readout that emits logits; it is rank k-1
    (the all-ones logit direction is a gauge freedom and lies in the kernel).
    """

    def matrix(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        z = y - y.max()
        e = np.exp(z)
        p = e / e.sum()
        return np.diag(p) - np.outer(p, p)


class BlockOutputMetric(OutputMetric):
    """Block-diagonal metric for outputs that concatenate different types.

    Compose several OutputMetrics over contiguous slices of the output vector, e.g. a
    Gaussian latency and a Bernoulli reward: BlockOutputMetric([(ScaledGaussian(0.1), 1),
    (BernoulliFisher(), 1)]). Each entry is (metric, block_size); block sizes must sum to
    the output dimension.
    """

    def __init__(self, blocks):
        self.blocks = [(m, int(s)) for m, s in blocks]

    def matrix(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        total = sum(s for _, s in self.blocks)
        if total != y.size:
            raise ValueError(f"block sizes sum to {total} but y has size {y.size}.")
        G = np.zeros((y.size, y.size))
        i = 0
        for metric, s in self.blocks:
            G[i:i + s, i:i + s] = metric.matrix(y[i:i + s])
            i += s
        return G
