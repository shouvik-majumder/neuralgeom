"""
neuralgeom.fitting — the differentiable maps we pull metrics back through,
plus the cross-validation that keeps them honest.
=====================================================================

A pullback metric is only as meaningful as the map it comes from, so fitting
and validation belong together in one place. Every neural-data analysis used
to carry its own near-identical copy of these routines; the copies had drifted
(different hidden sizes, different fold logic, one with a prediction-scatter
bug), which is exactly the sort of inconsistency this module exists to remove.

WHAT IS HERE
------------
mlp                 a small tanh MLP (or a linear model) -> the map f
fit_map             fit f on (X, Y) by Adam, with an optional sample cap
trial_folds         K folds that never split a trial across folds
cv_fit_predict      out-of-fold predictions AND out-of-fold Jacobians
r2_score            variance explained, on held-out data only
shrink_cov          Ledoit-Wolf-style shrinkage covariance

WHY FOLDS MUST BE BY TRIAL
--------------------------
Consecutive time bins of one trial are a smoothed trajectory and are therefore
strongly dependent. If folds were split over SAMPLES, bins of the same trial
would land in both train and test, and the score would measure memorisation of
trials rather than generalisation of the map. ``trial_folds`` prevents that.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .geometry.jacobian import batch_jacobian

__all__ = ["mlp", "fit_map", "trial_folds", "cv_fit_predict", "r2_score",
           "shrink_cov"]


def mlp(d_in: int, d_out: int = 1, hidden: int = 32, layers: int = 1,
        linear: bool = False) -> nn.Module:
    """Small tanh MLP, or a bare linear map when ``linear=True``.

    The linear version is not a fallback but a CONTROL: if it matches the MLP
    on held-out data, the map is effectively linear and any curvature reported
    from the MLP's gradients is suspect.
    """
    if linear:
        return nn.Sequential(nn.Linear(d_in, d_out))
    mods: List[nn.Module] = [nn.Linear(d_in, hidden), nn.Tanh()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.Tanh()]
    mods += [nn.Linear(hidden, d_out)]
    return nn.Sequential(*mods)


def fit_map(X: np.ndarray, Y: np.ndarray, *, hidden: int = 32, layers: int = 1,
            linear: bool = False, steps: int = 200, lr: float = 5e-3,
            weight_decay: float = 1e-4, max_samples: Optional[int] = None,
            seed: int = 0) -> nn.Module:
    """Fit f: X -> Y. Y may be 1-D (scalar target) or 2-D (vector target).

    max_samples caps the training set by random subsampling. This is a SPEED
    control only and never changes which trials are eligible; it is recorded
    here rather than hidden in a script so its effect is auditable.
    """
    Y2 = Y.reshape(len(Y), -1)
    if max_samples is not None and len(X) > max_samples:
        sel = np.random.default_rng(seed).choice(len(X), max_samples,
                                                 replace=False)
        X, Y2 = X[sel], Y2[sel]
    torch.manual_seed(seed)
    net = mlp(X.shape[1], Y2.shape[1], hidden, layers, linear)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y2)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            ((net(Xt) - Yt) ** 2).mean().backward()
            opt.step()
    net.eval()
    return net


def trial_folds(n_trials: int, n_folds: int = 5, seed: int = 0
                ) -> List[np.ndarray]:
    """K folds of TRIAL indices, each returned sorted ascending.

    Sorted matters: callers scatter fold predictions back into a
    sample-indexed array using a boolean mask, which is in ascending trial
    order. Returning shuffled indices here caused predictions to be assigned
    to the wrong trials (a bug that showed up as R^2 below the shuffled null).
    """
    rng = np.random.default_rng(seed)
    return [np.sort(f) for f in
            np.array_split(rng.permutation(n_trials), n_folds)]


def cv_fit_predict(X: np.ndarray, Y: np.ndarray, trial_id: np.ndarray, *,
                   n_folds: int = 5, seed: int = 0,
                   transform: Optional[Callable] = None,
                   **fit_kw) -> dict:
    """Out-of-fold predictions and Jacobians, with folds split by trial.

    X         (samples, d_in)
    Y         (samples,) or (samples, d_out)
    trial_id  (samples,) which trial each sample came from
    transform optional callable(train_X, test_X) -> (train_X', test_X') applied
              INSIDE each fold, e.g. a reducer fit on training data only. Use
              this rather than reducing beforehand, which leaks test data into
              the coordinate system.

    Returns dict(pred, J, r2) where J is (samples, d_out, d_in) evaluated by
    the fold model that never saw that sample.
    """
    Y2 = Y.reshape(len(Y), -1)
    uniq = np.unique(trial_id)
    folds = trial_folds(len(uniq), n_folds, seed)
    pred = np.zeros_like(Y2)
    Js: Optional[np.ndarray] = None
    for i, f in enumerate(folds):
        te_tr = uniq[f]
        m_te = np.isin(trial_id, te_tr)
        m_tr = ~m_te
        Xtr, Xte = X[m_tr], X[m_te]
        if transform is not None:
            Xtr, Xte = transform(Xtr, Xte)
        net = fit_map(Xtr, Y2[m_tr], seed=seed + i, **fit_kw)
        with torch.no_grad():
            pred[m_te] = net(torch.as_tensor(Xte)).numpy()
        Jte = batch_jacobian(net, torch.as_tensor(Xte)).detach().numpy()
        if Js is None:
            Js = np.zeros((len(Y2),) + Jte.shape[1:])
        Js[m_te] = Jte
    return dict(pred=pred, J=Js, r2=r2_score(Y2, pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - SS_res/SS_tot. Negative means worse than predicting the mean."""
    yt = np.asarray(y_true).reshape(len(y_true), -1)
    yp = np.asarray(y_pred).reshape(len(y_pred), -1)
    return float(1.0 - ((yt - yp) ** 2).sum()
                 / ((yt - yt.mean(0)) ** 2).sum())


def shrink_cov(Xc: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Covariance shrunk toward a scaled identity: (1-a)C + a*tr(C)/n*I.

    Xc must already be centred. Shrinkage is needed because a sample
    covariance with fewer samples than dimensions is singular, and the
    affine-invariant SPD distance inverts it.
    """
    C = np.cov(Xc.T)
    return ((1 - alpha) * C
            + alpha * np.trace(C) / C.shape[0] * np.eye(C.shape[0]))
