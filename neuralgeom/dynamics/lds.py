"""Conditioned linear dynamical systems + decomposition of the fitted dynamics.

Design and rationale: ``docs/LDS_PIPELINE_PLAN.md``.  Constraints: ``docs/INSTRUCTIONS.md``.

Three things live here.

1. THREE LINEAR MODEL CLASSES, all fit to the same (z, dz/dt) samples so they are comparable
   (``fit_lds``, ``fit_sliding_lds``, ``fit_cubic_field`` + ``linearize_cubic``):
     M1  global conditioned LDS      dz/dt = A z + b                (one fixed point)
     M2  sliding-window LDS          dz/dt = A(tau) z + b(tau)      (local Jacobian vs time)
     M3  cubic field, linearized     A = dF/dz at a chosen state    (keeps the cubic generator)

2. INPUT INFERENCE -- shared field + per-condition input, in two parameterizations
   (``fit_shared_input_free``, ``fit_shared_input_lowrank``), plus the per-timepoint split of
   the velocity into a field-driven and an input-driven part (``velocity_budget``).

3. METRIC-AWARE POTENTIAL/ROTATIONAL SPLIT (``helmholtz_split``, ``metric_dependence``).
   The split is NOT unique -- 'gradient' is defined only relative to a metric M:

       A = A_grad + A_rot,   A_grad = M^-1 Sym(MA),   A_rot = M^-1 Skew(MA)
       dz/dt|_grad = -M^-1 grad V(z),   V(z) = -1/2 z^T Sym(MA) z - (M b)^T z
       A_rot conserves the M-energy 1/2 z^T M z.

   M = I reproduces the classic S/W split used in ``demo_metriplectic_linear.py``;
   M = Sigma^-1 gives the data-adapted (Mahalanobis) split.

Every fit reports a trial-grouped cross-validated velocity R^2 and, on request, a within-trial
roll-shuffle null.  Nothing here plots; the demos do that.

Self-test:  python -m neuralgeom.dynamics.lds
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures

__all__ = [
    "trial_velocities", "r2_score_multi",
    "fit_lds", "fit_sliding_lds", "fit_cubic_field", "cubic_predict", "linearize_cubic",
    "helmholtz_split", "metric_dependence", "potential",
    "fit_shared_input_free", "fit_shared_input_lowrank", "velocity_budget",
    "behavior_relevant_fraction", "shuffle_null_r2",
]


# ======================================================================================
# velocities and scoring
# ======================================================================================
def _causal_boxcar(x, width):
    """Trailing moving average over ``width`` bins along axis 0: y[t] = mean(x[t-width+1..t]).

    Uses only PAST samples, so it cannot leak a later event backward across an onset.  The
    first ``width-1`` samples average over the available (shorter) history.
    """
    width = int(width)
    if width <= 1:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    out = np.empty_like(x)
    for t in range(len(x)):
        lo = max(0, t - width + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def trial_velocities(Z_trials, dt, gap=2, smooth_bins=2.0, keep=None, inst_lag=0,
                     causal=False, time=None):
    """Finite-difference velocities per trial, with the SAME convention for every model class.

    Parameters
    ----------
    Z_trials : list of (T_i, D) arrays, or a (n_trials, T, D) array -- state per trial.
    dt       : bin width (s).
    gap      : finite-difference gap in bins.
    smooth_bins : smoothing window applied to the trajectory before differencing (0 disables).
    causal   : if False (default), symmetric Gaussian smoothing (sigma = smooth_bins) + a
               forward difference (z[t+gap]-z[t])/(g dt) attributed to the LEFT endpoint t.
               Both are non-causal: they mix future samples into the estimate at time t, which
               LEAKS a post-onset event (e.g. the cue) BACKWARD across the onset.
               If True, use a CAUSAL boxcar (trailing average of ``smooth_bins`` bins) + a
               BACKWARD difference (z[t]-z[t-gap])/(g dt) attributed to the CURRENT time t, so
               the velocity at t depends only on samples at or before t.  Use this whenever the
               time axis matters (input-onset analyses).
    keep     : optional list of boolean masks (one per trial, length T_i) over ORIGINAL time,
               selecting which samples to keep after differencing.
    inst_lag : if > 0, also return an instrument Zinst = z at a lag (independent observation
               noise; see ``fit_lds(..., instrument=...)``).
    time     : optional (T,) time axis; if given, the true sample time is returned as ``tsec``.

    Returns
    -------
    dict with Z (n,D), V (n,D), group (n,), tindex (n,) = ORIGINAL time-bin index of each
    velocity sample, and (optionally) Zinst and tsec.
    """
    from scipy.ndimage import gaussian_filter1d

    Zs, Vs, Gs, Ts, Is = [], [], [], [], []
    for i, zt in enumerate(Z_trials):
        zt = np.asarray(zt, float)
        if zt.shape[0] < gap + inst_lag + 2:
            continue
        if causal:
            zz = _causal_boxcar(zt, smooth_bins) if smooth_bins else zt
            v = (zz[gap:] - zz[:-gap]) / (gap * dt)      # backward difference...
            z = zz[gap:]                                 # ...attributed to the CURRENT sample
            idx = np.arange(gap, zt.shape[0])            # its original time index
            zi = zz[gap - inst_lag: len(zz) - inst_lag] if inst_lag else None
            if zi is not None:
                zi = zi[: len(z)]
        else:
            zz = gaussian_filter1d(zt, smooth_bins, axis=0, mode="nearest") if smooth_bins else zt
            v = (zz[gap:] - zz[:-gap]) / (gap * dt)      # forward difference...
            z = zz[:-gap]                                # ...attributed to the LEFT endpoint
            idx = np.arange(z.shape[0])
            zi = None
            if inst_lag > 0:
                zi = zz[:-gap - inst_lag] if gap + inst_lag else zz
                z, v, idx = z[inst_lag:], v[inst_lag:], idx[inst_lag:]
                zi = zi[: len(z)]
        if keep is not None:
            m = np.asarray(keep[i], bool)[idx]
            if m.sum() < 2:
                continue
            z, v, idx = z[m], v[m], idx[m]
            if zi is not None:
                zi = zi[m]
        Zs.append(z); Vs.append(v); Gs.append(np.full(len(z), i)); Ts.append(idx)
        if zi is not None:
            Is.append(zi)
    if not Zs:
        return dict(Z=np.zeros((0, 1)), V=np.zeros((0, 1)), group=np.zeros(0, int),
                    tindex=np.zeros(0, int))
    out = dict(Z=np.vstack(Zs), V=np.vstack(Vs),
               group=np.concatenate(Gs), tindex=np.concatenate(Ts))
    if Is:
        out["Zinst"] = np.vstack(Is)
    if time is not None:
        out["tsec"] = np.asarray(time, float)[out["tindex"]]
    return out


def r2_score_multi(Y, Yhat):
    """Variance-weighted R^2 over all output dimensions jointly (matches existing demos)."""
    Y = np.asarray(Y, float); Yhat = np.asarray(Yhat, float)
    ss_res = np.sum((Y - Yhat) ** 2)
    ss_tot = np.sum((Y - Y.mean(0)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-300))


def _ridge_solve(D, Y, reg):
    """Solve (D^T D + diag(reg)) W = D^T Y."""
    reg = np.atleast_1d(np.asarray(reg, float)).ravel()
    if reg.size == 1:
        reg = np.full(D.shape[1], reg[0])
    return np.linalg.solve(D.T @ D + np.diag(reg), D.T @ Y)


def _cv_r2(design_fn, Y, groups, reg, n_splits=5):
    """Trial-grouped CV R^2 for a linear model whose design is a fixed matrix."""
    D = design_fn
    groups = np.asarray(groups)
    n_splits = int(min(n_splits, len(np.unique(groups))))
    if n_splits < 2:
        return float("nan")
    out = []
    for tr, te in GroupKFold(n_splits).split(D, Y, groups=groups):
        W = _ridge_solve(D[tr], Y[tr], reg)
        out.append(r2_score_multi(Y[te], D[te] @ W))
    return float(np.mean(out))


def shuffle_null_r2(fit_fn, Z, V, groups, n_shuffle=20, seed=0):
    """Within-trial roll-shuffle null: roll V within each trial, destroying the z->v map.

    ``fit_fn(Z, V, groups) -> cv_r2``.  Returns (mean, sd) of the null CV R^2.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    vals = []
    for _ in range(n_shuffle):
        Vs = V.copy()
        for g in np.unique(groups):
            m = np.where(groups == g)[0]
            if len(m) > 2:
                Vs[m] = np.roll(V[m], int(rng.integers(1, len(m))), axis=0)
        vals.append(fit_fn(Z, Vs, groups))
    return float(np.mean(vals)), float(np.std(vals))


# ======================================================================================
# M1: global conditioned LDS
# ======================================================================================
def fit_lds(Z, V, groups=None, ridge=1e-3, cv=True, instrument=None):
    """dz/dt = A z + b, ridge least squares (the intercept is NOT penalized).

    Returns dict: A, b, cv_r2, r2_in, fixed_point, eig, strength, rotation, n.
      strength = -mean Re eig(A)   (>0: contracting toward the fixed point)
      rotation = median |Im eig(A)| (rad/s)

    ``instrument`` (n, D): if given, A and b are estimated by INSTRUMENTAL VARIABLES
    (two-stage least squares) instead of OLS,

        A_iv = solve( Winst^T Z_aug ,  Winst^T V )

    This removes the errors-in-variables bias.  Observation noise eps enters the regressor
    z[t] = z_true[t] + eps[t] and, with the opposite sign, the finite difference
    (z[t+gap] - z[t])/(gap*dt), so E[z^T v] acquires a term -Var(eps)/(gap*dt) * I: OLS
    manufactures a SYMMETRIC contraction that is pure measurement noise, inflating both the
    apparent attractor strength and the apparent "gradient" fraction of the Helmholtz split.
    A lagged state has independent observation noise and is a valid instrument.
    """
    Z = np.asarray(Z, float); V = np.asarray(V, float)
    D = np.hstack([Z, np.ones((len(Z), 1))])
    reg = np.concatenate([np.full(Z.shape[1], float(ridge)), [0.0]])
    if instrument is None:
        W = _ridge_solve(D, V, reg)
        Wi = None
    else:
        # two-stage least squares; handles an over-identified instrument (more columns than D)
        Wi = np.hstack([np.asarray(instrument, float), np.ones((len(Z), 1))])
        W = _iv_solve(D, Wi, V, reg)
    A, b = W[:-1].T, W[-1]
    ev = np.linalg.eigvals(A)
    try:
        zfix = np.linalg.solve(A, -b)
    except np.linalg.LinAlgError:
        zfix = np.full(Z.shape[1], np.nan)
    out = dict(A=A, b=b, eig=ev,
               strength=float(-np.mean(ev.real)),
               rotation=float(np.median(np.abs(ev.imag))),
               max_real=float(np.max(ev.real)),        # >0 => an UNSTABLE direction exists,
               fixed_point=zfix,                       # so strength>0 alone is not an attractor
               r2_in=r2_score_multi(V, D @ W), n=len(Z))
    if Wi is not None:
        # WEAK-INSTRUMENT DIAGNOSTIC.  2SLS with a weak instrument is biased BACK toward OLS,
        # so these first-stage R^2 values must be reported with any IV estimate.
        Dh = Wi @ np.linalg.solve(Wi.T @ Wi + 1e-8 * np.eye(Wi.shape[1]), Wi.T @ D)
        ss = np.sum((D[:, :-1] - D[:, :-1].mean(0)) ** 2, axis=0) + 1e-300
        out["first_stage_r2"] = 1 - np.sum((D[:, :-1] - Dh[:, :-1]) ** 2, axis=0) / ss
    if cv and groups is not None:
        out["cv_r2"] = (_cv_r2(D, V, groups, reg) if Wi is None
                        else _cv_r2_iv(D, Wi, V, groups, reg))
    else:
        out["cv_r2"] = float("nan")
    return out


def _iv_solve(D, Wi, V, reg):
    """Two-stage least squares.  Note the ridge shrinks toward ZERO, i.e. opposite in sign to
    the OLS errors-in-variables bias, so it cannot manufacture a rotation-dominant answer."""
    Dhat = Wi @ np.linalg.solve(Wi.T @ Wi + 1e-8 * np.eye(Wi.shape[1]), Wi.T @ D)
    return _ridge_solve(Dhat, V, reg)


def _cv_r2_iv(D, Wi, V, groups, reg, n_splits=5):
    """Trial-grouped CV for an IV fit: refit 2SLS on the training folds and score the held-out
    velocities with the resulting A, b.  (Scoring an IV fit with the OLS CV of the same design
    would report the OLS model's predictive power, not the IV model's.)"""
    groups = np.asarray(groups)
    n_splits = int(min(n_splits, len(np.unique(groups))))
    if n_splits < 2:
        return float("nan")
    out = []
    for tr, te in GroupKFold(n_splits).split(D, V, groups=groups):
        W = _iv_solve(D[tr], Wi[tr], V[tr], reg)
        out.append(r2_score_multi(V[te], D[te] @ W))
    return float(np.mean(out))


def fit_sliding_lds(Z, V, tindex, groups=None, centers=None, halfwidth=3,
                    ridge=1e-3, min_samples_per_dim=10):
    """M2: local LDS in a sliding time window -> A(tau), the local Jacobian through the trial.

    ``tindex`` is the time-bin index of each sample.  A fit is attempted at every centre in
    ``centers`` using samples with |tindex - centre| <= halfwidth; windows with fewer than
    ``min_samples_per_dim * D`` samples are skipped (reported as NaN) rather than fit badly.
    """
    Z = np.asarray(Z, float); V = np.asarray(V, float); tindex = np.asarray(tindex)
    Dd = Z.shape[1]
    if centers is None:
        centers = np.arange(tindex.min(), tindex.max() + 1)
    rows = []
    for c in centers:
        m = np.abs(tindex - c) <= halfwidth
        if m.sum() < min_samples_per_dim * Dd:
            rows.append(dict(center=int(c), n=int(m.sum()), A=None, b=None,
                             strength=np.nan, rotation=np.nan, max_real=np.nan,
                             fixed_point=np.full(Dd, np.nan), cv_r2=np.nan))
            continue
        g = None if groups is None else np.asarray(groups)[m]
        f = fit_lds(Z[m], V[m], groups=g, ridge=ridge, cv=g is not None)
        rows.append(dict(center=int(c), n=int(m.sum()), A=f["A"], b=f["b"],
                         strength=f["strength"], rotation=f["rotation"],
                         max_real=float(np.max(f["eig"].real)),
                         fixed_point=f["fixed_point"], cv_r2=f["cv_r2"]))
    return rows


# ======================================================================================
# M3: cubic-polynomial field and its linearization
# ======================================================================================
def fit_cubic_field(Z, V, degree=3, ridge=2.0, groups=None, cv=True):
    """Global nonlinear generator: dz/dt = F(z), F a polynomial vector field (ridge LSQ).

    A cubic field can hold two attractors plus a saddle, which an LDS cannot -- this is the
    generator required by INSTRUCTIONS.md; the LDS is then read off it by ``linearize_cubic``.
    """
    Z = np.asarray(Z, float); V = np.asarray(V, float)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    P = poly.fit_transform(Z)
    W = _ridge_solve(P, V, ridge)
    out = dict(W=W, poly=poly, degree=degree, r2_in=r2_score_multi(V, P @ W), n=len(Z))
    out["cv_r2"] = _cv_r2(P, V, groups, ridge) if (cv and groups is not None) else float("nan")
    return out


def cubic_predict(fit, z):
    """Evaluate the fitted polynomial field at states z (n,D) or (D,)."""
    z = np.atleast_2d(np.asarray(z, float))
    return fit["poly"].transform(z) @ fit["W"]


def linearize_cubic(fit, z0, eps=1e-3):
    """M3: local LDS from the cubic field at z0 -- A = dF/dz|_{z0} (central differences).

    Returns the same readouts as ``fit_lds`` so the three model classes are directly comparable.
    """
    z0 = np.asarray(z0, float).ravel()
    Dd = z0.size
    h = eps * max(np.linalg.norm(z0) / max(np.sqrt(Dd), 1.0), 1.0)
    A = np.zeros((Dd, Dd))
    for j in range(Dd):
        e = np.zeros(Dd); e[j] = h
        A[:, j] = (cubic_predict(fit, z0 + e)[0] - cubic_predict(fit, z0 - e)[0]) / (2 * h)
    b = cubic_predict(fit, z0)[0] - A @ z0
    ev = np.linalg.eigvals(A)
    try:
        zfix = np.linalg.solve(A, -b)
    except np.linalg.LinAlgError:
        zfix = np.full(Dd, np.nan)
    return dict(A=A, b=b, eig=ev, strength=float(-np.mean(ev.real)),
                rotation=float(np.median(np.abs(ev.imag))), fixed_point=zfix, z0=z0)


def cubic_fixed_point(fit, z0, n_iter=60, tol=1e-9):
    """Newton solve F(z)=0 starting from z0 (uses the numeric Jacobian)."""
    z = np.asarray(z0, float).ravel().copy()
    for _ in range(n_iter):
        F = cubic_predict(fit, z)[0]
        if np.linalg.norm(F) < tol:
            break
        J = linearize_cubic(fit, z)["A"]
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break
        z = z + np.clip(step, -1.0, 1.0)
    return z


# ======================================================================================
# metric-aware potential / rotational split
# ======================================================================================
def helmholtz_split(A, b=None, M=None, Z=None, V=None):
    """Split A into an M-gradient part and an M-rotational part.

        A_grad = M^-1 Sym(MA),  A_rot = M^-1 Skew(MA)
        V(z)   = -1/2 z^T Sym(MA) z - (M b)^T z          so   A_grad z + b = -M^-1 grad V

    NOTE on interpretation: A_rot conserves the M-energy (1/2) z^T M z, but it does NOT conserve
    V.  This is an M-orthogonal decomposition of the generator, not an energy-conserving
    metriplectic split: the rotational part does carry the state across level sets of V.

    M defaults to the identity (the classic S = (A+A^T)/2, W = (A-A^T)/2 split).
    Fractions are computed on MA -- the object that actually lives in the metric -- so results
    under different metrics are on the same footing; grad_frac^2 + rot_frac^2 = 1.

    If measured (Z, V) are supplied, also returns the median M-cosine between the measured
    velocity and the downhill direction -M^-1 grad V (>0 means the motion descends V).
    """
    A = np.asarray(A, float)
    Dd = A.shape[0]
    M = np.eye(Dd) if M is None else np.asarray(M, float)
    b = np.zeros(Dd) if b is None else np.asarray(b, float).ravel()
    if np.linalg.eigvalsh((M + M.T) / 2).min() <= 0:
        raise ValueError("helmholtz_split: the metric M must be symmetric positive definite "
                         "(otherwise 'downhill' and the conserved M-energy are meaningless)")

    MA = M @ A
    Sym = (MA + MA.T) / 2
    Skew = (MA - MA.T) / 2
    Minv = np.linalg.inv(M)
    A_grad = Minv @ Sym
    A_rot = Minv @ Skew

    nMA = np.linalg.norm(MA) + 1e-300
    out = dict(M=M, A_grad=A_grad, A_rot=A_rot, Sym=Sym, Skew=Skew,
               grad_frac=float(np.linalg.norm(Sym) / nMA),
               rot_frac=float(np.linalg.norm(Skew) / nMA),
               eig_sym=np.linalg.eigvalsh(Sym),          # potential curvature; <0 => convex bowl
               eig_skew=np.linalg.eigvals(Skew),         # pure imaginary => rotation rates
               convex=bool(np.all(np.linalg.eigvalsh(Sym) < 0)),
               b=b)

    def gradV(z):
        z = np.atleast_2d(np.asarray(z, float))
        return -(z @ Sym.T) - (M @ b)          # grad V(z) = -Sym(MA) z - M b
    out["gradV"] = gradV
    out["V"] = lambda z: potential(np.atleast_2d(z), Sym, M @ b)

    if Z is not None and V is not None and len(Z):
        Z = np.asarray(Z, float); V = np.asarray(V, float)
        down = -(Z @ Sym.T) - (M @ b)          # = grad V
        down = -(Minv @ down.T).T              # downhill direction -M^-1 grad V
        num = np.einsum("ni,ij,nj->n", V, M, down)
        nv = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", V, M, V), 1e-300))
        nd = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", down, M, down), 1e-300))
        out["cos_descend"] = float(np.median(num / (nv * nd)))
    return out


def potential(Z, Sym, Mb):
    """V(z) = -1/2 z^T Sym z - (Mb)^T z."""
    Z = np.atleast_2d(np.asarray(Z, float))
    return -0.5 * np.einsum("ni,ij,nj->n", Z, Sym, Z) - Z @ np.asarray(Mb, float).ravel()


def metric_dependence(A, b, Sigma, alphas=None, Z=None, V=None, eps=1e-6):
    """How much is 'the flow is mostly gradient' a property of the flow vs of the metric?

    Interpolates M(alpha) = (1-alpha) I + alpha Sigma^-1 (both normalized to unit mean
    eigenvalue so alpha is a fair mixture) and reports grad_frac along the path.
    """
    Dd = A.shape[0]
    Sigma = np.asarray(Sigma, float) + eps * np.eye(Dd)
    Sinv = np.linalg.inv(Sigma)
    Sinv = Sinv / np.mean(np.linalg.eigvalsh(Sinv))
    I = np.eye(Dd)
    alphas = np.linspace(0, 1, 11) if alphas is None else np.asarray(alphas, float)
    rows = []
    for a in alphas:
        M = (1 - a) * I + a * Sinv
        s = helmholtz_split(A, b, M, Z=Z, V=V)
        rows.append(dict(alpha=float(a), grad_frac=s["grad_frac"], rot_frac=s["rot_frac"],
                         convex=s["convex"], cos_descend=s.get("cos_descend", np.nan)))
    return rows


# ======================================================================================
# input inference: shared field + per-condition input
# ======================================================================================
def _input_bins(tau_s, bin_s):
    """Map time-since-onset (s) to non-negative input bins; -1 for pre-onset samples."""
    k = np.floor(np.asarray(tau_s, float) / bin_s).astype(int)
    k[np.asarray(tau_s, float) < 0] = -1
    return k


def fit_shared_input_free(Z, V, cond, tau_s, groups=None, field="lds", input_bin_s=0.05,
                          ridge_field=1e-3, ridge_input=8.0, degree=3, cv=True):
    """I-A: ONE shared field + a per-condition, time-resolved, ONSET-ONLY input.

        dz/dt = F(z) + I_c(tau),    tau = time since cue onset, free for tau >= 0

    No cue duration is assumed anywhere (hard constraint): the input is a free ridge-penalized
    function of time-since-onset, and the data decide how long it lasts.
    ``field`` is 'lds' (F(z) = A z + b) or 'cubic' (polynomial field).

    Only ACROSS-condition input variation is identifiable -- a constant input is absorbed into
    the field's intercept -- so ``I_centered`` (mean over conditions removed) is the object to
    read, and ``input_variation`` summarizes its spread.
    """
    Z = np.asarray(Z, float); V = np.asarray(V, float)
    cond = np.asarray(cond); tau_s = np.asarray(tau_s, float)
    Dd = Z.shape[1]

    if field == "cubic":
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        Phi = poly.fit_transform(Z)
        reg_f = np.full(Phi.shape[1], float(max(ridge_field, 1e-3)))
    else:
        poly = None
        Phi = np.hstack([Z, np.ones((len(Z), 1))])
        reg_f = np.concatenate([np.full(Dd, float(ridge_field)), [0.0]])
    nf = Phi.shape[1]

    conds = np.unique(cond)
    kbin = _input_bins(tau_s, input_bin_s)
    post = kbin >= 0
    ktab = np.arange(0, (kbin[post].max() + 1) if post.any() else 0)
    col = {(c, k): j for j, (c, k) in enumerate([(c, k) for c in conds for k in ktab])}
    Din = np.zeros((len(Z), len(col)))
    for i in np.where(post)[0]:
        key = (cond[i], kbin[i])
        if key in col:
            Din[i, col[key]] = 1.0

    Des = np.hstack([Phi, Din])
    reg = np.concatenate([reg_f, np.full(Din.shape[1], float(ridge_input))])
    W = _ridge_solve(Des, V, reg)

    Wf, Iflat = W[:nf], W[nf:]
    I = np.zeros((len(conds), len(ktab), Dd))
    for (c, k), j in col.items():
        I[list(conds).index(c), k] = Iflat[j]
    I_centered = I - I.mean(0, keepdims=True)

    field_fit = (dict(W=Wf, poly=poly, degree=degree) if field == "cubic"
                 else dict(A=Wf[:-1].T, b=Wf[-1]))
    out = dict(field_kind=field, field=field_fit, I=I, I_centered=I_centered,
               tau=(ktab + 0.5) * input_bin_s, conds=conds, design=Des, W=W,
               r2_in=r2_score_multi(V, Des @ W),
               input_norm=np.linalg.norm(I, axis=2),                 # (n_cond, n_tau)
               input_norm_centered=np.linalg.norm(I_centered, axis=2),
               impulse=I.sum(1) * input_bin_s)                        # (n_cond, D)
    out["input_variation"] = float(np.std(out["impulse"] - out["impulse"].mean(0)))
    out["cv_r2"] = _cv_r2(Des, V, groups, reg) if (cv and groups is not None) else float("nan")
    if field != "cubic":
        A, b = field_fit["A"], field_fit["b"]
        ev = np.linalg.eigvals(A)
        out["strength"] = float(-np.mean(ev.real)); out["rotation"] = float(np.median(np.abs(ev.imag)))
        out["eig"] = ev
    return out


def fit_shared_input_lowrank(Z, V, cond, tau_s, groups=None, rank=2, input_bin_s=0.05,
                             ridge_field=1e-3, ridge_input=1e-2, n_iter=40, init=None,
                             tol=1e-9, cv=True, seed=0):
    """I-B: shared field + LOW-RANK input,  dz/dt = A z + b + B u_c(tau).

    B (D x r) is a shared input subspace; u_c(tau) is an r-vector per (condition, time bin).
    With r = 2 this is the paper's picture directly: B spans the cue-response plane, ||u_c||
    is cue AMPLITUDE and the direction of u_c within the plane is cue ANGLE.

    Fit by alternating least squares (jointly linear in (A, b, B) given u; linear in u given
    B), initialized from the SVD of a free-input fit unless ``init`` supplies a B.
    """
    Z = np.asarray(Z, float); V = np.asarray(V, float)
    cond = np.asarray(cond); tau_s = np.asarray(tau_s, float)
    n, Dd = Z.shape
    conds = np.unique(cond)
    kbin = _input_bins(tau_s, input_bin_s)
    post = kbin >= 0
    n_tau = int(kbin[post].max() + 1) if post.any() else 0
    cidx = np.searchsorted(conds, cond)

    if init is not None:
        B = np.asarray(init, float)[:, :rank].copy()
    else:
        free = fit_shared_input_free(Z, V, cond, tau_s, groups=None, field="lds",
                                     input_bin_s=input_bin_s, ridge_field=ridge_field,
                                     ridge_input=8.0, cv=False)
        Ifl = free["I"].reshape(-1, Dd)
        if np.allclose(Ifl, 0):
            B = np.linalg.qr(np.random.default_rng(seed).normal(size=(Dd, rank)))[0]
        else:
            B = np.linalg.svd(Ifl, full_matrices=False)[2][:rank].T
    B = np.asarray(B, float).reshape(Dd, rank)

    n_tau_eff = max(n_tau, 1)
    slot = cidx * n_tau_eff + np.clip(kbin, 0, None)
    cells = [np.where(post & (slot == s))[0] for s in range(len(conds) * n_tau_eff)]

    # initialize the field from an input-free LDS fit
    f0 = fit_lds(Z, V, ridge=ridge_field, cv=False)
    A, b = f0["A"], f0["b"]
    U = np.zeros((len(conds), n_tau_eff, rank))
    reg2 = np.concatenate([np.full(Dd, float(ridge_field)), [0.0], np.full(rank, 1e-6)])
    prev, obj = np.inf, np.inf
    for _ in range(n_iter):
        # --- step 1: u | A, b, B  (ridge least squares on B within each (cond, tau) cell) ---
        Rres = V - Z @ A.T - b
        Uf = U.reshape(-1, rank)
        BtB = B.T @ B
        for s, m in enumerate(cells):
            if len(m) == 0:
                Uf[s] = 0.0
                continue
            Uf[s] = np.linalg.solve(BtB + (ridge_input / len(m)) * np.eye(rank),
                                    B.T @ Rres[m].mean(0))
        U = Uf.reshape(len(conds), n_tau_eff, rank)

        # --- step 2: (A, b, B) | u  (jointly linear in all three) ---
        Ur = np.zeros((n, rank))
        Ur[post] = Uf[slot[post]]
        Des = np.hstack([Z, np.ones((n, 1)), Ur])
        W = _ridge_solve(Des, V, reg2)
        A, b, B = W[:Dd].T, W[Dd], W[Dd + 1:].T

        obj = float(np.sum((V - Des @ W) ** 2))
        if abs(prev - obj) < tol * max(prev, 1.0):
            break
        prev = obj

    # gauge fix: orthonormalize B, push the scale into u (B u is unchanged)
    Q, R = np.linalg.qr(B)
    sgn = np.sign(np.diag(R)); sgn[sgn == 0] = 1.0
    Q, R = Q * sgn, (sgn[:, None] * R)
    B, U = Q, np.einsum("ij,ctj->cti", R, U)

    Ur = np.zeros((n, rank)); Ur[post] = U.reshape(-1, rank)[slot[post]]
    Des = np.hstack([Z, np.ones((n, 1)), Ur])
    W = np.vstack([A.T, b[None, :], B.T])
    ev = np.linalg.eigvals(A)
    out = dict(A=A, b=b, B=B, U=U, conds=conds, tau=(np.arange(max(n_tau, 1)) + 0.5) * input_bin_s,
               rank=rank, r2_in=r2_score_multi(V, Des @ W),
               amplitude=np.linalg.norm(U, axis=2),                 # (n_cond, n_tau)
               eig=ev, strength=float(-np.mean(ev.real)),
               rotation=float(np.median(np.abs(ev.imag))),
               I=np.einsum("dr,ctr->ctd", B, U))
    out["I_centered"] = out["I"] - out["I"].mean(0, keepdims=True)
    out["input_variation"] = float(np.std(out["I"].sum(1) - out["I"].sum(1).mean(0)))
    if rank >= 2:
        # gauge note: the QR fix pins the sign of diag(R) but not the orientation of the
        # B-plane, so only RELATIVE angles between conditions are meaningful (those are exactly
        # invariant); the absolute value shifts with the initialization.
        out["angle"] = np.arctan2(U[..., 1], U[..., 0])
    if cv and groups is not None:
        # honest CV: refit the WHOLE alternating fit (including u) on the training folds.
        # Reusing a U estimated on all samples leaks the held-out data and lets a spurious
        # input model beat the no-input baseline.
        groups = np.asarray(groups)
        ns = int(min(5, len(np.unique(groups))))
        sc = []
        if ns >= 2:
            for tr, te in GroupKFold(ns).split(Z, V, groups=groups):
                f = fit_shared_input_lowrank(Z[tr], V[tr], cond[tr], tau_s[tr], groups=None,
                                             rank=rank, input_bin_s=input_bin_s,
                                             ridge_field=ridge_field, ridge_input=ridge_input,
                                             n_iter=n_iter, cv=False, seed=seed)
                kb_te = _input_bins(tau_s[te], input_bin_s)
                ci_te = np.searchsorted(f["conds"], cond[te])
                Ute = np.zeros((len(te), rank))
                okm = (kb_te >= 0) & (kb_te < f["U"].shape[1]) & \
                      (np.isin(cond[te], f["conds"]))
                Ute[okm] = f["U"][ci_te[okm], kb_te[okm]]
                pred = Z[te] @ f["A"].T + f["b"] + Ute @ f["B"].T
                sc.append(r2_score_multi(V[te], pred))
        out["cv_r2"] = float(np.mean(sc)) if sc else float("nan")
    else:
        out["cv_r2"] = float("nan")
    return out


def velocity_budget(Z, V, A, b, I_samples):
    """Per-timepoint split of the measured velocity into field-driven and input-driven parts.

    ``I_samples`` is the fitted input at each sample (n, D) -- zero before cue onset.
    Returns, per sample:
      field_share / input_share : projections onto the measured velocity, <v, term>/||v||^2
                                  (they sum to ~1 where the model fits well),
      field_norm / input_norm   : magnitudes of the two terms.
    """
    Z = np.asarray(Z, float); V = np.asarray(V, float); I_samples = np.asarray(I_samples, float)
    Fterm = Z @ np.asarray(A, float).T + np.asarray(b, float)
    v2 = np.maximum(np.sum(V * V, axis=1), 1e-300)
    return dict(field_share=np.sum(V * Fterm, axis=1) / v2,
                input_share=np.sum(V * I_samples, axis=1) / v2,
                field_norm=np.linalg.norm(Fterm, axis=1),
                input_norm=np.linalg.norm(I_samples, axis=1),
                field_term=Fterm, input_term=I_samples)


# ======================================================================================
# behavior link
# ======================================================================================
def behavior_relevant_fraction(component, g, k=1, eps=1e-12):
    """Fraction of a dynamical component that lies in the behavior-relevant subspace.

    ``g`` is the (state-space) pullback metric g = J^T g_Y J; its top-k eigenvectors span the
    directions behavior is sensitive to.  Returns ||P_g x|| / ||x|| per row of ``component``.
    Chance level for random vectors in D dimensions is sqrt(k/D) -- report it alongside.
    """
    component = np.atleast_2d(np.asarray(component, float))
    w, Vv = np.linalg.eigh(np.asarray(g, float))
    P = Vv[:, -k:]
    proj = component @ P
    num = np.linalg.norm(proj, axis=1)
    den = np.linalg.norm(component, axis=1) + eps
    return num / den


# ======================================================================================
# self-test
# ======================================================================================
def _selftest():
    rng = np.random.default_rng(0)
    print("=" * 78)
    print("1) analytic split recovery on a known A (Euclidean metric)")
    S_true = -np.diag([1.0, 2.0, 0.5])
    W_true = np.array([[0, 3.0, 0], [-3.0, 0, 0], [0, 0, 0]])
    A_true = S_true + W_true
    sp = helmholtz_split(A_true)
    assert np.allclose(sp["Sym"], S_true, atol=1e-10), sp["Sym"]
    assert np.allclose(sp["Skew"], W_true, atol=1e-10)
    assert sp["convex"]
    print(f"   Sym/Skew recovered exactly; grad_frac={sp['grad_frac']:.3f} "
          f"rot_frac={sp['rot_frac']:.3f} (sum of squares={sp['grad_frac']**2+sp['rot_frac']**2:.3f})")

    print("2) metric-aware split: a flow that is PURE GRADIENT under M, not under I")
    M = np.array([[4.0, 2.2, 0.0], [2.2, 1.6, 0.0], [0.0, 0.0, 1.0]])
    H = np.diag([1.0, 6.0, 2.0])
    A_M = -np.linalg.inv(M) @ H                    # exact M-gradient flow
    e = helmholtz_split(A_M, M=np.eye(3))
    m = helmholtz_split(A_M, M=M)
    print(f"   grad_frac under Euclidean = {e['grad_frac']:.3f}   under M = {m['grad_frac']:.3f}")
    assert m["grad_frac"] > 0.999 and e["grad_frac"] < 0.999
    assert np.allclose(m["Skew"], 0, atol=1e-10)
    print("   -> confirms the split is metric-dependent (the point of metric_dependence()).")

    print("3) LDS recovery from simulated trajectories + CV + shuffle null")
    Dd, dt, nT, T = 3, 0.02, 60, 120
    b_true = np.array([0.3, -0.2, 0.1])
    trials = []
    for i in range(nT):
        z = rng.normal(size=Dd) * 0.5
        zs = [z]
        for _ in range(T - 1):
            z = z + dt * (A_true @ z + b_true) + np.sqrt(dt) * 0.02 * rng.normal(size=Dd)
            zs.append(z)
        trials.append(np.array(zs))
    d = trial_velocities(trials, dt, gap=2, smooth_bins=0.0)
    f = fit_lds(d["Z"], d["V"], groups=d["group"])
    err = np.linalg.norm(f["A"] - A_true) / np.linalg.norm(A_true)
    print(f"   ||A_hat - A||/||A|| = {err:.4f}, CV R2 = {f['cv_r2']:.3f}, "
          f"strength = {f['strength']:.3f} (true {-np.mean(np.linalg.eigvals(A_true).real):.3f}), "
          f"rotation = {f['rotation']:.2f} (true 3.00)")
    assert err < 0.05 and f["cv_r2"] > 0.9
    null_m, null_s = shuffle_null_r2(
        lambda Z, V, g: fit_lds(Z, V, groups=g)["cv_r2"], d["Z"], d["V"], d["group"], n_shuffle=5)
    print(f"   roll-shuffle null CV R2 = {null_m:+.3f} +/- {null_s:.3f} (should be ~0 or negative)")
    assert null_m < 0.2

    print("4) cubic field + linearization agrees with the LDS on linear data")
    cf = fit_cubic_field(d["Z"], d["V"], degree=3, ridge=1e-2, groups=d["group"])
    lin = linearize_cubic(cf, d["Z"].mean(0))
    print(f"   cubic CV R2 = {cf['cv_r2']:.3f}; ||A_cubic - A_true||/||A|| = "
          f"{np.linalg.norm(lin['A'] - A_true)/np.linalg.norm(A_true):.4f}")
    assert np.linalg.norm(lin["A"] - A_true) / np.linalg.norm(A_true) < 0.15

    print("5) input inference: shared field + per-condition input, both parameterizations")
    Btrue = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    amps = [0.4, 1.0, 1.8]                      # per-condition cue amplitude (input mechanism)
    ang = 25 * np.pi / 180
    Ztr, Vtr, Ctr, TAUtr, Gtr = [], [], [], [], []
    for i in range(90):
        c = i % 3
        z = rng.normal(size=Dd) * 0.2
        zs, us = [z], []
        for k in range(T - 1):
            t = k * dt
            u = (amps[c] * np.array([np.cos(ang), np.sin(ang)])) if 0 <= t < 0.2 else np.zeros(2)
            us.append(u)
            z = z + dt * (A_true @ z + b_true + Btrue @ u) + np.sqrt(dt) * 0.02 * rng.normal(size=Dd)
            zs.append(z)
        zs = np.array(zs)
        v = (zs[2:] - zs[:-2]) / (2 * dt)
        Ztr.append(zs[:-2]); Vtr.append(v); Ctr.append(np.full(len(v), c))
        TAUtr.append(np.arange(len(v)) * dt); Gtr.append(np.full(len(v), i))
    Zi = np.vstack(Ztr); Vi = np.vstack(Vtr); Ci = np.concatenate(Ctr)
    TAUi = np.concatenate(TAUtr); Gi = np.concatenate(Gtr)

    fa = fit_shared_input_free(Zi, Vi, Ci, TAUi, groups=Gi, input_bin_s=0.05, ridge_input=1e-2)
    peak = fa["input_norm"][:, 0]
    print(f"   I-A  CV R2 = {fa['cv_r2']:.3f}; recovered |I| at onset per condition = "
          f"{np.round(peak, 2)}  (true amplitudes {amps})")
    print(f"        corr(recovered, true amplitude) = {np.corrcoef(peak, amps)[0,1]:+.3f}")
    assert np.corrcoef(peak, amps)[0, 1] > 0.95
    late = fa["input_norm"][:, -3:].mean()
    print(f"        temporal profile: |I| at onset {peak.mean():.2f} vs late {late:.2f} "
          f"-> transient ratio {peak.mean()/max(late,1e-6):.1f}x (transient = a real input)")

    fb = fit_shared_input_lowrank(Zi, Vi, Ci, TAUi, groups=Gi, rank=2, input_bin_s=0.05)
    amp_lr = fb["amplitude"][:, 0]
    sub = np.abs(np.linalg.svd(Btrue.T @ fb["B"], compute_uv=False))
    print(f"   I-B  CV R2 = {fb['cv_r2']:.3f}; rank-2 amplitudes = {np.round(amp_lr, 2)}; "
          f"corr with truth = {np.corrcoef(amp_lr, amps)[0,1]:+.3f}")
    print(f"        input-subspace alignment with the true B (principal cosines) = {np.round(sub, 3)}")
    assert np.corrcoef(amp_lr, amps)[0, 1] > 0.95
    assert sub.min() > 0.9

    print("6) velocity budget + behavior projection")
    Isamp = np.zeros_like(Zi)
    kb = _input_bins(TAUi, 0.05)
    for i in np.where(kb >= 0)[0]:
        ci = list(fa["conds"]).index(Ci[i])
        if kb[i] < fa["I"].shape[1]:
            Isamp[i] = fa["I"][ci, kb[i]]
    bud = velocity_budget(Zi, Vi, fa["field"]["A"], fa["field"]["b"], Isamp)
    early = TAUi < 0.2; late_m = TAUi > 0.6
    print(f"   input share of velocity: during cue {np.median(bud['input_share'][early]):+.2f}, "
          f"later {np.median(bud['input_share'][late_m]):+.2f}")
    print(f"   field share of velocity: during cue {np.median(bud['field_share'][early]):+.2f}, "
          f"later {np.median(bud['field_share'][late_m]):+.2f}")
    assert np.median(bud["input_share"][early]) > np.median(bud["input_share"][late_m])

    g = np.outer([1.0, 0, 0], [1.0, 0, 0])       # behavior sensitive only to dim 0
    frac_grad = behavior_relevant_fraction(Zi @ sp["A_grad"].T, g, k=1).mean()
    frac_rot = behavior_relevant_fraction(Zi @ sp["A_rot"].T, g, k=1).mean()
    print(f"   behavior-relevant fraction: gradient part {frac_grad:.2f}, rotational part "
          f"{frac_rot:.2f}, chance = {np.sqrt(1/3):.2f}")

    print("7) OBSERVATION-NOISE BIAS: finite-difference OLS invents symmetric contraction")
    tru = helmholtz_split(A_true)
    print(f"   truth: strength={-np.mean(np.linalg.eigvals(A_true).real):.2f}, "
          f"grad_frac={tru['grad_frac']:.2f}, rot_frac={tru['rot_frac']:.2f}, "
          f"convex={tru['convex']}")
    for obs in (0.0, 0.05, 0.15):
        noisy = [tr + obs * rng.normal(size=tr.shape) for tr in trials]
        do = trial_velocities(noisy, dt, gap=2, smooth_bins=0.0)
        fo = fit_lds(do["Z"], do["V"], cv=False)
        so = helmholtz_split(fo["A"])
        di = trial_velocities(noisy, dt, gap=2, smooth_bins=0.0, inst_lag=3)
        fi = fit_lds(di["Z"], di["V"], cv=False, instrument=di["Zinst"])
        si = helmholtz_split(fi["A"])
        print(f"   obs noise {obs:.2f}:  OLS strength={fo['strength']:6.2f} "
              f"grad={so['grad_frac']:.2f} convex={str(so['convex']):5s} | "
              f"IV  strength={fi['strength']:6.2f} grad={si['grad_frac']:.2f} "
              f"err={np.linalg.norm(fi['A']-A_true)/np.linalg.norm(A_true):.3f}")
    # with substantial observation noise, OLS must be badly biased and IV must not be
    noisy = [tr + 0.15 * rng.normal(size=tr.shape) for tr in trials]
    do = trial_velocities(noisy, dt, gap=2, smooth_bins=0.0)
    di = trial_velocities(noisy, dt, gap=2, smooth_bins=0.0, inst_lag=3)
    e_ols = np.linalg.norm(fit_lds(do["Z"], do["V"], cv=False)["A"] - A_true)
    e_iv = np.linalg.norm(fit_lds(di["Z"], di["V"], cv=False,
                                  instrument=di["Zinst"])["A"] - A_true)
    print(f"   ||A_hat - A_true||:  OLS {e_ols:.2f}  vs  IV {e_iv:.2f}")
    assert e_iv < 0.5 * e_ols

    print("=" * 78)
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _selftest()
