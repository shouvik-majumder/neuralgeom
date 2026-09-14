"""A plain, held-out regression suite for the dynamics: every X -> Y stated explicitly, with a
shuffle control, evaluated separately for pre-cue and post-cue time bins.

Two groups of regressions (see the demos for the figures):

  PREDICT VELOCITY  (target = measured dz/dt; regressor = neural state z)
    - full flow    A       dz/dt ~ A z + b
    - gradient     S       dz/dt ~ S z + b        (S = (A+A^T)/2)
    - rotation     W       dz/dt ~ W z            (W = (A-A^T)/2)
    - S+W                  dz/dt ~ (S z + b)+(W z) = A z + b   (exact-additivity check == full A)
    shuffle: the flow is refit on velocities time-rolled within each trial.

  PREDICT TIME-TO-LICK  (target = known time-to-lick = lick - t; regressor as named)
    - state z
    - velocity full     v_A = A z + b
    - velocity gradient v_S = S z + b
    - velocity rotation v_W = W z
    - velocity S+W      (= v_A)
    a ridge decoder is fit regressor -> time-to-lick.
    shuffle: the regressor is time-rolled within each trial before fitting.

The flow A (and its S, W parts) and the decoders are fit SEPARATELY within each epoch: a
pre-cue model on pre-cue training samples, a post-cue model on post-cue training samples, each
scored on its own held-out samples.  Also returned, for the raw 'target vs regressor' scatter,
is the regressor's leading component vs the target with its Pearson correlation, per epoch.

``time_to_lick_regression`` (separate) asks how much of the time to lick is predictable, and from
what: linear vs nonlinear (quadratic) state, and whether adding velocity / acceleration helps
(if they do, the observed state is a partial observation and recent history matters).
"""

from __future__ import annotations

import numpy as np

from .lds import trial_velocities, fit_lds

__all__ = ["run_regression_suite", "time_to_lick_regression"]


def _r2(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    return float(1 - np.sum((y - yh) ** 2) / (np.sum((y - y.mean(0)) ** 2) + 1e-12))


def _roll_within(M, groups, rng):
    """Circularly shift rows within each trial (group) by a random offset."""
    M = np.array(M, float)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) > 2:
            M[idx] = np.roll(M[idx], int(rng.integers(1, len(idx))), axis=0)
    return M


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def run_regression_suite(Z, lick, t, use, bin_s=0.05, base=-0.6, gap=2, smooth_bins=1.0,
                         n_splits=5, ridge=1.0, seed=0, scatter_n=2500):
    """Return {'velocity': [...], 'time_to_lick': [...]}; see module docstring.  Each entry is a dict with
    held-out CV R2 and shuffle R2 (mean +/- sd) for pre-cue and post-cue, the target-vs-regressor
    correlation per epoch, and pooled held-out points for the scatters."""
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge

    use = np.asarray(use)
    rng = np.random.default_rng(seed)

    def samples(idx):
        trials = [Z[i] for i in idx]
        keep = [(t >= base) & (t < lick[i] + 0.05) for i in idx]
        d = trial_velocities(trials, bin_s, gap=gap, smooth_bins=smooth_bins, causal=True,
                             time=t, keep=keep)
        tid = np.asarray(idx)[d["group"]]
        time_to_lick = lick[tid] - d["tsec"]
        return dict(z=d["Z"], v=d["V"], time_to_lick=time_to_lick, grp=d["group"], tsec=d["tsec"],
                    post=d["tsec"] >= 0)

    vel_names = ["full flow  A", "gradient  S", "rotation  W", "S + W"]
    # NOTE: these are MODEL velocities (A z + b etc.) -- invertible linear maps of the state, so
    # regressing them on time-to-lick recovers the same R2 as decoding from the state directly.
    # (The MEASURED velocity dz/dt is a different object; see time_to_lick_regression.)
    time_to_lick_names = ["state z", "model velocity full  (A z + b)", "model velocity gradient  (S z + b)",
                 "model velocity rotation  (W z)", "model velocity  S+W"]
    # accumulators
    V = {n: dict(cv_pre=[], cv_post=[], sh_pre=[], sh_post=[],
                 sc_pre=([], []), sc_post=([], []), tvr=([], [], [])) for n in vel_names}
    T = {n: dict(cv_pre=[], cv_post=[], sh_pre=[], sh_post=[],
                 sc_pre=([], []), sc_post=([], []), tvr=([], [], [])) for n in time_to_lick_names}

    def comp(A, b):
        S = (A + A.T) / 2; W = (A - A.T) / 2
        return {"full flow  A": (A, b), "gradient  S": (S, b), "rotation  W": (W, 0 * b),
                "S + W": (A, b)}, S, W

    # Fit a SEPARATE flow + decoders within each epoch (pre-cue / post-cue) and score on that
    # epoch's held-out samples.
    for key in ("pre", "post"):
        for tri, tei in GroupKFold(n_splits).split(use, groups=use):
            tr = samples(use[tri]); te = samples(use[tei])
            trm = (~tr["post"]) if key == "pre" else tr["post"]
            tem = (~te["post"]) if key == "pre" else te["post"]
            if trm.sum() < 10 * Z.shape[-1] or tem.sum() < 5:
                continue
            ztr, vtr, gtr = tr["z"][trm], tr["v"][trm], tr["grp"][trm]
            zte, vte = te["z"][tem], te["v"][tem]
            A, b = (lambda f: (f["A"], f["b"]))(fit_lds(ztr, vtr))
            As, bs = (lambda f: (f["A"], f["b"]))(fit_lds(ztr, _roll_within(vtr, gtr, rng)))
            models, S, W = comp(A, b); models_sh, _, _ = comp(As, bs)

            for nm in vel_names:
                M, bc = models[nm]; Ms, bcs = models_sh[nm]
                pv = zte @ M.T + bc
                V[nm][f"cv_{key}"].append(_r2(vte, pv))
                V[nm][f"sh_{key}"].append(_r2(vte, zte @ Ms.T + bcs))
                V[nm][f"sc_{key}"][0].extend(vte.ravel()); V[nm][f"sc_{key}"][1].extend(pv.ravel())
                V[nm]["tvr"][0].extend(zte[:, 0]); V[nm]["tvr"][1].extend(vte[:, 0])
                V[nm]["tvr"][2].extend([1 if key == "post" else 0] * len(zte))

            def regressor(name, z):
                if name == "state z":
                    return z
                vA = z @ A.T + b; vS = z @ S.T + b; vW = z @ W.T
                return {"model velocity full  (A z + b)": vA,
                        "model velocity gradient  (S z + b)": vS,
                        "model velocity rotation  (W z)": vW, "model velocity  S+W": vA}[name]
            for nm in time_to_lick_names:
                Xtr = regressor(nm, ztr); Xte = regressor(nm, zte)
                yte = te["time_to_lick"][tem]
                dec = Ridge(ridge).fit(Xtr, tr["time_to_lick"][trm])
                decs = Ridge(ridge).fit(_roll_within(Xtr, gtr, rng), tr["time_to_lick"][trm])
                T[nm][f"cv_{key}"].append(_r2(yte, dec.predict(Xte)))
                T[nm][f"sh_{key}"].append(_r2(yte, decs.predict(Xte)))
                T[nm][f"sc_{key}"][0].extend(yte); T[nm][f"sc_{key}"][1].extend(dec.predict(Xte))
                T[nm]["tvr"][0].extend(Xte[:, 0]); T[nm]["tvr"][1].extend(yte)
                T[nm]["tvr"][2].extend([1 if key == "post" else 0] * len(yte))

    # ---- summarize ----
    def sub(a, b, c=None):
        a, b = np.asarray(a), np.asarray(b)
        if c is not None:
            c = np.asarray(c)
        if len(a) <= scatter_n:
            return (a, b) if c is None else (a, b, c)
        k = np.random.default_rng(0).choice(len(a), scatter_n, replace=False)
        return (a[k], b[k]) if c is None else (a[k], b[k], c[k])

    def pack(store, names, reg_label, tgt_label):
        out = []
        for nm in names:
            s = store[nm]
            xr, yt, ep = np.asarray(s["tvr"][0]), np.asarray(s["tvr"][1]), np.asarray(s["tvr"][2])
            out.append(dict(
                name=nm, regressor=reg_label(nm), target=tgt_label,
                cv_pre=float(np.mean(s["cv_pre"])) if s["cv_pre"] else np.nan,
                cv_post=float(np.mean(s["cv_post"])) if s["cv_post"] else np.nan,
                cv_pre_sd=float(np.std(s["cv_pre"])) if s["cv_pre"] else np.nan,
                cv_post_sd=float(np.std(s["cv_post"])) if s["cv_post"] else np.nan,
                sh_pre=float(np.mean(s["sh_pre"])) if s["sh_pre"] else np.nan,
                sh_post=float(np.mean(s["sh_post"])) if s["sh_post"] else np.nan,
                corr_pre=_pearson(xr[ep == 0], yt[ep == 0]),
                corr_post=_pearson(xr[ep == 1], yt[ep == 1]),
                scat_pre=sub(*s["sc_pre"]), scat_post=sub(*s["sc_post"]),
                tvr=sub(xr, yt, ep)))
        return out

    vel = pack(V, vel_names, lambda nm: "neural state z", "velocity dz/dt")
    time_to_lick = pack(T, time_to_lick_names, lambda nm: ("neural state z" if nm == "state z" else nm),
               "time-to-lick (s)")
    return {"velocity": vel, "time_to_lick": time_to_lick}


# ======================================================================================
# How much of time-to-lick is predictable, and from what?  (nonlinearity + dynamical order)
# ======================================================================================
def _state_vel_acc(Z, lick, t, idx, bin_s, base, gap, smooth_bins):
    """Per post-cue-window sample: state z, velocity v = dz/dt, acceleration a = d2z/dt2,
    time-to-lick, epoch flag, trial group -- all causal (past-only), so nothing leaks forward."""
    from scipy.ndimage import uniform_filter1d
    zs, vs, as_, time_to_lick_list, post, grp = [], [], [], [], [], []
    dt = gap * bin_s
    for gi, i in enumerate(idx):
        m = (t >= base) & (t < lick[i] + 0.05)
        z = np.asarray(Z[i])[m]; ti = t[m]
        if len(z) < 2 * gap + 2:
            continue
        w = int(max(smooth_bins, 1))
        zz = uniform_filter1d(z, w, axis=0, origin=(w - 1) // 2, mode="nearest") if w > 1 else z
        k = np.arange(2 * gap, len(zz))
        v = (zz[k] - zz[k - gap]) / dt
        a = (zz[k] - 2 * zz[k - gap] + zz[k - 2 * gap]) / dt ** 2
        zs.append(zz[k]); vs.append(v); as_.append(a)
        time_to_lick_list.append(lick[i] - ti[k]); post.append(ti[k] >= 0); grp.append(np.full(len(k), gi))
    return (np.vstack(zs), np.vstack(vs), np.vstack(as_), np.concatenate(time_to_lick_list),
            np.concatenate(post), np.concatenate(grp))


def time_to_lick_regression(Z, lick, t, use, bin_s=0.05, base=-0.6, gap=2, smooth_bins=2,
                       n_splits=5, seed=0):
    """Predict time-to-lick from increasingly rich features, fit and scored WITHIN each epoch:

        1. state z                        (linear)          -- is a linear code enough?
        2. state z, quadratic             (z + all z_i z_j) -- is the code nonlinear?
        3. state z + MEASURED velocity    ([z, dz/dt])      -- does the observed motion add over z?
        4. state z + meas. vel + meas. accel ([z, dz/dt, d2z/dt2])
        5. quadratic z + meas. vel + accel (everything)

    Here 'velocity' is the MEASURED finite difference dz/dt = (z[t]-z[t-gap])/dt taken from the
    data -- NOT the model velocity A z + b.  Because it depends on z[t] AND the past state
    z[t-gap], [z, dz/dt] is effectively a two-tap DELAY EMBEDDING.  So if (3)/(4) beat (1), the
    6-PC state is a PARTIAL observation and recent history carries extra timing information (the
    model velocity A z + b, being an invertible linear map of z, could never add -- see
    run_regression_suite).  If (2) beats (1), the time-to-lick map is nonlinear in the state.
    Ridge throughout; shuffle = time-to-lick permuted.  CV R2 (mean +/- sd) per epoch.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    use = np.asarray(use); rng = np.random.default_rng(seed)
    feats = ["state (linear)", "state (quadratic)", "state + meas. velocity",
             "state + meas. vel + accel", "quadratic + meas. vel + accel"]
    R = {f: dict(pre=[], post=[], sh_pre=[], sh_post=[]) for f in feats}

    def design(name, z, v, a, poly=None):
        if name == "state (linear)":
            return z
        if name == "state (quadratic)":
            return poly.transform(z)
        if name == "state + meas. velocity":
            return np.hstack([z, v])
        if name == "state + meas. vel + accel":
            return np.hstack([z, v, a])
        return np.hstack([poly.transform(z), v, a])

    for key in ("pre", "post"):
        for tri, tei in GroupKFold(n_splits).split(use, groups=use):
            ztr, vtr, atr, ytr, ptr, gtr = _state_vel_acc(Z, lick, t, use[tri], bin_s, base, gap, smooth_bins)
            zte, vte, ate, yte, pte, _ = _state_vel_acc(Z, lick, t, use[tei], bin_s, base, gap, smooth_bins)
            trm = (ptr if key == "post" else ~ptr); tem = (pte if key == "post" else ~pte)
            if trm.sum() < 50 or tem.sum() < 20:
                continue
            poly = PolynomialFeatures(2, include_bias=False).fit(ztr[trm])
            yshuf = rng.permutation(ytr[trm])
            for f in feats:
                sc = StandardScaler()
                Xtr = sc.fit_transform(design(f, ztr[trm], vtr[trm], atr[trm], poly))
                Xte = sc.transform(design(f, zte[tem], vte[tem], ate[tem], poly))
                dec = Ridge(1.0).fit(Xtr, ytr[trm])
                R[f][key].append(_r2(yte[tem], dec.predict(Xte)))
                R[f][f"sh_{key}"].append(_r2(yte[tem], Ridge(1.0).fit(Xtr, yshuf).predict(Xte)))
    return [dict(name=f,
                 pre=float(np.mean(R[f]["pre"])) if R[f]["pre"] else np.nan,
                 post=float(np.mean(R[f]["post"])) if R[f]["post"] else np.nan,
                 pre_sd=float(np.std(R[f]["pre"])) if R[f]["pre"] else np.nan,
                 post_sd=float(np.std(R[f]["post"])) if R[f]["post"] else np.nan,
                 sh_post=float(np.mean(R[f]["sh_post"])) if R[f]["sh_post"] else np.nan)
            for f in feats]
