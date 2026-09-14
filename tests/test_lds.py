"""neuralgeom.dynamics estimator tests: LDS fit, the IV correction for the
finite-difference velocity bias, the Helmholtz split, causal velocities, and
the regression suite.

Run:  python tests/test_lds.py     (or: pytest)
The full estimator self-test (with printed diagnostics) is
``python -m neuralgeom.dynamics.lds``.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import neuralgeom.dynamics as dyn                                  # noqa: E402


def _sim(nT=60, T=120, dt=0.02, obs=0.0, seed=0):
    rng = np.random.default_rng(seed)
    S = -np.diag([1.0, 2.0, 0.5]); W = np.array([[0, 3., 0], [-3., 0, 0], [0, 0, 0]])
    A = S + W; b = np.array([0.3, -0.2, 0.1])
    trials = []
    for _ in range(nT):
        z = rng.normal(size=3) * 0.5; zs = [z]
        for _ in range(T - 1):
            z = z + dt * (A @ z + b) + np.sqrt(dt) * 0.02 * rng.normal(size=3); zs.append(z)
        zs = np.array(zs)
        if obs:
            zs = zs + obs * rng.normal(size=zs.shape)
        trials.append(zs)
    return trials, A, dt


def test_lds_recovers_known_A():
    trials, A, dt = _sim()
    d = dyn.trial_velocities(trials, dt, gap=2, smooth_bins=0.0)
    f = dyn.fit_lds(d["Z"], d["V"], groups=d["group"])
    assert np.linalg.norm(f["A"] - A) / np.linalg.norm(A) < 0.06 and f["cv_r2"] > 0.9


def test_iv_beats_ols_under_noise():
    trials, A, dt = _sim(obs=0.15)
    d = dyn.trial_velocities(trials, dt, gap=2, smooth_bins=0.0)
    di = dyn.trial_velocities(trials, dt, gap=2, smooth_bins=0.0, inst_lag=3)
    e_ols = np.linalg.norm(dyn.fit_lds(d["Z"], d["V"])["A"] - A)
    e_iv = np.linalg.norm(dyn.fit_lds(di["Z"], di["V"], instrument=di["Zinst"])["A"] - A)
    assert e_iv < 0.5 * e_ols


def test_split_half_pca_is_a_valid_instrument():
    """End-to-end: noisy 'units' -> split-half PCA states -> IV fit beats OLS."""
    from neuralgeom.data import from_arrays, state_pca, state_pca_split_half
    rng = np.random.default_rng(1)
    trials, A, dt = _sim(nT=80)
    Zlat = np.stack(trials)                                    # (n, T, 3) true state
    L = rng.standard_normal((24, 3))
    X = np.einsum("ntd,md->ntm", Zlat, L) + 0.8 * rng.normal(size=(80, Zlat.shape[1], 24))
    s = from_arrays(X, np.arange(Zlat.shape[1]) * dt, name="iv-check")
    # USAGE RULE: each half's state instruments the OTHER half's regressor and
    # velocity. (Instrumenting Z_all with one of its own halves is invalid —
    # they share units, so their observation noise is correlated.)
    Zall, Za, Zb = state_pca_split_half(s, n_pc=3)
    db = dyn.trial_velocities(list(Zb), dt, gap=2, smooth_bins=0.0)
    da = dyn.trial_velocities(list(Za), dt, gap=2, smooth_bins=0.0)
    f_ols = dyn.fit_lds(db["Z"], db["V"], cv=False)
    f_iv = dyn.fit_lds(db["Z"], db["V"], cv=False, instrument=da["Z"])
    # OLS on noisy states invents symmetric contraction (inflated strength and
    # gradient fraction); the split-half instrument must undo most of it.
    true_mean_negative_real_eigenvalue = 7.0 / 6.0                                  # -mean Re eig(A)
    assert f_iv["mean_negative_real_eigenvalue"] < 0.3 * f_ols["mean_negative_real_eigenvalue"]
    assert abs(f_iv["mean_negative_real_eigenvalue"] - true_mean_negative_real_eigenvalue) < abs(f_ols["mean_negative_real_eigenvalue"] - true_mean_negative_real_eigenvalue) / 4
    gf_ols = dyn.helmholtz_split(f_ols["A"])["symmetric_part_norm_fraction"]
    gf_iv = dyn.helmholtz_split(f_iv["A"])["symmetric_part_norm_fraction"]
    assert gf_iv < gf_ols - 0.15                               # de-biased toward the truth
    assert min(f_iv["first_stage_r2"]) > 0.05                  # instrument has bite


def test_helmholtz_split_identity():
    A = np.array([[-1., 3, 0], [-3, -2., 0], [0, 0, -0.5]])
    sp = dyn.helmholtz_split(A)
    assert np.allclose(sp["A_grad"] + sp["A_rot"], A) and sp["convex"]
    assert abs(sp["symmetric_part_norm_fraction"] ** 2 + sp["antisymmetric_part_norm_fraction"] ** 2 - 1) < 1e-9


def test_causal_velocity_no_preonset_leak():
    # a trajectory that is FLAT before t=0 and ramps after: causal differencing must report
    # zero velocity before the onset (no future leakage), unlike the non-causal default.
    dt = 0.05
    t = -0.6 + np.arange(40) * dt
    zt = np.zeros((40, 2)); zt[:, 0] = np.where(t >= 0, np.clip(t, 0, None) * 10, 0.0)
    d = dyn.trial_velocities([zt], dt, gap=2, smooth_bins=3, causal=True, time=t)
    pre = d["tsec"] < -0.05
    assert np.allclose(d["V"][pre], 0, atol=1e-9), "causal velocity leaks before onset"


def test_regression_suite():
    # synthetic ramp-to-threshold trials: time-to-lick should be decodable, and S+W == A exactly
    rng = np.random.default_rng(0)
    n, T, D = 40, 60, 5
    t = -0.6 + np.arange(T) * 0.05
    Z = []; lick = np.zeros(n)
    for i in range(n):
        L = rng.uniform(0.4, 0.9); lick[i] = L
        ramp = np.clip((t) / L, 0, 1.2)[:, None]
        z = ramp * rng.normal(size=D) + 0.05 * rng.normal(size=(T, D))
        Z.append(z)
    Z = np.array(Z)
    R = dyn.run_regression_suite(Z, lick, t, np.arange(n), bin_s=0.05, base=-0.6, n_splits=3)
    assert set(R) == {"velocity", "time_to_lick"}
    vel = {r["name"]: r for r in R["velocity"]}
    # S+W must equal the full flow A exactly (additivity)
    assert abs(vel["S + W"]["cv_post"] - vel["full flow  A"]["cv_post"]) < 1e-9
    # every field present
    for r in R["time_to_lick"]:
        for k in ("cv_pre", "cv_post", "sh_pre", "sh_post", "corr_pre", "corr_post"):
            assert k in r


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"{len(fns)}/{len(fns)} passed")
