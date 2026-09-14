"""
neuralgeom.synth.attractor — two-attractor model of cue-triggered timing.
========================================================================

A 2-D latent state ``[X (cue mode), Y (ramping mode)]`` moves in a vector
field made of two localised (Gaussian) attractors, a rest attractor and a
"lick" attractor, following the two-attractor model of Majumder et al. The
cue is a brief pulse; after it the flow is autonomous. A "lick" is registered
when ``Y`` crosses a threshold. The latent trajectories are embedded into
``n_neurons`` noisy units (Gaussian or Poisson) so that the estimators in
:mod:`neuralgeom.dynamics` and the pullback tools in
:mod:`neuralgeom.geometry` can be validated against known dynamics.

Two generative mechanisms produce families of differently timed ramps:

  * mechanism "input"      the vector field is fixed and the cue amplitude
                           varies by condition (a larger push crosses the
                           threshold sooner);
  * mechanism "landscape"  the cue is fixed and the vector field varies by
                           condition through the pull strengths of the lick
                           attractor. This changes the state-dependent field
                           (the Jacobian), unlike an additive input.

Both yield similar-looking ramps. Given only the embedded activity, the
question for the estimators is whether the two mechanisms can be told apart.

Note on identifiability: a constant additive tilt of the field enters the
dynamics exactly like a constant input and would be indistinguishable from
an input change, which is why the landscape mechanism varies the
state-dependent pull strengths instead.
"""
import numpy as np


# --------------------------------------------------------------------------- #
# Two-attractor model (Majumder et al.). State = [X (cue mode), Y (ramping
# mode)] in scaled coordinates. The field is a sum of two localised (Gaussian)
# pulls, one toward each attractor, with axis-specific strengths and a ramp gain
# that increases the pull with Y. The cue is a brief pulse (amplitude, angle);
# after it the flow is autonomous. A lick is registered when Y crosses
# LICK_THRESHOLD.
# --------------------------------------------------------------------------- #
REST_ATTRACTOR = np.array([1.5, 1.5])      # position of the rest attractor (scaled coordinates)
LICK_ATTRACTOR = np.array([15.0, 15.0])    # position of the lick attractor
LICK_THRESHOLD = 15.0                      # lick registered when Y crosses this value
DT = 0.001; T_BASE = 0.6; T_CUE = 0.6; CUE_DUR = 0.2; T_END = 3.0   # seconds; 600 ms baseline pre-cue
T_WARMUP = 0.5   # extra pre-baseline settling time, simulated then sliced off (edge transient)
WELL_WIDTH = 6.0                                    # Gaussian well width (scaled coords)
# per-attractor axis-specific pull strengths (paper Eq.6): (ax, ay) for baseline & lick attractor
BASE_PULLS = dict(ax1=1.0, ay1=1.0, ax2=1.0, ay2=1.0)


def attractor_field(X, Y, ax1=1.0, ay1=1.0, ax2=1.0, ay2=1.0, goal=15.0, couple=0.0):
    """Intrinsic vector field F(s), paper Eq.6 (RAW, not normalized -- normalization is figure-only).
    Two Gaussian wells (baseline REST_ATTRACTOR, lick attractor at (goal,goal)) with AXIS-SPECIFIC strengths:
        Fx = -sum_k ax_k (x-x_k) exp(-||s-a_k||^2 / 2 sig^2)
        Fy = -sum_k ay_k (y-y_k) exp(-||s-a_k||^2 / 2 sig^2)
    The per-attractor (ax_k, ay_k) set the direction & magnitude of the pull, and `goal` the lick
    attractor position -> together they control the angle/amplitude of the trajectory. Changing any
    of them is a LANDSCAPE (flow) manipulation."""
    a2x = a2y = goal
    r1 = ((X - REST_ATTRACTOR[0]) ** 2 + (Y - REST_ATTRACTOR[1]) ** 2) / (2 * WELL_WIDTH ** 2)
    r2 = ((X - a2x) ** 2 + (Y - a2y) ** 2) / (2 * WELL_WIDTH ** 2)
    G1 = np.exp(-r1); G2 = np.exp(-r2)
    # optional cue->ramp coupling: gate the lick-attractor's Y-pull by cue-mode (X) progress, so the
    # ramp only builds once X has advanced (paper 'mod' term). couple=0 off, couple=1 full gating.
    gate = np.clip((X - REST_ATTRACTOR[0]) / (a2x - REST_ATTRACTOR[0] + 1e-9), 0.0, 1.0)
    gy = (1 - couple) + couple * gate
    Fx = -ax1 * (X - REST_ATTRACTOR[0]) * G1 - ax2 * (X - a2x) * G2
    Fy = -ay1 * (Y - REST_ATTRACTOR[1]) * G1 - ay2 * (Y - a2y) * G2 * gy
    return np.array([Fx, Fy])


# back-compat alias used by some figures
def two_attractor_speed(X, Y, **kw):
    keep = {k: kw[k] for k in ("ax1", "ay1", "ax2", "ay2") if k in kw}
    return attractor_field(X, Y, **keep)


def simulate_two_attractor_trial(cue_amp, cue_ang=np.pi / 4, ax1=1.0, ay1=1.0, ax2=1.0, ay2=1.0,
                   goal=15.0, couple=0.0, field_scale=1.0, sigma=0.2, seed=0):
    """One trial. Baseline phase [0,T_BASE) at rest; brief cue force A[cos,sin] during
    [T_CUE, T_CUE+CUE_DUR]; autonomous otherwise. ds/dt = field_scale*F(s) + I(t). Lick = Y
    crossing 0.99*goal. Returns z (T,2), cue_on (T,), lick, time."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, T_END, DT); T = len(t)
    z = np.zeros((T, 2)); z[0] = REST_ATTRACTOR.copy()
    cue_on = (t >= T_CUE) & (t < T_CUE + CUE_DUR)
    lick = np.nan
    kick = cue_amp * np.array([np.cos(cue_ang), np.sin(cue_ang)])
    for i in range(1, T):
        F = attractor_field(z[i - 1, 0], z[i - 1, 1], ax1, ay1, ax2, ay2, goal, couple) * field_scale
        sp = F + (kick if cue_on[i] else 0.0)
        z[i] = z[i - 1] + DT * sp + sigma * np.sqrt(DT) * rng.standard_normal(2)
        if np.isnan(lick) and z[i, 1] > goal * 0.99 and t[i] > T_CUE:
            lick = t[i]
    return z, cue_on.astype(float), lick, t


# Default conditions. The cue angle of 25 deg (X-heavier than the diagonal) makes the trajectory
# curve right-then-up and lands the state at moderate x, so that changes of ax2, ay2 and their
# ratio all affect timing. Post-cue lick times are then roughly 0.5-1.2 s.
FIELD_SCALE = 14.0                                 # overall speed of the field (s^-1 scale)
FIXED_CUE = (90.0, 25.0)                           # (amplitude, angle deg) used by the landscape mechanism
CUE_AMPLITUDE_LEVELS = [(82, 25), (88, 25), (95, 25), (104, 25), (118, 25)]   # input mechanism
# landscape mechanism: each entry is a set of simulate_two_attractor_trial kwargs (cue = FIXED_CUE)
LICK_ATTRACTOR_X_PULL_LEVELS = [dict(ax2=v) for v in [0.3, 0.55, 1.0, 1.8, 3.2]]
LICK_ATTRACTOR_Y_PULL_LEVELS = [dict(ay2=v) for v in [0.72, 0.9, 1.1, 1.35, 1.75]]
LICK_THRESHOLD_LEVELS = [dict(goal=v) for v in [16, 17.5, 19, 21, 23]]


def lick_attractor_pull_ratio_family(ratios=(0.45, 0.7, 1.0, 1.5, 2.3), strength=1.0):
    """Landscape conditions parametrised by the ratio rho = ax2/ay2 of the lick attractor's x and y
    pull at fixed geometric-mean pull strength. rho > 1 (more x-pull, less y-pull) gives a flatter
    path and a slower ramp, hence a later lick; rho < 1 a steeper path and an earlier lick.
    Returns a list of kwarg dicts with the rho value recorded."""
    return [dict(ax2=strength * np.sqrt(r), ay2=strength / np.sqrt(r), rho=r) for r in ratios]


LICK_ATTRACTOR_PULL_RATIO_LEVELS = lick_attractor_pull_ratio_family()
# (ax2, ay2) pairs whose per-condition trajectories and lick times match those of the input
# mechanism's conditions (cue = FIXED_CUE); used for the behaviourally matched recovery test.
LICK_ATTRACTOR_BEHAVIOUR_MATCHED_LEVELS = [dict(ax2=0.88, ay2=0.85), dict(ax2=0.88, ay2=1.0), dict(ax2=1.46, ay2=1.0),
                dict(ax2=2.92, ay2=1.0), dict(ax2=3.5, ay2=1.75)]


def simulate_batch(n_trials, cue_amp, cue_ang=np.pi / 4, ax1=1.0, ay1=1.0, ax2=1.0, ay2=1.0,
                   goal=15.0, couple=0.0, field_scale=1.0, sigma=0.2, amp_jitter=0.12, seed=0,
                   warmup=T_WARMUP):
    """Vectorized: simulate n_trials at once (loop over time only). Returns Z (n,T,2), cue_on (T,),
    lick (n,), time (T,), amp_used (n,). `amp_jitter` = SD (fraction of cue_amp) of per-trial cue
    amplitude, so cue-response amplitude varies across trials like real data -> the weakest trials
    fail to cross the saddle and decay to baseline (NO-LICK trials; paper Fig 3c-e).

    `warmup` (s): the state starts as a delta at the baseline attractor, so its trial-to-trial
    spread is zero at the initial condition and only grows to the stationary noise-driven
    distribution over the first tens of ms -- an edge transient we do not care about.  We
    therefore simulate an extra ``warmup`` seconds of pure baseline before the analysed baseline
    begins.  The warm-up is KEPT in the returned data as a pre-window BUFFER (time < 0 on the
    internal clock, i.e. before the -0.6 s analysis baseline once re-zeroed to the cue): it lets
    causal filters have real history at the start of the analysis window, and downstream code
    excludes it with a simple time mask.  The internal clock has t=0 at the start of the analysed
    baseline, cue at T_CUE, and the buffer spans [-warmup, 0)."""
    rng = np.random.default_rng(seed)
    t_full = np.arange(-warmup, T_END, DT); T = len(t_full)
    Z = np.zeros((n_trials, T, 2)); Z[:, 0, :] = REST_ATTRACTOR
    cue_on = (t_full >= T_CUE) & (t_full < T_CUE + CUE_DUR)
    amp = cue_amp * (1 + amp_jitter * rng.standard_normal(n_trials))        # per-trial cue amplitude
    kick = amp[:, None] * np.array([np.cos(cue_ang), np.sin(cue_ang)])[None, :]   # (n,2)
    lick = np.full(n_trials, np.nan); nz = sigma * np.sqrt(DT)
    for i in range(1, T):
        F = attractor_field(Z[:, i - 1, 0], Z[:, i - 1, 1], ax1, ay1, ax2, ay2, goal, couple)
        sp = (F.T) * field_scale
        if cue_on[i]:
            sp = sp + kick
        Z[:, i, :] = Z[:, i - 1, :] + DT * sp + nz * rng.standard_normal((n_trials, 2))
        new = np.isnan(lick) & (Z[:, i, 1] > goal * 0.99) & (t_full[i] > T_CUE)
        lick[new] = t_full[i]
    return Z, cue_on.astype(float), lick, t_full, amp


def make_two_attractor_dataset(mechanism, levels=None, n_trials=30, seed=0, bin_s=0.02,
                       field_scale=FIELD_SCALE, **fixed):
    """Two-attractor dataset for one mechanism (paper-faithful, no overshoot, with baseline phase).
      mechanism='input'     : FIXED landscape; cue AMPLITUDE & ANGLE covary by condition.
                              levels = list of (amp, angle_deg); default CUE_AMPLITUDE_LEVELS.
      mechanism='landscape' : FIXED cue (FIXED_CUE); the lick-attractor Y-pull ay2 varies by condition
                              (shallower well -> slower ramp -> later lick), a genuine flow change.
                              levels = list of ay2 values; default LICK_ATTRACTOR_Y_PULL_LEVELS.
    Returns Zlat (n,T,2), U (cue on/off), cond, lick, time, plus the per-condition cue (amp,angle)
    and ay2 so the ground-truth input can be reconstructed."""
    step = int(round(bin_s / DT))
    Z, cue, cond, lick, rho_of = [], [], [], [], []
    cue_amp_of, cue_ang_of, ay2_of = [], [], []
    rng = np.random.default_rng(seed)
    if levels is None:
        levels = CUE_AMPLITUDE_LEVELS if mechanism == "input" else LICK_ATTRACTOR_Y_PULL_LEVELS
    for ci, lev in enumerate(levels):
        rho = np.nan
        if mechanism == "input":
            amp, ang = lev; kw = {}
        elif mechanism == "landscape":
            amp, ang = FIXED_CUE                              # FIXED cue for all landscape conditions
            if isinstance(lev, dict):
                kw = dict(lev); rho = kw.pop("rho", np.nan)   # 'rho' is metadata, not a sim arg
            elif isinstance(lev, (tuple, list)):
                kw = dict(ax2=lev[0], ay2=lev[1])
            else:
                kw = dict(ay2=float(lev))
        else:
            raise ValueError(mechanism)
        zb, con, lkb, t, amp_used = simulate_batch(n_trials, amp, cue_ang=np.deg2rad(ang),
                                                    field_scale=field_scale,
                                                    seed=int(rng.integers(1, 2**31)), **kw)
        for k in range(n_trials):
            Z.append(zb[k, ::step]); cue.append(con[::step]); cond.append(ci); lick.append(lkb[k])
            cue_amp_of.append(amp_used[k]); cue_ang_of.append(np.deg2rad(ang))
            ay2_of.append(kw.get("ay2", 1.0)); rho_of.append(rho)
    return dict(Zlat=np.array(Z), U=np.array(cue), cond=np.array(cond), lick=np.array(lick),
                time=t[::step], mechanism=mechanism, levels=list(levels), bin_s=bin_s,
                cue_amp=np.array(cue_amp_of), cue_ang=np.array(cue_ang_of), ay2=np.array(ay2_of),
                rho=np.array(rho_of), field_scale=field_scale)


def embed_rates(Zlat, n_neurons=80, gain=1.0, noise=0.15, seed=0, nonneg=True):
    """Linear (optionally softplus) embedding of the 2-D latent into n_neurons 'firing rates'
    plus observation noise. Returns rates (n, T, n_neurons) and the loading matrix L (n_neurons,2)."""
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((n_neurons, 2))
    b0 = rng.uniform(0.2, 1.0, n_neurons)
    # normalize the latent (it can range over tens) so the softplus embedding stays near-linear
    mu = Zlat.reshape(-1, Zlat.shape[-1]).mean(0)
    sd = Zlat.reshape(-1, Zlat.shape[-1]).std(0) + 1e-6
    Zn = (Zlat - mu) / sd
    lin = np.einsum("ntd,md->ntm", Zn, L) * gain + b0[None, None, :]
    if nonneg:
        lin = np.log1p(np.exp(lin))            # softplus -> nonneg, mildly nonlinear
    rates = lin + noise * rng.standard_normal(lin.shape)
    return rates, L


def generate(mechanism="input", levels=None, n_trials=120, seed=0, bin_s=0.02,
             n_neurons=80, obs_noise=0.15, gain=1.0, poisson=False, mean_count=None,
             field_scale=FIELD_SCALE, embed_seed=5, **fixed):
    """One call: simulate the 2-attractor latent, embed into 'neural' activity, and return a
    plain dict ready for ``neuralgeom.data.from_synthetic``.

    Time is re-zeroed so the CUE ONSET is at t = 0 (matching the real sessions).

    Observation model:
      * poisson=False (default): additive Gaussian observation noise of SD ``obs_noise`` on the
        softplus embedding (fast, mildly nonlinear).
      * poisson=True: draw spike COUNTS ~ Poisson(rate*bin); if ``mean_count`` is given the
        rates are rescaled so the mean count per unit per bin matches it (use this to reproduce
        the real sessions' ~0.24 spikes/unit/50 ms and its counting-noise level).

    Returns a dict with keys:
        X (n_trials, T, n_neurons)   embedded activity,
        time (T,)                    cue-relative seconds (cue at 0),
        lick (n_trials,)             first post-cue lick time (NaN = no lick),
        cond (n_trials,)             within-mechanism condition index,
        latent (n_trials, T, 2)      the NOISELESS ground-truth state,
        plus cue_amp / cue_ang / ay2 / mechanism / levels for ground-truth checks.
    """
    d = make_two_attractor_dataset(mechanism, levels, n_trials=n_trials, seed=seed, bin_s=bin_s,
                           field_scale=field_scale, **fixed)
    if poisson:
        rates, _ = embed_rates(d["Zlat"], n_neurons=n_neurons, gain=gain, noise=0.0,
                               seed=embed_seed)
        rates = np.clip(rates, 1e-3, None)
        if mean_count is not None:
            rates = rates * (mean_count / rates.mean())
        X = np.random.default_rng(seed + 1).poisson(rates).astype(float)
    else:
        X, _ = embed_rates(d["Zlat"], n_neurons=n_neurons, gain=gain, noise=obs_noise,
                           seed=embed_seed)
    time = np.asarray(d["time"], float) - T_CUE               # cue onset -> t = 0
    out = dict(X=X, time=time, lick=np.asarray(d["lick"], float) - T_CUE,
               cond=d["cond"], latent=d["Zlat"], mechanism=mechanism,
               levels=d["levels"], cue_amp=d["cue_amp"], cue_ang=d["cue_ang"], ay2=d["ay2"],
               bin_s=bin_s)
    return out
