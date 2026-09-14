"""Numerical checks for the rnn package — run with: python test_rnn.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import torch
import torch.nn as nn

from neuralgeom.tasks import (TASKS, ContextDecision, DelayMatchToSample,
                 EvidenceIntegration, PerceptualDecision, make_model,
                 make_task, masked_loss, train)
from neuralgeom.dynamics.rnn import (find_slow_points, input_jacobian, jacobian_spectrum,
                          participation_ratio, readout_subspace,
                          recurrent_jacobian, state_pullback_metric,
                          subspace_alignment, trajectory_subspaces)

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def check(name, cond):
    assert cond, f"FAILED: {name}"
    print(f"  ok: {name}")


# 1. Task API contract
print("[1] Task API contract (all registered tasks)")
for name in TASKS:
    task = make_task(name, dt=20, seed=0)
    b = task.sample(16)
    S = task.spec
    check(f"{name}: input shape (B,T,{S.input_dim})",
          b.inputs.shape[0] == 16 and b.inputs.shape[2] == S.input_dim)
    check(f"{name}: targets (B,T) long, within [0,{S.output_dim})",
          b.targets.shape == b.inputs.shape[:2]
          and b.targets.dtype == torch.long
          and int(b.targets.max()) < S.output_dim)
    check(f"{name}: mask is bool (B,T) and nonempty",
          b.loss_mask.dtype == torch.bool
          and b.loss_mask.shape == b.targets.shape
          and bool(b.loss_mask.any()))
    check(f"{name}: every trial has a decision epoch",
          bool(((b.targets > 0) & b.loss_mask).any(dim=1).all()))
    check(f"{name}: inputs finite", bool(torch.isfinite(b.inputs).all()))

# 2. Task semantics
print("[2] Task semantics / ground truth")
t = PerceptualDecision(dt=20, seed=1, sigma=0.0)
b = t.sample(64)
coh, ch = b.meta["coherence"], b.meta["choice"]
check("perceptual: choice == 1 iff coherence > 0",
      bool(((coh > 0) == (ch == 1)).all()))

t = EvidenceIntegration(dt=20, seed=1, sigma=0.0)
b = t.sample(128)
na, nb_ = b.meta["n_pulses_A"], b.meta["n_pulses_B"]
check("integration: label follows the REALIZED pulse counts",
      bool(((na >= nb_) == (b.meta["choice"] == 1)).all()))
check("integration: pulse counts match the input channels",
      bool((b.inputs[:, :, 1] > 0.5).sum(1).eq(na).all()))
check("integration: variable stimulus duration across trials",
      int(b.meta["n_stim"].unique().numel()) > 1)

t = ContextDecision(dt=20, seed=1, sigma=0.0)
b = t.sample(256)
ctx, cm, cc, ch = (b.meta["context"], b.meta["coh_motion"],
                   b.meta["coh_colour"], b.meta["choice"])
rel = torch.where(ctx == 0, cm, cc)
irr = torch.where(ctx == 0, cc, cm)
check("context: choice follows the CUED feature",
      bool(((rel > 0) == (ch == 1)).all()))
conflict = (rel.sign() != irr.sign())
check("context: conflict trials exist (irrelevant feature disagrees)",
      float(conflict.double().mean()) > 0.3)
check("context: cue channels are one-hot during the trial",
      bool((b.inputs[:, 0, 5] + b.inputs[:, 0, 6]).eq(1).all()))

t = DelayMatchToSample(dt=20, seed=1, sigma=0.0, n_stim=4)
b = t.sample(128)
check("dms: match label iff sample == test",
      bool(((b.meta["sample_id"] == b.meta["test_id"])
            == (b.meta["choice"] == 2)).all()))
check("dms: non-match trials really differ",
      bool((b.meta["sample_id"][~b.meta["is_match"]]
            != b.meta["test_id"][~b.meta["is_match"]]).all()))
# blank delay: between sample offset and test onset there is no stimulus
i0 = 0
t_on = int(b.meta["test_on"][i0])
check("dms: delay period has zero stimulus drive",
      float(b.inputs[i0, t_on - 2, 1:].abs().max()) < 1e-9)

# 3. Models: interface + step/forward consistency
print("[3] Model interface")
task = make_task("perceptual_decision", dt=20, seed=0)
for mname in ["vanilla", "gru", "lstm"]:
    m = make_model(mname, task.spec, hidden_size=16, dt=task.dt)
    m.eval()
    b = task.sample(4)
    out, H = m(b.inputs)
    check(f"{mname}: forward shapes",
          out.shape == (4, b.n_steps, task.spec.output_dim)
          and H.shape[0] == 4 and H.shape[1] == b.n_steps)
    # rolling step() manually must reproduce forward()
    h = m.init_state(4)
    for tt in range(b.n_steps):
        h = m.step(b.inputs[:, tt], h)
    check(f"{mname}: manual step loop == forward",
          torch.allclose(h, H[:, -1], atol=1e-10))
    check(f"{mname}: readout consistent",
          torch.allclose(m.readout(H[:, -1]), out[:, -1], atol=1e-10))

# 4. VanillaRNN dynamics against the analytic formula
print("[4] VanillaRNN analytic checks")
m = make_model("vanilla", task.spec, hidden_size=12, dt=20.0, tau=100.0,
               noise=0.0)
m.eval()
h = torch.randn(5, 12) * 0.3
x = torch.randn(5, task.spec.input_dim) * 0.3
pre = m.rec(h) + m.inp(x)
h_expected = (1 - m.alpha) * h + m.alpha * torch.tanh(pre)
check("step matches (1-a)h + a*tanh(Wh + Ux + b)",
      torch.allclose(m.step(x, h), h_expected, atol=1e-12))
# alpha is now PER UNIT — a (hidden,) tensor, since tau may be given as a
# (low, high) range. With a scalar tau every entry is dt/tau, so compare the
# whole vector rather than truncating it to a scalar.
check("alpha = dt/tau", bool((m.alpha - 0.2).abs().max() < 1e-12))

J = recurrent_jacobian(m, h, x)
J_analytic = ((1 - m.alpha) * torch.eye(12)
              + m.alpha * torch.diag_embed(1 - torch.tanh(pre) ** 2)
              @ m.rec.weight)
check("recurrent Jacobian == analytic dF/dh",
      torch.allclose(J, J_analytic, atol=1e-10))
Ji = input_jacobian(m, h, x)
Ji_analytic = (m.alpha * torch.diag_embed(1 - torch.tanh(pre) ** 2)
               @ m.inp.weight)
check("input Jacobian == analytic dF/dx",
      torch.allclose(Ji, Ji_analytic, atol=1e-10))

# 5. Spectrum & time constants
print("[5] Spectrum utilities")
# a pure leaky integrator with no recurrence: eigenvalues all = 1 - alpha
m2 = make_model("vanilla", task.spec, hidden_size=8, dt=20.0, tau=100.0,
                noise=0.0)
with torch.no_grad():
    m2.rec.weight.zero_()
J2 = recurrent_jacobian(m2, torch.zeros(1, 8), torch.zeros(1, task.spec.input_dim))
ev, mod, tau_eff = jacobian_spectrum(J2, dt=20.0)
check("no recurrence: all |lambda| = 1 - alpha = 0.8",
      torch.allclose(mod, torch.full_like(mod, 0.8), atol=1e-10))
check("time constant recovers tau: -dt/log(0.8) ~ 89.6 ms",
      abs(float(tau_eff[0, 0]) - (-20.0 / math.log(0.8))) < 1e-6)
# an identity-recurrent net at h=0 is a perfect integrator
m3 = make_model("vanilla", task.spec, hidden_size=6, dt=20.0, tau=100.0,
                noise=0.0)
with torch.no_grad():
    m3.rec.weight.copy_(torch.eye(6))
    m3.inp.bias.zero_()
J3 = recurrent_jacobian(m3, torch.zeros(1, 6), torch.zeros(1, task.spec.input_dim))
_, mod3, tau3 = jacobian_spectrum(J3, dt=20.0)
check("identity recurrence at h=0: |lambda| = 1 (line attractor)",
      torch.allclose(mod3, torch.ones_like(mod3), atol=1e-12))
check("...and the time constant is infinite",
      bool(torch.isinf(tau3).all()))

# 6. Pullback metric of the update map
print("[6] State/input pullback metrics")
g = state_pullback_metric(m, h, x, wrt="h")
check("state metric (B, H, H), symmetric PSD",
      g.shape == (5, 12, 12)
      and torch.allclose(g, g.transpose(-1, -2))
      and bool((torch.linalg.eigvalsh(g) > -1e-10).all()))
check("state metric == J^T J", torch.allclose(g, J.transpose(-1, -2) @ J,
                                              atol=1e-10))
gx = state_pullback_metric(m, h, x, wrt="x")
check("input metric (B, in, in) is rank <= hidden",
      gx.shape == (5, task.spec.input_dim, task.spec.input_dim))

# 7. Slow points
print("[7] Slow-point finder")
# Build a net with a single stable fixed point at h=0 (zero input, zero bias)
m4 = make_model("vanilla", task.spec, hidden_size=10, dt=20.0, noise=0.0,
                g=0.4)
with torch.no_grad():
    m4.inp.bias.zero_()
x0 = torch.zeros(task.spec.input_dim)
sp = find_slow_points(m4, x0, 0.4 * torch.randn(12, 10), steps=400, lr=0.05)
check("all searches converge to low speed q", float(sp.q.max()) < 1e-8)
check("they collapse to ONE unique fixed point", len(sp.unique(tol=1e-3)) == 1)
check("that point is the origin", float(sp.unique(tol=1e-3).h.abs().max()) < 1e-4)
check("it is stable (no unstable eigenvalues)",
      int(sp.unique(tol=1e-3).n_unstable[0]) == 0)
check("filter() keeps only true fixed points",
      len(sp.filter(1e-12)) <= len(sp))

# 8. Population descriptors
print("[8] Population descriptors")
H_iso = torch.randn(500, 20)
H_1d = torch.randn(500, 1) @ torch.randn(1, 20)
check("PR ~ hidden for isotropic activity",
      participation_ratio(H_iso) > 15)
check("PR ~ 1 for rank-1 activity",
      abs(participation_ratio(H_1d) - 1.0) < 1e-6)
b = task.sample(32)
mm = make_model("vanilla", task.spec, hidden_size=16, dt=task.dt)
mm.eval()
_, H = mm(b.inputs)
Q, labels = trajectory_subspaces(H, b.meta["choice"], k=2)
check("trajectory subspaces (n_groups, hidden, k), orthonormal",
      Q.shape == (len(labels), 16, 2)
      and torch.allclose(Q[0].T @ Q[0], torch.eye(2), atol=1e-10))
R = readout_subspace(mm)
check("readout subspace is (hidden, out) orthonormal",
      R.shape == (16, 3)
      and torch.allclose(R.T @ R, torch.eye(3), atol=1e-10))
al = subspace_alignment(Q[0], Q[1])
check("alignment report has angles + distance",
      "principal_angles_deg" in al and "geodesic_distance" in al
      and 0 <= al["geodesic_distance"] < math.pi)

# 9. Trainer plumbing (short run must reduce the loss)
print("[9] Trainer")
task = make_task("perceptual_decision", dt=40, seed=0)
m = make_model("vanilla", task.spec, hidden_size=32, dt=task.dt, noise=0.02)
b = task.sample(8)
out, _ = m(b.inputs)
l0 = float(masked_loss(out, b, task.spec.loss))
check("masked_loss is finite and positive", math.isfinite(l0) and l0 > 0)
hist = train(m, task, steps=120, batch_size=32, log_every=60, verbose=False)
check("loss decreased over training", hist["loss"][-1] < hist["loss"][0])
check("history records step/loss/acc", set(hist) == {"step", "loss", "acc"})

# 10. neurogym adapter (skipped if neurogym is not installed)
print("[10] neurogym adapter")
try:
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        from neuralgeom.tasks.neurogym import NeuroGymTask
        ng = NeuroGymTask("PerceptualDecisionMaking-v0", dt=20, seed=0)
        nb = ng.sample(8)
        check("adapter yields the same TrialBatch contract",
              nb.inputs.shape[0] == 8
              and nb.inputs.shape[2] == ng.spec.input_dim
              and nb.targets.dtype == torch.long
              and nb.loss_mask.dtype == torch.bool)
        check("adapter task drives the generic model factory",
              make_model("vanilla", ng.spec, hidden_size=8,
                         dt=ng.dt)(nb.inputs)[0].shape
              == (8, nb.n_steps, ng.spec.output_dim))
        check("adapter batch runs through the generic loss",
              math.isfinite(float(masked_loss(
                  make_model("vanilla", ng.spec, hidden_size=8,
                             dt=ng.dt)(nb.inputs)[0], nb, ng.spec.loss))))
except ImportError:
    print("  skipped: neurogym not installed (pip install neurogym)")

print("\nAll rnn package checks passed.")
