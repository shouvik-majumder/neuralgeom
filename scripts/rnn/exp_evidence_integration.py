"""
EXPERIMENT 2 — Evidence integration over a variable window.
===========================================================

Question: the network must count pulses over a window whose length it does
not know in advance. What structure does it build to do that, and can the
geometry of that structure predict how well it remembers?

Arc:
  1 TRAINING    why this task forbids the shortcuts the last one allowed
  2 BEHAVIOUR   accuracy vs evidence and duration; the integration kernel
  3 ACTIVITY    the population rides a one-dimensional manifold
  4 GEOMETRY    the line attractor: slow points, their eigenvalues, and a
                memory-decay prediction tested against behaviour

Run:  python exp_evidence_integration.py
"""

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuralgeom.paths import PDF_DIR  # noqa: E402

import numpy as np
import torch

import matplotlib.pyplot as plt
from exp_common import (Report, banner, decode_axis, figdir, leak_prediction,
                        pca, plot_state_traj, project, psth,
                        reverse_correlation, savefig, selectivity)

from neuralgeom.tasks import EvidenceIntegration
from neuralgeom.dynamics.rnn import (find_slow_points, jacobian_spectrum,
                          participation_ratio, recurrent_jacobian)
from train_rnn_models import CONFIG, load_or_train

TASK = "evidence_integration"
FIG = figdir(TASK)
torch.manual_seed(0)
np.random.seed(0)

# --------------------------------------------------------------------------- #
banner("1  TRAINING", "Load the network; describe the task's demands")
_, model, hist = load_or_train(TASK, "vanilla", verbose=True)
model.eval()
tkw, mkw, _ = CONFIG[TASK]
dt = tkw["dt"]
alpha = model.alpha
print(f"  units={model.hidden_size}  dt={dt} ms  tau={dt/alpha:.0f} ms")

# analysis task: fixed maximum window so trials align; duration still varies
task = EvidenceIntegration(dt=dt, sigma=tkw["sigma"], seed=11,
                           t_fix=200, t_stim=(400, 1200), t_dec=300)
n_fix = task._steps(200)
n_max = task._steps(1200)
batch = task.sample(4000)
out, H = model(batch.inputs)
T = batch.n_steps
tms = (np.arange(T) - n_fix) * dt

n_stim = batch.meta["n_stim"]
ev_true = batch.meta["evidence"]                  # realized pulse difference
dec_on = n_fix + n_stim
choice = torch.stack([torch.mode(out[i, dec_on[i]:dec_on[i] + 12].argmax(-1)
                                 ).values for i in range(batch.batch_size)])
correct = (choice == batch.meta["choice"]).double()
print(f"  {batch.batch_size} trials, accuracy {float(correct.mean()):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.0), constrained_layout=True)
if hist:
    axes[0].plot(hist["step"], hist["loss"], color="C3")
    axes[0].set_xlabel("training step"); axes[0].set_ylabel("masked loss")
    axes[0].set_title("Training loss")
    axes[1].plot(hist["step"], hist["acc"], "o-", ms=3, color="C0")
    axes[1].set_ylim(0, 1.02); axes[1].axhline(0.5, color="0.6", ls=":")
    axes[1].set_xlabel("training step"); axes[1].set_title("Learning curve")
i0 = int(torch.argmax(n_stim))
axes[2].eventplot([np.where(batch.inputs[i0, :, 1].numpy() > 0.5)[0] * dt
                   - n_fix * dt,
                   np.where(batch.inputs[i0, :, 2].numpy() > 0.5)[0] * dt
                   - n_fix * dt],
                  colors=["#c0392b", "#2471a3"], lineoffsets=[1, 0],
                  linelengths=0.7)
axes[2].axvline(0, color="0.5"); axes[2].axvline(float(n_stim[i0]) * dt,
                                                 color="0.5")
axes[2].set_yticks([0, 1], ["pulses B", "pulses A"])
axes[2].set_xlabel("time from stimulus onset (ms)")
axes[2].set_title(f"One trial: A={int(batch.meta['n_pulses_A'][i0])} vs "
                  f"B={int(batch.meta['n_pulses_B'][i0])} pulses")
f_train = savefig(fig, FIG, "ei_1_training.png")

# --------------------------------------------------------------------------- #
banner("2  BEHAVIOUR", "Accuracy vs evidence and duration; the kernel")
absev = ev_true.abs()
bins = torch.tensor([0, 1, 2, 3, 4, 6, 9, 100.0])
xs, ys, ns = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (absev >= lo) & (absev < hi)
    if int(m.sum()) > 25:
        xs.append(float(absev[m].mean())); ys.append(float(correct[m].mean()))
        ns.append(int(m.sum()))
# accuracy vs duration, at matched evidence (the key control)
dur_bins = torch.quantile(n_stim.double(),
                          torch.linspace(0, 1, 5, dtype=torch.float64))
dur_x, dur_y, dur_y_matched = [], [], []
matched = (absev >= 1) & (absev <= 3)
for lo, hi in zip(dur_bins[:-1], dur_bins[1:]):
    m = (n_stim >= lo) & (n_stim <= hi)
    if int(m.sum()) > 25:
        dur_x.append(float(n_stim[m].double().mean()) * dt)
        dur_y.append(float(correct[m].mean()))
        mm = m & matched
        dur_y_matched.append(float(correct[mm].mean()) if int(mm.sum()) > 20
                             else np.nan)
print("  accuracy by |evidence|:",
      [f"{x:.1f}:{y:.2f}" for x, y in zip(xs, ys)])
print("  accuracy by duration (ms):",
      [f"{x:.0f}:{y:.2f}" for x, y in zip(dur_x, dur_y)])

# integration kernel over the first 400 ms window common to ALL trials
n_common = int(n_stim.min())
mom = (batch.inputs[:, :, 1] - batch.inputs[:, :, 2])
kern, kern_sem = reverse_correlation(
    mom[:, n_fix:n_fix + n_common], choice)
kern_n = (kern / kern.mean()).numpy()
ts_k = tms[n_fix:n_fix + n_common]

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.2), constrained_layout=True)
axes[0].plot(xs, ys, "o-", color="C2")
axes[0].axhline(0.5, color="0.6", ls=":")
axes[0].set_xlabel("|pulse-count difference| (realized evidence)")
axes[0].set_ylabel("accuracy"); axes[0].set_ylim(0.4, 1.02)
axes[0].set_title("Accuracy tracks the REALIZED count,\nnot the underlying rate")
axes[1].plot(dur_x, dur_y, "o-", label="all trials")
axes[1].plot(dur_x, dur_y_matched, "s--", label="|evidence| 1-3 (matched)")
axes[1].set_xlabel("stimulus duration (ms)"); axes[1].set_ylabel("accuracy")
axes[1].legend(fontsize=7); axes[1].set_ylim(0.4, 1.02)
axes[1].set_title("Performance is stable across durations:\nthe window is not "
                  "hard-coded")
axes[2].fill_between(ts_k, (kern - kern_sem).numpy(), (kern + kern_sem).numpy(),
                     color="C4", alpha=0.3)
axes[2].plot(ts_k, kern.numpy(), color="C4")
axes[2].axhline(0, color="0.6", ls=":")
axes[2].set_xlabel("time from stimulus onset (ms)")
axes[2].set_ylabel("influence on choice")
axes[2].set_title("Integration kernel over the window\nshared by every trial")
f_behav = savefig(fig, FIG, "ei_2_behaviour.png")

# --------------------------------------------------------------------------- #
banner("3  ACTIVITY", "A one-dimensional manifold carries the count")
t_ref = n_fix + n_common
mu, comps, evr = pca(H[:, n_fix:t_ref].reshape(-1, model.hidden_size), k=6)
Z = project(H, mu, comps).numpy()
pr = participation_ratio(H[:, t_ref])
axis_dv = decode_axis(H[:, t_ref], (ev_true > 0).long())
dv = (H @ axis_dv).numpy()
sel = selectivity(H, (ev_true > 0).long(), slice(t_ref - 6, t_ref))
u_best = int(torch.argsort(sel, descending=True)[0])
print(f"  PR at t={t_ref}: {pr:.2f}; PC1 explains {float(evr[0]):.1%}")

# running integral of the input vs state coordinate
cum_ev = torch.cumsum(mom, dim=1)
r_running = [float(np.corrcoef(cum_ev[:, t].numpy(), dv[:, t])[0, 1])
             for t in range(n_fix + 2, t_ref)]
print(f"  corr(running integral, state coordinate) over the window: "
      f"{np.mean(r_running):.3f} (min {np.min(r_running):.3f})")

fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.0), constrained_layout=True)
labels, m_, s_ = psth(H, torch.sign(ev_true).long(), u_best)
for lab, mm, ss in zip(labels, m_, s_):
    c = plt.cm.coolwarm(0.5 + 0.5 * float(lab))
    axes[0, 0].plot(tms, mm.numpy(), color=c, label=f"evidence {int(lab):+d}")
    axes[0, 0].fill_between(tms, (mm - ss).numpy(), (mm + ss).numpy(),
                            color=c, alpha=0.25)
axes[0, 0].axvline(0, color="0.5"); axes[0, 0].legend(fontsize=7)
axes[0, 0].set_title(f"Most selective unit ({u_best})")
axes[0, 0].set_xlabel("time from stim onset (ms)")

qs = torch.quantile(ev_true, torch.linspace(0, 1, 6))
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (ev_true >= lo) & (ev_true <= hi)
    if int(m.sum()) < 20:
        continue
    c = plt.cm.coolwarm(float((0.5 * (lo + hi) - qs[0]) / (qs[-1] - qs[0])))
    axes[0, 1].plot(tms[:t_ref], dv[m][:, :t_ref].mean(0), color=c)
axes[0, 1].axvline(0, color="0.5")
axes[0, 1].set_xlabel("time from stim onset (ms)")
axes[0, 1].set_ylabel("state coordinate")
axes[0, 1].set_title("Trajectories separate by evidence\nand HOLD their "
                     "separation")

axes[0, 2].plot(tms[n_fix + 2:t_ref], r_running, "o-", ms=3, color="C2")
axes[0, 2].set_ylim(0, 1.02)
axes[0, 2].set_xlabel("time from stim onset (ms)")
axes[0, 2].set_ylabel("correlation")
axes[0, 2].set_title("The state coordinate TRACKS the running\nintegral of "
                     "the input, moment by moment")

sm = plot_state_traj(axes[1, 0], Z, ev_true.numpy(), every=8,
                     tslice=slice(n_fix, t_ref))
fig.colorbar(sm, ax=axes[1, 0], label="realized evidence")
axes[1, 0].set_xlabel("PC1"); axes[1, 0].set_ylabel("PC2")
axes[1, 0].set_title("Population trajectories fan out\nalong a single axis")

axes[1, 1].scatter(ev_true.numpy(), dv[:, t_ref], s=4, alpha=0.25)
r_end = float(np.corrcoef(ev_true.numpy(), dv[:, t_ref])[0, 1])
axes[1, 1].set_xlabel("realized evidence (pulse difference)")
axes[1, 1].set_ylabel("state coordinate at t = window end")
axes[1, 1].set_title(f"The coordinate IS the count (|r| = {abs(r_end):.2f})")

axes[1, 2].bar(range(1, 7), evr.numpy() * 100, color="C0")
axes[1, 2].set_xlabel("principal component"); axes[1, 2].set_ylabel("% variance")
axes[1, 2].set_title(f"Effectively 1-D: PR = {pr:.2f}")
f_act = savefig(fig, FIG, "ei_3_activity.png")

# --------------------------------------------------------------------------- #
banner("4  GEOMETRY", "The line attractor and a memory-decay prediction")
Hs = H[:, n_fix:t_ref].reshape(-1, model.hidden_size)
x_bar = batch.inputs[:, n_fix:t_ref].reshape(-1, batch.inputs.shape[-1]).mean(0)
h_init = Hs[torch.randint(0, Hs.shape[0], (80,))]
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sp = find_slow_points(model, x_bar, h_init, steps=700, lr=0.05)
sp_u = sp.unique(tol=0.05)
_, mod_sp, tau_sp = jacobian_spectrum(sp_u.jac, dt=dt)
lam_line = float(mod_sp[:, 0].median())
n_saddle = int((sp_u.n_unstable > 0).sum())
# |lambda| >= 1 means no measurable decay: tau is unbounded (and slightly
# above 1 means weak amplification). Report honestly instead of dividing by
# a log that is ~0.
if lam_line < 0.9995:
    tau_line = -dt / np.log(lam_line)
    tau_str = f"{tau_line:.0f} ms"
else:
    tau_line = float("inf")
    over = (lam_line ** n_common - 1) * 100
    tau_str = (f"unbounded (|&#955;| = {lam_line:.4f} &#8805; 1: no decay; "
               f"a {over:+.0f}% change over the whole window)")
print(f"  {len(sp)} converged -> {len(sp_u)} distinct slow points; "
      f"{n_saddle} saddles")
print(f"  median top |lambda| = {lam_line:.4f}  -> memory tau: "
      f"{tau_line if np.isfinite(tau_line) else 'unbounded (no decay)'}")
print(f"  cumulative gain over the {n_common}-step window: "
      f"{lam_line**n_common:.3f}x")

Zsp = ((sp_u.h - mu) @ comps).numpy()
sp_coord = ((sp_u.h - mu) @ comps[:, 0]).numpy()
# how well do the slow points lie on a LINE?
sp_c = (sp_u.h - sp_u.h.mean(0, keepdim=True))
sv = torch.linalg.svdvals(sp_c)
line_frac = float(sv[0].square() / sv.square().sum())
print(f"  slow points are {line_frac:.1%} one-dimensional "
      f"(first singular value of their spread)")

# PREDICTION: with per-step retention lam, evidence at time t is discounted
# by lam^(T-t). Test against the behavioural kernel measured above.
pred_kernel = leak_prediction(n_common, lam_line)
third = max(3, n_common // 3)
meas_ratio = kern_n[-third:].mean() / kern_n[:third].mean()
pred_ratio = pred_kernel[-third:].mean() / pred_kernel[:third].mean()
print(f"  kernel recency: measured {meas_ratio:.2f}x, predicted from the "
      f"attractor {pred_ratio:.2f}x")

print(f"  longest stimulus in the task: {float(n_stim.max())*dt:.0f} ms")

fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4), constrained_layout=True)
sm = plot_state_traj(axes[0], Z, ev_true.numpy(), every=8,
                     tslice=slice(n_fix, t_ref), mark_start=False,
                     mark_end=False)
axes[0].scatter(Zsp[:, 0], Zsp[:, 1], c="k", s=30, zorder=5,
                label=f"{len(sp_u)} slow points")
axes[0].legend(fontsize=7)
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].set_title(f"The slow points form a LINE\n({line_frac:.0%} of their "
                  f"spread is 1-D)")
axes[1].hist(mod_sp[:, 0].numpy(), bins=16, color="C4")
axes[1].axvline(1.0, color="r", ls="--", label="$|\\lambda|=1$")
axes[1].legend(fontsize=7)
axes[1].set_xlabel("top $|\\lambda|$ at each slow point")
axes[1].set_title(f"Marginally stable: median {lam_line:.3f}\n"
                  f"({n_saddle} saddles among them)")
axes[2].scatter(sp_coord, mod_sp[:, 0].numpy(), s=25, c=sp_u.n_unstable.numpy(),
                cmap="coolwarm")
axes[2].axhline(1.0, color="r", ls="--")
axes[2].set_xlabel("position along the attractor (PC1)")
axes[2].set_ylabel("top $|\\lambda|$")
axes[2].set_title("Stability along the line\n(red = saddle)")
axes[3].plot(ts_k, kern_n, color="C4", lw=0.8, alpha=0.35)
sm5 = np.convolve(kern_n, np.ones(5) / 5, mode="same")
sm5[:2], sm5[-2:] = np.nan, np.nan
axes[3].plot(ts_k, sm5, color="C4", lw=2, label="measured (smoothed)")
axes[3].plot(ts_k, pred_kernel, "--", color="k", lw=1.4,
             label=f"predicted, $|\\lambda|$={lam_line:.3f}")
axes[3].axhline(1.0, color="0.7", ls=":")
axes[3].legend(fontsize=7)
axes[3].set_xlabel("time from stimulus onset (ms)")
axes[3].set_ylabel("normalized influence")
axes[3].set_title(f"Near-flat kernel = near-perfect memory\n"
                  f"measured {meas_ratio:.2f}x vs predicted {pred_ratio:.2f}x")
f_geom = savefig(fig, FIG, "ei_4_geometry.png")

# --------------------------------------------------------------------------- #
banner("REPORT", "Building the PDF")
rep = Report(PDF_DIR / "experiment_2_evidence_integration.pdf",
             "Experiment 2 — Evidence Integration",
             "Counting pulses over an unpredictable window, and the line "
             "attractor that makes it possible")

rep.h1("1. What the network was trained to do")
rep.p(
    "Two input channels emit discrete <b>pulses</b>, independently, at "
    "slightly different rates. After a window whose length varies from 400 "
    "to 1200&#160;ms — unknown to the network in advance — it must report "
    "which channel emitted more pulses. Three design choices make this "
    "harder than it looks, and each removes a shortcut:")
rep.deflist([
    ("Discrete pulses",
     "The input is sparse and binary rather than a steady drive, so there is "
     "no instantaneous quantity to read off. Information exists only in the "
     "accumulated count."),
    ("Realized, not expected",
     "The correct answer is the <i>actual</i> pulse difference on that "
     "trial, not the side with the higher underlying rate. On a substantial "
     "fraction of trials the sampling goes against the rate, and the network "
     "is scored on the sample. Inferring the rate is not enough — it must "
     "count."),
    ("Variable duration",
     "The response epoch begins at an unpredictable time, so no fixed-latency "
     "or ramp-to-threshold-by-time-T strategy works. The running total has "
     "to be available and correct at any moment."),
], w0=1.35)
rep.p(
    f"The architecture provides no counter: a plain leaky tanh RNN whose "
    f"units forget with &#964; = {dt/alpha:.0f}&#160;ms, against stimuli up "
    f"to 1200&#160;ms long. Whatever accumulates must be built from "
    f"recurrent connectivity.")
rep.figure(f_train,
           "<b>Left, middle:</b> training loss and accuracy. <b>Right:</b> the "
           "pulse trains on one long trial — the entire stimulus is these two "
           "sparse event sequences.")
rep.box(f"<b>Outcome.</b> {float(correct.mean()):.1%} correct on "
        f"{batch.batch_size} held-out trials.")
rep.pagebreak()

rep.h1("2. Behaviour: it really is integrating")
rep.figure(f_behav,
           "<b>Left:</b> accuracy rises with the realized pulse difference — "
           "the network is sensitive to the actual sample, not just the "
           "generating rate. <b>Middle:</b> accuracy against stimulus duration, "
           "both overall and with evidence held fixed; performance does not "
           "collapse for long trials. <b>Right:</b> the integration kernel over "
           "the 400&#160;ms window present on every trial.")
rep.p(
    "The middle panel is the control that matters. If the network had "
    "learned a fixed-window strategy — integrate for 600&#160;ms then stop — "
    "accuracy on long trials would fall apart, because the pulses that "
    "arrived later would be ignored while the correct answer still counted "
    "them. With evidence held in a narrow band, accuracy stays roughly "
    f"level from {dur_x[0]:.0f} to {dur_x[-1]:.0f}&#160;ms.")
rep.p(
    "The kernel is close to flat, meaning early pulses influenced the choice "
    "about as much as late ones. That is the behavioural signature of "
    "near-lossless integration, and it sets up a quantitative question: how "
    "close to lossless, and what in the network's dynamics sets the limit?")
rep.pagebreak()

rep.h1("3. Activity: a one-dimensional running total")
rep.figure(f_act,
           "<b>Top row:</b> the most evidence-selective unit; mean state "
           "coordinate for trials binned by evidence, showing separation that "
           "persists rather than decaying; and the moment-by-moment "
           "correlation between the state coordinate and the running integral "
           "of the input. <b>Bottom row:</b> population trajectories fanning "
           "out along one axis; the state coordinate at window end against "
           "the true count; variance per principal component.")
rep.p(
    f"The population is close to one-dimensional during the stimulus "
    f"(participation ratio {pr:.2f}; PC1 alone captures {float(evr[0]):.0%} "
    f"of the variance). Along that single axis, the state coordinate tracks "
    f"the running integral of the input with a correlation averaging "
    f"{np.mean(r_running):.2f} at every moment of the window, and by the end "
    f"of the shared window it predicts the realized pulse difference at "
    f"|r| = {abs(r_end):.2f}.")
rep.p(
    "So the network has an analogue register: a single direction of state "
    "space whose coordinate is the running count. The activity analysis "
    "establishes that this register exists. It does not explain <i>why the "
    "register does not leak</i> — why a network of 200&#160;ms units can hold "
    "a total for over a second. For that we need the dynamics.")
rep.pagebreak()

rep.h1("4. Geometry: the line attractor")
rep.p(
    "A register that neither decays nor explodes requires the dynamics to be "
    "<b>marginally stable</b> along the direction it uses. The way to check "
    "is to find the states where the dynamics stop — the fixed and slow "
    "points — and examine the eigenvalues of the recurrent Jacobian there.")
rep.p(
    "Starting from 80 states sampled from real trajectories, minimizing the "
    f"speed &#189;&#8214;F(h,x)&#8722;h&#8214;<super>2</super> converged to "
    f"{len(sp_u)} distinct points. They are not scattered: "
    f"{line_frac:.0%} of their spread lies along a single dimension. The "
    "network built a <b>line attractor</b>.")
rep.figure(f_geom,
           "<b>Panel 1:</b> the slow points (black) overlaid on the "
           "trajectories — they trace out the axis the trajectories ride. "
           "<b>Panel 2:</b> the top |&#955;| at each slow point, clustered at 1. "
           "<b>Panel 3:</b> stability along the line; saddles in red. "
           "<b>Panel 4:</b> the behavioural kernel against the decay predicted "
           "by the attractor's eigenvalue.")
rep.table(
    ["Measurement", "Value", "Interpretation"],
    [["distinct slow points", f"{len(sp_u)}", "found from trajectory states"],
     ["one-dimensionality of their spread", f"{line_frac:.0%}",
      "they form a line, not a cloud"],
     ["median top |&#955;|", f"{lam_line:.4f}",
      "marginal stability — the definition of a line attractor"],
     ["cumulative gain over the window",
      f"{lam_line**n_common:.2f}&#215;",
      f"what an input from the start of the window is worth at the end "
      f"(1.00 = lossless)"],
     ["implied memory &#964;", tau_str,
      f"vs single-unit &#964; = {dt/alpha:.0f} ms"],
     ["saddles among them", f"{n_saddle} of {len(sp_u)}",
      "one unstable direction each; they separate the two choice basins"]],
    widths=[2.3, 1.1, 3.0])
rep.p(
    f"<b>The prediction.</b> A retention factor of {lam_line:.4f} per "
    f"{dt}&#160;ms step means evidence from t steps ago is still worth "
    f"{lam_line:.4f}<super>t</super> of its original weight. Over the "
    f"{n_common}-step shared window that predicts a recency ratio of "
    f"{pred_ratio:.2f}&#215; between the last and the first third of the "
    f"kernel — that is, a <i>flat</i> kernel. The measured ratio is "
    f"{meas_ratio:.2f}&#215;. Both numbers sit within a few percent of 1.0: "
    "the kernel is flat because the attractor is marginal, the same fact "
    "arrived at from behaviour and from dynamics independently.")
rep.p(
    "It is worth appreciating how different these two measurements are. The "
    "kernel came from 4000 trials of stimulus-and-choice data, with no "
    "access to the network. The eigenvalue came from a single "
    "eigen-decomposition at states found by an optimizer, with no reference "
    "to behaviour. Contrast this with Experiment 1, where the same "
    "prediction succeeded near threshold and failed on strong trials: there "
    "the state left the region where the linearization held. Here it does "
    "not, because riding a line attractor is precisely the strategy of "
    "<i>staying</i> in a region where the dynamics are linear and marginal.")
rep.box(
    "<b>What the geometry bought us.</b> The behavioural kernel says the "
    "network integrates well but cannot say why, and cannot say when it "
    "would stop working. The attractor's eigenvalue supplies the mechanism — "
    "a marginally stable line built by W<sub>rec</sub> — and converts "
    f"'integrates well' into a number: a cumulative gain of "
    f"{lam_line**n_common:.2f}&#215; across the window, i.e. lossless to "
    f"within a few percent. It also identifies the failure mode to watch "
    f"for. Because |&#955;| sits a hair <i>above</i> 1, the register very "
    f"slowly amplifies rather than decays; on stimuli far longer than the "
    f"{float(n_stim.max())*dt:.0f}&#160;ms used in training that "
    f"amplification would eventually saturate the units and destroy the "
    f"count.")
rep.h2("Caveats specific to this measurement")
rep.p(
    "The attractor is conditional on the constant input used in the search "
    "(the mean stimulus): a different held input can deform or destroy it. "
    "The slow points are the output of an optimizer, so absence of a point "
    "is not proof of absence. And 'line attractor' is a description of "
    f"{len(sp_u)} sampled points with {line_frac:.0%} one-dimensional spread, "
    "not a proof of a continuous invariant manifold — establishing that "
    "would require continuation methods.")
rep.build()
print("Done.")
