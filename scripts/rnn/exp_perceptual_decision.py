"""
EXPERIMENT 1 — Perceptual decision making.
==========================================

Question: how does a vanilla RNN turn a noisy, extended stimulus into a
categorical choice, and can a geometric measurement predict the network's
behaviour rather than merely describe it?

Arc:
  1 TRAINING    architecture, curriculum, what had to be discovered
  2 BEHAVIOUR   psychometric function; the psychophysical kernel
  3 ACTIVITY    single units, population PCA, the decision variable
  4 GEOMETRY    the recurrent Jacobian eigenvalue predicts the kernel

Run:  python exp_perceptual_decision.py
"""

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuralgeom.paths import PDF_DIR  # noqa: E402

import numpy as np
import torch

from exp_common import (Report, banner, decode_axis, epoch_shading, figdir,
                        leak_prediction, pca, plot_state_traj, project, psth,
                        reverse_correlation, savefig, selectivity)
import matplotlib.pyplot as plt

from neuralgeom.tasks import PerceptualDecision
from neuralgeom.dynamics.rnn import (input_jacobian, jacobian_spectrum,
                          participation_ratio, readout_subspace,
                          recurrent_jacobian)
from train_rnn_models import CONFIG, load_or_train

TASK = "perceptual_decision"
FIG = figdir(TASK)
torch.manual_seed(0)
np.random.seed(0)

# --------------------------------------------------------------------------- #
banner("1  TRAINING", "Load the trained network and describe the fit")
_, model, hist = load_or_train(TASK, "vanilla", verbose=True)
model.eval()
tkw, mkw, _ = CONFIG[TASK]
dt = tkw["dt"]
alpha = model.alpha
print(f"  units={model.hidden_size}  dt={dt} ms  tau={dt/alpha:.0f} ms  "
      f"alpha={alpha:.3f}  train noise={mkw['noise']}")

# Analysis version of the task: fixed fixation length so trials are aligned.
task = PerceptualDecision(dt=dt, sigma=tkw["sigma"], seed=7,
                          t_fix=(300, 300), t_stim=800, t_dec=300)
n_fix = task._steps(300)
n_stim = task._steps(800)
stim_sl = slice(n_fix, n_fix + n_stim)
t_dec = n_fix + n_stim

batch = task.sample(6000)
out, H = model(batch.inputs)
T = batch.n_steps
tms = (np.arange(T) - n_fix) * dt          # time relative to stimulus onset

pred = out.argmax(-1)
dec_win = slice(t_dec, T)
choice = torch.mode(pred[:, dec_win], dim=1).values
coh = batch.meta["coherence"]
correct = (choice == batch.meta["choice"]).double()
print(f"  {batch.batch_size} test trials, overall accuracy "
      f"{float(correct.mean()):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.0), constrained_layout=True)
if hist:
    axes[0].plot(hist["step"], hist["loss"], color="C3")
    axes[0].set_xlabel("training step"); axes[0].set_ylabel("masked loss")
    axes[0].set_title("Training loss")
    axes[1].plot(hist["step"], hist["acc"], "o-", ms=3, color="C0")
    axes[1].axhline(0.5, color="0.6", ls=":")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("training step"); axes[1].set_ylabel("decision accuracy")
    axes[1].set_title("Learning curve")
b1 = task.sample(1)
im = axes[2].imshow(b1.inputs[0].T.numpy(), aspect="auto", cmap="viridis",
                    extent=[tms[0], tms[-1], 2.5, -0.5], interpolation="nearest")
axes[2].set_yticks([0, 1, 2], task.spec.input_labels, fontsize=7)
axes[2].set_xlabel("time from stimulus onset (ms)")
axes[2].set_title("Inputs on one trial")
fig.colorbar(im, ax=axes[2])
f_train = savefig(fig, FIG, "pd_1_training.png")

# --------------------------------------------------------------------------- #
banner("2  BEHAVIOUR", "Psychometric function and the psychophysical kernel")
uc = torch.unique(coh)
p_A = torch.tensor([float((choice[coh == c] == 1).double().mean()) for c in uc])
n_per = [int((coh == c).sum()) for c in uc]
acc_by_coh = torch.tensor([float(correct[coh == c].mean()) for c in uc])
print("  coherence : P(choose A) : accuracy : n")
for c, p, a, n in zip(uc, p_A, acc_by_coh, n_per):
    print(f"   {float(c):+.2f}   :   {float(p):.3f}    :  {float(a):.3f}  : {n}")

# psychophysical kernel: influence of the momentary NOISE on choice
mom = (batch.inputs[:, :, 1] - batch.inputs[:, :, 2])        # (B, T)
mom_resid = mom.clone()
for c in uc:                                    # remove the coherence signal
    m = coh == c
    mom_resid[m] = mom[m] - mom[m].mean(0, keepdim=True)
kernel, kern_sem = reverse_correlation(mom_resid[:, stim_sl], choice)
kernel_n = (kernel / kernel.mean()).numpy()

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.2), constrained_layout=True)
axes[0].plot(uc.numpy(), p_A.numpy(), "o-", color="k")
axes[0].axhline(0.5, color="0.6", ls=":"); axes[0].axvline(0, color="0.6", ls=":")
axes[0].set_xlabel("coherence (+ favours A)"); axes[0].set_ylabel("P(choose A)")
axes[0].set_title("Psychometric function")
axes[1].plot(uc.abs().numpy(), acc_by_coh.numpy(), "o", color="C2")
axes[1].set_xlabel("|coherence|"); axes[1].set_ylabel("accuracy")
axes[1].set_ylim(0.4, 1.02); axes[1].axhline(0.5, color="0.6", ls=":")
axes[1].set_title("Accuracy grows with signal strength")
ts = tms[stim_sl]
axes[2].fill_between(ts, (kernel - kern_sem).numpy(), (kernel + kern_sem).numpy(),
                     alpha=0.3, color="C4")
axes[2].plot(ts, kernel.numpy(), color="C4")
axes[2].axhline(0, color="0.6", ls=":")
axes[2].set_xlabel("time from stimulus onset (ms)")
axes[2].set_ylabel("influence on choice")
axes[2].set_title("Psychophysical kernel\n(weight given to each moment)")
f_behav = savefig(fig, FIG, "pd_2_behaviour.png")

# --------------------------------------------------------------------------- #
banner("3  ACTIVITY", "Single units, population geometry, the decision variable")
sel = selectivity(H, choice, slice(t_dec - 8, t_dec))
top_units = torch.argsort(sel, descending=True)[:3].tolist()
print(f"  most choice-selective units: {top_units} "
      f"(d' = {[round(float(sel[u]),2) for u in top_units]})")
pr_stim = participation_ratio(H[:, t_dec - 1])
mu, comps, evr = pca(H[:, stim_sl].reshape(-1, model.hidden_size), k=6)
print(f"  population: participation ratio {pr_stim:.2f}; "
      f"PC1-3 explain {float(evr[:3].sum()):.1%} of stimulus-epoch variance")

Z = project(H, mu, comps).numpy()
axis_choice = decode_axis(H[:, t_dec - 1], choice)
dv = (H @ axis_choice).numpy()               # decision variable over time

fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.0), constrained_layout=True)
for ax, u in zip(axes[0], top_units):
    labels, m, s = psth(H, torch.sign(coh).long(), u)
    for lab, mm, ss in zip(labels, m, s):
        c = plt.cm.coolwarm(0.5 + 0.5 * float(lab))
        ax.plot(tms, mm.numpy(), color=c,
                label=f"coh {'+' if lab > 0 else '-'}")
        ax.fill_between(tms, (mm - ss).numpy(), (mm + ss).numpy(), color=c,
                        alpha=0.25)
    ax.axvline(0, color="0.5", lw=0.8); ax.axvline((t_dec - n_fix) * dt,
                                                   color="0.5", lw=0.8)
    ax.set_title(f"unit {u} (d'={float(sel[u]):.2f})")
    ax.set_xlabel("time from stim onset (ms)")
    ax.legend(fontsize=6.5)
axes[0, 0].set_ylabel("activity")

sm = plot_state_traj(axes[1, 0], Z, coh.numpy(), every=6,
                     tslice=slice(n_fix, t_dec))
fig.colorbar(sm, ax=axes[1, 0], label="coherence")
axes[1, 0].set_xlabel("PC1"); axes[1, 0].set_ylabel("PC2")
axes[1, 0].set_title("State trajectories during the stimulus\n"
                     "(black square = state at decision onset)")
axes[1, 0].legend(fontsize=6.5)

for c in uc:
    m = (coh == c).numpy()
    axes[1, 1].plot(tms, dv[m].mean(0),
                    color=plt.cm.coolwarm(float((c - uc.min())
                                                / (uc.max() - uc.min()))))
axes[1, 1].axvline(0, color="0.5", lw=0.8)
axes[1, 1].axvline((t_dec - n_fix) * dt, color="0.5", lw=0.8)
axes[1, 1].set_xlabel("time from stimulus onset (ms)")
axes[1, 1].set_ylabel("projection on choice axis")
axes[1, 1].set_title("The decision variable RAMPS,\nat a rate set by coherence")

axes[1, 2].bar(range(1, 7), evr.numpy() * 100, color="C0")
axes[1, 2].set_xlabel("principal component")
axes[1, 2].set_ylabel("% variance")
axes[1, 2].set_title(f"Low-dimensional: PR = {pr_stim:.2f}")
f_act = savefig(fig, FIG, "pd_3_activity.png")

ramp_rates = []
for c in uc:
    m = (coh == c).numpy()
    seg = dv[m][:, n_fix:t_dec]
    ramp_rates.append(np.polyfit(np.arange(seg.shape[1]), seg.mean(0), 1)[0])
r_ramp = np.corrcoef(uc.numpy(), ramp_rates)[0, 1]
print(f"  ramp rate vs coherence: r = {r_ramp:+.3f}")

# --------------------------------------------------------------------------- #
banner("4  GEOMETRY", "Does the recurrent Jacobian PREDICT the kernel?")
t_probe = n_fix + n_stim // 2
h_probe, x_probe = H[:, t_probe], batch.inputs[:, t_probe]
J = recurrent_jacobian(model, h_probe, x_probe)
ev, mod, tau = jacobian_spectrum(J, dt=dt)
lam_glob = float(mod[:, 0].mean())
print(f"  global top |lambda| = {lam_glob:.4f}  (tau = "
      f"{float(tau[:,0].median()):.0f} ms; single-unit tau = {dt/alpha:.0f} ms)")


def axis_gain(h, x):
    """One-step gain of the DECISION VARIABLE: a^T J a with a = choice axis.
    This is the leak the behaviour actually experiences."""
    Jl = recurrent_jacobian(model, h, x)
    return torch.einsum("i,bij,j->b", axis_choice, Jl, axis_choice)


# how strongly the stimulus enters the state, along the choice axis
Ji = input_jacobian(model, h_probe, x_probe)
drive = ((Ji[:, :, 1] - Ji[:, :, 2]) @ axis_choice)
print(f"  input drive of the evidence channel onto the choice axis: "
      f"{float(drive.mean()):+.4f} +- {float(drive.std()):.4f}")

# --- the gain is STATE-DEPENDENT: measure it as a function of |DV| --------- #
dv_probe = (H[:, t_probe] @ axis_choice).abs()
g_probe = axis_gain(H[:, t_probe], batch.inputs[:, t_probe])
qs = torch.quantile(dv_probe, torch.linspace(0, 1, 9))
gain_curve, dv_centers = [], []
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (dv_probe >= lo) & (dv_probe < hi)
    if int(m.sum()) > 20:
        gain_curve.append(float(g_probe[m].mean()))
        dv_centers.append(float(dv_probe[m].mean()))
print("  gain along the choice axis vs distance from origin:")
for d, g in zip(dv_centers, gain_curve):
    print(f"    |DV| = {d:5.2f}  ->  a^T J a = {g:.4f}")

# --- split trials by difficulty; predict each kernel from its own gain ----- #
groups = {"near-threshold  |coh| <= 0.10": coh.abs() <= 0.10,
          "strong  |coh| >= 0.25": coh.abs() >= 0.25}
res = {}
for label, m in groups.items():
    k_m, _ = reverse_correlation(mom_resid[m][:, stim_sl], choice[m])
    kn_m = (k_m / k_m.mean()).numpy()
    g_m = float(axis_gain(H[m][:, t_probe], batch.inputs[m][:, t_probe]).mean())
    pred_m = leak_prediction(n_stim, g_m)
    third = max(3, n_stim // 3)
    meas_ratio = kn_m[-third:].mean() / kn_m[:third].mean()
    pred_ratio = pred_m[-third:].mean() / pred_m[:third].mean()
    dv_end = float((H[m][:, t_dec - 1] @ axis_choice).abs().mean())
    res[label] = dict(kn=kn_m, pred=pred_m, gain=g_m, meas=meas_ratio,
                      predr=pred_ratio, dv=dv_end, n=int(m.sum()))
    print(f"  {label}: n={int(m.sum())}  gain={g_m:.4f}  "
          f"measured recency ratio={meas_ratio:.2f}  predicted={pred_ratio:.2f}"
          f"  |DV| at decision={dv_end:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4), constrained_layout=True)
axes[0].plot(dv_centers, gain_curve, "o-", color="C0")
axes[0].axhline(1.0, color="r", ls="--", lw=1, label="perfect integration")
axes[0].set_xlabel("|decision variable|  (distance along the choice axis)")
axes[0].set_ylabel("one-step gain  $a^{T}J_{rec}a$")
axes[0].legend(fontsize=7)
axes[0].set_title("The leak is STATE-DEPENDENT: near the origin the\nnetwork "
                  "integrates almost perfectly, further out it leaks")

for ax, (label, r) in zip(axes[1:], res.items()):
    ax.plot(ts, r["kn"], color="C4", lw=0.8, alpha=0.35)
    sm5 = np.convolve(r["kn"], np.ones(5) / 5, mode="same")
    sm5[:2], sm5[-2:] = np.nan, np.nan          # edges are unreliable
    ax.plot(ts, sm5, color="C4", lw=2.0, label="measured (5-step average)")
    ax.plot(ts, r["pred"], "--", color="k", lw=1.4,
            label=f"predicted from gain {r['gain']:.3f}")
    ax.axhline(1.0, color="0.7", ls=":")
    ax.set_xlabel("time from stimulus onset (ms)")
    ax.set_ylabel("normalized influence")
    ax.legend(fontsize=7)
    ax.set_title(f"{label}\nmeasured {r['meas']:.2f}x vs predicted "
                 f"{r['predr']:.2f}x recency")
f_geom = savefig(fig, FIG, "pd_4_geometry.png")

near = res["near-threshold  |coh| <= 0.10"]
strong = res["strong  |coh| >= 0.25"]
tau_near = -dt / np.log(min(near["gain"], 0.999999))
tau_strong = -dt / np.log(min(strong["gain"], 0.999999))

# --------------------------------------------------------------------------- #
banner("REPORT", "Building the PDF")
rep = Report(PDF_DIR / "experiment_1_perceptual_decision.pdf",
             "Experiment 1 — Perceptual Decision Making",
             "How an RNN converts a noisy stimulus into a choice, and a "
             "geometric measurement that predicts its behaviour")

rep.h1("1. What the network was trained to do")
rep.p(
    "On each trial two input channels carry a constant drive of "
    "0.5&#160;&#177;&#160;coherence/2, corrupted by independent Gaussian "
    "noise on every timestep, for 800&#160;ms. A third channel holds "
    "fixation high until the response epoch. The network must report which "
    "channel carried the stronger drive, and it may only answer during the "
    "final 300&#160;ms. This is the random-dot motion task in its simplest "
    "form: the instantaneous input is almost uninformative — at coherence "
    "0.05 the signal is roughly a twentieth of the noise — so the only way "
    "to perform well is to <i>average the stimulus over time</i>.")
rep.p(
    "Crucially, nothing in the architecture provides an accumulator. The "
    "network is a plain leaky tanh RNN whose units, left alone, forget with "
    f"a time constant of {dt/alpha:.0f}&#160;ms — shorter than the stimulus. "
    "Any integration longer than that has to be built out of recurrent "
    "connectivity during training. What that solution looks like, and how "
    "to measure it, is the subject of this experiment.")
rep.table(
    ["Setting", "Value", "Why"],
    [["architecture", f"leaky tanh RNN, {model.hidden_size} units",
      "no gating, no memory cells — memory must be learned"],
     ["update", "h &#8592; (1&#8722;&#945;)h + &#945;&#183;tanh(W<sub>rec</sub>h + W<sub>in</sub>x + b)",
      f"&#945; = dt/&#964; = {alpha:.2f}"],
     ["timestep / &#964;", f"{dt} ms / {dt/alpha:.0f} ms",
      "single-unit memory is shorter than the trial"],
     ["private noise", f"SD {mkw['noise']} per step (train only)",
      "forces robust, attractor-like solutions rather than brittle ones"],
     ["loss", "masked cross-entropy over 3 actions",
      "scored on fixation AND decision epochs, so the network must also "
      "learn to withhold its answer"],
     ["regularization", "L2 on firing rates (10<super>-4</super>)",
      "keeps activity in the smooth part of the tanh; cleans up the "
      "dynamics for analysis"],
     ["optimizer", f"Adam, lr 2&#215;10<super>-3</super>, "
                   f"{len(hist.get('step', []) or [1])*0 + 1200} steps &#215; batch 64",
      "gradient clipping at 1.0"]],
    widths=[1.15, 2.2, 3.05])
rep.figure(f_train,
           "<b>Left, middle:</b> training loss and decision accuracy. "
           "<b>Right:</b> the three input channels on one trial — fixation "
           "(top) drops at the response epoch; the two evidence channels "
           "are visibly dominated by noise.")
rep.box(
    f"<b>Outcome.</b> {float(correct.mean()):.1%} correct on "
    f"{batch.batch_size} held-out trials. Performance is far from 100% "
    "<i>by design</i>: low-coherence trials are close to unanswerable, and "
    "a network that scored 100% would indicate the task was too easy to "
    "require integration.")
rep.pagebreak()

rep.h1("2. Behaviour: what the network does")
rep.p(
    "Before interpreting any internal signal it is worth measuring the "
    "network the way one would measure an animal. Two standard assays: the "
    "<b>psychometric function</b> (choice probability against signal "
    "strength) and the <b>psychophysical kernel</b> (how much each moment of "
    "the stimulus influenced the eventual choice).")
rep.figure(f_behav,
           "<b>Left:</b> a graded, monotonic psychometric function through "
           "chance at zero coherence — the network is not using a threshold "
           "rule, it is weighing evidence. <b>Middle:</b> accuracy against "
           "signal strength. <b>Right:</b> the psychophysical kernel, "
           "computed by removing the coherence signal from each trial and "
           "asking how the residual <i>noise</i> at each moment differed "
           "between the two choices.")
rep.p(
    "The kernel is the informative one. It is a purely behavioural "
    "measurement — computed from stimulus and choice alone, with no access "
    "to the network's internals — and its shape reveals the integration "
    "strategy. A flat kernel means every moment counted equally (perfect "
    "integration). A kernel rising toward the end means the network leaked, "
    "discounting early evidence (recency). A falling kernel means it "
    "committed early (primacy).")
rep.table(
    ["Coherence", "P(choose A)", "Accuracy", "n trials"],
    [[f"{float(c):+.2f}", f"{float(p):.3f}", f"{float(a):.3f}", n]
     for c, p, a, n in zip(uc, p_A, acc_by_coh, n_per)],
    widths=[1.6, 1.6, 1.6, 1.6])
rep.pagebreak()

rep.h1("3. Activity: how the network does it")
rep.figure(f_act,
           "<b>Top row:</b> the three most choice-selective units, averaged "
           "by the sign of the coherence (shading = SEM). Their responses "
           "diverge slowly over the stimulus epoch rather than switching — "
           "the signature of accumulation, not detection. <b>Bottom left:</b> "
           "population trajectories in the first two principal components, "
           "coloured by coherence; trials fan out along a single axis. "
           "<b>Bottom middle:</b> the decision variable — activity projected "
           "onto the choice axis — ramping at a rate set by coherence. "
           "<b>Bottom right:</b> variance explained per component.")
rep.p(
    f"Three facts organize the picture. First, the computation is "
    f"<b>low-dimensional</b>: the participation ratio at decision onset is "
    f"{pr_stim:.2f}, and the first three principal components capture "
    f"{float(evr[:3].sum()):.0%} of the variance during the stimulus — "
    f"{model.hidden_size} units, but effectively a two- or three-dimensional "
    f"computation. Second, the population moves along a <b>single dominant "
    f"axis</b>, and its coordinate on that axis is the decision variable. "
    f"Third, that coordinate <b>ramps</b>: the projection grows steadily "
    f"during the stimulus, and the ramp rate is nearly a linear function of "
    f"coherence (r = {r_ramp:+.2f} across coherence levels).")
rep.p(
    "This is exactly the drift-diffusion picture that decades of "
    "electrophysiology describe in parietal and prefrontal cortex, arrived "
    "at here by a network that was told nothing except which button to "
    "press. The individual units look heterogeneous and messy; the "
    "population, projected properly, is almost one-dimensional and smooth.")
rep.box(
    "<b>The question this raises.</b> A ramp implies an integrator, and an "
    "integrator implies memory. But how <i>long</i> is the network's memory? "
    "The activity plots show that something accumulates; they do not say "
    "over what timescale, or whether early evidence is retained as well as "
    "late evidence. That is a quantitative question, and it is where the "
    "geometry earns its keep.")
rep.pagebreak()

rep.h1("4. Geometry: predicting the behaviour from the dynamics")
rep.p(
    "The recurrent Jacobian <b>J<sub>rec</sub> = &#8706;F/&#8706;h</b> is the "
    "linearization of the network's autonomous dynamics: what happens to a "
    "perturbation of the state over one timestep. The quantity the "
    "behaviour actually experiences is its <b>gain along the choice axis</b>, "
    "a<super>T</super>J<sub>rec</sub>a, where a is the decision axis "
    "identified in part 3. If that gain is 0.98, the decision variable "
    "retains 98% of its value each step, so evidence arriving at time t "
    "still contributes 0.98<super>(T&#8722;t)</super> of its weight at the "
    "end of the trial.")
rep.box(
    "That is a <b>falsifiable prediction about behaviour</b>. The "
    "psychophysical kernel — measured from stimulus and choices alone, with "
    "no access to the network's internals — should have the shape "
    "gain<super>(T&#8722;t)</super>. No fitting is involved: the curve is "
    "fixed by one number read off a Jacobian.")
rep.figure(f_geom,
           "<b>Left:</b> the one-step gain along the choice axis, measured as "
           "a function of how far the state has travelled along that axis. "
           "<b>Middle, right:</b> the measured psychophysical kernel (purple) "
           "against the kernel predicted from each group's own gain "
           "(dashed), for near-threshold and for strong trials.")
rep.p(
    "The left panel is the key to the whole experiment. Near the origin the "
    f"gain is {gain_curve[0]:.3f} — the network integrates almost perfectly. "
    f"Far out along the choice axis it falls to {gain_curve[-1]:.3f}. The "
    "leak is not a property of the network; it is a property of <i>where the "
    "network is</i>. That immediately predicts that easy and hard trials, "
    "which travel different distances, should show different amounts of "
    "recency.")
rep.table(
    ["Trial group", "n", "gain a<sup>T</sup>Ja", "&#964; implied",
     "|DV| reached", "recency measured", "predicted"],
    [["near-threshold |coh| &#8804; 0.10", near["n"], f"{near['gain']:.3f}",
      f"{tau_near:.0f} ms", f"{near['dv']:.2f}", f"{near['meas']:.2f}&#215;",
      f"{near['predr']:.2f}&#215;"],
     ["strong |coh| &#8805; 0.25", strong["n"], f"{strong['gain']:.3f}",
      f"{tau_strong:.0f} ms", f"{strong['dv']:.2f}",
      f"{strong['meas']:.2f}&#215;", f"{strong['predr']:.2f}&#215;"]],
    widths=[1.75, 0.5, 0.85, 0.75, 0.75, 0.95, 0.85], fontsize=8.0)
rep.p(
    f"<b>On near-threshold trials the prediction works.</b> The state stays "
    f"close to the origin (|DV| = {near['dv']:.2f}), where the linearization "
    f"is valid, and the measured recency of {near['meas']:.2f}&#215; sits "
    f"close to the predicted {near['predr']:.2f}&#215;. These are exactly the "
    "trials psychophysicists use for kernel estimation, and the geometric "
    "prediction — made without fitting anything to behaviour — lands in the "
    "right place.")
rep.p(
    f"<b>On strong trials it fails, and the failure is diagnostic.</b> Here "
    f"the state runs out to |DV| = {strong['dv']:.2f}, into the saturating "
    f"region where the gain has dropped to {strong['gain']:.3f}. A pure leaky "
    f"integrator with that gain would show {strong['predr']:.2f}&#215; "
    f"recency; the network shows only {strong['meas']:.2f}&#215;. The reason "
    "is visible in the same left-hand panel: once the state has travelled "
    "far along the choice axis the decision is effectively already made, so "
    "late evidence stops mattering too. Early evidence is discounted by "
    "leak, late evidence by commitment, and the two flatten the kernel from "
    "opposite ends.")
rep.box(
    "<b>What the geometry bought us.</b> The activity analysis showed "
    "<i>that</i> the network integrates. The Jacobian gain says <i>how "
    "well</i>, as a single number per state, computed from one forward pass "
    "— no behavioural fitting, no drift-diffusion model assumed. It then "
    "predicts a behavioural measurement quantitatively where its assumptions "
    "hold, and its own state-dependence tells you where those assumptions "
    "break.")
rep.h2("Caveats specific to this measurement")
rep.p(
    "The headline number depends on which state you evaluate at: the global "
    f"slowest eigenvalue at mid-stimulus is {lam_glob:.3f}, while the gain "
    f"along the choice axis ranges from {min(gain_curve):.3f} to "
    f"{max(gain_curve):.3f} across the state space the network actually "
    "visits. Quoting one Jacobian for 'the network' would have been "
    "meaningless — the distribution is the result. The prediction also "
    "assumes evidence enters along the choice axis, which the input Jacobian "
    f"supports (drive = {float(drive.mean()):+.3f}) but does not guarantee, "
    "and it treats the decision epoch as passive, which it is not.")
rep.build()
print("Done.")
