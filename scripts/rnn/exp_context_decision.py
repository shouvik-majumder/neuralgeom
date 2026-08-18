"""
EXPERIMENT 3 — Context-dependent decision making (Mante / Sussillo).
====================================================================

Question: the same two features arrive on every trial; a cue says which one
matters. Where does the irrelevant feature go? Is it filtered at the input,
or represented and then ignored? And can a geometric measurement predict
how much it leaks into behaviour?

Arc:
  1 TRAINING    the task, and why input filtering is impossible here
  2 BEHAVIOUR   psychometrics per context; regression weights on both features
  3 ACTIVITY    both features are represented regardless of context
  4 GEOMETRY    input-Jacobian gain onto the decision plane PREDICTS the
                behavioural regression weights

Run:  python exp_context_decision.py
"""

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuralgeom.paths import PDF_DIR  # noqa: E402

import numpy as np
import torch

import matplotlib.pyplot as plt
from exp_common import (Report, banner, decode_axis, figdir, logistic_weights,
                        pca, plot_state_traj, project, savefig)

from neuralgeom.geometry.grassmann import grassmann_distance, principal_angles
from neuralgeom.tasks import ContextDecision
from neuralgeom.dynamics.rnn import (input_jacobian, participation_ratio,
                          readout_subspace, recurrent_jacobian,
                          trajectory_subspaces)
from train_rnn_models import CONFIG, load_or_train

TASK = "context_decision"
FIG = figdir(TASK)
torch.manual_seed(0)
np.random.seed(0)
CTX_NAME = {0: "motion", 1: "colour"}

# --------------------------------------------------------------------------- #
banner("1  TRAINING", "Load the network; state the computational problem")
_, model, hist = load_or_train(TASK, "vanilla", verbose=True)
model.eval()
tkw, mkw, _ = CONFIG[TASK]
dt = tkw["dt"]
alpha = model.alpha
task = ContextDecision(dt=dt, sigma=tkw["sigma"], seed=13)
n_fix = task._steps(task.t_fix)
n_stim = task._steps(task.t_stim)
t_dec = n_fix + n_stim

batch = task.sample(6000)
out, H = model(batch.inputs)
T = batch.n_steps
tms = (np.arange(T) - n_fix) * dt
choice = torch.mode(out[:, t_dec:].argmax(-1), dim=1).values
ctx = batch.meta["context"]
coh_m, coh_c = batch.meta["coh_motion"], batch.meta["coh_colour"]
rel = batch.meta["relevant_coh"]
irr = torch.where(ctx == 0, coh_c, coh_m)
correct = (choice == batch.meta["choice"]).double()
print(f"  units={model.hidden_size}  accuracy {float(correct.mean()):.3f} "
      f"({int((ctx==0).sum())} motion / {int((ctx==1).sum())} colour trials)")

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.0), constrained_layout=True)
if hist:
    axes[0].plot(hist["step"], hist["loss"], color="C3")
    axes[0].set_xlabel("training step"); axes[0].set_ylabel("masked loss")
    axes[0].set_title("Training loss")
    axes[1].plot(hist["step"], hist["acc"], "o-", ms=3, color="C0")
    axes[1].set_ylim(0, 1.02); axes[1].axhline(0.5, color="0.6", ls=":")
    axes[1].set_xlabel("training step"); axes[1].set_title("Learning curve")
im = axes[2].imshow(batch.inputs[0].T.numpy(), aspect="auto", cmap="viridis",
                    extent=[tms[0], tms[-1], 6.5, -0.5],
                    interpolation="nearest")
axes[2].set_yticks(range(7), task.spec.input_labels, fontsize=6)
axes[2].set_xlabel("time from stimulus onset (ms)")
axes[2].set_title("Inputs on one trial (both features present)")
fig.colorbar(im, ax=axes[2])
f_train = savefig(fig, FIG, "cd_1_training.png")

# --------------------------------------------------------------------------- #
banner("2  BEHAVIOUR", "Psychometrics per context; behavioural weights")
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.3), constrained_layout=True)
ucoh = torch.unique(coh_m)
beh_w = {}
for c in [0, 1]:
    m = ctx == c
    y = (choice[m] == 1).double().numpy()
    X = np.stack([coh_m[m].numpy(), coh_c[m].numpy()], axis=1)
    w = logistic_weights(X, y)
    beh_w[c] = (float(w[0]), float(w[1]))
    print(f"  context={CTX_NAME[c]}: behavioural weight on motion "
          f"{w[0]:+.2f}, on colour {w[1]:+.2f}")
    ax = axes[c]
    for feat, name, style in [(coh_m, "motion", "-o"), (coh_c, "colour", "--s")]:
        p = [float((choice[m & (feat == u)] == 1).double().mean())
             for u in ucoh]
        ax.plot(ucoh.numpy(), p, style, label=name, ms=4)
    ax.axhline(0.5, color="0.6", ls=":")
    ax.set_xlabel("coherence"); ax.set_ylabel("P(choose A)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title(f"context = {CTX_NAME[c]}\n(solid = cued feature)")

congr = rel.sign() == irr.sign()
acc_cg, acc_cf = float(correct[congr].mean()), float(correct[~congr].mean())
w_rel = 0.5 * (beh_w[0][0] + beh_w[1][1])
w_irr = 0.5 * (beh_w[0][1] + beh_w[1][0])
beh_ratio = w_rel / max(abs(w_irr), 1e-9)
print(f"  congruent {acc_cg:.3f} vs conflict {acc_cf:.3f}")
print(f"  behavioural weight ratio relevant/irrelevant = {beh_ratio:.2f}")
axes[2].bar([0, 1], [w_rel, w_irr], color=["C2", "0.7"], width=0.55)
axes[2].set_xticks([0, 1], ["cued feature", "ignored feature"])
axes[2].set_ylabel("logistic regression weight on choice")
axes[2].set_title(f"Behavioural weights:\ncued counts {beh_ratio:.1f}x more")
f_behav = savefig(fig, FIG, "cd_2_behaviour.png")

# --------------------------------------------------------------------------- #
banner("3  ACTIVITY", "Is the ignored feature represented at all?")
t_end = t_dec - 1
mu, comps, evr = pca(H[:, n_fix:t_dec].reshape(-1, model.hidden_size), k=6)
Z = project(H, mu, comps).numpy()
pr = participation_ratio(H[:, t_end])

# de-mix: axes for context, motion and colour (independent of relevance)
ax_ctx = decode_axis(H[:, t_end], ctx)
ax_mot = decode_axis(H[:, t_end], (coh_m > 0).long())
ax_col = decode_axis(H[:, t_end], (coh_c > 0).long())
print(f"  PR = {pr:.2f}; angle(motion axis, colour axis) = "
      f"{np.degrees(np.arccos(abs(float(ax_mot @ ax_col)))):.0f} deg; "
      f"angle(context axis, motion axis) = "
      f"{np.degrees(np.arccos(abs(float(ax_ctx @ ax_mot)))):.0f} deg")

# how well can each feature be decoded WITHIN each context?
dec_strength = {}
for c in [0, 1]:
    m = ctx == c
    row = []
    for feat, axv in [(coh_m, ax_mot), (coh_c, ax_col)]:
        proj = (H[m][:, t_end] @ axv)
        row.append(abs(float(np.corrcoef(feat[m].numpy(), proj.numpy())[0, 1])))
    dec_strength[c] = row
    print(f"  context={CTX_NAME[c]}: |corr| of state with motion "
          f"{row[0]:.2f}, with colour {row[1]:.2f}")

fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.2), constrained_layout=True)
for c, ax in zip([0, 1], axes[0, :2]):
    m = (ctx == c).numpy()
    sc = ax.scatter(Z[m, t_end, 0], Z[m, t_end, 1], c=coh_m[m].numpy(),
                    cmap="PiYG", s=6)
    fig.colorbar(sc, ax=ax, label="motion coherence")
    ax.set_title(f"context = {CTX_NAME[c]}: states coloured by MOTION")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
sm = plot_state_traj(axes[0, 2], Z, ctx.numpy().astype(float), cmap="coolwarm",
                     every=12, tslice=slice(n_fix, t_dec))
axes[0, 2].set_title("Trajectories, coloured by context")
axes[0, 2].set_xlabel("PC1"); axes[0, 2].set_ylabel("PC2")

x = np.arange(2)
axes[1, 0].bar(x - 0.18, dec_strength[0], 0.36, label="context = motion")
axes[1, 0].bar(x + 0.18, dec_strength[1], 0.36, label="context = colour")
axes[1, 0].set_xticks(x, ["motion feature", "colour feature"])
axes[1, 0].set_ylabel("|corr| of state with feature")
axes[1, 0].legend(fontsize=7)
axes[1, 0].set_title("BOTH features are encoded in BOTH contexts\n"
                     "(the ignored one is not filtered away)")

for c, ls in [(0, "-"), (1, "--")]:
    m = ctx == c
    for feat, axv, col in [(coh_m, ax_mot, "C2"), (coh_c, ax_col, "C4")]:
        tr = [abs(float(np.corrcoef(feat[m].numpy(),
                                    (H[m][:, t] @ axv).numpy())[0, 1]))
              for t in range(n_fix, t_dec)]
        axes[1, 1].plot(tms[n_fix:t_dec], tr, ls, color=col, lw=1.2)
axes[1, 1].set_xlabel("time from stimulus onset (ms)")
axes[1, 1].set_ylabel("|corr| with feature")
axes[1, 1].set_title("Feature encoding over time\n(green = motion, purple = "
                     "colour; dashed = colour context)")

Q_ctx, labels = trajectory_subspaces(H[:, n_fix:t_dec], ctx, k=3)
ang = torch.rad2deg(principal_angles(Q_ctx[0], Q_ctx[1]))
d_ctx = float(grassmann_distance(Q_ctx[0], Q_ctx[1]))
axes[1, 2].bar(range(len(ang)), ang.numpy(), color="C4", width=0.6)
axes[1, 2].set_xticks(range(len(ang)),
                      [f"$\\theta_{i+1}$" for i in range(len(ang))])
axes[1, 2].set_ylabel("principal angle (deg)")
axes[1, 2].set_title(f"Context subspaces on Gr(3, {model.hidden_size}):\n"
                     f"geodesic distance {d_ctx:.2f} rad")
f_act = savefig(fig, FIG, "cd_3_activity.png")

# --------------------------------------------------------------------------- #
banner("4  GEOMETRY", "Input-Jacobian gain predicts the behavioural weights")
R = readout_subspace(model, k=2)                     # the decision plane
t_probe = n_fix + n_stim // 2
geo_gain = {}
for c in [0, 1]:
    m = ctx == c
    h, x = H[m][:, t_probe], batch.inputs[m][:, t_probe]
    Ji = input_jacobian(model, h, x)
    v_m = Ji[:, :, 1] - Ji[:, :, 2]                  # motion drive direction
    v_c = Ji[:, :, 3] - Ji[:, :, 4]                  # colour drive direction
    gm = float((v_m @ R).norm(dim=1).mean())
    gc = float((v_c @ R).norm(dim=1).mean())
    geo_gain[c] = (gm, gc)
    print(f"  context={CTX_NAME[c]}: geometric drive into the decision plane "
          f"— motion {gm:.4f}, colour {gc:.4f}")

g_rel = 0.5 * (geo_gain[0][0] + geo_gain[1][1])
g_irr = 0.5 * (geo_gain[0][1] + geo_gain[1][0])
geo_ratio = g_rel / g_irr
print(f"  geometric gain ratio relevant/irrelevant = {geo_ratio:.2f}")
print(f"  behavioural weight ratio                  = {beh_ratio:.2f}")


def accumulated_sensitivity(h_all, x_all, a, t_from, t_to):
    """Total linear influence of a unit input perturbation at each time on the
    FINAL decision variable, accounting for all the recurrent processing in
    between.

    A one-step Jacobian says how much an input nudges the state now. What
    behaviour depends on is how much it moves the decision variable at the
    END of the trial, after the nudge has been propagated through
    J_rec(t+1) ... J_rec(T-1). Backward (adjoint) recursion:

        z_T = a                       (the choice axis)
        z_s = J_rec(s)^T z_{s+1}
        sensitivity at t = z_{t+1}^T J_inp(t) v

    Returns per-trial totals for the motion and colour drive directions.
    """
    B = h_all.shape[0]
    z = a.expand(B, -1).clone()
    tot_m = torch.zeros(B)
    tot_c = torch.zeros(B)
    for t in range(t_to - 1, t_from - 1, -1):
        Ji_t = input_jacobian(model, h_all[:, t], x_all[:, t])
        vm = Ji_t[:, :, 1] - Ji_t[:, :, 2]
        vc = Ji_t[:, :, 3] - Ji_t[:, :, 4]
        tot_m += torch.einsum("bi,bi->b", z, vm)
        tot_c += torch.einsum("bi,bi->b", z, vc)
        Jr_t = recurrent_jacobian(model, h_all[:, t], x_all[:, t])
        z = torch.einsum("bji,bj->bi", Jr_t, z)        # z <- J^T z
    return tot_m, tot_c


axis_dv = decode_axis(H[:, t_end], choice)
n_sub = 200
acc_gain = {}
for c in [0, 1]:
    idx = torch.where(ctx == c)[0][:n_sub]
    tm, tc = accumulated_sensitivity(H[idx], batch.inputs[idx], axis_dv,
                                     n_fix, t_dec)
    # SIGNED mean = the consistent push on the decision variable (this is what
    # a behavioural regression weight measures).
    # ABSOLUTE mean = the raw magnitude of influence, sign ignored.
    acc_gain[c] = (float(tm.mean()), float(tc.mean()),
                   float(tm.abs().mean()), float(tc.abs().mean()))
    print(f"  context={CTX_NAME[c]}: accumulated sensitivity — "
          f"motion signed {acc_gain[c][0]:+.3f} |{acc_gain[c][2]:.3f}|, "
          f"colour signed {acc_gain[c][1]:+.3f} |{acc_gain[c][3]:.3f}|")
a_rel = 0.5 * (acc_gain[0][0] + acc_gain[1][1])
a_irr = 0.5 * (acc_gain[0][1] + acc_gain[1][0])
a_rel_abs = 0.5 * (acc_gain[0][2] + acc_gain[1][3])
a_irr_abs = 0.5 * (acc_gain[0][3] + acc_gain[1][2])
acc_ratio = abs(a_rel) / max(abs(a_irr), 1e-12)
acc_ratio_abs = a_rel_abs / max(a_irr_abs, 1e-12)
print(f"  ACCUMULATED signed ratio = {acc_ratio:.1f}   "
      f"(magnitude-only ratio {acc_ratio_abs:.1f}; "
      f"one-step {geo_ratio:.2f}; behaviour {beh_ratio:.1f})")
print(f"  -> the irrelevant feature's influence is {a_irr_abs:.2f} in "
      f"magnitude but only {abs(a_irr):.2f} on average: it pushes the "
      f"decision variable INCONSISTENTLY across trials, so it cancels.")

# total input drive (ignoring the readout) — is the input itself gated?
tot = {}
for c in [0, 1]:
    m = ctx == c
    h, x = H[m][:, t_probe], batch.inputs[m][:, t_probe]
    Ji = input_jacobian(model, h, x)
    tot[c] = (float((Ji[:, :, 1] - Ji[:, :, 2]).norm(dim=1).mean()),
              float((Ji[:, :, 3] - Ji[:, :, 4]).norm(dim=1).mean()))
    print(f"  context={CTX_NAME[c]}: TOTAL input drive (any direction) — "
          f"motion {tot[c][0]:.3f}, colour {tot[c][1]:.3f}")
tot_ratio = (0.5 * (tot[0][0] + tot[1][1])) / (0.5 * (tot[0][1] + tot[1][0]))
print(f"  total-drive ratio = {tot_ratio:.2f}  (vs {geo_ratio:.2f} into the "
      f"decision plane)")

fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4), constrained_layout=True)
axes[0].bar([0, 1, 2.5, 3.5],
            [tot[0][0], tot[0][1], tot[1][0], tot[1][1]],
            color=["C2", "0.7", "0.7", "C2"], width=0.8)
axes[0].set_xticks([0, 1, 2.5, 3.5], ["mot", "col", "mot", "col"], fontsize=8)
axes[0].set_xlabel("ctx = motion          ctx = colour")
axes[0].set_ylabel("$\\|J_{inp}v\\|$ (any direction)")
axes[0].set_title(f"TOTAL input drive is barely gated\n(ratio "
                  f"{tot_ratio:.2f}x) — the input is not filtered")
axes[1].bar([0, 1, 2.5, 3.5],
            [geo_gain[0][0], geo_gain[0][1], geo_gain[1][0], geo_gain[1][1]],
            color=["C2", "0.7", "0.7", "C2"], width=0.8)
axes[1].set_xticks([0, 1, 2.5, 3.5], ["mot", "col", "mot", "col"], fontsize=8)
axes[1].set_xlabel("ctx = motion          ctx = colour")
axes[1].set_ylabel("drive INTO the decision plane")
axes[1].set_title(f"Drive into the READOUT plane is gated\n({geo_ratio:.2f}x) "
                  f"— selection is a routing effect")
axes[2].bar([0, 1, 2], [geo_ratio, acc_ratio, beh_ratio],
            color=["C0", "C1", "C4"], width=0.6)
axes[2].set_xticks([0, 1, 2], ["one-step\n$J_{inp}$ gain",
                               "accumulated\nlinear response",
                               "behaviour\n(logistic)"], fontsize=7.5)
axes[2].set_yscale("log")
axes[2].set_ylabel("relevant / irrelevant ratio (log)")
axes[2].axhline(1.0, color="r", ls="--", lw=1, label="no selection")
for i, v in enumerate([geo_ratio, acc_ratio, beh_ratio]):
    axes[2].text(i, v * 1.15, f"{v:.1f}x", ha="center", fontsize=7.5)
axes[2].legend(fontsize=7)
axes[2].set_title("Linearization detects selection but\nUNDERSTATES it: the "
                  "rest is nonlinear")
gvals = [geo_gain[0][0], geo_gain[0][1], geo_gain[1][0], geo_gain[1][1]]
bvals = [beh_w[0][0], beh_w[0][1], beh_w[1][0], beh_w[1][1]]
axes[3].scatter(gvals, bvals, s=60, c=["C2", "0.6", "0.6", "C2"])
for g_, b_, lab in zip(gvals, bvals, ["mot|ctx=mot", "col|ctx=mot",
                                      "mot|ctx=col", "col|ctx=col"]):
    axes[3].annotate(lab, (g_, b_), fontsize=6.5,
                     textcoords="offset points", xytext=(4, 3))
rr = float(np.corrcoef(gvals, bvals)[0, 1])
axes[3].set_xlabel("geometric gain into decision plane")
axes[3].set_ylabel("behavioural regression weight")
axes[3].set_title(f"Per-condition agreement (r = {rr:.2f})")
f_geom = savefig(fig, FIG, "cd_4_geometry.png")
print(f"  per-condition correlation geometry vs behaviour: r = {rr:.3f}")

# --------------------------------------------------------------------------- #
banner("REPORT", "Building the PDF")
rep = Report(PDF_DIR / "experiment_3_context_decision.pdf",
             "Experiment 3 — Context-Dependent Decision Making",
             "Two features, one cue: where does the ignored evidence go?")

rep.h1("1. What the network was trained to do")
rep.p(
    "Every trial presents <b>both</b> features at once — a noisy "
    "'motion' signal and an independent noisy 'colour' "
    "signal, each with its own coherence drawn independently — plus a cue "
    "indicating which of the two determines the correct answer. The network "
    "integrates the cued feature and reports its sign; the other feature is "
    "irrelevant on that trial but equally present, equally strong, and "
    "equally informative-looking.")
rep.p(
    "This is Mante and Sussillo's task, and its importance is that it "
    "<b>rules out input filtering</b>. The two features arrive through fixed "
    "input weights that cannot change between trials, and the context is "
    "only revealed by a cue that arrives through yet another set of fixed "
    "weights. Whatever selection happens must be implemented by the "
    "<i>dynamics</i>, in the same network, on a trial-by-trial basis. The "
    "question this experiment answers is where in the processing chain that "
    "selection actually occurs.")
rep.table(
    ["Setting", "Value"],
    [["architecture", f"leaky tanh RNN, {model.hidden_size} units, "
                      f"&#964; = {dt/alpha:.0f} ms"],
     ["inputs", "7 channels: fixation, motion&#215;2, colour&#215;2, "
                "context cue&#215;2"],
     ["coherences", "&#177;0.15, &#177;0.50, drawn independently per feature"],
     ["stimulus", f"{task.t_stim} ms, then a {task.t_dec} ms response epoch"],
     ["accuracy", f"{float(correct.mean()):.3f} on {batch.batch_size} "
                  f"held-out trials"]],
    widths=[1.3, 5.1])
rep.figure(f_train,
           "<b>Left, middle:</b> training. <b>Right:</b> one trial's inputs — "
           "note that both feature pairs (rows 1-4) are active simultaneously; "
           "only the cue rows (5-6) distinguish trial types.")
rep.pagebreak()

rep.h1("2. Behaviour: the ignored feature barely counts")
rep.figure(f_behav,
           "<b>Left, middle:</b> psychometric curves within each context, "
           "plotted against both features. The cued feature (solid) drives "
           "choice; the ignored feature (dashed) produces a nearly flat line. "
           "<b>Right:</b> logistic-regression weights on choice, averaged "
           "across contexts.")
rep.p(
    f"Fitting a logistic model of choice on both coherences within each "
    f"context gives a weight of {w_rel:+.2f} on the cued feature and "
    f"{w_irr:+.2f} on the ignored one — a ratio of "
    f"<b>{beh_ratio:.2f}&#215;</b>. Accuracy is {acc_cg:.2f} on congruent "
    f"trials and {acc_cf:.2f} on conflict trials, i.e. essentially no "
    f"congruency cost. Behaviourally, the network looks like it is not "
    f"seeing the irrelevant feature at all.")
rep.box(
    "<b>The question.</b> 'Looks like it is not seeing it' has two very "
    "different mechanistic readings. Either the irrelevant feature never "
    "enters the network's state (filtered at the input), or it enters, is "
    "represented, and is then prevented from reaching the decision. "
    "Behaviour cannot distinguish these. Activity can.")
rep.pagebreak()

rep.h1("3. Activity: both features are represented, always")
rep.figure(f_act,
           "<b>Top:</b> states at the end of the stimulus, coloured by motion "
           "coherence, shown separately for each context; and trajectories "
           "coloured by context. <b>Bottom left:</b> how strongly each feature "
           "can be read out of the state, in each context. <b>Bottom middle:</b> "
           "the same across time. <b>Bottom right:</b> principal angles between "
           "the two context subspaces.")
rep.table(
    ["", "motion encoded", "colour encoded"],
    [["context = motion", f"{dec_strength[0][0]:.2f}", f"{dec_strength[0][1]:.2f}"],
     ["context = colour", f"{dec_strength[1][0]:.2f}", f"{dec_strength[1][1]:.2f}"]],
    widths=[2.0, 2.2, 2.2])
rep.p(
    "The irrelevant feature is <b>not</b> filtered out. In both contexts the "
    "state carries a clearly decodable representation of both features, at "
    "comparable strength — the top-row scatter plots show motion coherence "
    "organizing the state cloud even on colour-context trials, when motion "
    "is behaviourally almost ignored. This is the central empirical finding "
    "of the original Mante study, reproduced here.")
rep.p(
    f"What the context <i>does</i> change is where the computation lives. The "
    f"cue displaces the entire trajectory to a different region of state "
    f"space, and the two context subspaces sit {d_ctx:.2f} rad apart on the "
    f"Grassmannian, with principal angles of "
    f"{', '.join(f'{float(a):.0f}&#176;' for a in ang)} — overlapping but "
    f"clearly distinct. The activity analysis thus sharpens the question "
    f"rather than answering it: if the irrelevant feature is in there, what "
    f"stops it from reaching the output?")
rep.pagebreak()

rep.h1("4. Geometry: selection is a routing effect, and it is quantitative")
rep.p(
    "The input Jacobian <b>J<sub>inp</sub> = &#8706;F/&#8706;x</b> gives the "
    "direction in state space that each input channel pushes the state, "
    "evaluated at the state the network is actually in. Combining it with "
    "the readout subspace R — the plane the output units read — separates "
    "two questions that behaviour conflates:")
rep.deflist([
    ("Is the input gated?",
     "&#8214;J<sub>inp</sub>v&#8214; for the feature's drive direction v: "
     "how hard the feature pushes the state <i>in any direction</i>."),
    ("Is the input routed?",
     "&#8214;(J<sub>inp</sub>v)<super>T</super>R&#8214;: how much of that "
     "push lands in the decision plane, where it can affect the answer."),
], w0=1.5)
rep.figure(f_geom,
           "<b>Panel 1:</b> total input drive — nearly identical for cued and "
           "ignored features. <b>Panel 2:</b> drive into the readout plane — "
           "strongly context-dependent. <b>Panel 3:</b> the geometric selection "
           "ratio next to the behavioural one. <b>Panel 4:</b> the four "
           "condition-specific values plotted against each other.")
rep.table(
    ["Measurement", "Cued", "Ignored", "Ratio", "What it says"],
    [["total input drive &#8214;J<sub>inp</sub>v&#8214;",
      f"{0.5*(tot[0][0]+tot[1][1]):.3f}", f"{0.5*(tot[0][1]+tot[1][0]):.3f}",
      f"{tot_ratio:.2f}&#215;", "the input is NOT gated"],
     ["one-step drive into the decision plane", f"{g_rel:.3f}", f"{g_irr:.3f}",
      f"{geo_ratio:.2f}&#215;", "the routing IS gated"],
     ["accumulated linear response to the end of the trial",
      f"{a_rel:.2f}", f"{a_irr:.2f}", f"{acc_ratio:.1f}&#215;",
      "recurrence amplifies the gating"],
     ["behavioural logistic weight", f"{w_rel:.2f}", f"{w_irr:.2f}",
      f"{beh_ratio:.1f}&#215;", "measured from choices alone"]],
    widths=[2.05, 0.8, 0.8, 0.75, 2.0], fontsize=8.0)
rep.p(
    f"The first two rows answer the mechanistic question. Both features push "
    f"the state about equally hard ({tot_ratio:.2f}&#215;), confirming from "
    f"the dynamics what the decoding analysis showed from the activity: "
    f"nothing is filtered on the way in. But the fraction of that push which "
    f"lands in the readout plane differs by <b>{geo_ratio:.2f}&#215;</b>. The "
    f"context cue does not turn the irrelevant input off; it moves the state "
    f"to a region where that input's effect is largely <i>orthogonal to the "
    f"decision</i>. Selection is routing, not filtering.")
rep.h2("How far does the linear prediction get?")
rep.p(
    "A one-step Jacobian understates the case, because what matters "
    "behaviourally is not how much an input nudges the state now but how "
    "much it moves the decision variable at the <i>end</i> of the trial, "
    "after the nudge has been propagated through every subsequent "
    "J<sub>rec</sub>. That accumulated linear response is computable by a "
    "backward adjoint recursion (z<sub>T</sub> = a, "
    "z<sub>s</sub> = J<sub>rec</sub>(s)<super>T</super>z<sub>s+1</sub>), and "
    f"it raises the estimated selection from {geo_ratio:.2f}&#215; to "
    f"{acc_ratio:.1f}&#215;: the recurrent dynamics amplify the initial "
    "routing advantage over the course of the trial.")
rep.box(
    f"<b>An honest gap.</b> Behaviour shows {beh_ratio:.0f}&#215;. The full "
    f"linear-response calculation reaches {acc_ratio:.1f}&#215;. The "
    f"linearization therefore captures the <i>mechanism</i> of selection and "
    f"its <i>direction</i> — across the four feature&#215;context conditions "
    f"the geometric and behavioural values correlate at r = {rr:.2f} — but it "
    f"underestimates the <i>magnitude</i> by roughly an order of magnitude. "
    f"The missing suppression is nonlinear: the state does not simply drift "
    f"under perturbation, it is pulled back onto the context-appropriate "
    f"manifold by the same attractor dynamics that make the computation "
    f"stable. A first-order expansion cannot see a restoring force that only "
    f"exists at second order and beyond.")
rep.p(
    "This is worth stating plainly because it is the opposite of the result "
    "in Experiment 2, where the geometric prediction matched behaviour to "
    "within a few percent. There, the network's strategy was to stay in a "
    "region where the dynamics are linear and marginal, so a linear tool was "
    "exactly right. Here, the strategy is to use curvature — and a linear "
    "tool measures the part of it that lives in the derivative, which is "
    "real but partial.")
rep.h2("Caveats specific to this measurement")
rep.p(
    "The accumulated response is evaluated along the network's own "
    "trajectories, so it inherits their statistics; a perturbation large "
    "enough to leave that neighbourhood would behave differently. The "
    "decision plane is taken to be the readout subspace, and the adjoint "
    "recursion assumes the decision variable at the end of the stimulus is "
    "what determines the choice, ignoring the response epoch's own dynamics. "
    "Most importantly, everything here is correlational: showing that the "
    "irrelevant drive misses the readout plane is not the same as showing "
    "that redirecting it would change the choice. The decisive experiment is "
    "causal — inject a perturbation along the irrelevant drive direction "
    "during the trial and measure the change in behaviour — and the geometry "
    "provides exactly the direction and magnitude to inject.")
rep.build()
print("Done.")
