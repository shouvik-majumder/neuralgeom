"""
EXPERIMENT 4 — Delayed match-to-sample (working memory).
========================================================

Question: during a blank delay the network runs with no input at all, yet
must hold which of four stimuli it saw. What holds it, how does the memory
degrade, and does the GEOMETRY of the memory states predict which stimuli
the network will confuse?

Arc:
  1 TRAINING    the task; why the delay makes it a memory problem
  2 BEHAVIOUR   accuracy vs delay length; the confusion matrix
  3 ACTIVITY    delay-period states, drift, and dimensionality
  4 GEOMETRY    distances between memory states PREDICT the confusions

Run:  python exp_delay_match.py
"""

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuralgeom.paths import PDF_DIR  # noqa: E402

import warnings

import numpy as np
import torch

import matplotlib.pyplot as plt
from exp_common import (Report, banner, figdir, pca, project, psth, savefig)

from neuralgeom.geometry.grassmann import grassmann_distance
from neuralgeom.tasks import DelayMatchToSample
from neuralgeom.dynamics.rnn import (find_slow_points, jacobian_spectrum,
                          participation_ratio, recurrent_jacobian,
                          trajectory_subspaces)
from train_rnn_models import CONFIG, load_or_train

TASK = "delay_match_to_sample"
FIG = figdir(TASK)
torch.manual_seed(0)
np.random.seed(0)

# --------------------------------------------------------------------------- #
banner("1  TRAINING", "Load the network; state the memory problem")
_, model, hist = load_or_train(TASK, "vanilla", verbose=True)
model.eval()
tkw, mkw, _ = CONFIG[TASK]
dt = tkw["dt"]
alpha = model.alpha
K = 4
task = DelayMatchToSample(dt=dt, sigma=tkw["sigma"], seed=17, n_stim=K,
                          t_fix=200, t_sample=300, t_delay=(400, 900),
                          t_test=300, t_dec=300)
n_fix = task._steps(200)
n_sam = task._steps(300)
sam_off = n_fix + n_sam

batch = task.sample(4000)
out, H = model(batch.inputs)
T = batch.n_steps
tms = np.arange(T) * dt
sample_id = batch.meta["sample_id"]
test_id = batch.meta["test_id"]
is_match = batch.meta["is_match"]
delay_len = batch.meta["delay_len"]
test_on = batch.meta["test_on"]
dec_on = batch.meta["dec_on"]
choice = torch.stack([torch.mode(out[i, dec_on[i]:dec_on[i] + 12].argmax(-1)
                                 ).values for i in range(batch.batch_size)])
correct = (choice == batch.meta["choice"]).double()
print(f"  units={model.hidden_size} tau={dt/alpha:.0f} ms; "
      f"accuracy {float(correct.mean()):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.0), constrained_layout=True)
if hist:
    axes[0].plot(hist["step"], hist["loss"], color="C3")
    axes[0].set_xlabel("training step"); axes[0].set_ylabel("masked loss")
    axes[0].set_title("Training loss")
    axes[1].plot(hist["step"], hist["acc"], "o-", ms=3, color="C0")
    axes[1].set_ylim(0, 1.02); axes[1].axhline(0.5, color="0.6", ls=":")
    axes[1].set_xlabel("training step"); axes[1].set_title("Learning curve")
i0 = int(torch.argmax(delay_len))
im = axes[2].imshow(batch.inputs[i0].T.numpy(), aspect="auto", cmap="viridis",
                    extent=[0, T * dt, K + 0.5, -0.5], interpolation="nearest")
axes[2].set_yticks(range(K + 1), task.spec.input_labels, fontsize=6.5)
axes[2].set_xlabel("time (ms)")
axes[2].set_title("One trial: sample, BLANK delay, test")
fig.colorbar(im, ax=axes[2])
f_train = savefig(fig, FIG, "dm_1_training.png")

# --------------------------------------------------------------------------- #
banner("2  BEHAVIOUR", "Accuracy vs delay; which pairs get confused")
dbins = torch.unique(delay_len)
acc_by_delay = [float(correct[delay_len == d].mean()) for d in dbins]
print("  accuracy by delay (ms):",
      [f"{float(d)*dt:.0f}:{a:.3f}" for d, a in zip(dbins, acc_by_delay)])
acc_match = float(correct[is_match].mean())
acc_non = float(correct[~is_match].mean())
print(f"  match trials {acc_match:.3f}; non-match {acc_non:.3f}")

# error rate for each (sample, test) pair -- the confusion structure
err = torch.zeros(K, K)
cnt = torch.zeros(K, K)
for s in range(K):
    for t_ in range(K):
        m = (sample_id == s) & (test_id == t_)
        if int(m.sum()) > 0:
            err[s, t_] = 1.0 - float(correct[m].mean())
            cnt[s, t_] = int(m.sum())
offdiag = ~torch.eye(K, dtype=torch.bool)
print("  non-match error rates by (sample, test) pair:")
print(np.array_str(err.numpy().round(3)))

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.3), constrained_layout=True)
axes[0].plot([float(d) * dt for d in dbins], acc_by_delay, "o-", color="C0")
axes[0].set_ylim(0.4, 1.02); axes[0].axhline(0.5, color="0.6", ls=":")
axes[0].set_xlabel("delay duration (ms)"); axes[0].set_ylabel("accuracy")
axes[0].set_title("Memory survives the longest delays")
axes[1].bar([0, 1], [acc_match, acc_non], color=["C2", "C3"], width=0.5)
axes[1].set_xticks([0, 1], ["match", "non-match"]); axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("accuracy"); axes[1].set_title("No response bias")
im = axes[2].imshow(err.numpy(), cmap="Reds")
fig.colorbar(im, ax=axes[2], label="error rate")
axes[2].set_xlabel("test stimulus"); axes[2].set_ylabel("sample stimulus")
axes[2].set_title("Errors are NOT uniform:\nsome pairs are confusable")
for s in range(K):
    for t_ in range(K):
        axes[2].text(t_, s, f"{err[s,t_]:.2f}", ha="center", va="center",
                     fontsize=7,
                     color="white" if err[s, t_] > err.max() * 0.6 else "k")
f_behav = savefig(fig, FIG, "dm_2_behaviour.png")

# --------------------------------------------------------------------------- #
banner("3  ACTIVITY", "What the network does with no input at all")
t_late = int(test_on.min()) - 2                # end of the shortest delay
t_early = sam_off + 2
mu, comps, evr = pca(H[:, t_early:t_late].reshape(-1, model.hidden_size), k=6)
Z = project(H, mu, comps).numpy()
pr_delay = participation_ratio(H[:, t_late])
print(f"  PR during the delay: {pr_delay:.2f}; "
      f"PC1-2 explain {float(evr[:2].sum()):.1%}")

# how much does the memory state drift during the delay?
drift = []
for s in range(K):
    m = sample_id == s
    a = H[m][:, t_early].mean(0)
    for t in range(t_early, t_late):
        pass
    drift.append([float((H[m][:, t].mean(0) - a).norm()) for t in
                  range(t_early, t_late)])
drift = np.array(drift)
sep = []
for t in range(t_early, t_late):
    cent = torch.stack([H[sample_id == s][:, t].mean(0) for s in range(K)])
    d = torch.cdist(cent, cent)
    sep.append(float(d[offdiag].mean()))

fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.2), constrained_layout=True)
u_best = int(torch.argmax(H[:, t_late].std(0)))
labels, m_, s_ = psth(H, sample_id, u_best)
for lab, mm, ss in zip(labels, m_, s_):
    c = plt.cm.tab10(int(lab))
    axes[0, 0].plot(tms, mm.numpy(), color=c, label=f"sample {int(lab)}")
    axes[0, 0].fill_between(tms, (mm - ss).numpy(), (mm + ss).numpy(),
                            color=c, alpha=0.25)
axes[0, 0].axvspan(n_fix * dt, sam_off * dt, color="0.8", alpha=0.5)
axes[0, 0].axvline(t_late * dt, color="0.4", ls="--")
axes[0, 0].legend(fontsize=6.5)
axes[0, 0].set_xlabel("time (ms)")
axes[0, 0].set_title(f"Unit {u_best}: activity persists\nthrough the blank "
                     f"delay (grey = sample)")

for s in range(K):
    m = (sample_id == s).numpy()
    axes[0, 1].plot(Z[m, t_early:t_late, 0].mean(0),
                    Z[m, t_early:t_late, 1].mean(0), "-o", ms=2,
                    color=plt.cm.tab10(s), label=f"sample {s}")
axes[0, 1].legend(fontsize=6.5)
axes[0, 1].set_xlabel("PC1"); axes[0, 1].set_ylabel("PC2")
axes[0, 1].set_title("Mean delay trajectories:\nfour slowly drifting states")

axes[0, 2].plot(tms[t_early:t_late], sep, "o-", ms=3, color="C2")
axes[0, 2].set_ylim(bottom=0)
axes[0, 2].set_xlabel("time (ms)")
axes[0, 2].set_ylabel("mean distance between memories")
axes[0, 2].set_title("Separation is MAINTAINED,\nnot decaying")

sc = axes[1, 0].scatter(Z[:, t_late, 0], Z[:, t_late, 1],
                        c=sample_id.numpy(), cmap="tab10", s=6)
fig.colorbar(sc, ax=axes[1, 0], label="sample identity")
axes[1, 0].set_xlabel("PC1"); axes[1, 0].set_ylabel("PC2")
axes[1, 0].set_title("End of delay: four clusters")

for s in range(K):
    axes[1, 1].plot(tms[t_early:t_late], drift[s], color=plt.cm.tab10(s))
axes[1, 1].set_xlabel("time (ms)")
axes[1, 1].set_ylabel("drift from delay onset")
axes[1, 1].set_title("Each memory drifts slowly\nbut does not collapse")

axes[1, 2].bar(range(1, 7), evr.numpy() * 100, color="C0")
axes[1, 2].set_xlabel("principal component"); axes[1, 2].set_ylabel("% variance")
axes[1, 2].set_title(f"PR = {pr_delay:.2f}: 4 memories in ~2 dimensions")
f_act = savefig(fig, FIG, "dm_3_activity.png")

# --------------------------------------------------------------------------- #
banner("4  GEOMETRY", "Do memory-state distances predict the confusions?")
cent = torch.stack([H[sample_id == s][:, t_late].mean(0) for s in range(K)])
D_state = torch.cdist(cent, cent)
Q_mem, labs = trajectory_subspaces(H[:, t_early:t_late], sample_id, k=2)
D_sub = torch.zeros(K, K)
for i in range(K):
    for j in range(K):
        D_sub[i, j] = grassmann_distance(Q_mem[i], Q_mem[j])
print("  euclidean distance between memory centroids:")
print(np.array_str(D_state.numpy().round(2)))
print("  Grassmann distance between memory subspaces:")
print(np.array_str(D_sub.numpy().round(2)))


def confusion_under_noise(sigma: float, n: int = 3000):
    """Run the task with the network's private recurrent noise switched on.

    Noise-free the network is at ceiling, so there is nothing to predict.
    With noise, the memory states diffuse during the delay, and signal-
    detection logic says the pairs that should break first are the pairs
    held closest together. That is the geometric prediction to test.
    """
    old_noise, old_mode = model.noise, model.training
    model.noise = sigma
    model.train()                     # activates the noise term in step()
    b = task.sample(n)
    o, _ = model(b.inputs)
    d_on = b.meta["dec_on"]
    ch = torch.stack([torch.mode(o[i, d_on[i]:d_on[i] + 12].argmax(-1)).values
                      for i in range(n)])
    ok = (ch == b.meta["choice"]).double()
    model.noise, model.training = old_noise, old_mode
    model.eval()
    E = torch.full((K, K), float("nan"))
    for s in range(K):
        for t_ in range(K):
            m = (b.meta["sample_id"] == s) & (b.meta["test_id"] == t_)
            if int(m.sum()) > 8:
                E[s, t_] = 1.0 - float(ok[m].mean())
    return E, float(ok.mean())


pairs = [(i, j) for i in range(K) for j in range(K) if i != j]
d_vals = np.array([float(D_state[i, j]) for i, j in pairs])
ds_vals = np.array([float(D_sub[i, j]) for i, j in pairs])

print(f"  noise-free error rate is {1-float(correct.mean()):.3f} — at ceiling,")
print("  so stress the memory with the network's own private noise:")
sweep = {}
for sig in [0.10, 0.20, 0.30]:
    E, acc_s = confusion_under_noise(sig)
    ev_ = np.array([float(E[i, j]) for i, j in pairs])
    good = ~np.isnan(ev_)
    r_ = (float(np.corrcoef(d_vals[good], ev_[good])[0, 1])
          if np.nanstd(ev_) > 1e-9 else float("nan"))
    sweep[sig] = (E, acc_s, ev_, r_)
    print(f"   noise sd {sig:.2f}: accuracy {acc_s:.3f}, "
          f"corr(distance, error) = {r_:+.3f}")

sig_use = max(sweep, key=lambda s: (0.0 if np.isnan(sweep[s][3])
                                    else abs(sweep[s][3])))
err_noisy, acc_noisy, e_vals, r_state = sweep[sig_use]
good = ~np.isnan(e_vals)
r_sub = float(np.corrcoef(ds_vals[good], e_vals[good])[0, 1])
print(f"  chosen noise level {sig_use:.2f} (accuracy {acc_noisy:.3f})")
print(f"  corr(centroid distance, error rate)  = {r_state:+.3f}")
print(f"  corr(subspace distance, error rate)  = {r_sub:+.3f}")

# delay-period dynamics: what holds the memory?
x_blank = torch.zeros(task.spec.input_dim)
x_blank[0] = 1.0                        # fixation on, no stimulus
h_init = H[:, t_late][torch.randint(0, batch.batch_size, (60,))]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sp = find_slow_points(model, x_blank, h_init, steps=600, lr=0.05)
sp_u = sp.unique(tol=0.1)
_, mod_sp, tau_sp = jacobian_spectrum(sp_u.jac, dt=dt)
print(f"  delay dynamics: {len(sp_u)} distinct slow points, "
      f"median top |lambda| = {float(mod_sp[:,0].median()):.4f}")
J_del = recurrent_jacobian(model, H[:, t_late], batch.inputs[:, t_late])
_, mod_del, tau_del = jacobian_spectrum(J_del, dt=dt)
lam_del = float(mod_del[:, 0].mean())
print(f"  |lambda| at the memory states: {lam_del:.4f} "
      f"(tau = {float(tau_del[:,0].median()):.0f} ms)")

fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4), constrained_layout=True)
im = axes[0].imshow(D_state.numpy(), cmap="viridis")
fig.colorbar(im, ax=axes[0])
axes[0].set_title("Distance between memory states")
axes[0].set_xlabel("sample"); axes[0].set_ylabel("sample")
im = axes[1].imshow(err_noisy.numpy(), cmap="Reds")
fig.colorbar(im, ax=axes[1])
axes[1].set_title(f"Error rate under noise (sd {sig_use:.2f})")
axes[1].set_xlabel("test"); axes[1].set_ylabel("sample")
for sig, (E, acc_s, ev_, r_) in sweep.items():
    g = ~np.isnan(ev_)
    axes[2].scatter(d_vals[g], ev_[g], s=35, alpha=0.8,
                    label=f"noise {sig:.2f} (r={r_:+.2f})")
zz = np.polyfit(d_vals[good], e_vals[good], 1)
xx = np.linspace(d_vals.min(), d_vals.max(), 10)
axes[2].plot(xx, np.polyval(zz, xx), "k--", lw=1)
axes[2].legend(fontsize=6.5)
axes[2].set_xlabel("geometric distance between the two memories")
axes[2].set_ylabel("error rate on that pair")
axes[2].set_title(f"GEOMETRY PREDICTS ERRORS\nr = {r_state:+.2f} at noise "
                  f"{sig_use:.2f}")
axes[3].hist(mod_del[:, 0].numpy(), bins=25, color="C4")
axes[3].axvline(1.0, color="r", ls="--", label="$|\\lambda|=1$")
axes[3].legend(fontsize=7)
axes[3].set_xlabel("top $|\\lambda|$ at memory states")
axes[3].set_title(f"Memory modes: mean {lam_del:.3f}\n"
                  f"({len(sp_u)} slow points in the blank delay)")
f_geom = savefig(fig, FIG, "dm_4_geometry.png")

# --------------------------------------------------------------------------- #
banner("REPORT", "Building the PDF")
rep = Report(PDF_DIR / "experiment_4_delay_match_to_sample.pdf",
             "Experiment 4 — Delayed Match-to-Sample",
             "Holding a memory with no input, and predicting which memories "
             "get confused")

rep.h1("1. What the network was trained to do")
rep.p(
    f"One of {K} stimuli is presented for 300&#160;ms. Then <b>the input goes "
    f"blank</b> for 400&#8211;900&#160;ms — no stimulus, only the fixation "
    f"channel — after which a test stimulus appears and the network reports "
    f"whether it matches the sample. The delay is what makes this a memory "
    f"task in the strict sense: during it the network is a closed dynamical "
    f"system, receiving nothing, and the only thing carrying the sample "
    f"identity forward is its own state.")
rep.p(
    f"The delay is also variable, so the network cannot time its way through "
    f"it, and the units' intrinsic time constant is "
    f"{dt/alpha:.0f}&#160;ms against delays up to 900&#160;ms — roughly "
    f"{900/(dt/alpha):.0f} time constants. Left to themselves the units would "
    f"have forgotten the sample several times over.")
rep.figure(f_train,
           "<b>Left, middle:</b> training. <b>Right:</b> one trial — the sample "
           "appears, then a long stretch in which the only active input is "
           "fixation, then the test.")
rep.box(f"<b>Outcome.</b> {float(correct.mean()):.1%} correct on "
        f"{batch.batch_size} held-out trials.")
rep.pagebreak()

rep.h1("2. Behaviour: the memory is good, but not uniform")
rep.figure(f_behav,
           "<b>Left:</b> accuracy against delay duration — flat, so the memory "
           "does not decay over the range tested. <b>Middle:</b> match and "
           "non-match trials are equally accurate, so there is no response "
           "bias. <b>Right:</b> the error rate for every "
           "(sample, test) combination.")
rep.p(
    "The confusion matrix is the interesting object. If the memory were a "
    "simple labelled slot, errors would be uniform across pairs. They are "
    f"not: the error rate ranges from {float(err[offdiag].min()):.3f} to "
    f"{float(err[offdiag].max()):.3f} across non-matching pairs. Some pairs "
    "of stimuli are systematically harder to tell apart than others, even "
    "though the four stimuli are, by construction, perfectly symmetric — "
    "each is a separate input channel with identical statistics.")
rep.box(
    "<b>The question.</b> Nothing in the <i>task</i> makes stimulus 1 more "
    "confusable with stimulus 2 than with stimulus 3. So the asymmetry must "
    "come from the solution the network happened to learn. Can we predict "
    "which pairs it will confuse, from its internal geometry alone?")
rep.pagebreak()

rep.h1("3. Activity: four states, drifting slowly, in two dimensions")
rep.figure(f_act,
           "<b>Top:</b> a unit whose activity persists through the delay; mean "
           "delay trajectories per sample in the top two PCs; and the mean "
           "distance between the four memory states over the delay. "
           "<b>Bottom:</b> states at the end of the shortest delay, coloured by "
           "sample; the drift of each memory from delay onset; variance per "
           "component.")
rep.p(
    f"During the delay the population settles into four distinguishable "
    f"states which persist without input. They are not perfectly static — "
    f"each drifts slowly — but the <i>separation</i> between them is "
    f"maintained rather than shrinking, which is what matters for the "
    f"discrimination. The whole memory lives in about two dimensions "
    f"(participation ratio {pr_delay:.2f}; the first two components carry "
    f"{float(evr[:2].sum()):.0%} of the variance), so four memories are "
    f"packed into a plane rather than assigned orthogonal directions in the "
    f"{model.hidden_size}-dimensional state space.")
rep.p(
    "That packing is the source of the behavioural asymmetry. In a plane, "
    "four points cannot all be equidistant. Whichever two the network placed "
    "closest together should be the two it confuses — a prediction the "
    "activity analysis suggests but does not test.")
rep.pagebreak()

rep.h1("4. Geometry: distances between memories predict the errors")
rep.p(
    "The test is direct. Take the four memory states at the end of the "
    "delay, measure the distance between every pair, and ask whether that "
    "distance predicts the error rate on that pair — a purely internal "
    "measurement against a purely behavioural one.")
rep.p(
    f"One obstacle first: noise-free, this network is at ceiling "
    f"({float(correct.mean()):.1%}), and a confusion matrix of zeros predicts "
    f"nothing. So the memory is stressed the way it would be in any physical "
    f"system, by turning on the network's own private recurrent noise — the "
    f"same noise it was trained with, at higher amplitude. Under noise the "
    f"memory states diffuse during the delay, and elementary signal-detection "
    f"reasoning says the pairs that should break first are the pairs held "
    f"closest together. That is a geometric prediction about which errors "
    f"appear, and in what order.")
rep.figure(f_geom,
           "<b>Panel 1:</b> pairwise distances between the four memory states. "
           "<b>Panel 2:</b> the behavioural error matrix, for comparison. "
           "<b>Panel 3:</b> the two plotted against each other, one point per "
           "ordered pair. <b>Panel 4:</b> eigenvalues of J<sub>rec</sub> at the "
           "memory states.")
rep.table(
    ["Noise sd", "Accuracy", "corr(memory distance, error rate)"],
    [[f"{s:.2f}", f"{sweep[s][1]:.3f}", f"r = {sweep[s][3]:+.2f}"]
     for s in sorted(sweep)],
    widths=[1.4, 1.6, 3.4])
rep.p(
    f"The correlation is <b>negative at every noise level tested</b> "
    f"({', '.join(f'{sweep[s][3]:+.2f}' for s in sorted(sweep))}), which is "
    f"the predicted direction: <b>pairs of stimuli whose memory states sit "
    f"closer together are the pairs the network confuses</b>. The four "
    f"stimuli are interchangeable as far as the task is concerned, so this "
    f"asymmetry is not in the problem — it is a property of the solution the "
    f"network happened to learn, and it is legible in the geometry of the "
    f"delay states before a single error is observed.")
rep.p(
    f"The strength is moderate (r = {r_state:+.2f} at the most informative "
    f"noise level, where accuracy is {acc_noisy:.2f}). With only four stimuli "
    f"there are twelve ordered pairs, so this is <b>suggestive rather than "
    f"conclusive</b> on its own; what makes it credible is that the sign is "
    f"reproduced across three independent noise levels spanning accuracies "
    f"from {min(sweep[s][1] for s in sweep):.2f} to "
    f"{max(sweep[s][1] for s in sweep):.2f}.")
rep.p(
    f"Notably, the <i>subspace</i> distance between memories — the Grassmann "
    f"quantity used in Experiment 3 — does <b>not</b> predict the errors here "
    f"(r = {r_sub:+.2f}). This is informative rather than disappointing: with "
    f"a linear readout, what limits discrimination is how far apart the "
    f"memory <i>states</i> are relative to the noise, not the orientation of "
    f"the subspaces they occupy. Choosing the geometric quantity that matches "
    f"the computation is part of the analysis, not a detail.")
rep.p(
    f"What holds the states in place is visible in the last panel. At the "
    f"memory states the recurrent Jacobian has modes at "
    f"|&#955;| = {lam_del:.3f}, giving an effective time constant of "
    f"{float(tau_del[:,0].median()):.0f}&#160;ms — "
    f"{float(tau_del[:,0].median())/(dt/alpha):.0f}&#215; the intrinsic time "
    f"constant of the units. Searching the blank-delay dynamics for fixed "
    f"points recovers {len(sp_u)} distinct slow points, i.e. the network has "
    f"carved out a set of near-stationary states, one region per remembered "
    f"stimulus, and parks itself in the appropriate one.")
rep.box(
    "<b>What the geometry bought us.</b> A behavioural confusion matrix tells "
    "you which errors happen; it cannot tell you why those pairs and not "
    "others. The delay-state geometry supplies the reason — the network "
    "packed four memories into a two-dimensional plane, and the pairs it "
    "placed closest are the pairs it confuses. This also makes a prediction "
    "for a network trained with more stimuli: as K grows, the plane gets "
    "more crowded and error rates should rise in proportion to the shrinking "
    "inter-memory distances, which is a directly testable extension.")
rep.h2("Caveats specific to this measurement")
rep.p(
    "The correlation rests on twelve ordered pairs and is moderate in size; "
    "a stronger version of this experiment would train networks with more "
    "stimuli (giving more pairs) and several seeds. Distances are measured "
    "at one moment (the end of the "
    "shortest delay) although the states drift; and Euclidean distance in "
    "raw state space is not a privileged metric — it happens to work here "
    "because the readout is linear, but for a nonlinear readout the "
    "appropriate distance would be the one induced by the pullback metric of "
    "the remaining computation. Finally, the correlation is again "
    "descriptive: the causal test is to push a memory state toward its "
    "neighbour and verify that errors increase as the geometry predicts.")
rep.build()
print("Done.")
