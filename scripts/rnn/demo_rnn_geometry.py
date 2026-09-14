"""
Demo 6 — RNNs on cognitive tasks: the geometry of recurrent computation.
========================================================================

Four vanilla tanh RNNs, trained (by ``train_rnn_models.py``) on tasks that
cannot be solved without using time:

    perceptual_decision    accumulate a noisy constant stimulus, then report
    evidence_integration   count pulses over a VARIABLE window, then report
    context_decision       integrate the cued feature, ignore the other
    delay_match_to_sample  hold a stimulus through a blank delay, compare

For a feedforward net the object of interest was the input->output map. For
an RNN it is the one-step update h_{t+1} = F(h_t, x_t), and its two
derivatives:

    J_rec = dF/dh   how the state evolves on its own   -> memory, attractors
    J_inp = dF/dx   how evidence enters the state      -> input gain/selection

Pipeline:
  Step 0  The tasks themselves (input rasters) + learning curves
  Step 1  Behaviour: psychometric curves, integration over time
  Step 2  J_rec spectra: the eigenvalue that sits at |lambda| ~ 1 is the
          integrator; time constants tau = -dt/log|lambda|
  Step 3  Fixed / slow points and the state-space portrait (line attractor)
  Step 4  Context task: the SAME stimulus routed differently — Grassmann
          distance between context subspaces, and input-gain selection
  Step 5  Delay task + what these tools CANNOT do

Run:  python train_rnn_models.py && python demo_rnn_geometry.py
"""

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


from demo_common import banner, savefig                      # noqa: E402
from neuralgeom.geometry.grassmann import grassmann_distance, principal_angles  # noqa: E402
from neuralgeom.tasks import run_trials                                    # noqa: E402
from neuralgeom.dynamics.rnn import (find_slow_points, input_jacobian,    # noqa: E402
                          jacobian_spectrum, participation_ratio,
                          readout_subspace, recurrent_jacobian,
                          recurrent_update_pullback_metric, subspace_alignment,
                          trajectory_subspaces)
from train_rnn_models import load_or_train                    # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(0)

TASKS = ["perceptual_decision", "evidence_integration", "context_decision",
         "delay_match_to_sample"]
SHORT = {"perceptual_decision": "perceptual", "evidence_integration": "integration",
         "context_decision": "context", "delay_match_to_sample": "DMS"}

banner("Step 0", "Load the four trained RNNs (train_rnn_models.py caches them)")
bundle = {}
for name in TASKS:
    task, model, hist = load_or_train(name, "vanilla", verbose=True)
    model.eval()
    bundle[name] = (task, model, hist)

fig, axes = plt.subplots(2, 4, figsize=(15, 5.6), constrained_layout=True,
                         gridspec_kw={"height_ratios": [2, 1]})
for j, name in enumerate(TASKS):
    task, model, hist = bundle[name]
    b = task.sample(1)
    im = axes[0, j].imshow(b.inputs[0].T.numpy(), aspect="auto", cmap="viridis",
                           interpolation="nearest")
    axes[0, j].set_yticks(range(task.spec.input_dim),
                          task.spec.input_labels, fontsize=6)
    axes[0, j].set_title(f"{SHORT[name]}\ninputs (one trial)")
    axes[0, j].set_xlabel("time step")
    if hist:
        axes[1, j].plot(hist["step"], hist["acc"], "o-", ms=3)
        axes[1, j].set_ylim(0, 1.02)
        axes[1, j].axhline(0.5, color="0.6", ls=":")
        axes[1, j].set_xlabel("training step")
        axes[1, j].set_title("decision accuracy")
savefig(fig, "rnn_step0_tasks.png")

# --------------------------------------------------------------------------- #
banner("Step 1", "Behaviour first — is the computation actually being done?")
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5), constrained_layout=True)

# psychometric: perceptual decision
task, model, _ = bundle["perceptual_decision"]
batch, out, H = run_trials(model, task, 1500)
dec = batch.loss_mask & (batch.targets > 0)
pred = out.argmax(-1)
choose_A = torch.stack([(pred[i][dec[i]] == 1).double().mean()
                        for i in range(pred.shape[0])])
cohs = batch.meta["coherence"]
uc = torch.unique(cohs)
axes[0].plot(uc.numpy(), [float(choose_A[cohs == c].mean()) for c in uc], "o-")
axes[0].axhline(0.5, color="0.6", ls=":")
axes[0].set_xlabel("coherence  (+ favours A)")
axes[0].set_ylabel("P(choose A)")
axes[0].set_title("Perceptual decision: psychometric curve")

# integration: accuracy vs realized evidence, and vs stimulus duration
task, model, _ = bundle["evidence_integration"]
batch, out, H = run_trials(model, task, 2000)
dec = batch.loss_mask & (batch.targets > 0)
pred = out.argmax(-1)
corr = torch.stack([(pred[i][dec[i]] == batch.targets[i][dec[i]]).double().mean()
                    for i in range(pred.shape[0])])
ev = batch.meta["evidence"].abs()
bins = torch.tensor([0, 1, 2, 3, 5, 8, 100.0])
xs, ys = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (ev >= lo) & (ev < hi)
    if int(m.sum()) > 10:
        xs.append(float(ev[m].mean()))
        ys.append(float(corr[m].mean()))
axes[1].plot(xs, ys, "o-", color="C2")
axes[1].axhline(0.5, color="0.6", ls=":")
axes[1].set_xlabel("|pulse count difference| (realized evidence)")
axes[1].set_ylabel("accuracy")
axes[1].set_title("Evidence integration: more evidence,\nbetter decisions")

# context: congruent vs conflict trials
task, model, _ = bundle["context_decision"]
batch, out, H = run_trials(model, task, 2000)
dec = batch.loss_mask & (batch.targets > 0)
pred = out.argmax(-1)
corr_c = torch.stack([(pred[i][dec[i]] == batch.targets[i][dec[i]]).double().mean()
                      for i in range(pred.shape[0])])
ctx = batch.meta["context"]
rel = batch.meta["relevant_coh"]
irr = torch.where(ctx == 0, batch.meta["coh_colour"], batch.meta["coh_motion"])
congruent = rel.sign() == irr.sign()
vals = [float(corr_c[congruent].mean()), float(corr_c[~congruent].mean())]
axes[2].bar([0, 1], vals, color=["C0", "C3"], width=0.55)
axes[2].set_xticks([0, 1], ["congruent", "conflict"])
axes[2].set_ylim(0, 1.05)
axes[2].axhline(0.5, color="0.6", ls=":")
for i, v in enumerate(vals):
    axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
axes[2].set_title("Context task: NO congruency cost — accuracy is\nunchanged "
                  "when the ignored feature disagrees")
savefig(fig, "rnn_step1_behaviour.png")
print(f"  context task: congruent {vals[0]:.3f} vs conflict {vals[1]:.3f} "
      f"(difference {vals[1] - vals[0]:+.3f})")
print("  -> essentially no congruency cost: the irrelevant feature does not")
print("     leak into the decision. (A network that failed to gate would")
print("     show conflict accuracy well BELOW congruent.)")

# --------------------------------------------------------------------------- #
banner("Step 2", "Recurrent Jacobian spectra: where the memory is")
fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), constrained_layout=True)
summary = {}
for j, name in enumerate(TASKS):
    task, model, _ = bundle[name]
    batch, out, H = run_trials(model, task, 64)
    # sample states from the middle of the stimulus/delay epoch
    t_mid = int(batch.meta["dec_on"].double().mean().item() * 0.7)
    h, x = H[:, t_mid], batch.inputs[:, t_mid]
    J = recurrent_jacobian(model, h, x)
    ev, mod, tau = jacobian_spectrum(J, dt=task.dt)
    summary[name] = (float(mod[:, 0].mean()), float(tau[:, 0].median()))
    ax = axes[j]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="0.6", lw=0.8)
    evf = ev.reshape(-1)
    ax.scatter(evf.real.numpy(), evf.imag.numpy(), s=4, alpha=0.25)
    ax.axhline(0, color="0.85", lw=0.6)
    ax.axvline(0, color="0.85", lw=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"{SHORT[name]}\ntop $|\\lambda|$ = {summary[name][0]:.3f}, "
                 f"$\\tau$ = {summary[name][1]:.0f} ms")
    ax.set_xlabel("Re $\\lambda$")
fig.suptitle("Eigenvalues of $J_{rec} = \\partial F/\\partial h$ (unit circle "
             "in grey). Eigenvalues hugging $|\\lambda|=1$ are directions "
             "that neither decay nor explode — memory.", y=1.09, fontsize=9)
savefig(fig, "rnn_step2_spectra.png")
for name, (m_, t_) in summary.items():
    print(f"  {SHORT[name]:12s}: top |lambda| = {m_:.4f}   "
          f"effective tau = {t_:8.0f} ms   (single-unit tau = "
          f"{bundle[name][1].alpha and bundle[name][0].dt / bundle[name][1].alpha:.0f} ms)")
print("  -> effective time constants far exceed the single-unit tau: the")
print("     memory is a property of the RECURRENT CIRCUIT, not of the units.")

# --------------------------------------------------------------------------- #
banner("Step 3", "Slow points and the state-space portrait (line attractor)")
task, model, _ = bundle["evidence_integration"]
batch, out, H = run_trials(model, task, 300)
t_dec = int(batch.meta["dec_on"].min())
# PCA on states during the stimulus epoch
Hs = H[:, 5:t_dec].reshape(-1, H.shape[-1])
mu = Hs.mean(0, keepdim=True)
U, S, Vh = torch.linalg.svd(Hs - mu, full_matrices=False)
PC = Vh[:3].transpose(0, 1)

# slow points of the autonomous dynamics under the MEAN stimulus input
x_bar = batch.inputs[:, 5:t_dec].reshape(-1, batch.inputs.shape[-1]).mean(0)
h_init = Hs[torch.randint(0, Hs.shape[0], (60,))]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sp = find_slow_points(model, x_bar, h_init, steps=600, lr=0.05)
sp_u = sp.unique(tol=0.05)
print(f"  found {len(sp)} converged points -> {len(sp_u)} distinct; "
      f"speed q range [{float(sp_u.q.min()):.2e}, {float(sp_u.q.max()):.2e}]")
_, mod_sp, tau_sp = jacobian_spectrum(sp_u.jac, dt=task.dt)
print(f"  their top |lambda| range: [{float(mod_sp[:,0].min()):.4f}, "
      f"{float(mod_sp[:,0].max()):.4f}]   unstable dirs: "
      f"{sp_u.n_unstable.tolist()[:12]}")

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
Z = ((H - mu) @ PC).numpy()
evid = batch.meta["evidence"].numpy()
norm = plt.Normalize(np.percentile(evid, 5), np.percentile(evid, 95))
for i in range(0, 300, 2):
    axes[0].plot(Z[i, 5:t_dec, 0], Z[i, 5:t_dec, 1], lw=0.5, alpha=0.45,
                 color=plt.cm.coolwarm(norm(evid[i])))
Zsp = ((sp_u.h - mu) @ PC).numpy()
axes[0].scatter(Zsp[:, 0], Zsp[:, 1], c="k", s=28, zorder=5, marker="o",
                label="slow points")
axes[0].legend(fontsize=7)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=axes[0],
             label="realized evidence (A - B)")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].set_title("Integration task: trajectories in state space.\nThe slow "
                  "points form a LINE — the integrator's memory")

# state along the attractor vs accumulated evidence
proj = ((H[:, t_dec - 1] - mu) @ PC[:, 0]).numpy()
axes[1].scatter(evid, proj, s=6, alpha=0.35)
r = np.corrcoef(evid, proj)[0, 1]
axes[1].set_xlabel("realized evidence (pulse difference)")
axes[1].set_ylabel("PC1 of state at decision onset")
axes[1].set_title(f"The attractor coordinate ENCODES the integral\n"
                  f"(|r| = {abs(r):.3f}; PC sign is arbitrary)")

# stability of the slow points
n_saddle = int((sp_u.n_unstable > 0).sum())
axes[2].hist(mod_sp[:, 0].numpy(), bins=20, color="C4")
axes[2].axvline(1.0, color="r", ls="--", label="$|\\lambda| = 1$")
axes[2].set_xlabel("top $|\\lambda|$ of each slow point")
axes[2].legend(fontsize=8)
axes[2].set_title(f"Slow points cluster at $|\\lambda|\\approx 1$ "
                  f"(marginal stability);\n{n_saddle}/{len(sp_u)} are saddles "
                  f"with one unstable direction")
savefig(fig, "rnn_step3_fixed_points.png")
print(f"  |correlation|(evidence, attractor coordinate) = {abs(r):.3f} "
      f"(sign of a PC is arbitrary)")
print(f"  {n_saddle}/{len(sp_u)} slow points are saddles (1 unstable "
      f"direction) — these sit between the two choice basins;")
print("  the rest are marginally stable points making up the attractor line.")

# --------------------------------------------------------------------------- #
banner("Step 4", "Context task: same stimulus, two routings (Grassmann + gain)")
task, model, _ = bundle["context_decision"]
batch, out, H = run_trials(model, task, 800)
ctx = batch.meta["context"]
t_stim = int(batch.meta["stim_on"][0]) + 5
t_end = int(batch.meta["dec_on"][0])

# (a) activity subspaces per context
Q_ctx, labels = trajectory_subspaces(H[:, t_stim:t_end], ctx, k=3)
d_ctx = float(grassmann_distance(Q_ctx[0], Q_ctx[1]))
ang = torch.rad2deg(principal_angles(Q_ctx[0], Q_ctx[1]))
print(f"  context subspaces (k=3): geodesic distance {d_ctx:.3f}, "
      f"principal angles {[round(float(a),1) for a in ang]} deg")

# (b) input gain: how strongly each feature drives the state, per context
gains = {}
for c in [0, 1]:
    sel = ctx == c
    h = H[sel][:, t_stim]
    x = batch.inputs[sel][:, t_stim]
    Ji = input_jacobian(model, h, x)                 # (B, hidden, 7)
    # drive direction for each feature = difference of its two channels
    v_motion = (Ji[:, :, 1] - Ji[:, :, 2])
    v_colour = (Ji[:, :, 3] - Ji[:, :, 4])
    R = readout_subspace(model, k=2)                 # decision plane
    # component of each drive that lands in the readout (decision) subspace
    gm = (v_motion @ R).norm(dim=1).mean()
    gc = (v_colour @ R).norm(dim=1).mean()
    gains[c] = (float(gm), float(gc))
    print(f"  context={'motion' if c == 0 else 'colour'}: drive into the "
          f"decision plane — motion {gains[c][0]:.4f}, colour {gains[c][1]:.4f}")
sel_idx = ((gains[0][0] / gains[0][1]) * (gains[1][1] / gains[1][0])) ** 0.5
print(f"  selection index (relevant/irrelevant gain, geometric mean) = "
      f"{sel_idx:.2f}x")

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), constrained_layout=True)
mu_c = H[:, t_stim:t_end].reshape(-1, H.shape[-1]).mean(0, keepdim=True)
Hc = H[:, t_stim:t_end].reshape(-1, H.shape[-1]) - mu_c
Vh_c = torch.linalg.svd(Hc, full_matrices=False)[2]
PCc = Vh_c[:2].transpose(0, 1)
Zc = ((H[:, t_end - 1] - mu_c) @ PCc).numpy()
for c, col, lab in [(0, "C0", "context = motion"), (1, "C3", "context = colour")]:
    m = (ctx == c).numpy()
    axes[0].scatter(Zc[m, 0], Zc[m, 1], s=7, alpha=0.5, c=col, label=lab)
axes[0].legend(fontsize=7)
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].set_title("End-of-stimulus states separate by CONTEXT\n"
                  "(the cue moves the whole computation)")
axes[1].bar([0, 1, 2.5, 3.5],
            [gains[0][0], gains[0][1], gains[1][0], gains[1][1]],
            color=["C2", "0.7", "0.7", "C2"], width=0.8)
axes[1].set_xticks([0, 1, 2.5, 3.5], ["motion", "colour", "motion", "colour"],
                   fontsize=8)
axes[1].set_xlabel("context = motion            context = colour")
axes[1].set_ylabel("drive into decision plane")
axes[1].set_title(f"Input GAIN is context-dependent:\nthe cued feature drives "
                  f"{sel_idx:.1f}x harder (green)")
axes[2].bar(range(len(ang)), ang.numpy(), color="C4", width=0.6)
axes[2].set_xticks(range(len(ang)), [f"$\\theta_{i+1}$" for i in range(len(ang))])
axes[2].set_ylabel("principal angle [deg]")
axes[2].set_title(f"Context subspaces on Gr(3, {H.shape[-1]}):\n"
                  f"geodesic distance {d_ctx:.2f} rad")
savefig(fig, "rnn_step4_context.png")

# --------------------------------------------------------------------------- #
banner("Step 5", "Working memory + limitations")
task, model, _ = bundle["delay_match_to_sample"]
batch, out, H = run_trials(model, task, 400)
t_test = int(batch.meta["test_on"].min())
# recurrent Jacobian DURING the blank delay (no input at all)
t_delay = t_test - 3
h, x = H[:, t_delay], batch.inputs[:, t_delay]
J = recurrent_jacobian(model, h, x)
ev_d, mod_d, tau_d = jacobian_spectrum(J, dt=task.dt)
pr_delay = participation_ratio(H[:, t_delay])
print(f"  DMS delay period: top |lambda| = {float(mod_d[:,0].mean()):.4f}, "
      f"tau = {float(tau_d[:,0].median()):.0f} ms, "
      f"participation ratio = {pr_delay:.2f}")
Q_s, labs = trajectory_subspaces(H[:, t_delay - 5:t_delay],
                                 batch.meta["sample_id"], k=2)
dmat = torch.zeros(len(labs), len(labs))
for i in range(len(labs)):
    for j in range(len(labs)):
        dmat[i, j] = grassmann_distance(Q_s[i], Q_s[j])
print(f"  memory subspaces for the {len(labs)} sample identities are distinct "
      f"(mean off-diagonal Grassmann distance "
      f"{float(dmat[dmat > 0].mean()):.2f} rad)")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
mu_d = H[:, t_delay].mean(0, keepdim=True)
Vd = torch.linalg.svd(H[:, t_delay] - mu_d, full_matrices=False)[2]
Zd = ((H[:, t_delay] - mu_d) @ Vd[:2].transpose(0, 1)).numpy()
sid = batch.meta["sample_id"].numpy()
sc = axes[0].scatter(Zd[:, 0], Zd[:, 1], c=sid, cmap="tab10", s=10)
fig.colorbar(sc, ax=axes[0], label="sample identity")
axes[0].set_title("DMS: states at the END of the blank delay\ncluster by the "
                  "REMEMBERED stimulus")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
im = axes[1].imshow(dmat.numpy(), cmap="inferno")
fig.colorbar(im, ax=axes[1], label="Grassmann distance")
axes[1].set_title("Pairwise distance between the memory\nsubspaces of each "
                  "sample identity")
axes[1].set_xlabel("sample id"); axes[1].set_ylabel("sample id")
th = np.linspace(0, 2 * np.pi, 200)
axes[2].plot(np.cos(th), np.sin(th), color="0.6", lw=0.8)
evf = ev_d.reshape(-1)
axes[2].scatter(evf.real.numpy(), evf.imag.numpy(), s=4, alpha=0.25)
axes[2].set_aspect("equal")
axes[2].set_title(f"$J_{{rec}}$ during the blank delay:\nmemory modes at "
                  f"$|\\lambda|\\approx${float(mod_d[:,0].mean()):.3f}")
axes[2].set_xlabel("Re $\\lambda$")
savefig(fig, "rnn_step5_memory.png")

print("""
LIMITATIONS — what this RNN geometry cannot do

(5a) J_rec is a LOCAL LINEARIZATION. It describes dynamics in an
     infinitesimal neighbourhood of one state at one time. Trajectories
     that traverse very different regions are not summarized by any single
     Jacobian — always report spectra as distributions over states, as
     above, never as one number for "the network".

(5b) Slow points are found by OPTIMIZATION, not proof. The search returns
     what it converges to from the initializations you supply (here: real
     trajectory states). Missing structure is always possible; q > 0 means
     "slow", not "fixed"; and the whole portrait is conditional on the
     constant input you chose to hold.

(5c) Fixed-point structure is input-conditional. The line attractor shown
     above exists under the mean stimulus; a different input can destroy or
     create attractors. There is no single "the dynamics" of the network.

(5d) Grassmann/SPD comparisons need a COMMON ambient space. Hidden states of
     two networks with different widths (or different random seeds, whose
     units are permuted) cannot be compared directly — align them first
     (e.g. Procrustes/CCA) or compare basis-free quantities such as
     eigenvalue spectra and participation ratios.

(5e) These are descriptions, not causes. A high-gain direction is not proof
     the network USES it; confirm with perturbation/ablation experiments.

(5f) Everything here assumes smooth dynamics. ReLU RNNs are piecewise
     linear (see demo 1) and LSTMs have gates that saturate; the tools run,
     but "the Jacobian at x" becomes region-dependent in the same way.
""")
print("Done. Figures in outputs/figures/")
