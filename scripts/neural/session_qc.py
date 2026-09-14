"""
session_qc.py — data quality control for the Neuropixels sessions.
===================================================================

Purpose: catch data-wrangling problems BEFORE any modelling. Nothing here
fits a model or computes a geometry; every panel describes what is actually
in the file after loading, with the interpretation written on the figure so
a wrong assumption is visible rather than buried.

Figures:
  qc1_session_overview   dataset-wide context: every session's yield/behaviour
  qc2_unit_quality       SNR, spike width, depth, rate, cell type, what was cut
  qc3_behaviour          lick times, trial types, criterion drift
  qc4_spiking_sanity     rate distributions, silent/runaway units, alignment
  qc5_rasters_psth       example units: raw-spike rasters vs processed PSTHs
  qc6_population         population activity, dimensionality, lick tuning

Run:  python scripts/neural/session_qc.py [SESSION_ID]
"""
from __future__ import annotations

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from neuralgeom.paths import DATA_DIR, fig_dir  # noqa: E402

from neuralgeom.data.loader import (DROPPED_FIELDS, RESPONSE_TYPES,  # noqa: E402
                          load_session, session_table)

DATA = DATA_DIR
FIG = fig_dir("neural")

SESSION = sys.argv[1] if len(sys.argv) > 1 else "SM259_20230417_g0"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 145, "font.size": 8,
                     "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.grid": True, "grid.alpha": 0.2,
                     "legend.fontsize": 7})
C_EARLY, C_REW = "#c0392b", "#2471a3"


def save(fig, name):
    p = FIG / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p.name}")
    return p


def note(ax, txt, right=True):
    ax.text(0.98 if right else 0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=6.4, va="top", ha="right" if right else "left",
            bbox=dict(fc="white", alpha=0.82, lw=0.4))


# =========================================================================== #
print(f"\n=== QC1: dataset overview ===")
tab = session_table(DATA)
fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.2), constrained_layout=True)
groups = sorted({r["group"] for r in tab})
gcol = {g: plt.cm.tab10(i % 10) for i, g in enumerate(groups)}

ax[0, 0].bar(range(len(tab)), [r["n_units"] for r in tab], color="0.8",
             label="all recorded")
ax[0, 0].bar(range(len(tab)), [r["n_units_qm"] for r in tab], color="C0",
             label="pass qm_pass")
ax[0, 0].set_xticks(range(len(tab)), [r["session_id"][:5] for r in tab],
                    rotation=90, fontsize=5)
ax[0, 0].set_ylabel("units"); ax[0, 0].legend()
ax[0, 0].set_title("Unit yield per session")

w = 0.4
ax[0, 1].bar(np.arange(len(tab)) - w / 2, [r["n_early"] for r in tab], w,
             color=C_EARLY, label="early lick")
ax[0, 1].bar(np.arange(len(tab)) + w / 2, [r["n_rewarded"] for r in tab], w,
             color=C_REW, label="rewarded")
ax[0, 1].set_xticks(range(len(tab)), [r["session_id"][:5] for r in tab],
                    rotation=90, fontsize=5)
ax[0, 1].set_ylabel("valid trials"); ax[0, 1].legend()
ax[0, 1].set_title("Valid trials by outcome\n(no-response/no-cue excluded by "
                   "validity flag)")

for r in tab:
    ax[0, 2].scatter(r["median_lick_early"], r["median_lick_rewarded"],
                     color=gcol[r["group"]], s=26)
ax[0, 2].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[0, 2].set_xlabel("median lick, early trials (s)")
ax[0, 2].set_ylabel("median lick, rewarded (s)")
ax[0, 2].set_title("Rewarded licks are later than early licks\n"
                   "(sanity: the task is about withholding)")
ax[0, 2].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c,
                                    label=g, ms=4) for g, c in gcol.items()],
                fontsize=5, loc="lower right")

ax[1, 0].scatter([r["training_day"] for r in tab],
                 [r["n_rewarded"] / max(1, r["n_valid"]) for r in tab],
                 c=[gcol[r["group"]] for r in tab], s=26)
ax[1, 0].set_xlabel("training day"); ax[1, 0].set_ylabel("fraction rewarded")
ax[1, 0].set_title("Performance vs training day")

ax[1, 1].scatter([r["n_units_qm"] for r in tab], [r["n_valid"] for r in tab],
                 c=[gcol[r["group"]] for r in tab], s=26)
for r in tab:
    if r["session_id"] == SESSION:
        ax[1, 1].scatter([r["n_units_qm"]], [r["n_valid"]], s=150,
                         facecolor="none", edgecolor="k", lw=1.4)
        ax[1, 1].annotate("this session", (r["n_units_qm"], r["n_valid"]),
                          fontsize=6, xytext=(6, 6),
                          textcoords="offset points")
ax[1, 1].set_xlabel("units passing QC"); ax[1, 1].set_ylabel("valid trials")
ax[1, 1].set_title("Yield trade-off across sessions")

cnt = {}
for r in tab:
    cnt[r["group"]] = cnt.get(r["group"], 0) + 1
ax[1, 2].barh(list(cnt), list(cnt.values()), color=[gcol[g] for g in cnt])
ax[1, 2].set_xlabel("sessions"); ax[1, 2].tick_params(labelsize=6)
ax[1, 2].set_title("Experimental groups")
fig.suptitle("QC1 — dataset overview (one point/bar per recording session)",
             fontsize=10)
save(fig, "qc1_session_overview.png")

# =========================================================================== #
print(f"\n=== QC2-6: session {SESSION} ===")
path = DATA / f"{SESSION}.h5"
s = load_session(path)
s_all = load_session(path, qm_only=False, min_rate_hz=0.0)
print(f"  {s}")
print(f"  kept {s.n_units}/{s_all.n_units} units, {s.n_trials} trials, "
      f"bin {s.bin_s*1000:.0f} ms, t {s.t[0]:.2f}..{s.t[-1]:.2f} s")
print(f"  dropped fields: {list(DROPPED_FIELDS)}")

U, Uall = s.units, s_all.units
kept = np.isin(Uall["unit_index"], U["unit_index"])
fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.2), constrained_layout=True)

ax[0, 0].hist([Uall["snr"][kept], Uall["snr"][~kept]], bins=25, stacked=True,
              color=["C0", "0.75"], label=["kept", "excluded"])
ax[0, 0].set_xlabel("SNR"); ax[0, 0].set_ylabel("units"); ax[0, 0].legend()
ax[0, 0].set_title("Unit SNR")

ax[0, 1].scatter(Uall["spike_width_ms"][~kept], Uall["mean_fr_hz"][~kept],
                 s=12, c="0.75", label="excluded")
ax[0, 1].scatter(Uall["spike_width_ms"][kept], Uall["mean_fr_hz"][kept], s=14,
                 c=Uall["is_fsi"][kept], cmap="coolwarm", label="kept")
ax[0, 1].set_yscale("log")
ax[0, 1].set_xlabel("spike width (ms)"); ax[0, 1].set_ylabel("mean rate (Hz)")
ax[0, 1].legend()
ax[0, 1].set_title("Waveform width vs rate\n(red = flagged FSI)")

for pr_ in np.unique(Uall["probe"]):
    m = (Uall["probe"] == pr_) & kept
    lab = pr_.decode() if isinstance(pr_, bytes) else str(pr_)
    ax[0, 2].scatter(Uall["mean_fr_hz"][m], Uall["depth_um"][m], s=14, label=lab)
ax[0, 2].set_xscale("log"); ax[0, 2].invert_yaxis()
ax[0, 2].set_xlabel("mean rate (Hz)"); ax[0, 2].set_ylabel("depth (um)")
ax[0, 2].legend(); ax[0, 2].set_title("Depth vs rate, by probe")

steps = ["recorded", "qm_pass", "+rate>=0.5Hz"]
vals = [s_all.n_units, int(Uall["qm_pass"].sum()), s.n_units]
ax[1, 0].bar(steps, vals, color=["0.75", "C0", "C2"])
for i, v in enumerate(vals):
    ax[1, 0].text(i, v + 1, str(v), ha="center", fontsize=7)
ax[1, 0].set_ylabel("units"); ax[1, 0].set_title("Unit inclusion cascade")

ax[1, 1].hist(np.log10(np.clip(U["mean_fr_hz"], 1e-2, None)), bins=25,
              color="C0")
ax[1, 1].set_xlabel("log10 mean rate (Hz)"); ax[1, 1].set_ylabel("units")
ax[1, 1].set_title("Rate distribution, kept units")
note(ax[1, 1], f"median {np.median(U['mean_fr_hz']):.1f} Hz\n"
               f"range {U['mean_fr_hz'].min():.1f}-{U['mean_fr_hz'].max():.0f}")

types = {"pyramidal": int(U["is_pyramidal"].sum()), "FSI": int(U["is_fsi"].sum())}
types["neither"] = int(len(U["is_fsi"]) - types["pyramidal"] - types["FSI"])
ax[1, 2].bar(list(types), list(types.values()), color=["C0", "C3", "0.7"])
ax[1, 2].set_ylabel("units"); ax[1, 2].set_title("Putative cell types (kept)")
fig.suptitle(f"QC2 — unit quality and inclusion: {SESSION}", fontsize=10)
save(fig, "qc2_unit_quality.png")

# =========================================================================== #
rt = s.trials["response_type"]
rew, early = s.cond["rewarded"], s.cond["early_lick"]
with h5py.File(path, "r") as f:
    rt_raw = f["trials/response_type"][:]
fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.2), constrained_layout=True)

ax[0, 0].bar([RESPONSE_TYPES[k] for k in range(4)],
             [int((rt_raw == k).sum()) for k in range(4)], color="0.75",
             label="in file")
ax[0, 0].bar([RESPONSE_TYPES[k] for k in range(4)],
             [int((rt == k).sum()) for k in range(4)], color="C0",
             label="after filtering")
ax[0, 0].tick_params(axis="x", rotation=20, labelsize=6.5)
ax[0, 0].set_ylabel("trials"); ax[0, 0].legend()
ax[0, 0].set_title("Trial types before/after the valid filter")

bins = np.linspace(0, float(np.nanpercentile(s.lick, 99.5)), 50)
ax[0, 1].hist(s.lick[early], bins=bins, color=C_EARLY, alpha=0.7,
              label=f"early (n={int(early.sum())})")
ax[0, 1].hist(s.lick[rew], bins=bins, color=C_REW, alpha=0.7,
              label=f"rewarded (n={int(rew.sum())})")
ax[0, 1].set_xlabel("first_lick_s (cue-relative)"); ax[0, 1].set_ylabel("trials")
ax[0, 1].legend(); ax[0, 1].set_title("THE BEHAVIOUR: when the animal licks")
ov = float(np.mean(s.lick[early] > s.lick[rew].min())) if rew.sum() else np.nan
note(ax[0, 1], f"early max {s.lick[early].max():.3f}\n"
               f"rew min {s.lick[rew].min():.3f}\noverlap {ov:.0%}")

k = np.arange(s.n_trials)
ax[0, 2].scatter(k[early], s.lick[early], s=3, c=C_EARLY, alpha=0.45,
                 label="early")
ax[0, 2].scatter(k[rew], s.lick[rew], s=5, c=C_REW, alpha=0.8, label="rewarded")
w = max(40, s.n_trials // 15)
run = [s.lick[max(0, j - w):j + w][rew[max(0, j - w):j + w]].min()
       if rew[max(0, j - w):j + w].sum() > 3 else np.nan
       for j in range(s.n_trials)]
ax[0, 2].plot(k, run, "k-", lw=1.5, label="running min rewarded lick")
ax[0, 2].set_xlabel("trial (in session)"); ax[0, 2].set_ylabel("first_lick_s")
ax[0, 2].legend(fontsize=6)
ax[0, 2].set_title("CRITERION DRIFT (auto-learn): rewarded vs early\n"
                   "is partly confounded with time in session")

edges = np.linspace(0, s.n_trials, 11).astype(int)
ax[1, 0].plot(0.5 * (edges[:-1] + edges[1:]),
              [rew[a:b].mean() for a, b in zip(edges[:-1], edges[1:])], "o-")
ax[1, 0].set_xlabel("trial (in session)"); ax[1, 0].set_ylabel("fraction rewarded")
ax[1, 0].set_title("Performance across the session")

lb = s.lick_bins(5)
ax[1, 1].bar(range(lb["n_bins"]), lb["counts"], color="C4")
ax[1, 1].set_xticks(range(lb["n_bins"]),
                    [f"{v*1000:.0f}" for v in lb["median"]], fontsize=7)
ax[1, 1].set_xlabel("lick-time bin (labelled by MEDIAN lick, ms)")
ax[1, 1].set_ylabel("trials")
ax[1, 1].set_title("Lick-time quantile bins\n(the conditions used downstream)")

if "second_lick_s" in s.trials:
    d = np.asarray(s.trials["second_lick_s"], float) - s.lick
    good = np.isfinite(d)
    ax[1, 2].hist(d[good], bins=40, color="C2")
    ax[1, 2].set_xlabel("second lick - first lick (s)")
    ax[1, 2].set_ylabel("trials"); ax[1, 2].set_title("Inter-lick interval")
    note(ax[1, 2], f"median {np.nanmedian(d[good])*1000:.0f} ms")
fig.suptitle(f"QC3 — behaviour: {SESSION} ({s.group}, day {s.training_day})",
             fontsize=10)
save(fig, "qc3_behaviour.png")

# =========================================================================== #
fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.2), constrained_layout=True)
mean_rate = s.X.mean(axis=(0, 1))
ax[0, 0].hist(mean_rate, bins=30, color="C0")
ax[0, 0].set_xlabel("mean rate in window (Hz)"); ax[0, 0].set_ylabel("units")
ax[0, 0].set_title("Rates AFTER loading (the analysis tensor)")
note(ax[0, 0], f"min {mean_rate.min():.2f} Hz\nmax {mean_rate.max():.1f} Hz\n"
               f"any zero: {bool((mean_rate <= 0).any())}")

peak = s.X.max(axis=(0, 1))
ax[0, 1].scatter(mean_rate, peak, s=12)
ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log")
ax[0, 1].set_xlabel("mean rate (Hz)"); ax[0, 1].set_ylabel("peak rate (Hz)")
ax[0, 1].set_title("Peak vs mean — runaway-unit check")
note(ax[0, 1], f"max peak {peak.max():.0f} Hz\n1 spike/bin = {1/s.bin_s:.0f} Hz")

ax[0, 2].plot(s.t, s.X.mean(axis=(0, 2)), color="k")
ax[0, 2].axvline(0, color="C3", ls="--", label="cue onset")
ax[0, 2].axvline(float(np.median(s.lick)), color="C2", ls="--",
                 label=f"median lick {np.median(s.lick):.2f}s")
ax[0, 2].set_xlabel("time from cue (s)"); ax[0, 2].set_ylabel("pop rate (Hz)")
ax[0, 2].legend()
ax[0, 2].set_title("Population mean — ALIGNMENT CHECK\n"
                   "(a response should follow t=0)")

ax[1, 0].hist((s.X == 0).mean(axis=(0, 1)), bins=30, color="C1")
ax[1, 0].set_xlabel("fraction of bins with zero rate")
ax[1, 0].set_ylabel("units")
ax[1, 0].set_title("Sparsity per unit (1.0 = silent)")

half = s.n_trials // 2
m1, m2 = s.X[:half].mean(0), s.X[half:].mean(0)
rel = np.array([np.corrcoef(m1[:, u], m2[:, u])[0, 1]
                for u in range(s.n_units)])
ax[1, 1].hist(rel[np.isfinite(rel)], bins=30, color="C2")
ax[1, 1].set_xlabel("corr(PSTH 1st half, 2nd half)"); ax[1, 1].set_ylabel("units")
ax[1, 1].set_title("Split-half PSTH reliability")
note(ax[1, 1], f"median r = {np.nanmedian(rel):.2f}\n"
               f"{int((rel > 0.5).sum())}/{s.n_units} units r>0.5")

ax[1, 2].hist(s.lick, bins=40, color="C4")
ax[1, 2].axvline(s.t[-1], color="r", ls="--", label="window end")
ax[1, 2].set_xlabel("first lick time (s)"); ax[1, 2].set_ylabel("trials")
ax[1, 2].legend()
ax[1, 2].set_title("Do licks fall INSIDE the loaded window?")
note(ax[1, 2], f"window {s.t[0]:.2f}..{s.t[-1]:.2f}s\n"
               f"licks after window: {int((s.lick > s.t[-1]).sum())}\n"
               f"licks <= 0: {int((s.lick <= 0).sum())}")
fig.suptitle(f"QC4 — spiking sanity after preprocessing "
             f"({s.bin_s*1000:.0f} ms bins, causal boxcar)", fontsize=10)
save(fig, "qc4_spiking_sanity.png")

# =========================================================================== #
mod = (np.abs(s.X[rew].mean(0) - s.X[early].mean(0)).max(0)
       if rew.sum() > 5 else s.X.std(axis=(0, 1)))
top = np.argsort(mod)[::-1][:4]
with h5py.File(path, "r") as f:
    st = f["spikes/spike_times_flat"][:]
    tid = f["spikes/spike_trial_ids_flat"][:]
    off = f["spikes/unit_spike_offsets"][:]
    uidx_all = f["units/unit_index"][:]

tr_ids_kept = np.asarray(s.trials.get("trial_index_0based",
                                      np.arange(s.n_trials))).astype(int)
order = np.argsort(s.lick)
rank = np.empty_like(order); rank[order] = np.arange(len(order))
row_of_trial = {int(tr): int(rank[i]) for i, tr in enumerate(tr_ids_kept)}

fig, ax = plt.subplots(2, 4, figsize=(14, 6.0), constrained_layout=True)
for j, u in enumerate(top):
    ui = int(np.where(uidx_all == U["unit_index"][u])[0][0])
    sp_t = st[off[ui]:off[ui + 1]]
    sp_tr = tid[off[ui]:off[ui + 1]]
    inwin = (sp_t >= s.t[0]) & (sp_t <= s.t[-1])
    xs, ys = [], []
    for tt, tr in zip(sp_t[inwin], sp_tr[inwin]):
        r_ = row_of_trial.get(int(tr))
        if r_ is not None:
            xs.append(tt); ys.append(r_)
    a = ax[0, j]
    a.scatter(xs, ys, s=0.12, c="k", alpha=0.5)
    a.plot(np.sort(s.lick), np.arange(s.n_trials), color=C_REW, lw=1.0,
           label="first lick")
    a.axvline(0, color="C3", ls="--", lw=0.8)
    a.set_xlim(s.t[0], s.t[-1]); a.set_ylim(0, s.n_trials)
    a.set_xlabel("time from cue (s)")
    a.set_title(f"unit {int(U['unit_index'][u])} — raw spike raster\n"
                f"(trials sorted by lick time)")
    if j == 0:
        a.set_ylabel("trial (sorted by lick)"); a.legend(fontsize=6)

    b = ax[1, j]
    for m, c, lab in [(early, C_EARLY, "early"), (rew, C_REW, "rewarded")]:
        if m.sum() < 5:
            continue
        mu = s.X[m][:, :, u].mean(0)
        se = s.X[m][:, :, u].std(0) / np.sqrt(int(m.sum()))
        b.plot(s.t, mu, color=c, label=f"{lab} (n={int(m.sum())})")
        b.fill_between(s.t, mu - se, mu + se, color=c, alpha=0.25)
    b.axvline(0, color="0.4", lw=0.8)
    b.set_xlabel("time from cue (s)"); b.legend(fontsize=6)
    b.set_title("PSTH from the processed tensor")
    if j == 0:
        b.set_ylabel("firing rate (Hz)")
fig.suptitle("QC5 — raw spike rasters (top) vs processed PSTHs (bottom). "
             "These must agree; if not, the wrangling is wrong.", fontsize=10)
save(fig, "qc5_rasters_psth.png")

# =========================================================================== #
fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.2), constrained_layout=True)
Z = s.zscored()
ax[0, 0].plot(s.t, Z.mean(axis=(0, 2)), "k")
ax[0, 0].axvline(0, color="C3", ls="--")
ax[0, 0].set_xlabel("time from cue (s)"); ax[0, 0].set_ylabel("z-scored rate")
ax[0, 0].set_title("Population mean (z-scored per unit)")

lb = s.lick_bins(5)
for b in range(lb["n_bins"]):
    m = lb["label"] == b
    if m.sum() < 10:
        continue
    ax[0, 1].plot(s.t, Z[m].mean(axis=(0, 2)),
                  color=plt.cm.viridis(b / max(1, lb["n_bins"] - 1)),
                  label=f"{lb['median'][b]*1000:.0f} ms")
ax[0, 1].axvline(0, color="0.4", lw=0.8)
ax[0, 1].set_xlabel("time from cue (s)"); ax[0, 1].set_ylabel("z-scored rate")
ax[0, 1].legend(title="median lick", fontsize=6, title_fontsize=6)
ax[0, 1].set_title("Population activity by LICK-TIME BIN (cue-aligned)")

sl = s.lick_aligned((-0.6, 0.4))
Zl = sl.zscored(baseline=(-0.6, -0.4))
lbl = sl.lick_bins(5)
for b in range(lbl["n_bins"]):
    m = lbl["label"] == b
    if m.sum() < 10:
        continue
    ax[0, 2].plot(sl.t, Zl[m].mean(axis=(0, 2)),
                  color=plt.cm.viridis(b / max(1, lbl["n_bins"] - 1)),
                  label=f"{lbl['median'][b]*1000:.0f} ms")
ax[0, 2].axvline(0, color="C2", ls="--", label="lick")
ax[0, 2].set_xlabel("time from LICK (s)"); ax[0, 2].legend(fontsize=6)
ax[0, 2].set_title(f"Same, LICK-aligned ({sl.n_trials}/{s.n_trials} trials fit)")

srt = np.argsort(np.argmax(Z.mean(0), axis=0))
im = ax[1, 0].imshow(Z.mean(0)[:, srt].T, aspect="auto", cmap="RdBu_r",
                     vmin=-2, vmax=2, extent=[s.t[0], s.t[-1], s.n_units, 0])
fig.colorbar(im, ax=ax[1, 0], label="z-scored rate")
ax[1, 0].set_xlabel("time from cue (s)"); ax[1, 0].set_ylabel("unit (sorted)")
ax[1, 0].set_title("All units, trial-averaged (sorted by peak time)")

Xc = Z.reshape(-1, s.n_units)
Xc = Xc - Xc.mean(0)
sv = np.linalg.svd(Xc, compute_uv=False)
var = sv ** 2 / (sv ** 2).sum()
ax[1, 1].plot(np.arange(1, 21), np.cumsum(var)[:20], "o-")
ax[1, 1].set_xlabel("principal component"); ax[1, 1].set_ylabel("cumulative variance")
ax[1, 1].set_ylim(0, 1)
pr = float((sv ** 2).sum() ** 2 / (sv ** 4).sum())
ax[1, 1].set_title(f"Single-trial dimensionality\nparticipation ratio = {pr:.1f}")

r_lick = np.array([np.corrcoef(s.lick, s.epoch_mean(0.0, 0.3)[:, u])[0, 1]
                   for u in range(s.n_units)])
ax[1, 2].hist(r_lick[np.isfinite(r_lick)], bins=30, color="C4")
ax[1, 2].axvline(0, color="k", lw=0.8)
ax[1, 2].set_xlabel("corr(unit rate 0-300 ms, lick time)")
ax[1, 2].set_ylabel("units")
ax[1, 2].set_title("Single-unit tuning to lick TIME\n(is behaviour encoded?)")
note(ax[1, 2], f"|r|>0.1: {int((np.abs(r_lick) > 0.1).sum())}/{s.n_units}\n"
               f"max |r| = {np.nanmax(np.abs(r_lick)):.2f}")
fig.suptitle(f"QC6 — population activity and behavioural encoding: {SESSION}",
             fontsize=10)
save(fig, "qc6_population.png")

print("\nSummary")
print(f"  session       : {s.session_id}  {s.group}  day {s.training_day}")
print(f"  tensor        : {s.X.shape} (trials, time, units), "
      f"{s.bin_s*1000:.0f} ms bins, causal smoothing")
print(f"  behaviour     : lick {np.nanmin(s.lick):.3f}-{np.nanmax(s.lick):.3f}s,"
      f" median {np.nanmedian(s.lick):.3f}s")
print(f"  outcomes      : {int(rew.sum())} rewarded / {int(early.sum())} early")
print(f"  reliability   : median split-half PSTH r = {np.nanmedian(rel):.2f}")
print(f"  dimensionality: participation ratio {pr:.1f} of {s.n_units} units")
print(f"  lick tuning   : {int((np.abs(r_lick) > 0.1).sum())} units |r|>0.1")
print(f"  lick-aligned  : {sl.n_trials}/{s.n_trials} trials fit the window")
print("\nDone — inspect the six QC figures before any modelling.")
