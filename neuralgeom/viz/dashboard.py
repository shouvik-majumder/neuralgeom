"""
Rolling dashboard for the neural-data analysis.
===============================================

Regenerates outputs/pdf/neural_dashboard.pdf from whatever figures currently
exist. Re-run it any time; it is cheap (no analysis, just assembly) and always
reflects the latest saved figures.

    python build_dashboard.py

Layout
------
  page 1   status board: what has been run, headline numbers, open issues
  then     one page per figure, in pipeline order, each with a caption
           explaining what is plotted and what to look for

Adding a figure: drop it in one of the scanned directories and, if you want a
caption, add an entry to CAPTIONS keyed by filename. Uncaptioned figures are
still included (flagged as needing a caption) so nothing silently disappears.

Numbers shown on the status page are read from results.json when present
(written by the analysis scripts via log_result), else from STATUS_FALLBACK.
"""
from __future__ import annotations

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path

from ..paths import FIG_ROOT, OUTPUT_DIR, PDF_DIR
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from datetime import datetime
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

FIGROOT = FIG_ROOT
OUT = PDF_DIR / "neural_dashboard.pdf"
RESULTS = OUTPUT_DIR / "results.json"

# All neural figures live FLAT in outputs/figures/neural/ (stage identity is
# in the filename prefix). Scanned in this order; label = the section header.
# A file belongs to the first section whose prefix matches its name.
SECTIONS = [
    (("qc",), "STAGE 0 — data QC and wrangling checks"),
    (("s1_",), "STAGE 1 — descriptive population structure"),
    (("s2_",), "STAGE 2 — map-free geometry (binned covariances)"),
    (("s2b_",), "STAGE 2b — sweeping the arbitrary choices"),
    (("s3_",), "STAGE 3 — pullback metric, per-time-bin decoders"),
    (("s3b_",), "STAGE 3b — pullback metric FIELD, pooled decoder"),
    (("s3c_",), "STAGE 3c — permutation controls for the metric field"),
    (("s4a_", "s4"), "STAGE 4a — condition-manifold pullback (rank-2 geometry)"),
    (("pullback_real_",), "ATTRACTOR PIPELINE — rank-1 pullback of the lick-time decoder"),
    (("dynamics_real_",), "ATTRACTOR PIPELINE — LDS fit + input inference + decomposition"),
]

CAPTIONS = {
    "qc1_session_overview.png":
        "All 26 sessions. Unit yield, valid trials by outcome, and the "
        "sanity check that rewarded licks are always later than early licks. "
        "LOOK FOR: sessions with too few units or too few rewarded trials.",
    "qc2_unit_quality.png":
        "Unit inclusion for the analysed session. SNR, waveform width vs "
        "rate (FSI flagging), depth by probe, and the inclusion cascade "
        "(recorded -> qm_pass -> rate >= 0.5 Hz). LOOK FOR: how many units "
        "each criterion removes.",
    "qc3_behaviour.png":
        "Behaviour. Trial types before/after filtering; the lick-time "
        "distribution split by outcome; and CRITERION DRIFT — the running "
        "minimum rewarded lick time climbs within a session (auto-learn), so "
        "rewarded vs early is partly confounded with time in session.",
    "qc4_spiking_sanity.png":
        "Post-preprocessing sanity. Rate distribution, runaway-unit check, "
        "population mean (ALIGNMENT CHECK: flat before t=0, rising after), "
        "sparsity, split-half PSTH reliability, and whether licks fall inside "
        "the loaded window.",
    "qc5_rasters_psth.png":
        "THE KEY WRANGLING CHECK. Raw spike rasters (top) directly above "
        "PSTHs computed from the processed tensor (bottom) for the same "
        "units. If these disagree, the preprocessing is wrong.",
    "qc6_population.png":
        "Population activity and behavioural encoding. Note the cue-aligned "
        "curves for all lick-time bins rise together at t=0 while the "
        "lick-aligned ones are staggered: the dominant response is "
        "CUE-locked, and lick timing is a secondary modulation.",
    "s1_0_window_choice.png":
        "Choosing the pre-lick window W. Longer W keeps more of the timing "
        "dynamics but discards fast trials, and because slow trials are more "
        "often rewarded the retained set becomes biased. LOOK FOR: the "
        "trade-off; at W=0.4 s the kept trials are ~90% rewarded.",
    "s1_A_prelick_structure.png":
        "EPOCH A (post-cue pre-lick) structure. Dimensionality with a "
        "cross-validated curve (gap = overfit noise dimensions), split-half "
        "subspace angles (only the top ~3 PCs reproduce), trajectories by "
        "lick-time bin, and lick-time encoding (PC1 beats every single unit).",
    "s1_B_full_structure.png":
        "EPOCH B (full window) structure, same panels. More time bins and "
        "trials make the subspace far more reproducible than in Epoch A.",
    "s1_compare_epochs.png":
        "The two epochs side by side. They are never pooled.",
    "s2_A_prelick_geometry.png":
        "Binned-covariance geometry, Epoch A. Each bin's population "
        "covariance is a point on the SPD manifold; distances between bins "
        "are compared with |difference in median lick time|. Bottom row: "
        "confound controls (learning drift, reward content, firing rate).",
    "s2_B_full_geometry.png":
        "Same for Epoch B. Its stronger numbers are discounted: the window "
        "contains the lick and outcome, so bins with separated lick times "
        "are trivially misaligned in time.",
    "s2b_surface_A_prelick.png":
        "Sweeping reducer x bin count, Epoch A. Top: Mantel r. Middle: "
        "split-half reliability of the distance matrix. Bottom: Mantel p "
        "(NOTE the floor — with 4 bins p cannot go below ~0.04). LOOK FOR: "
        "configurations that are strong AND reliable, not just strong.",
    "s2b_surface_B_full.png": "Same sweep for Epoch B.",
    "s2b_tradeoff.png":
        "Every configuration as one point: effect size against reliability. "
        "Upper right is what we want. Many high-r configurations have low "
        "reliability and should not be interpreted.",
    "s2b_null_reducer.png":
        "NULL REDUCER. Same analysis in the top-k PC space (blue) vs k RANDOM "
        "orthonormal directions (red). At low k PCA wins clearly; by k=15-20 "
        "random projections catch up, so the raw effect size at high k is "
        "largely about dimensionality rather than structure.",
    "s3_A_prelick_pullback.png":
        "PULLBACK METRIC of a decoder f: state -> lick time, fitted "
        "separately at each time bin. |grad f| is the sensitivity "
        "(sqrt pseudo-det g); its direction is the decoding axis; the "
        "remaining k-1 dimensions are the behaviour's null space. LOOK FOR: "
        "R^2 crossing zero at ~50 ms (sensory delay), and the linear decoder "
        "failing where the MLP succeeds.",
    "s3_reducer_comparison.png":
        "Does the pullback need PCA's directions? PCA-10 vs random "
        "projections at matched k, plus full space and other k.",
    "s3b_A_pooled_field.png":
        "STAGE 3b, EPOCH A. ONE decoder over all (trial, bin) pairs, target "
        "= time remaining until the lick, CV split BY TRIAL and PCA fit "
        "inside each fold. The metric is now a FIELD over state space "
        "(bottom right: colour = sensitivity, arrows = grad f). LOOK FOR: "
        "the gap between the real fit and the shuffled/shifted nulls.",
    "qc_behaviour_check.png":
        "Ad-hoc check made while validating the file format: what defines a "
        "rewarded trial, and what delay_duration_s actually is. Confirmed "
        "first_lick_s is cue-relative and reward = licking after a required "
        "delay (clean 0.75 s step in SM318). delay_duration_s is unrelated "
        "to lick time and is dropped from the loader.",
    "s2_Ap_prelick_earlyonly_geometry.png":
        "OPT-IN CONTROL, not part of the default pipeline: Epoch A "
        "restricted to early-lick trials so outcome is held constant. "
        "Retained only for reference; per instruction the reward/lick-time "
        "collinearity is accepted rather than corrected by subsetting.",
    "s2_compare_epochs.png":
        "Stage 2 summary across epochs and metrics, with split-half "
        "reliability and shrinkage sensitivity.",
    "s3b_B_pooled_field.png":
        "STAGE 3b, EPOCH B. The nulls nearly match the real fit here — in "
        "the full window almost all apparent performance is the network "
        "reading elapsed time, not upcoming behaviour. Epoch B is "
        "uninformative for this question.",
}

CAPTIONS["s3c_A_controls.png"] = (
    "PERMUTATION CONTROLS on every reported quantity, not just R^2. Red line "
    "= real data, grey = null draws (C1 shuffled lick times, C3 circular "
    "time shift within trial, C4 random projection, C5 within-bin trial "
    "shuffle). KEY RESULT (bottom right): the distribution of decoding-axis "
    "rotation for real data overlaps the shuffled null, so the apparent "
    "curvature of the metric field in Stage 3b is largely estimation noise. "
    "The |grad f| SPREAD is likewise identical to null (2.55 vs 2.53). Only "
    "R^2 and the sensitivity-vs-time correlation stand clear of the nulls.")

CAPTIONS["s4a_A_condition_geometry.png"] = (
    "CONDITION-MANIFOLD PULLBACK — the direction both reference papers use. "
    "Map f: (time t, lick time tau) -> population state, so the domain is 2-D "
    "and rank(g)=2: volume element, anisotropy AND curvature all exist "
    "(none of them do for a scalar decoder, where g is rank 1). Columns: "
    "area magnification sqrt(det g); log anisotropy (is the code more "
    "sensitive to t or to tau?); Gaussian curvature K (warped vs merely "
    "stretched); distribution of magnification. Top row Euclidean codomain "
    "metric, bottom row noise-weighted (g = J^T Sigma^-1 J, i.e. Fisher "
    "metric for Gaussian noise, distances in d' units). LOOK FOR: the "
    "magnification ridge running along small lick times and its dependence "
    "on t, and the sign structure of K.")
CAPTIONS["s4a_A_controls.png"] = (
    "Stage 4a controls. Left: does noise weighting change the geometry, or "
    "is it just a rescaling? Middle: N1, is the magnification pattern above "
    "the shuffled-lick-time null? Right: N2, does including lick time in the "
    "domain explain more population variance than time alone?")

STATUS_FALLBACK = {
    "session": "SM239_20230302_g0 (WT_control, day 2)",
    "rows": [
        ["Stage 0 QC", "done",
         "95 units kept of 161; split-half PSTH r = 0.84; edge artifact from "
         "causal smoothing found and fixed"],
        ["Stage 1 structure", "done",
         "Epoch A: only top ~3 PCs reproduce (split-half angles 8/14/18 then "
         "31/45 deg); PC1 vs lick time r = +0.39"],
        ["Stage 2 binned geometry", "superseded",
         "Told us structure exists; effect size not trustworthy (moved "
         "0.52-0.96 with arbitrary settings). Retired as a screen."],
        ["Stage 2b sweeps", "done",
         "PCA beats random projections only at low k; by k=15-20 the null "
         "catches up"],
        ["Stage 3 per-bin pullback", "done",
         "CV R^2 peaks +0.345 at 130 ms, crosses 0 at ~50 ms; linear decoder "
         "fails (<= +0.07); axis split-half stability 28 deg"],
        ["Stage 3b pooled field", "done",
         "Epoch A R^2 = +0.321 vs shuffled +0.122, shifted +0.095, random "
         "proj +0.180. Epoch B: nulls ~= real fit, uninformative."],
        ["Stage 3c controls", "done",
         "R^2 +0.321 vs C1 null +0.100+-0.008 (~28 SD) — solid. BUT |grad f| "
         "spread 2.55 vs null 2.53 and axis rotation 58 vs null 67 deg: the "
         "metric FIELD structure is not above chance."],
        ["Stage 4a condition manifold", "done",
         "rank-2 geometry. NO trial thresholding: all 964 trials contribute "
         "via a per-sample pre-lick mask (13439 samples). Volume spread "
         "42x vs shuffled null 242+-47 (real is LOWER); anisotropy 51 vs null "
         "1722; K median -0.019 vs null +0.008, 73% negative. Support mask "
         "restricts the metric to the triangular region t < lick time."],
        ["Stage 4b-d autoencoder / RNN / Fisher-Rao", "planned",
         "shared core in neuro/geometry.py already supports arbitrary domain "
         "dim and either codomain metric"],
    ],
    "issues": [
        "RESOLVED by Stage 3c: the Epoch-A metric field is effectively a "
        "CONSTANT direction. Axis rotation (58 deg) is below the shuffled "
        "null (67 deg) and sensitivity spread matches null exactly, so the "
        "field's apparent curvature was estimation noise. Consistent with "
        "MLP = linear in R^2. What survives is the decoding axis itself and "
        "its magnitude, not a spatially varying geometry.",
        "Stage 4a: the shuffled-lick-time null gives LARGER volume spread "
        "and anisotropy than the real data, because shuffling makes the map "
        "depend on t alone and hence degenerate in the tau direction. The "
        "real map being less degenerate is the meaningful direction, but a "
        "shuffle that preserves the tau marginal per time bin would be a "
        "cleaner null. Not yet implemented.",
        "Stage 3c p-values floor at 1/(n_draws+1) = 0.20 with 4 draws; the "
        "informative comparison there is the effect size in null-SD units, "
        "not p. More draws needed before quoting p.",
        "PCA leak in Stage 3 (reducer fit on all trials) is fixed in Stage 3b "
        "but Stage 3's numbers still carry it.",
        "Reward and lick time are collinear by task design (r = 0.90 across "
        "bins); per instruction this is accepted, not corrected.",
        "Criterion drift within learning sessions is a known, accepted "
        "confound; normalized trial number is available as a regressor "
        "(Session.trial_frac) but not yet used.",
    ],
}


def load_status():
    if RESULTS.exists():
        try:
            return json.loads(RESULTS.read_text())
        except Exception:
            pass
    return STATUS_FALLBACK


# --------------------------------------------------------------------------- #
ACCENT = HexColor("#1d4e89")
GREY = HexColor("#555555")
ss = getSampleStyleSheet()
S_TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=18,
                         textColor=ACCENT, spaceAfter=2)
S_SUB = ParagraphStyle("s", parent=ss["Normal"], fontSize=9.5,
                       textColor=GREY, alignment=TA_CENTER, spaceAfter=12)
S_H = ParagraphStyle("h", parent=ss["Heading1"], fontSize=12.5,
                     textColor=ACCENT, spaceBefore=6, spaceAfter=4)
S_CAP = ParagraphStyle("c", parent=ss["Normal"], fontSize=8.6, leading=11.4,
                       textColor=HexColor("#333333"), spaceBefore=4)
S_CELL = ParagraphStyle("cl", parent=ss["Normal"], fontSize=8.2, leading=10.8)
S_MISS = ParagraphStyle("m", parent=S_CAP, textColor=HexColor("#b00020"))


def build():
    st = load_status()
    story = [Paragraph("Neural-data pullback-metric dashboard", S_TITLE),
             Paragraph(f"session {st['session']} &nbsp;|&nbsp; regenerated "
                       f"{datetime.now():%Y-%m-%d %H:%M}", S_SUB)]

    story.append(Paragraph("Pipeline status", S_H))
    rows = [[Paragraph(f"<b>{c}</b>", S_CELL) for c in
             ("stage", "state", "headline")]]
    for r in st["rows"]:
        rows.append([Paragraph(str(c), S_CELL) for c in r])
    t = Table(rows, colWidths=[1.6 * inch, 0.9 * inch, 6.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#cfe0f2")),
        ("GRID", (0, 0), (-1, -1), 0.4, GREY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story += [t, Spacer(1, 10),
              Paragraph("Open issues / caveats", S_H)]
    for i, s in enumerate(st["issues"], 1):
        story.append(Paragraph(f"{i}. {s}", S_CAP))

    n_fig = 0
    d = FIGROOT / "neural"
    all_figs = sorted(d.glob("*.png")) if d.exists() else []
    assigned = set()

    def section_figs(prefixes):
        out = [f for f in all_figs
               if f.name not in assigned and f.name.startswith(prefixes)]
        assigned.update(f.name for f in out)
        return out

    for prefixes, label in SECTIONS:
        figs = section_figs(tuple(prefixes))
        if not figs:
            continue
        story.append(PageBreak())
        story.append(Paragraph(label, S_H))
        for i, f in enumerate(figs):
            if i:
                story.append(PageBreak())
                story.append(Paragraph(label, S_H))
            w, h = PILImage.open(f).size
            width = 9.2 * inch
            height = width * h / w
            if height > 5.6 * inch:               # keep caption on the page
                height = 5.6 * inch
                width = height * w / h
            story.append(Image(str(f), width=width, height=height))
            cap = CAPTIONS.get(f.name)
            story.append(Paragraph(
                f"<b>{f.name}</b> &mdash; {cap}" if cap else
                f"<b>{f.name}</b> &mdash; (no caption registered yet)",
                S_CAP if cap else S_MISS))
            n_fig += 1

    doc = SimpleDocTemplate(str(OUT), pagesize=landscape(letter),
                            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                            topMargin=0.45 * inch, bottomMargin=0.4 * inch,
                            title="Neural pullback dashboard")
    doc.build(story)
    print(f"wrote {OUT}  ({n_fig} figures)")
    missing = [f.name for f in all_figs if f.name not in CAPTIONS]
    orphans = [f.name for f in all_figs if f.name not in assigned]
    if missing:
        print("  figures without captions:", ", ".join(missing))
    if orphans:
        print("  figures matching no section (add a prefix to SECTIONS):",
              ", ".join(orphans))


if __name__ == "__main__":
    build()
