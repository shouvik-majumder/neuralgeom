"""
make_architecture_diagram.py — render the neuralgeom architecture / data-flow map.

Draws the package structure and the flow of the shared Trajectory object into
the analysis subpackages, marking optional-extra modules. Outputs docs/architecture.svg
and docs/architecture.png. Pure matplotlib (no graphviz binary needed).

    python scripts/make_architecture_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- palette (dark text on light fills; restrained, high-contrast) -------- #
INK = "#0f172a"
SPINE_F, SPINE_E = "#fde68a", "#b45309"      # amber — the Trajectory object
SRC_F, SRC_E = "#dcfce7", "#15803d"          # green — data sources
PB_F, PB_E = "#ccfbf1", "#0f766e"            # teal — pullback metric / dynamics
SUB_F, SUB_E = "#e0e7ff", "#4338ca"          # indigo — subspace / topology
SHARE_F, SHARE_E = "#e2e8f0", "#334155"      # slate — shared geometry
SUP_F, SUP_E = "#f1f5f9", "#64748b"          # gray — support modules

fig, ax = plt.subplots(figsize=(15.5, 10.5))
ax.set_xlim(0, 16); ax.set_ylim(0, 11.6); ax.axis("off")


def box(x, y, w, h, title, subs=None, fc="#fff", ec="#333", optional=False,
        title_size=11, sub_size=8.4):
    style = "round,pad=0.02,rounding_size=0.12"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec,
                                lw=1.8, ls=(":" if optional else "-"), zorder=2))
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=INK, zorder=3)
    if subs:
        ax.text(x + w / 2, y + h - 0.66, "\n".join(subs), ha="center", va="top",
                fontsize=sub_size, color=INK, zorder=3, linespacing=1.35)


def arrow(x1, y1, x2, y2, color=INK, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, lw=lw, color=color,
                                 ls=ls, zorder=1,
                                 connectionstyle="arc3,rad=0.0"))


ax.text(8, 11.25, "neuralgeom — architecture and data flow", ha="center",
        fontsize=17, fontweight="bold", color=INK)
ax.text(8, 10.82, "analyses of one shared data object, the Trajectory",
        ha="center", fontsize=11, color="#475569", style="italic")

# ---- sources (top) -------------------------------------------------------- #
box(0.4, 9.2, 4.4, 1.25, "synth  (generators)",
    ["attractor · lowrank_rnn · fixtures",
     "subspace_rnn  (random/lowrank/ring)"], SRC_F, SRC_E)
box(5.3, 9.2, 4.0, 1.25, "data.loader / adapters",
    ["Session ← Neuropixels HDF5",
     "from_arrays · from_synthetic"], SRC_F, SRC_E)
box(9.8, 9.2, 5.8, 1.25, "external / trained nets",
    ["any (n_trials, T, N) states + labels",
     "write the HDF5 schema to analyse them"], SRC_F, SRC_E, optional=True)

# ---- the shared data object ----------------------------------------------- #
box(0.4, 7.7, 15.2, 1.1, "data.Trajectory  —  the shared data object",
    ["X (n_trials,T,N) · time · dt,tau · inputs · condition · outputs · "
     "W/U/V · aux · meta      |      HDF5 round-trip · from_session · legacy rnn_{tag}.h5"],
    SPINE_F, SPINE_E, title_size=12, sub_size=8.6)
for sx in (2.6, 7.3, 12.7):
    arrow(sx, 9.2, sx, 8.82, color="#166534")

# ---- shared geometry ------------------------------------------------------ #
box(4.9, 6.05, 6.2, 1.15, "geometry.grassmann  (shared by both)",
    ["torch model API  +  numpy/geomstats frame API",
     "GrassmannManifold · frame_distance · √2 metric"], SHARE_F, SHARE_E,
    sub_size=8.4)
arrow(8.0, 7.7, 8.0, 7.22, color=SPINE_E)

# ---- analysis subpackages ------------------------------------------------- #
# pullback / dynamics (left)
box(0.4, 0.7, 7.1, 5.1, "Pullback metric / dynamics   (core · torch)", None,
    PB_F, PB_E, title_size=12)
box(0.7, 3.9, 6.5, 1.5, "geometry",
    ["jacobian · manifold · pullback · riemann",
     "output_metrics · maps · torch_readouts · spd"], "#ffffff", PB_E)
box(0.7, 2.25, 3.15, 1.4, "dynamics",
    ["rnn · lds (IV)", "regression", "torch_geometry"], "#ffffff", PB_E)
box(4.05, 2.25, 3.15, 1.4, "tasks",
    ["cognitive · models", "training"], "#ffffff", PB_E)
box(0.7, 0.95, 6.5, 1.05, "fitting · stats · viz · paths",
    ["CV maps · permutation nulls · PDF reports · data/output paths"],
    SUP_F, SUP_E, sub_size=8.2)

# subspace / topology (right)
box(8.5, 0.7, 7.1, 5.1, "Subspace trajectory / topology", None, SUB_F, SUB_E,
    title_size=12)
box(8.8, 3.9, 6.5, 1.5, "subspace   (generic manifold interface)",
    ["embed  (Gr(k,N) frames · sv_gap)",
     "kinematics (speed/curv · Karcher · tangent-PCA) · pooling"],
    "#ffffff", SUB_E)
box(8.8, 2.25, 6.5, 1.4, "topology",
    ["persistence  (𝔽₂ PH · bottleneck)   [topology]",
     "dec  [dec]     ·     direct (state-space PH)  [topology]"], "#ffffff", SUB_E)
box(8.8, 0.95, 6.5, 1.05, "optional extras (imported lazily)",
    ["[geom] geomstats · [topology] ripser+persim · [dec] dxtr · [report] reportlab"],
    SUP_F, SUP_E, optional=True, sub_size=8.0)

# Trajectory → analyses, and shared geometry → both
arrow(3.9, 7.7, 3.9, 5.8, color=PB_E)
arrow(12.1, 7.7, 12.1, 5.8, color=SUB_E)
arrow(6.3, 6.05, 4.2, 5.42, color=SHARE_E, ls="--")
arrow(9.7, 6.05, 11.8, 5.42, color=SHARE_E, ls="--")

# ---- legend --------------------------------------------------------------- #
lx, ly = 0.4, 0.12
items = [("Trajectory", SPINE_F, SPINE_E), ("sources", SRC_F, SRC_E),
         ("pullback / dynamics", PB_F, PB_E), ("subspace / topology", SUB_F, SUB_E),
         ("shared geometry", SHARE_F, SHARE_E)]
for i, (lab, fc, ec) in enumerate(items):
    ax.add_patch(FancyBboxPatch((lx + i * 3.05, ly), 0.34, 0.24,
                 boxstyle="round,pad=0.01", fc=fc, ec=ec, lw=1.5))
    ax.text(lx + i * 3.05 + 0.45, ly + 0.12, lab, va="center", fontsize=8.6,
            color=INK)
ax.text(8, 10.5, "dotted boxes = optional extras / external inputs", ha="center",
        va="center", fontsize=9, style="italic", color="#94a3b8")

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs"
fig.savefig(out / "architecture.svg", bbox_inches="tight")
fig.savefig(out / "architecture.png", dpi=150, bbox_inches="tight")
print("wrote", out / "architecture.svg", "and .png")
