"""
Shared infrastructure for the per-task RNN experiments.
=======================================================

Each experiment script (exp_<task>.py) follows the same four-part structure:

    1. TRAINING      what the network was asked to learn, and how it learned
    2. BEHAVIOUR     what it does, measured the way a psychophysicist would
    3. ACTIVITY      what the units and the population do (PSTHs, PCA)
    4. GEOMETRY      one geometric measurement that EXPLAINS part 2 and 3

This module holds the helpers they share: population analysis helpers,
plotting conventions, psychophysical reverse correlation, and a small
report builder so each experiment can emit a self-contained PDF.
"""
from __future__ import annotations

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from neuralgeom.paths import fig_dir  # noqa: E402
from torch import Tensor

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
ROOT = EXAMPLES.parent

torch.set_grad_enabled(False)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "axes.grid": True,
    "grid.alpha": 0.22, "legend.fontsize": 7.5,
})

CHOICE_COLORS = {1: "#c0392b", 2: "#2471a3"}


def figdir(task: str) -> Path:
    # all RNN-experiment figures live flat in outputs/figures/rnn; the task
    # identity is carried by the filename prefix (cd_, dm_, ei_, pd_)
    return fig_dir(f"exp_{task}")


def savefig(fig, path: Path, name: str) -> Path:
    p = path / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")
    return p


def banner(step: str, text: str) -> None:
    print(f"\n=== {step} ===\n{text}")


# --------------------------------------------------------------------------- #
# Population analysis
# --------------------------------------------------------------------------- #
def pca(H: Tensor, k: int = 6):
    """PCA of a state cloud. H: (N, hidden) -> (mean, components (hidden,k),
    explained-variance-ratio (k,))."""
    mu = H.mean(0, keepdim=True)
    Hc = H - mu
    U, S, Vh = torch.linalg.svd(Hc, full_matrices=False)
    var = S.square()
    return mu, Vh[:k].transpose(0, 1), (var / var.sum())[:k]


def project(H: Tensor, mu: Tensor, comps: Tensor) -> Tensor:
    """(B, T, hidden) or (N, hidden) -> same leading shape, k trailing dims."""
    return (H - mu) @ comps


def psth(H: Tensor, groups: Tensor, unit: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Condition-averaged activity of one unit. Returns (labels, mean, sem)."""
    labels = torch.unique(groups)
    m = torch.stack([H[groups == g, :, unit].mean(0) for g in labels])
    s = torch.stack([H[groups == g, :, unit].std(0)
                     / max(1.0, float((groups == g).sum()) ** 0.5)
                     for g in labels])
    return labels, m, s


def selectivity(H: Tensor, groups: Tensor, window: slice) -> Tensor:
    """|d'|-style selectivity of every unit for a binary grouping, in a window."""
    g = torch.unique(groups)
    assert len(g) == 2, "selectivity expects a binary grouping"
    a = H[groups == g[0]][:, window].mean(1)
    b = H[groups == g[1]][:, window].mean(1)
    num = (a.mean(0) - b.mean(0)).abs()
    den = (0.5 * (a.var(0) + b.var(0))).clamp_min(1e-12).sqrt()
    return num / den


def decode_axis(H: Tensor, labels: Tensor) -> Tensor:
    """Unit-norm 'choice axis': difference of class means (a 1-dim readout).

    H : (N, hidden) states, labels : (N,) binary. This is the population
    vector that best separates the two conditions in the mean sense — the
    standard 'decision axis' of the systems-neuroscience literature.
    """
    u = torch.unique(labels)
    v = H[labels == u[0]].mean(0) - H[labels == u[1]].mean(0)
    return v / v.norm().clamp_min(1e-12)


def reverse_correlation(stim: Tensor, choice: Tensor, valid: Optional[Tensor] = None
                        ) -> Tuple[Tensor, Tensor]:
    """Psychophysical kernel: influence of the stimulus at each time step on
    the eventual choice.

    stim   : (B, T) signed momentary evidence (positive favours choice 1)
    choice : (B,) in {1, 2}
    valid  : (B, T) bool mask of time steps where the stimulus was on

    Returns (kernel, sem): the difference in mean momentary evidence between
    choice-1 and choice-2 trials at each time step. A flat kernel means the
    network weights all moments equally (perfect integration); a rising
    kernel means recency (leak); a falling kernel means primacy.
    """
    s = stim.clone()
    if valid is not None:
        s = torch.where(valid, s, torch.full_like(s, float("nan")))
    c1, c2 = choice == 1, choice == 2

    def nanmean_sem(x):
        m = torch.nanmean(x, dim=0)
        n = (~torch.isnan(x)).sum(0).clamp_min(1)
        v = torch.nanmean((x - m) ** 2, dim=0)
        return m, (v / n).sqrt()

    m1, e1 = nanmean_sem(s[c1])
    m2, e2 = nanmean_sem(s[c2])
    return m1 - m2, (e1 ** 2 + e2 ** 2).sqrt()


def leak_prediction(n_steps: int, lam: float) -> np.ndarray:
    """Kernel predicted by a leaky integrator with per-step decay ``lam``.

    Evidence arriving at step t survives to the end multiplied by
    lam^(T-1-t), so the influence profile is an exponential rising toward
    the end of the trial. Normalized to unit mean for shape comparison.
    """
    t = np.arange(n_steps)
    w = lam ** (n_steps - 1 - t)
    return w / w.mean()


def logistic_weights(X: np.ndarray, y: np.ndarray, iters: int = 400,
                     lr: float = 0.5) -> np.ndarray:
    """Tiny logistic regression (no sklearn dependency). Returns [w..., bias].

    Used to obtain BEHAVIOURAL weights on task variables, which the geometry
    is then asked to predict.
    """
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        grad = Xb.T @ (y - p) / len(y)
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb / len(y)
        w = w + np.linalg.solve(H + 1e-6 * np.eye(len(w)), grad)
    return w


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def plot_state_traj(ax, Z: np.ndarray, color_vals: np.ndarray, cmap="coolwarm",
                    every: int = 1, lw=0.6, alpha=0.5, mark_start=True,
                    mark_end=True, tslice=None):
    """Trajectories in a 2-D projection, coloured by a per-trial value."""
    norm = plt.Normalize(np.nanpercentile(color_vals, 5),
                         np.nanpercentile(color_vals, 95))
    sl = tslice or slice(None)
    for i in range(0, Z.shape[0], every):
        c = plt.get_cmap(cmap)(norm(color_vals[i]))
        ax.plot(Z[i, sl, 0], Z[i, sl, 1], color=c, lw=lw, alpha=alpha)
    if mark_start:
        ax.scatter(Z[::every, sl, 0][:, 0], Z[::every, sl, 1][:, 0],
                   s=6, c="k", zorder=4, marker="o", label="start")
    if mark_end:
        ax.scatter(Z[::every, sl, 0][:, -1], Z[::every, sl, 1][:, -1],
                   s=10, c="k", zorder=4, marker="s", label="end")
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def epoch_shading(ax, marks: Sequence[Tuple[float, float, str, str]]):
    """Shade task epochs: list of (start, stop, colour, label)."""
    for a, b, c, lab in marks:
        ax.axvspan(a, b, color=c, alpha=0.14, lw=0)
        ax.text((a + b) / 2, ax.get_ylim()[1], lab, ha="center", va="bottom",
                fontsize=6.5, color="0.35")


# --------------------------------------------------------------------------- #
# PDF report builder
# --------------------------------------------------------------------------- #
from PIL import Image as PILImage                                # noqa: E402
from reportlab.lib.colors import HexColor                        # noqa: E402
from reportlab.lib.enums import TA_CENTER                        # noqa: E402
from reportlab.lib.pagesizes import letter                       # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch                             # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph,     # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)


class Report:
    """Minimal document builder shared by the experiment scripts."""

    def __init__(self, out_path: Path, title: str, subtitle: str,
                 accent: str = "#0f766e"):
        self.out = Path(out_path)
        self.accent = HexColor(accent)
        self.grey = HexColor("#555555")
        ss = getSampleStyleSheet()
        self.s_title = ParagraphStyle("t", parent=ss["Title"], fontSize=19,
                                      textColor=self.accent, spaceAfter=3)
        self.s_sub = ParagraphStyle("st", parent=ss["Normal"], fontSize=10.5,
                                    textColor=self.grey, alignment=TA_CENTER,
                                    spaceAfter=15)
        self.s_h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13.5,
                                   textColor=self.accent, spaceBefore=14,
                                   spaceAfter=5)
        self.s_h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11,
                                   spaceBefore=9, spaceAfter=3)
        self.s_body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.7,
                                     leading=13.4, spaceAfter=6.5)
        self.s_cap = ParagraphStyle("c", parent=ss["Normal"], fontSize=8.5,
                                    leading=11.3, textColor=self.grey,
                                    spaceBefore=2, spaceAfter=11,
                                    leftIndent=12, rightIndent=12)
        self.s_box = ParagraphStyle("bx", parent=self.s_body,
                                    backColor=HexColor("#eaf3f2"),
                                    borderPadding=6, borderColor=self.accent,
                                    borderWidth=0.7, spaceBefore=5,
                                    spaceAfter=9)
        self.s_cell = ParagraphStyle("cl", parent=self.s_body, fontSize=8.8,
                                     leading=11.8, spaceAfter=0)
        self.s_mono = ParagraphStyle("m", parent=self.s_body,
                                     fontName="Courier", fontSize=8.2,
                                     leading=11.2,
                                     backColor=HexColor("#f2f5f5"),
                                     borderPadding=5, spaceAfter=8)
        self.story = [Paragraph(title, self.s_title),
                      Paragraph(subtitle, self.s_sub)]

    # -- content ---------------------------------------------------------- #
    def h1(self, t): self.story.append(Paragraph(t, self.s_h1)); return self
    def h2(self, t): self.story.append(Paragraph(t, self.s_h2)); return self
    def p(self, t): self.story.append(Paragraph(t, self.s_body)); return self
    def box(self, t): self.story.append(Paragraph(t, self.s_box)); return self
    def mono(self, t): self.story.append(Paragraph(t, self.s_mono)); return self
    def pagebreak(self): self.story.append(PageBreak()); return self
    def space(self, h=8): self.story.append(Spacer(1, h)); return self

    def figure(self, path: Path, caption: str = "", width: float = 6.4):
        w, h = PILImage.open(path).size
        self.story.append(Image(str(path), width=width * inch,
                                height=width * inch * h / w))
        if caption:
            self.story.append(Paragraph(caption, self.s_cap))
        return self

    def table(self, header: Sequence[str], rows: Sequence[Sequence],
              widths: Optional[Sequence[float]] = None, fontsize=8.8):
        st = ParagraphStyle("tc", parent=self.s_cell, fontSize=fontsize)
        data = [[Paragraph(f"<b>{c}</b>", st) for c in header]] + \
               [[Paragraph(str(c), st) for c in r] for r in rows]
        n = len(header)
        widths = widths or [6.4 / n] * n
        t = Table(data, colWidths=[w * inch for w in widths])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#d5e8e5")),
            ("GRID", (0, 0), (-1, -1), 0.45, self.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        self.story.append(t)
        self.space(7)
        return self

    def deflist(self, rows: Sequence[Tuple[str, str]], w0: float = 1.5):
        st_term = ParagraphStyle("dt", parent=self.s_cell,
                                 textColor=self.accent)
        data = [[Paragraph(f"<b>{a}</b>", st_term),
                 Paragraph(b, self.s_cell)] for a, b in rows]
        t = Table(data, colWidths=[w0 * inch, (6.4 - w0) * inch])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#9fc7c2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [HexColor("#ffffff"), HexColor("#f3f8f7")]),
        ]))
        self.story.append(t)
        self.space(7)
        return self

    def build(self):
        doc = SimpleDocTemplate(
            str(self.out), pagesize=letter, leftMargin=0.85 * inch,
            rightMargin=0.85 * inch, topMargin=0.75 * inch,
            bottomMargin=0.75 * inch, title=self.out.stem)
        doc.build(self.story)
        print(f"\nwrote {self.out}")
        return self.out
