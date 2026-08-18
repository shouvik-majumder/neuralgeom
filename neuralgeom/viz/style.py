"""
neuralgeom.viz.style — one plotting style and one figure-saving helper.
==================================================================

Previously ``examples/demo_common.py`` and ``examples/experiments/exp_common.py``
each set their own rcParams and defined their own ``savefig``; the two had
drifted apart. This is the single version.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["apply_style", "savefig", "note", "CHOICE_COLORS", "C_EARLY",
           "C_REW"]

C_EARLY, C_REW = "#c0392b", "#2471a3"
CHOICE_COLORS = {1: C_EARLY, 2: C_REW}

_STYLE = {
    "figure.dpi": 110, "savefig.dpi": 145, "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.grid": True,
    "grid.alpha": 0.2, "legend.fontsize": 7,
}


def apply_style(**overrides) -> None:
    """Apply the shared style; pass rcParams to override per script."""
    plt.rcParams.update({**_STYLE, **overrides})


def savefig(fig, directory: Path, name: str, close: bool = True) -> Path:
    """Save under ``directory`` and print the path. Returns the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    fig.savefig(p, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"  -> {p.name}")
    return p


def note(ax, text: str, right: bool = True, fontsize: float = 6.4) -> None:
    """Small annotation box in a corner of an axis."""
    ax.text(0.98 if right else 0.02, 0.97, text, transform=ax.transAxes,
            fontsize=fontsize, va="top", ha="right" if right else "left",
            bbox=dict(fc="white", alpha=0.82, lw=0.4))
