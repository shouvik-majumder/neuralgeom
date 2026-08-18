"""Shared utilities for the demo scripts: data, models, training, plotting."""
from __future__ import annotations

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from neuralgeom.paths import fig_dir  # noqa: E402
import torch.nn as nn


FIGDIR = fig_dir("demos")

torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float64)
# Demos only *evaluate* geometry — no autograd graphs needed. torch.func
# Jacobians work fine under no_grad; training re-enables grad locally.
torch.set_grad_enabled(False)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def banner(step: str, text: str) -> None:
    print(f"\n=== {step} ===\n{text}")


def savefig(fig, name: str) -> None:
    path = FIGDIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {path.name}")


# --------------------------------------------------------------------------- #
# Data & models
# --------------------------------------------------------------------------- #
def two_moons(n: int = 400, noise: float = 0.07):
    """Classic two-moons dataset in R^2 (labels 0/1)."""
    m = n // 2
    t = torch.rand(m) * math.pi
    outer = torch.stack([t.cos(), t.sin()], dim=1)
    t = torch.rand(n - m) * math.pi
    inner = torch.stack([1.0 - t.cos(), 0.5 - t.sin()], dim=1)
    X = torch.cat([outer, inner]) + noise * torch.randn(n, 2)
    y = torch.cat([torch.zeros(m, dtype=torch.long),
                   torch.ones(n - m, dtype=torch.long)])
    return X, y


def make_classifier(act=nn.Tanh, hidden: int = 32) -> nn.Sequential:
    """2 -> hidden -> hidden -> 2 logits. Tanh by default (smooth Jacobians)."""
    return nn.Sequential(
        nn.Linear(2, hidden), act(),
        nn.Linear(hidden, hidden), act(),
        nn.Linear(hidden, 2),
    )


def train_classifier(model: nn.Module, X, y, steps: int = 400, lr: float = 5e-3,
                     weight_decay: float = 0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    with torch.enable_grad():
        for i in range(steps):
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
    model.eval()
    acc = (model(X).argmax(1) == y).double().mean()
    print(f"  trained {steps} steps: loss={loss.item():.4f}, train acc={acc:.3f}")
    return model


def input_grid(X, res: int = 60, pad: float = 0.4):
    """Regular grid covering the data, as (res*res, 2) tensor + meshgrid arrays."""
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    gx = torch.linspace(x_min, x_max, res)
    gy = torch.linspace(y_min, y_max, res)
    GX, GY = torch.meshgrid(gx, gy, indexing="xy")
    P = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=1)
    return P, GX.numpy(), GY.numpy()


def decision_background(ax, model, X, res: int = 200, alpha: float = 0.35):
    """Softmax-probability background + decision boundary contour."""
    P, GX, GY = input_grid(X, res)
    with torch.no_grad():
        prob = torch.softmax(model(P), dim=1)[:, 1].reshape(res, res)
    ax.contourf(GX, GY, prob.numpy(), levels=20, cmap="RdBu_r", alpha=alpha)
    ax.contour(GX, GY, prob.numpy(), levels=[0.5], colors="k", linewidths=1.2)


def scatter_data(ax, X, y, s: int = 8):
    Xn = X.numpy()
    ax.scatter(*Xn[y.numpy() == 0].T, s=s, c="#b40426", label="class 0")
    ax.scatter(*Xn[y.numpy() == 1].T, s=s, c="#3b4cc0", label="class 1")


def metric_ellipse(ax, center, g, scale=0.15, **kw):
    """Draw the image of the unit circle under J (semi-axes = singular values
    of J = sqrt(eigvals of g)), i.e. how the map locally stretches space."""
    from matplotlib.patches import Ellipse
    w, V = np.linalg.eigh(g)
    w = np.clip(w, 0.0, None)
    r = np.sqrt(w)
    angle = math.degrees(math.atan2(V[1, 1], V[0, 1]))  # leading eigvec
    e = Ellipse(center, width=2 * r[1] * scale, height=2 * r[0] * scale,
                angle=angle, **kw)
    ax.add_patch(e)
