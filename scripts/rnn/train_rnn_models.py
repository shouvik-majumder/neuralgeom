"""
Train RNNs on the cognitive tasks and cache the checkpoints.
============================================================

Kept separate from the analysis demo so training happens once and the
geometry can be re-run instantly.

    python train_rnn_models.py                 # all tasks (skips cached)
    python train_rnn_models.py context_decision --steps 4000
    python train_rnn_models.py --model gru     # swap the architecture
    python train_rnn_models.py --force         # retrain from scratch

Checkpoints land in outputs/checkpoints/<task>_<model>.pt and store the
model state plus the exact task/model construction arguments, so the demo
can rebuild everything reproducibly.
"""
from __future__ import annotations

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuralgeom.paths import CKPT_DIR  # noqa: E402

import argparse
import sys
from pathlib import Path

import torch


from neuralgeom.tasks import evaluate, make_model, make_task, train  # noqa: E402

CKPT = CKPT_DIR

# task -> (task kwargs, model kwargs, training steps)
CONFIG = {
    "perceptual_decision": (dict(dt=25, sigma=0.15), dict(hidden_size=64,
                            tau=100, noise=0.05, g=1.0), 1200),
    "evidence_integration": (dict(dt=25, sigma=0.05), dict(hidden_size=64,
                             tau=200, noise=0.05, g=0.9), 2000),
    "context_decision": (dict(dt=25, sigma=0.15), dict(hidden_size=96,
                         tau=150, noise=0.05, g=1.0), 3000),
    "delay_match_to_sample": (dict(dt=25, sigma=0.1), dict(hidden_size=96,
                              tau=200, noise=0.05, g=1.0), 2500),
}


def ckpt_path(task_name: str, model_name: str) -> Path:
    return CKPT / f"{task_name}_{model_name}.pt"


def build(task_name: str, model_name: str = "vanilla", seed: int = 0):
    """Reconstruct (task, model) from the config — used by the demo too."""
    tkw, mkw, steps = CONFIG[task_name]
    task = make_task(task_name, seed=seed, **tkw)
    torch.manual_seed(seed)
    model = make_model(model_name, task.spec, dt=task.dt, **mkw)
    return task, model, steps


def load_or_train(task_name: str, model_name: str = "vanilla", seed: int = 0,
                  steps: int | None = None, force: bool = False,
                  resume: bool = False, verbose: bool = True):
    """Return (task, model, history).

    * cached checkpoint present and ``resume``/``force`` false -> just load.
    * ``resume=True``  -> load if present, train ``steps`` MORE steps, save.
      (Useful when a single training run would exceed a time limit: call
      repeatedly until ``history['total_steps']`` reaches the target.)
    * ``force=True``   -> retrain from scratch.
    """
    path = ckpt_path(task_name, model_name)
    task, model, default_steps = build(task_name, model_name, seed)
    steps = steps or default_steps
    prev_hist, done = {}, 0

    if path.exists() and not force:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        prev_hist = blob.get("history", {})
        done = int(blob.get("total_steps", steps))
        if not resume:
            if verbose:
                print(f"  loaded {path.name}  ({done} steps, "
                      f"acc {blob['final_acc']:.3f})")
            return task, model, prev_hist

    if verbose:
        print(f"  training {task_name}/{model_name}: {steps} steps "
              f"(already done: {done}) ...")
    hist = train(model, task, steps=steps, batch_size=64, lr=2e-3,
                 l2_rate=1e-4, log_every=max(1, steps // 4), verbose=verbose)
    # splice histories so learning curves stay continuous across chunks
    merged = {k: list(prev_hist.get(k, [])) for k in ("step", "loss", "acc")}
    merged["step"] += [s + done for s in hist["step"]]
    merged["loss"] += hist["loss"]
    merged["acc"] += hist["acc"]

    acc = float(evaluate(model, task, batch_size=512))
    torch.save({"state_dict": model.state_dict(), "history": merged,
                "final_acc": acc, "total_steps": done + steps,
                "task": task_name, "model": model_name, "seed": seed}, path)
    if verbose:
        print(f"  saved {path.name}  ({done + steps} steps, acc {acc:.3f})")
    return task, model, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=list(CONFIG),
                    help="task names (default: all)")
    ap.add_argument("--model", default="vanilla",
                    help="vanilla | gru | lstm")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="train --steps MORE steps on top of the checkpoint")
    args = ap.parse_args()

    for name in (args.tasks or list(CONFIG)):
        print(f"\n=== {name} ===")
        _, _, _ = load_or_train(name, args.model, args.seed, args.steps,
                                args.force, args.resume)


if __name__ == "__main__":
    main()
