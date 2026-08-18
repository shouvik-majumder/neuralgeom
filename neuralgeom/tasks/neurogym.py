"""
rnn.neurogym_adapter — use any neurogym environment through the Task API.
=========================================================================

The native tasks in ``rnn.tasks`` are self-contained and fast (fully batched
tensors, no Python stepping). This adapter exists so the rest of the code —
models, trainer, geometry analyses — can also run on neurogym's much larger
task catalogue without changing a line.

    pip install neurogym

    from neuralgeom.tasks.neurogym import NeuroGymTask
    task = NeuroGymTask("PerceptualDecisionMaking-v0", dt=20)
    batch = task.sample(64)          # same TrialBatch as the native tasks
    model = make_model("vanilla", task.spec, hidden_size=128, dt=task.dt)
    train(model, task, steps=2000)

Notes
-----
* neurogym generates trials one at a time in Python, so sampling is slower
  than the native tasks; trials are padded to the batch's longest trial and
  the padding is excluded via ``loss_mask``.
* neurogym's ground truth is a per-timestep integer action, which maps
  directly onto our cross-entropy convention (0 = fixate, 1.. = choices).
* Some envs use continuous targets; pass ``loss="mse"`` for those.
"""
from __future__ import annotations

from typing import Optional

import torch

from .cognitive import Task, TaskSpec, TrialBatch

__all__ = ["NeuroGymTask", "list_neurogym_tasks"]


def list_neurogym_tasks():
    """All registered neurogym environment ids (requires neurogym)."""
    import neurogym as ngym
    return sorted(ngym.all_tasks().keys()) if hasattr(ngym, "all_tasks") \
        else sorted(ngym.envs.ALL_ENVS.keys())


class NeuroGymTask(Task):
    """Wrap a neurogym environment so it behaves like a native ``Task``.

    Parameters
    ----------
    env_id : e.g. "PerceptualDecisionMaking-v0", "ContextDecisionMaking-v0",
        "DelayMatchSample-v0", "PulseDecisionMaking-v0".
    dt : timestep in ms passed to the environment.
    env_kwargs : forwarded to ``gym.make`` / ``ngym.make``.
    loss : "cross_entropy" (discrete actions) or "mse".
    """

    def __init__(self, env_id: str, dt: float = 20.0, seed: Optional[int] = None,
                 env_kwargs: Optional[dict] = None, loss: str = "cross_entropy"):
        super().__init__(dt=dt, sigma=0.0, seed=seed)
        try:
            import neurogym as ngym  # noqa: F401
            import gymnasium as gym
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "neurogym (and gymnasium) are required for NeuroGymTask: "
                "pip install neurogym"
            ) from e

        kwargs = dict(dt=dt, **(env_kwargs or {}))
        self.env = gym.make(env_id, **kwargs)
        if seed is not None:
            self.env.reset(seed=int(seed))
        else:
            self.env.reset()
        unwrapped = self.env.unwrapped

        obs_space = self.env.observation_space
        act_space = self.env.action_space
        in_dim = int(obs_space.shape[0])
        out_dim = int(getattr(act_space, "n", 0)) or int(act_space.shape[0])

        self.env_id = env_id
        self.spec = TaskSpec(
            name=f"neurogym:{env_id}", input_dim=in_dim, output_dim=out_dim,
            loss=loss,
            input_labels=tuple(getattr(unwrapped, "observation_space", None)
                               .name.keys())
            if hasattr(getattr(unwrapped, "observation_space", None), "name")
            else (),
            description=f"neurogym environment {env_id}",
        )

    def sample(self, batch_size: int) -> TrialBatch:
        unwrapped = self.env.unwrapped
        obs_list, gt_list = [], []
        for _ in range(batch_size):
            unwrapped.new_trial()
            obs_list.append(torch.as_tensor(unwrapped.ob, dtype=torch.get_default_dtype()))
            gt_list.append(torch.as_tensor(unwrapped.gt))
        T = max(o.shape[0] for o in obs_list)

        B, n_in = batch_size, self.spec.input_dim
        x = torch.zeros(B, T, n_in)
        y = torch.zeros(B, T, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.bool)
        for i, (o, g) in enumerate(zip(obs_list, gt_list)):
            t = o.shape[0]
            x[i, :t] = o
            y[i, :t] = g.long()
            mask[i, :t] = True

        return TrialBatch(x, y, mask, {"trial_len":
                                       torch.tensor([o.shape[0] for o in obs_list])})
