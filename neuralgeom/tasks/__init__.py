"""
neuralgeom.tasks — self-sufficient synthetic task generation and RNN training.
=========================================================================

Three independent layers; swap any one without touching the others.

    cognitive.py  what to solve  — TrialBatch API and four cognitive tasks
                  (perceptual decision, evidence integration, context-
                  dependent decision, delayed match-to-sample)
    models.py     what solves it — VanillaRNN / GRU / LSTM behind one
                  interface whose ``step(x, h)`` is a pure function, which is
                  what lets torch.func differentiate the dynamics
    training.py   how it learns  — task- and model-agnostic trainer

Adding a task: subclass ``Task``, set ``self.spec``, implement
``sample(batch_size) -> TrialBatch``, register in ``TASKS``.
Adding a model: subclass ``_BaseRNN`` and implement ``step(x, h)``.

The cue-triggered lick-timing task is NOT here. It lives in the separate
``timingtask`` repository, along with its agents and its RL trainer, and hands
its results back as a ``Trajectory`` HDF5 file that
``neuralgeom.data.load_trajectory`` reads. It carries its own copies of
``models.py`` and ``training.py`` so that it stands alone; those copies are
allowed to diverge from these, and that is the point of the split.
"""
from .cognitive import (ContextDecision, DelayMatchToSample,
                        EvidenceIntegration, PerceptualDecision, TASKS, Task,
                        TaskSpec, TrialBatch, make_task)
from .models import GRUModel, LSTMModel, MODELS, VanillaRNN, make_model
from .training import evaluate, masked_loss, run_trials, train

__all__ = ["Task", "TaskSpec", "TrialBatch", "TASKS", "make_task",
           "PerceptualDecision", "EvidenceIntegration", "ContextDecision",
           "DelayMatchToSample",
           "VanillaRNN", "GRUModel", "LSTMModel", "MODELS", "make_model",
           "train", "evaluate", "run_trials", "masked_loss"]
