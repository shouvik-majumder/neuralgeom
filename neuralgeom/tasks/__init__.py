"""
neuralgeom.tasks — cognitive-task generation and RNN training.
==============================================================

Three independent layers:

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

The cue-triggered timing task is maintained in a separate repository
(``timingtask``). It exports a ``Trajectory`` HDF5 file that
``neuralgeom.data.load_trajectory`` reads; the two packages share no code.
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
