"""
neuralgeom.tasks.cognitive — temporally demanding cognitive tasks, as batched tensors.
======================================================================================

All tasks share one small interface, so tasks, models and analyses stay
independent:

    task = EvidenceIntegration(dt=20, T=1000)
    batch = task.sample(64)            # -> TrialBatch
    batch.inputs                       # (B, T, input_dim)
    batch.targets                      # (B, T) int64  (or float for MSE tasks)
    batch.loss_mask                    # (B, T) bool — where the loss applies
    batch.meta                         # dict of per-trial condition variables

To use a different task, swap the object.

Design notes
------------
* Time is discretized with a step ``dt`` (ms). Trials in a batch share a
  common tensor length T_steps; per-trial epoch boundaries vary and are
  handled through ``loss_mask``, so variable timing is supported without
  ragged tensors.
* Inputs follow the standard neurophysiology convention: a fixation channel
  that goes low at the response epoch, plus stimulus channels.
* Decision tasks are scored only during the decision epoch; the target
  labels are 0 = fixate/withhold, 1..K = choices, so a single
  cross-entropy over K+1 classes covers the whole trial.
* Every task exposes ``spec`` (dims, loss type, epoch layout) so models and
  trainers can be built generically.

Tasks implemented
-----------------
PerceptualDecision     coherence-based 2AFC (random-dot motion analogue)
EvidenceIntegration    discrete pulse accumulation, variable duration
ContextDecision        Mante/Sussillo context-dependent integration
DelayMatchToSample     working memory: sample, delay, test, match?
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = [
    "TaskSpec",
    "TrialBatch",
    "Task",
    "PerceptualDecision",
    "EvidenceIntegration",
    "ContextDecision",
    "DelayMatchToSample",
    "TASKS",
    "make_task",
]


# --------------------------------------------------------------------------- #
@dataclass
class TaskSpec:
    """Static description of a task — everything a model/trainer needs."""
    name: str
    input_dim: int
    output_dim: int
    loss: str = "cross_entropy"           # or "mse"
    input_labels: Sequence[str] = ()
    output_labels: Sequence[str] = ()
    description: str = ""


@dataclass
class TrialBatch:
    """One batch of trials."""
    inputs: Tensor                        # (B, T, input_dim)
    targets: Tensor                       # (B, T) long | (B, T, out) float
    loss_mask: Tensor                     # (B, T) bool
    meta: Dict[str, Tensor] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.inputs.shape[0]

    @property
    def n_steps(self) -> int:
        return self.inputs.shape[1]

    def to(self, device) -> "TrialBatch":
        return TrialBatch(
            self.inputs.to(device), self.targets.to(device),
            self.loss_mask.to(device),
            {k: v.to(device) for k, v in self.meta.items()},
        )


class Task:
    """Base class. Subclasses implement ``sample`` and set ``spec``."""

    spec: TaskSpec

    def __init__(self, dt: float = 20.0, sigma: float = 0.15,
                 seed: Optional[int] = None):
        self.dt = float(dt)
        self.sigma = float(sigma)          # input noise SD (scaled by dt below)
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))
        else:
            self._gen.seed()

    # -- helpers ---------------------------------------------------------- #
    def _steps(self, ms: float) -> int:
        return max(1, int(round(ms / self.dt)))

    def _randn(self, *shape) -> Tensor:
        return torch.randn(*shape, generator=self._gen)

    def _rand(self, *shape) -> Tensor:
        return torch.rand(*shape, generator=self._gen)

    def _randint(self, high: int, shape) -> Tensor:
        return torch.randint(high, shape, generator=self._gen)

    def _noise(self, shape) -> Tensor:
        """Input noise scaled so its effect is dt-independent."""
        return self.sigma * math.sqrt(2.0 * 100.0 / self.dt) * self._randn(*shape)

    def sample(self, batch_size: int) -> TrialBatch:
        raise NotImplementedError

    def accuracy(self, outputs: Tensor, batch: TrialBatch) -> Tensor:
        """Fraction of trials whose majority decision-epoch output is correct.

        outputs : (B, T, out) logits. Uses the same mask the loss uses.
        """
        if self.spec.loss != "cross_entropy":
            raise NotImplementedError("accuracy defined for classification tasks")
        pred = outputs.argmax(-1)                              # (B, T)
        m = batch.loss_mask
        correct = ((pred == batch.targets) & m).sum(1).double()
        return correct / m.sum(1).clamp_min(1).double()

    def __repr__(self) -> str:
        return f"{self.spec.name}(dt={self.dt}, sigma={self.sigma})"


# --------------------------------------------------------------------------- #
class PerceptualDecision(Task):
    """Coherence-based two-alternative forced choice (random-dot analogue).

    Two stimulus channels receive constant drive 0.5 &plusmn; coh/2 plus noise
    during the stimulus epoch; the network must report which channel was
    stronger, during the response epoch (after fixation drops).

    Inputs  : [fixation, stim_A, stim_B]
    Outputs : [fixate, choose_A, choose_B]
    """

    def __init__(self, dt=20.0, sigma=0.15, seed=None,
                 t_fix=(200, 400), t_stim=800, t_delay=0, t_dec=300,
                 coherences=(-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5)):
        super().__init__(dt, sigma, seed)
        self.t_fix, self.t_stim = t_fix, t_stim
        self.t_delay, self.t_dec = t_delay, t_dec
        self.coherences = tuple(coherences)
        self.spec = TaskSpec(
            name="perceptual_decision", input_dim=3, output_dim=3,
            input_labels=("fixation", "stim_A", "stim_B"),
            output_labels=("fixate", "choose_A", "choose_B"),
            description="Coherence 2AFC; report the stronger of two channels.",
        )

    def sample(self, batch_size: int) -> TrialBatch:
        B = batch_size
        n_fix_max = self._steps(self.t_fix[1])
        n_stim = self._steps(self.t_stim)
        n_del = self._steps(self.t_delay) if self.t_delay else 0
        n_dec = self._steps(self.t_dec)
        T = n_fix_max + n_stim + n_del + n_dec

        # variable fixation duration per trial
        lo, hi = self._steps(self.t_fix[0]), n_fix_max
        n_fix = torch.randint(lo, hi + 1, (B,), generator=self._gen)

        coh = torch.tensor(self.coherences)[self._randint(len(self.coherences), (B,))]
        # coh > 0 means channel A carries the stronger drive -> choose_A (=1)
        choice = torch.where(coh > 0, 1, 2)

        x = torch.zeros(B, T, 3)
        y = torch.zeros(B, T, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.bool)
        tgrid = torch.arange(T).unsqueeze(0)               # (1, T)

        stim_on = n_fix.unsqueeze(1)
        stim_off = stim_on + n_stim
        dec_on = stim_off + n_del
        dec_off = dec_on + n_dec

        fix_period = tgrid < dec_on
        stim_period = (tgrid >= stim_on) & (tgrid < stim_off)
        dec_period = (tgrid >= dec_on) & (tgrid < dec_off)

        x[:, :, 0] = fix_period.to(x.dtype)
        drive_a = (0.5 + coh / 2).unsqueeze(1) * stim_period
        drive_b = (0.5 - coh / 2).unsqueeze(1) * stim_period
        x[:, :, 1] = drive_a
        x[:, :, 2] = drive_b
        x[:, :, 1:] += self._noise((B, T, 2)) * stim_period.unsqueeze(-1)

        y[fix_period] = 0
        y[dec_period] = choice.unsqueeze(1).expand(B, T)[dec_period]
        mask |= fix_period | dec_period                    # score whole trial

        return TrialBatch(x, y, mask, {
            "coherence": coh, "choice": choice,
            "stim_on": stim_on.squeeze(1), "dec_on": dec_on.squeeze(1),
        })


# --------------------------------------------------------------------------- #
class EvidenceIntegration(Task):
    """Discrete pulse accumulation with variable stimulus duration.

    On each stimulus step, each of two channels independently emits a pulse
    with probability p_A / p_B. The network must report which side emitted
    more pulses over the whole (variable-length) stimulus period — this
    requires genuine integration, not a leaky snapshot.

    Inputs  : [fixation, pulse_A, pulse_B]
    Outputs : [fixate, choose_A, choose_B]
    """

    def __init__(self, dt=20.0, sigma=0.05, seed=None,
                 t_fix=200, t_stim=(400, 1200), t_dec=300,
                 rates=(0.15, 0.35), base_rate=0.5):
        super().__init__(dt, sigma, seed)
        self.t_fix, self.t_stim, self.t_dec = t_fix, t_stim, t_dec
        self.rates, self.base_rate = rates, base_rate
        self.spec = TaskSpec(
            name="evidence_integration", input_dim=3, output_dim=3,
            input_labels=("fixation", "pulse_A", "pulse_B"),
            output_labels=("fixate", "choose_A", "choose_B"),
            description="Count pulses on two channels over a variable window; "
                        "report the larger count (needs a line attractor).",
        )

    def sample(self, batch_size: int) -> TrialBatch:
        B = batch_size
        n_fix = self._steps(self.t_fix)
        n_stim_max = self._steps(self.t_stim[1])
        n_stim_min = self._steps(self.t_stim[0])
        n_dec = self._steps(self.t_dec)
        T = n_fix + n_stim_max + n_dec

        n_stim = torch.randint(n_stim_min, n_stim_max + 1, (B,),
                               generator=self._gen)
        # per-trial pulse rates: one side favored
        strength = torch.tensor(self.rates)[self._randint(len(self.rates), (B,))]
        side = self._randint(2, (B,))                       # 0 -> A favored
        signed = torch.where(side == 0, strength, -strength)
        p_a = self.base_rate + signed / 2
        p_b = self.base_rate - signed / 2

        x = torch.zeros(B, T, 3)
        y = torch.zeros(B, T, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.bool)
        tgrid = torch.arange(T).unsqueeze(0)

        stim_on = torch.full((B, 1), n_fix)
        stim_off = stim_on + n_stim.unsqueeze(1)
        dec_on = stim_off
        dec_off = dec_on + n_dec

        stim_period = (tgrid >= stim_on) & (tgrid < stim_off)
        dec_period = (tgrid >= dec_on) & (tgrid < dec_off)
        fix_period = tgrid < dec_on

        pulses_a = (self._rand(B, T) < p_a.unsqueeze(1)) & stim_period
        pulses_b = (self._rand(B, T) < p_b.unsqueeze(1)) & stim_period
        x[:, :, 0] = fix_period.float()
        x[:, :, 1] = pulses_a.float()
        x[:, :, 2] = pulses_b.float()
        x[:, :, 1:] += self._noise((B, T, 2)) * stim_period.unsqueeze(-1)

        # ground truth is the ACTUAL realized pulse difference (not the rate)
        n_a, n_b = pulses_a.sum(1), pulses_b.sum(1)
        choice = torch.where(n_a >= n_b, 1, 2)

        y[dec_period] = choice.unsqueeze(1).expand(B, T)[dec_period]
        mask |= fix_period | dec_period

        return TrialBatch(x, y, mask, {
            "n_pulses_A": n_a, "n_pulses_B": n_b,
            "evidence": (n_a - n_b).float(), "choice": choice,
            "stim_on": stim_on.squeeze(1), "dec_on": dec_on.squeeze(1),
            "n_stim": n_stim,
        })


# --------------------------------------------------------------------------- #
class ContextDecision(Task):
    """Context-dependent integration (Mante, Sussillo et al. 2013).

    Two independent noisy features ("motion" and "colour") are presented
    simultaneously; two context cues indicate which feature is relevant on
    this trial. The network must integrate the relevant feature and IGNORE
    the irrelevant one — the canonical test of flexible, context-dependent
    computation, and the reason this task has such interesting geometry:
    the same stimulus must be routed into different decision subspaces
    depending on context.

    Inputs  : [fixation, motion_A, motion_B, colour_A, colour_B,
               ctx_motion, ctx_colour]
    Outputs : [fixate, choose_A, choose_B]
    """

    def __init__(self, dt=20.0, sigma=0.15, seed=None,
                 t_fix=200, t_stim=800, t_dec=300,
                 coherences=(-0.5, -0.15, 0.15, 0.5)):
        super().__init__(dt, sigma, seed)
        self.t_fix, self.t_stim, self.t_dec = t_fix, t_stim, t_dec
        self.coherences = tuple(coherences)
        self.spec = TaskSpec(
            name="context_decision", input_dim=7, output_dim=3,
            input_labels=("fixation", "motion_A", "motion_B", "colour_A",
                          "colour_B", "ctx_motion", "ctx_colour"),
            output_labels=("fixate", "choose_A", "choose_B"),
            description="Integrate the cued feature, ignore the other "
                        "(Mante-Sussillo context-dependent decision).",
        )

    def sample(self, batch_size: int) -> TrialBatch:
        B = batch_size
        n_fix, n_stim = self._steps(self.t_fix), self._steps(self.t_stim)
        n_dec = self._steps(self.t_dec)
        T = n_fix + n_stim + n_dec

        cohs = torch.tensor(self.coherences)
        coh_m = cohs[self._randint(len(cohs), (B,))]
        coh_c = cohs[self._randint(len(cohs), (B,))]
        context = self._randint(2, (B,))                    # 0 = motion
        relevant = torch.where(context == 0, coh_m, coh_c)
        # positive coherence -> the "_A" channel is stronger -> choose_A (=1)
        choice = torch.where(relevant > 0, 1, 2)

        x = torch.zeros(B, T, 7)
        y = torch.zeros(B, T, dtype=torch.long)
        tgrid = torch.arange(T).unsqueeze(0)
        stim_period = (tgrid >= n_fix) & (tgrid < n_fix + n_stim)
        dec_period = tgrid >= n_fix + n_stim
        fix_period = ~dec_period

        x[:, :, 0] = fix_period.float()
        x[:, :, 1] = (0.5 + coh_m / 2).unsqueeze(1) * stim_period
        x[:, :, 2] = (0.5 - coh_m / 2).unsqueeze(1) * stim_period
        x[:, :, 3] = (0.5 + coh_c / 2).unsqueeze(1) * stim_period
        x[:, :, 4] = (0.5 - coh_c / 2).unsqueeze(1) * stim_period
        x[:, :, 1:5] += self._noise((B, T, 4)) * stim_period.unsqueeze(-1)
        # context cue is on for the whole trial (before the response)
        x[:, :, 5] = ((context == 0).unsqueeze(1) & fix_period).float()
        x[:, :, 6] = ((context == 1).unsqueeze(1) & fix_period).float()

        y[dec_period.expand(B, T)] = choice.unsqueeze(1).expand(B, T)[
            dec_period.expand(B, T)]
        mask = torch.ones(B, T, dtype=torch.bool)

        return TrialBatch(x, y, mask, {
            "coh_motion": coh_m, "coh_colour": coh_c, "context": context,
            "relevant_coh": relevant, "choice": choice,
            "stim_on": torch.full((B,), n_fix),
            "dec_on": torch.full((B,), n_fix + n_stim),
        })


# --------------------------------------------------------------------------- #
class DelayMatchToSample(Task):
    """Delayed match-to-sample: hold a stimulus through a delay, compare.

    A sample stimulus (one of K rings) is shown, then a delay with NO input
    (pure working memory), then a test stimulus; report match / non-match.

    Inputs  : [fixation, stim_1 .. stim_K]
    Outputs : [fixate, non-match, match]
    """

    def __init__(self, dt=20.0, sigma=0.1, seed=None, n_stim=4,
                 t_fix=200, t_sample=300, t_delay=(400, 900), t_test=300,
                 t_dec=300):
        super().__init__(dt, sigma, seed)
        self.n_stim = int(n_stim)
        self.t_fix, self.t_sample = t_fix, t_sample
        self.t_delay, self.t_test, self.t_dec = t_delay, t_test, t_dec
        self.spec = TaskSpec(
            name="delay_match_to_sample", input_dim=1 + n_stim, output_dim=3,
            input_labels=("fixation",) + tuple(f"stim_{i}" for i in range(n_stim)),
            output_labels=("fixate", "non-match", "match"),
            description="Hold a sample through a blank delay, compare to a "
                        "test stimulus (working memory).",
        )

    def sample(self, batch_size: int) -> TrialBatch:
        B, K = batch_size, self.n_stim
        n_fix, n_sam = self._steps(self.t_fix), self._steps(self.t_sample)
        n_del_min, n_del_max = self._steps(self.t_delay[0]), self._steps(self.t_delay[1])
        n_test, n_dec = self._steps(self.t_test), self._steps(self.t_dec)
        T = n_fix + n_sam + n_del_max + n_test + n_dec

        n_del = torch.randint(n_del_min, n_del_max + 1, (B,), generator=self._gen)
        sample_id = self._randint(K, (B,))
        is_match = self._randint(2, (B,)).bool()
        offset = 1 + self._randint(K - 1, (B,))
        test_id = torch.where(is_match, sample_id, (sample_id + offset) % K)

        x = torch.zeros(B, T, 1 + K)
        y = torch.zeros(B, T, dtype=torch.long)
        tgrid = torch.arange(T).unsqueeze(0)

        sam_on = torch.full((B, 1), n_fix)
        sam_off = sam_on + n_sam
        test_on = sam_off + n_del.unsqueeze(1)
        test_off = test_on + n_test
        dec_on, dec_off = test_off, test_off + n_dec

        sam_period = (tgrid >= sam_on) & (tgrid < sam_off)
        test_period = (tgrid >= test_on) & (tgrid < test_off)
        dec_period = (tgrid >= dec_on) & (tgrid < dec_off)
        fix_period = tgrid < dec_on

        x[:, :, 0] = fix_period.float()
        bidx = torch.arange(B).unsqueeze(1)
        x[bidx, torch.arange(T).unsqueeze(0), 1 + sample_id.unsqueeze(1)] += \
            sam_period.float()
        x[bidx, torch.arange(T).unsqueeze(0), 1 + test_id.unsqueeze(1)] += \
            test_period.float()
        x[:, :, 1:] += self._noise((B, T, K)) * \
            (sam_period | test_period).unsqueeze(-1)

        label = is_match.long() + 1                        # 1 non-match, 2 match
        y[dec_period] = label.unsqueeze(1).expand(B, T)[dec_period]
        mask = fix_period | dec_period

        return TrialBatch(x, y, mask, {
            "sample_id": sample_id, "test_id": test_id,
            "is_match": is_match, "choice": label,
            "delay_len": n_del, "test_on": test_on.squeeze(1),
            "dec_on": dec_on.squeeze(1),
        })


# --------------------------------------------------------------------------- #
TASKS = {
    "perceptual_decision": PerceptualDecision,
    "evidence_integration": EvidenceIntegration,
    "context_decision": ContextDecision,
    "delay_match_to_sample": DelayMatchToSample,
}


def make_task(name: str, **kwargs) -> Task:
    """Factory: ``make_task('context_decision', dt=20, seed=0)``."""
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}. Available: {sorted(TASKS)}")
    return TASKS[name](**kwargs)
