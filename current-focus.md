# Current focus — neuralgeom

> Overwrite this file, never append. It describes the present, not history.
> Lines marked [VERIFY] were inferred by Claude from the repo, not stated by me.

**This repository is now the ANALYSIS half only.** The cue-triggered lick-timing
task, its agents and everything that trains them moved out on 2026-09-14 into
`C:\dev\timingtask` (package `timingtask`), which has its own
`current-focus.md` holding the state of that science. Read that one for where
the timing work stands; this one is about geometry and topology.

**The split.** `timingtask` produces neural activity; `neuralgeom` measures its
geometry. Neither imports the other. They meet at exactly one place and it is a
file: `timingtask.export` writes the canonical `Trajectory` HDF5 schema defined
in `neuralgeom/data/trajectory.py`, and `neuralgeom.data.load_trajectory` reads
it. `timingtask/tests/test_standalone.py` fails if an import ever creeps back
in — which matters because both packages are installed in the same conda
environment, so an accidental coupling would work perfectly on this machine and
break for anyone else.

```python
from neuralgeom.data import load_trajectory
traj = load_trajectory("runs/act_probe.h5")
traj.X                 # (n_trials, T, N)  cue-aligned recurrent states
traj.condition         # (n_trials,)       the required delay on that trial
traj.aux["n_steps"]    # per-trial length; everything past it is NaN, not zero
```

From there it is an ordinary `Trajectory` and the whole toolkit applies: the
sliding-window Grassmannian embedding, the kinematics, the persistent homology,
the pullback metric of the recurrent dynamics, the fixed-point finder.

## What is here

    geometry/   pullback metric, Jacobians, SPD and Grassmannian distances,
                Riemannian fields, Fisher–Rao output metrics
    subspace/   sliding-window frames, kinematics, pooling
    topology/   persistent homology, DEC, direct state-manifold cross-check
    dynamics/   LDS estimators, recurrent-dynamics geometry, regression
    data/       loader, adapters, Trajectory  ← the contract both halves speak
    synth/      attractor, low-rank and subspace RNN generators
    tasks/      the four cognitive tasks + models + the supervised trainer
    viz/        figures, PDF reports, the rolling dashboard

## Open

1. **Nothing has yet been analysed from a trained timing agent**, because
   nothing has yet learned to time — see `timingtask/current-focus.md`. The
   read path is tested end to end (`timingtask/tests/test_export.py` round-trips
   through the real `load_trajectory`), but only on an untrained agent's states.
2. **The `neurogym` extra in `pyproject.toml` is stale.** `neuralgeom/tasks/
   neurogym.py` was deleted in `5feeb04` ("Drop neurogym, gymnasium>=1.0") but
   the optional-dependency group and the README row that advertise it were left
   behind. [VERIFY] Harmless, but it promises a module that is not there.

## Reference

- `docs/formulation.tex` — the pullback-metric mathematics
- `docs/subspace_methods.tex` — the subspace/topology constructions
- `../timingtask/` — the task, the agents, and their own docs

**Machines:** WS1 `D:\dev\neuralgeom`, WS2 `C:\dev\neuralgeom`. The drive letter
is a per-machine accident; nothing in the repo depends on it. `timingtask` sits
beside it as `<drive>:\dev\timingtask`.

**Data is NOT in the repo.** `data_dir.local` per machine points at
`Z:/Users/Shouvik/Modelling/SampleData`. Verify with
`python -c "from neuralgeom.paths import describe_paths; print(describe_paths())"`

**Untracked but kept on disk** (see `.gitignore`): `HANDOFF.md` is the full API
map. `docs/PROJECTIVE_RESULTS.md` and `docs/PROJECTIVE_ROADMAP.md` hold the
projective-geometry findings and plan.
