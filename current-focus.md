# Current focus — neuralgeom

> Overwrite this file, never append. It describes the present, not history.
> Lines marked [VERIFY] were inferred by Claude from the repo, not stated by me.

**Task:** Gym timing-task module. Plan lives in the Claude project doc
`claude/gym_timing_task_plan.md`. Phase 0 (environment) is done; Phase 1 (port
and restructure `timing_task/` into `neuralgeom/tasks/timing/`) is next.

**Where:** whole repo; `neuralgeom/` is the package, `scripts/` the runnable
analyses, `tests/` the numerical checks.

**Last thing I did:** Set up WS2, which exposed the `.gitignore` bug below.
Settled the environment question and cleaned it up.

## RESOLVED — `.gitignore` was swallowing a source package

Line 28 was a bare `data/`, which matches a directory of that name at **any
depth**. It caught `neuralgeom/data/` — the source package. Eight files had never
been committed: `__init__.py`, `adapters.py`, `conditions.py`, `features.py`,
`loader.py`, `reduce.py`, `trajectory.py`, `trial_data.py`.

The GitHub repo was unimportable from a fresh clone, and WS1's working copy was
the only copy of `Trajectory`, the contract every analysis consumes. Fixed by
anchoring to `/data/` and committing the package.

Lesson: bare directory names in `.gitignore` are unanchored. Anchor anything that
could collide with a source directory name. **A fresh clone on a second machine
is the only reliable test that a repo is complete** — a working copy always looks
fine. 109 source files on disk, 15 ignored; the other 7 are `docs/*.md` handoff
files excluded on purpose.

## RESOLVED — one environment, and why numpy stays pinned

`environment.yml` builds and the whole stack works together at numpy 1.26.4.
Verified with `python examples/check_env.py`, which now runs functional tests on
the compiled paths, not just imports.

**The `numpy<2` pin stays, but not for the reason the old comment gave.** The
sole cause is geomstats 2.8.0 (latest):

    geomstats/_backend/numpy/__init__.py:4   from numpy import (... trapz ...)

`np.trapz` was removed in numpy 2.0 (renamed `trapezoid`), so `import geomstats`
raises ImportError under numpy 2. geomstats declares `numpy>=1.18.1` with **no
upper bound**, so a resolver will install a broken combination silently.
**ripser and persim are not the reason** — both tested fine on numpy 2.4.

**neurogym removed.** It requires `numpy==2.2.*`, unsatisfiable against the pin,
so `neuralgeom/tasks/neurogym.py` had never run in a conforming env. Module
deleted, extras dropped. Task environments use gymnasium directly.

**gymnasium upgraded to 1.3.0**, verified against every API the timing task uses.
One behaviour change from 0.29: wrappers no longer forward unknown attributes to
the wrapped env — use `env.unwrapped.x` or `env.get_wrapper_attr("x")`. A
`hasattr(self.env, ...)` guard now silently returns False instead of reaching
through, which fails quietly rather than loudly.

**Known benign:** h5py warns it runs against HDF5 2.2.0 but was built against
2.1.0. The gzip round-trip test passes. Only act if real `.h5` reads corrupt or
segfault, then `conda install -c conda-forge "h5py=*=*nompi*" --force-reinstall`.

## Open

- The README claims 51 tests (27 engine + 24 subspace/topology). There are 12
  test files. Does the count hold after the merge? [VERIFY] **The suite still has
  not been run since the merge** — `pytest -q` is the outstanding check.
- `paths.py` calls `mkdir` at import time for the four `outputs/` subfolders.
  A side effect on import; matters only if the package is imported read-only.

## Hazards — do not change without thinking

- `numpy` pinned `<2` — see above for the exact reason. Bumping it breaks
  `import geomstats` and therefore the whole subspace/topology half.
- `torch` is CORE. `neuralgeom/__init__.py` imports the entire package eagerly,
  so nothing imports without torch.
- The DEC dependency is `dxtr`. The PyPI package `pydec` is unrelated.
- Data is NOT in the repo. `data_dir.local` per machine points at
  `Z:/Users/Shouvik/Modelling/SampleData`. Verify with
  `python -c "from neuralgeom.paths import describe_paths; print(describe_paths())"`

**Machines:** WS1 `D:\dev\neuralgeom`, WS2 `C:\dev\neuralgeom`. The drive letter
is a per-machine accident; nothing in the repo depends on it.

**Untracked but kept on disk** (see `.gitignore`): `HANDOFF.md` is the full API
map — read it when you have forgotten how the two halves join.
`docs/PROJECTIVE_RESULTS.md` and `docs/PROJECTIVE_ROADMAP.md` hold findings and plan.
