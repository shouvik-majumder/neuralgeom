# Current focus — neuralgeom

> Overwrite this file, never append. It describes the present, not history.
> Lines marked [VERIFY] were inferred by Claude from the repo, not stated by me.

**Task:** Nothing active. Repo was just put under version control (v0.5.0).
The library is a merged substrate that has never been exercised end-to-end.

**Where:** whole repo; `neuralgeom/` is the package, `scripts/` the runnable
analyses, `tests/` the numerical checks.

**Last thing I did:** Initial commit and push to GitHub. Made `DATA_DIR`
machine-configurable (`data_dir.local`, gitignored) so the repo lives on D:
while recordings stay on Z:. Set up an nbstripout filter on the notebooks.

**Next:** Build the conda environment and run the test suite. This has not been
done since the merge, so the pass rate is unknown.

    conda env create -f environment.yml
    conda activate neuralgeom
    pip install -e .
    pytest -q

**Open questions:**
- Does `environment.yml` actually solve on Windows? [VERIFY] `dxtr` (the DEC
  dependency) has no conda-forge build and is pip-only. It is optional and
  imported lazily, so a failure there should not block the core.
- The README claims 51 tests (27 engine + 24 subspace/topology). There are 12
  test files. Does the count still hold after the merge? [VERIFY]
- `paths.py` calls `mkdir` at import time for the four `outputs/` subfolders.
  Harmless here but it is a side effect on import — worth knowing if the package
  is ever imported somewhere read-only.

**Hazards — do not change without thinking:**
- `numpy` is pinned `<2`. This is not arbitrary: geomstats and ripser require
  1.x. Bumping it silently breaks the whole subspace/topology half.
- `torch` is a CORE dependency, not optional. The pullback/dynamics engine
  cannot import without it.
- The DEC dependency is `dxtr`. The PyPI package named `pydec` is a different,
  unrelated library.
- Data is NOT in the repo. `data_dir.local` on each machine points at
  `Z:/Users/Shouvik/Modelling/SampleData`. Verify with
  `python -c "from neuralgeom.paths import describe_paths; print(describe_paths())"`

**Untracked but kept on disk** (see `.gitignore`): `HANDOFF.md` is the full API
map and is the file to read when you have forgotten how the two halves join.
`docs/PROJECTIVE_RESULTS.md` and `docs/PROJECTIVE_ROADMAP.md` hold the findings
and the plan.

WS2 reachable.
