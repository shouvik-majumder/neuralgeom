# neuralgeom — geometry & topology of neural representations and dynamics

`neuralgeom` is a single, unified library that merges two exploratory research
codebases into one consistent toolkit:

* **PullbackMetric** (`pbgeom`) — the **pullback-metric / dynamics engine**. For
  a differentiable map `f`, the pullback metric `g = Jᵀ M J` measures how `f`
  distorts its domain (which directions are magnified, which collapse, how that
  varies point to point), giving volume, anisotropy and curvature; plus
  distances between metric tensors (SPD) and between subspaces (Grassmannian),
  Fisher–Rao output metrics, recurrent-dynamics geometry, LDS estimators, and
  cognitive-task RNN training.
* **ProjectiveSpaceModels** — the **subspace / Grassmannian-trajectory + topology
  lens**. Instead of the raw state `x(t) ∈ ℝᴺ`, track the *k-dimensional subspace*
  the activity locally occupies as a point on the Grassmannian `Gr(k, N)`
  (`Gr(1, N) = ℝPᴺ⁻¹`, real projective space), and study how that point *moves*
  (Riemannian kinematics) and what shape its orbit traces (persistent homology).

They are two complementary views of the **same object** — a batch of
high-dimensional neural state trajectories — and now share one data contract
(`neuralgeom.data.Trajectory`) and one geometry layer.

> Both codebases are exploratory: the library is a stable, tested substrate, not
> a claim of conclusive results. Neither the Neuropixels recordings nor the
> synthetic datasets the analyses were developed against are bundled here — see
> "Working on this repo" at the end of this file for how to point the library at
> your own copy.

## The one idea (pullback lens) and the one move (subspace lens)

* **Pullback:** `rank(g) = dim(domain)`. Put a *scalar* behaviour in the codomain
  and `g` is rank-1 (one direction, no volume/curvature); put a *low-dimensional
  manifold in the domain* and the geometry becomes rich. Recurrent dynamics give
  full-rank `g` for free.
* **Subspace:** `k` is a *topological filter* — different `k` expose different
  structure; a k-frame is only trustworthy where `σ_k/σ_{k+1} ≫ 1` (report
  `sv_gap`); use uncentered windows (the occupied subspace) and 𝔽₂ coefficients
  for homology (they expose projective/non-orientable structure).

## Install

```bash
pip install -e .                       # core engine (numpy<2, scipy, sklearn, torch, h5py, matplotlib)
pip install -e '.[geom,topology]'      # + the subspace/topology lens (geomstats, ripser, persim)
pip install -e '.[full]'               # + DEC (dxtr), reports (reportlab…), neurogym
```

Or the conda environment: `conda env create -f environment.yml`.

**Dependency model — light core + optional extras.** The core is
`numpy(<2)/scipy/scikit-learn/torch/h5py/matplotlib`. Everything else is an
optional extra and is imported lazily with a clear install hint if missing, so
`import neuralgeom` works with only the core:

| extra | packages | enables |
|---|---|---|
| `geom` | geomstats | `GrassmannManifold` ⇒ all of `neuralgeom.subspace` kinematics + pooled topology |
| `topology` | ripser, persim | persistent homology & bottleneck (`neuralgeom.topology`) |
| `dec` | dxtr | discrete exterior calculus (`neuralgeom.topology.dec`) — *not* `pydec` |
| `report` | reportlab, pillow, pypdf | PDF reports & dashboard (`neuralgeom.viz`) |
| `neurogym` | neurogym, gymnasium | the neurogym task adapter |

`numpy` is pinned `<2` because geomstats/ripser require it. **torch is core**
(the pullback/dynamics engine cannot import without it) — this is a deliberate
deviation from treating torch as an extra. The genuinely optional dependencies
are imported lazily by the modules that need them and raise a clear install hint
if absent, so `import neuralgeom` works with only the core installed.

## Layout (modular, two levels deep)

```
neuralgeom/
  geometry/      map-agnostic geometry toolbox
    jacobian.py       feed-forward Jacobians, g = JᵀJ, volume, spectra (torch)
    manifold.py       low-D domain: volume, anisotropy, curvature, whitening
    pullback.py       PullbackMetric = ReadoutMap + OutputMetric → g(x)
    riemann.py        RiemannianField: Christoffels, geodesics, curvature
    output_metrics.py codomain metrics incl. Fisher–Rao families
    maps.py, torch_readouts.py   readout maps with Jacobians
    spd.py            distances/means between metric tensors (SPD & PSD)
    grassmann.py      distances/means between subspaces — BOTH the torch
                      model-facing API and the numpy/geomstats frame API
                      (merged from the two repos)
  subspace/      the subspace lens (manifold-agnostic)
    embed.py          sliding-window Grassmannian frames + sv_gap reliability
    kinematics.py     speed / covariant accel / curvature; Karcher mean;
                      tangent-PCA; chordal-vs-geodesic; transported velocity
    pooling.py        pool frames + scalar fields across trials
  topology/      topology of subspace trajectories
    persistence.py    persistent homology (𝔽₂) on distance matrices, bottleneck
    dec.py            OPTIONAL discrete exterior calculus (dxtr) + cross-projection
    direct.py         OPTIONAL direct state-manifold cross-check
  dynamics/      measure AND estimate dynamics (rnn, lds, regression, torch_geometry)
  data/          loader, adapters, reduce, conditions, features, trial_data,
                 trajectory  ← the unified Trajectory contract
  synth/         attractor, lowrank_rnn, fixtures, subspace_rnn (connectivity families)
  tasks/         cognitive tasks + RNN training (+ optional neurogym)
  viz/  paths.py  fitting.py  stats.py

scripts/   runnable analyses & demos (never imported by the package)
tests/     numerical checks (51 tests: 27 engine + 24 subspace/topology/contract)
docs/      formulation.tex, subspace_methods.tex (the formal mathematics)
outputs/   generated artefacts: figures/{demos,rnn,neural}, pdf/, checkpoints/, cache/
```

## Quick start

### The shared contract

```python
from neuralgeom.synth.subspace_rnn import SubspaceRNNConfig, make_trajectory

cfg  = SubspaceRNNConfig(connectivity="ring", ring_moving=True, N=50, n_trials=8)
traj = make_trajectory(cfg)          # a neuralgeom.data.Trajectory
traj.save("ring_moving.h5")          # HDF5 round-trip
# every analysis consumes this one object; adapters exist from raw arrays,
# the other synth generators, and a recording Session (Trajectory.from_session)
```

### Subspace lens (Grassmannian trajectory → kinematics → topology)

```python
from neuralgeom.subspace import EmbedConfig, embed_from_trajectory, compute_kinematics, tangent_pca
from neuralgeom.topology.persistence import single_trial_distances, ph, top_life

emb = embed_from_trajectory(traj, trial=0, cfg=EmbedConfig(k=1, win=50, stride=10))
kin = compute_kinematics(emb["frames"], emb["win_times"])   # speed, curvature, efficiency…
tp  = tangent_pca(emb["frames"])                            # intrinsic dimensionality

D   = single_trial_distances(traj, 0, EmbedConfig(k=1))     # geodesic distance matrix
h1  = top_life(ph(D, maxdim=1)[1])                          # persistence of the loop (ℝP¹)
```

### Pullback lens (feed-forward and recurrent geometry)

```python
import torch, torch.nn as nn
from neuralgeom.geometry import PullbackGeometry, spd_distance, grassmann_distance

model = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 10))
geo = PullbackGeometry(model)
X   = torch.randn(32, 3)
vol = geo.volume_element(X)          # local volume magnification √det g

from neuralgeom.tasks import make_task, make_model, train
task  = make_task("context_decision", dt=25, seed=0)
rnn   = make_model("vanilla", task.spec, hidden_size=96, dt=task.dt)
train(rnn, task, steps=3000)         # then neuralgeom.dynamics for J_rec, fixed points, LDS…
```

## Worked examples (executed notebooks with plots)

Two detailed, narrated notebooks work a single network end-to-end and explain what
each quantity describes. Their outputs are stripped in version control (see
"Working on this repo"), so run them to regenerate the plots:

- `examples/example_ring_attractor.ipynb` — the **subspace/topology** lens on a
  ring-attractor RNN (Grassmannian embedding → kinematics → persistent homology of
  the ℝP¹ loop → DEC → cross-check).
- `examples/example_task_trained_rnn.ipynb` — the **pullback/dynamics** lens on a
  vanilla RNN **trained** on evidence integration (Jacobian spectra → line
  attractor / slow points → state-space pullback metric → bridge to the subspace
  lens).

To re-run them, add a kernel: `pip install -e '.[notebook]'` (or `pip install
ipykernel`) and pick the env's kernel in VS Code / Jupyter. See `examples/README.md`.

## Running tests & scripts

```bash
python -m pytest tests -q                                   # 51 tests (skips extras not installed)
python examples/check_env.py                                # is this env usable? per-lens verdict
python examples/verify_all.py                               # run EVERY branch: PASS/SKIP/FAIL
python scripts/subspace/demo_subspace_pipeline.py --quick   # subspace lens end-to-end (needs geom,topology)
python scripts/demos/demo_pullback_metric.py                # pullback lens demos (recycled from pbgeom)
```

Scripts under `scripts/neural/` and the real-data `scripts/attractor/*_real.py`
expect the Neuropixels recordings, which are **not** bundled — point
`neuralgeom.paths.DATA_DIR` at your copy, as described under "Working on this
repo" below.

## Documentation

| file | what it is |
|---|---|
| `README.md` | this — orientation, install, layout, quick start, per-machine setup |
| `docs/formulation.tex` | the rigorous pullback-metric mathematics |
| `docs/subspace_methods.tex` | formal definitions of the subspace/topology constructions (from ProjectiveSpaceModels) |

## Background

Pullback lens: Zavatone-Veth et al. (*magnifying areas near decision
boundaries*); Cayco-Gajic & Pellegrino (*geometry-aware similarity metrics*,
the spectral-ratio SPD distance); Majumder et al. (the timing task & 2-attractor
model). Subspace lens: low-rank RNN theory (Mastrogiuseppe & Ostojic; Beiran et
al.), fixed-point reverse-engineering (Sussillo & Barak), and shared dynamical
motifs (Driscoll et al.). Full reference lists are in the two `.tex` documents.

---

## Working on this repo (once per machine)

The repository lives on fast local disk; the recordings live on the lab
network share. Three things are therefore machine-local and deliberately **not**
tracked by git:

**1. Where the data is.** Copy `data_dir.local.example` to `data_dir.local` and
put the absolute path to the recordings directory in it (one line). Or set the
`NEURALGEOM_DATA_DIR` environment variable, which takes precedence. Verify:

```bash
python -c "from neuralgeom.paths import describe_paths; print(describe_paths())"
```

That prints the resolved layout, whether `DATA_DIR` exists, how many `.h5`
files it holds, and *which rule* produced the path — which is the fastest way
to diagnose a "file not found" on a machine you have not used in a while.

**2. The environment.**

```bash
conda env create -f environment.yml
conda activate neuralgeom
pip install -e .
```

After any `conda install`, re-export and commit:
`conda env export --from-history > environment.yml`. An `environment.yml` that
has drifted out of date is worse than none, because it claims a reproducibility
it does not deliver.

**3. The notebook output filter.** `.gitattributes` declares that `.ipynb`
files pass through `nbstripout`, but the filter *definition* lives in
`.git/config`, which is not tracked. So per clone:

```bash
pip install nbstripout
where.exe nbstripout                    # copy the absolute path
git config filter.nbstripout.clean  "<abs-path>/nbstripout.exe"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
git config diff.ipynb.textconv "<abs-path>/nbstripout.exe -t"
```

Until this is done, committing a notebook fails with an error. That is
intentional: the alternative is silently committing megabytes of base64-encoded
figure output on every re-run, which the repository never recovers from.

### Day to day

```bash
git pull                                        # start of a session
# ... work ...
git add -A && git commit -m "..." && git push   # end of a session
```

`pull.rebase=true` is set globally, so switching between workstations keeps
history linear instead of accumulating merge commits.
