# examples/

Runnable material for verifying an installation and learning the library. All
examples use synthetic or self-trained networks; no datasets are required.

## Notebooks

| notebook | tools | what it works out on one example |
|---|---|---|
| `example_ring_attractor.ipynb` | subspace trajectory, topology | A ring-attractor RNN with a rotating bump. Raw activity → PCA → Grassmannian embedding and the singular-value gap → distance from the initial subspace and recurrence → Riemannian kinematics → tangent PCA → persistent homology of the `ℝP¹` loop → dependence on `k` → discrete exterior calculus → comparison with the homology of the raw state distances. |
| `example_task_trained_rnn.ipynb` | pullback metric, dynamics | A vanilla RNN trained on evidence integration. Training → behaviour and psychometrics → population geometry → recurrent-Jacobian spectra and effective time constants → slow points and the line attractor → readout versus input subspaces → state-space pullback metric → subspace trajectory of the same network. |
| `neuralgeom_tour.ipynb` | all | A brief guided tour of the main analyses. |

Notebook outputs are stripped on commit, so the repository holds the narrative
and the code; run a notebook to regenerate its figures. A Jupyter kernel is
needed:

```bash
pip install -e '.[notebook]'
python -m ipykernel install --user --name neuralgeom
```

## Scripts

| script | purpose |
|---|---|
| `check_env.py` | audit the current environment: per-package versions and a verdict per feature group (works before `neuralgeom` is installed) |
| `verify_all.py` | run a small check of every subpackage and print PASS / SKIP / FAIL (`--quick` for a faster pass) |
| `quickstart.py` | the shortest end-to-end example |

```bash
python examples/check_env.py
python examples/verify_all.py
python examples/quickstart.py
```
