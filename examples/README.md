# examples/

Runnable material for verifying and learning the library. Everything here uses
synthetic or self-trained networks — **no bundled datasets required**.

## Detailed, narrated notebooks (executed, with plots)

| notebook | lens | what it works out on one example |
|---|---|---|
| `example_ring_attractor.ipynb` | subspace / topology | A ring-attractor RNN with a rotating bump. Raw activity → PCA → Grassmannian embedding + reliability → drift & recurrence → Riemannian kinematics → tangent-PCA → **persistent homology** (the ℝP¹ loop) → `k` as a topological filter → DEC → direct-manifold cross-check. Narrates what each quantity means. |
| `example_task_trained_rnn.ipynb` | pullback / dynamics | A vanilla RNN **trained** on evidence integration. Training → behaviour/psychometrics → population geometry → recurrent-Jacobian spectra & effective time constants → **slow points / line attractor** → readout vs input subspaces → state-space pullback metric → bridge back to the subspace lens. |

Outputs are stripped when these notebooks are committed (an `nbstripout` filter
— see "Working on this repo" in the top-level README), so what you have here is
the narrative and the code, not the figures. Run a notebook to regenerate them.

**To re-run them** you need a Jupyter kernel in your env:
```bash
pip install ipykernel            # then open in VS Code / Cursor and pick the neuralgeom kernel
# or for browser Jupyter:
pip install jupyterlab ipykernel
python -m ipykernel install --user --name neuralgeom
jupyter lab
```

## Scripts (no Jupyter needed)

| script | purpose |
|---|---|
| `check_env.py` | audit the current environment; per-package + per-lens verdict (usable / update / missing) |
| `verify_all.py` | run **every** branch of the library and print PASS/SKIP/FAIL (add `--quick` for a fast pass) |
| `quickstart.py` | the smallest end-to-end touch of both lenses |

```bash
python examples/check_env.py
python examples/verify_all.py
```
