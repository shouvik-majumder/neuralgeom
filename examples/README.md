# Examples and scripts

All examples use synthetic or self-trained networks unless marked as requiring
recordings.

## examples/

| file | content |
|---|---|
| `example_ring_attractor.ipynb` | ring-attractor RNN: Grassmannian embedding, kinematics, tangent PCA, persistent homology, DEC |
| `example_task_trained_rnn.ipynb` | RNN trained on evidence integration: Jacobian spectra, slow points, readout subspaces, pullback metric |
| `neuralgeom_tour.ipynb` | short tour of the main analyses |
| `quickstart.py` | shortest end-to-end script |
| `check_env.py` | audit the environment (works before install) |
| `verify_all.py` | PASS / SKIP / FAIL check of every subpackage (`--quick` for a fast pass) |

Notebook outputs are stripped on commit; run a notebook to regenerate its figures
(`pip install -e '.[notebook]'`).

## scripts/

| folder | content |
|---|---|
| `demos/` | pullback metric, SPD geometry, Grassmannian geometry and regression demonstrations on small models |
| `subspace/` | subspace embedding, kinematics and persistent homology across connectivity families and regimes |
| `rnn/` | train RNNs on the four cognitive tasks and analyse their dynamics |
| `attractor/` | dynamics estimation and pullback analyses on the two-attractor model (synthetic, and on recordings) |
| `neural/` | quality control, population structure, reducer sweep, decoder pullback field with controls, condition-manifold pullback (requires recordings) |
