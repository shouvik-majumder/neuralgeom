# neuralgeom

Differential geometry and topology of neural population activity.

`neuralgeom` operates on batches of high-dimensional state trajectories from
recordings, simulations or trained networks. It computes the pullback metric of
a differentiable map on neural state space, tracks the subspace that activity
occupies as a trajectory on the Grassmannian, and characterises the topology
of that trajectory with persistent homology.

This is an exploratory research package under development. The API may change
between versions, and the datasets the methods were developed against are not
bundled.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .                       # core: numpy<2, scipy, scikit-learn, torch, h5py, matplotlib
pip install -e '.[geom,topology]'      # + Grassmannian geometry (geomstats), persistent homology (ripser, persim)
pip install -e '.[full]'               # + discrete exterior calculus (dxtr), PDF reports (reportlab)
```

Or `conda env create -f environment.yml`. Optional dependencies are imported
lazily; `import neuralgeom` needs only the core. `python examples/check_env.py`
audits an environment and `python examples/verify_all.py` exercises every
installed component.

## Capabilities

- **Pullback metrics** of differentiable maps (decoders, readouts, recurrent
  updates): metric tensor, volume element, anisotropy, spectrum, geodesics and
  curvature, with Euclidean or Fisher–Rao codomain metrics.
- **Distances between geometries**: metric tensors on the SPD manifold,
  subspaces on the Grassmannian, and their Fréchet means.
- **Subspace trajectories**: sliding-window Grassmannian embedding with a
  singular-value-gap diagnostic, Riemannian speed, covariant acceleration and
  curvature, Karcher mean, tangent PCA.
- **Topology**: persistent homology within and across trials, bottleneck
  distances between conditions, optional discrete exterior calculus.
- **Recurrent dynamics**: recurrent and input Jacobians, spectra and time
  constants, slow points, readout and input subspaces of trained networks.
- **Dynamics estimation from data**: linear and cubic vector fields with an
  instrumental-variable correction, input inference, Helmholtz decomposition.
- **Data and synthetic models**: a shared `Trajectory` object with an HDF5
  schema; loaders and adapters; a two-attractor timing model, low-rank and
  connectivity-family rate RNNs; cognitive tasks and RNN training.

## Examples

`examples/` contains two narrated notebooks (a ring-attractor RNN analysed with
the subspace and topology tools, and a task-trained RNN analysed with the
pullback-metric and dynamics tools), a short tour, and a quickstart script.
`scripts/` contains runnable demonstrations and analyses; see
[examples/README.md](examples/README.md).

## Documentation

- [docs/methods.md](docs/methods.md): definitions of every computed quantity.
- [docs/data_format.md](docs/data_format.md): the `Trajectory` object and its HDF5 schema.
