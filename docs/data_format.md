# Data format

Every analysis consumes a `neuralgeom.data.Trajectory`.

## Fields

| field | shape | required | meaning |
|---|---|---|---|
| `X` | `(n_trials, T, N)` | yes | states (firing rates or hidden units) |
| `time` | `(T,)` | yes | sample times in seconds |
| `dt` | scalar | yes | sample step (inferred from `time` if absent) |
| `tau` | scalar | no | time constant of the generating dynamics |
| `inputs` | `(n_trials, T, n_in)` | no | stimulus / input |
| `outputs` | `(n_trials, T, n_out)` | no | readout or target |
| `condition` | `(n_trials,)` | no | per-trial label |
| `W` | `(N, N)` | no | connectivity |
| `U`, `V` | `(N, R)` | no | low-rank loadings, `W = U Vᵀ / N` |
| `aux` | dict of arrays | no | generator-specific extras |
| `meta` | dict | yes | generator identifier and configuration |

Variable-length trials are padded with NaN; the per-trial length can be stored
in `aux["n_steps"]`.

## HDF5 layout

`Trajectory.save(path)` writes and `neuralgeom.data.load_trajectory(path)` reads:

```
/X, /time, /inputs, /outputs, /condition, /W, /U, /V     datasets (optional ones omitted)
/aux/<key>                                                one dataset per aux entry
attrs: dt, tau, meta (JSON string), neuralgeom_trajectory = True
```

Any external code that writes this layout can be analysed without importing
`neuralgeom`.

## Constructors

| constructor | from |
|---|---|
| `Trajectory.from_arrays(X, dt=..., condition=...)` | plain arrays |
| `Trajectory.from_synth_dict(d)` | the dict returned by the generators in `neuralgeom.synth` |
| `Trajectory.from_session(session)` | a recording `Session` from `neuralgeom.data.load_session` |
| `load_trajectory(path)` | an HDF5 file in the layout above |

`traj.trial(i)` returns `(X_i, time)` for one trial.

## Recording location

Scripts that read recordings resolve the data directory from, in order, the
`NEURALGEOM_DATA_DIR` environment variable, a one-line `data_dir.local` file in
the repository root (see `data_dir.local.example`), or `./SampleData`.
`python -c "from neuralgeom.paths import describe_paths; print(describe_paths())"`
prints the resolved layout.
