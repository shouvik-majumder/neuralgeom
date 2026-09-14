"""neuralgeom.synth test: generation, embedding, the ground-truth fields, and the
dependency rule (the dynamics estimators never import the generator).

Run:  python tests/test_synth.py     (or: pytest)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import neuralgeom.synth as syn                                     # noqa: E402
from neuralgeom.data import from_synthetic                         # noqa: E402


def test_generate_gaussian():
    d = syn.generate("input", syn.CUE_AMPLITUDE_LEVELS, n_trials=12, seed=0, bin_s=0.05)
    assert d["X"].ndim == 3 and d["latent"].shape[-1] == 2
    assert d["time"][0] < 0 and abs(d["time"][np.argmin(np.abs(d["time"]))]) < 0.05  # cue at 0
    assert np.isfinite(d["lick"]).mean() > 0.5


def test_generate_poisson_matches_count():
    d = syn.generate("landscape", syn.LICK_ATTRACTOR_BEHAVIOUR_MATCHED_LEVELS, n_trials=10, seed=1, bin_s=0.05,
                     poisson=True, mean_count=0.24)
    assert abs(d["X"].mean() - 0.24) < 0.05
    assert set(np.unique(d["X"])).issubset(set(range(0, int(d["X"].max()) + 1)))  # integer counts


def test_both_mechanisms_run():
    for mech, lv in (("input", syn.CUE_AMPLITUDE_LEVELS), ("landscape", syn.LICK_ATTRACTOR_Y_PULL_LEVELS)):
        d = syn.generate(mech, lv, n_trials=8, seed=2, bin_s=0.05)
        assert d["mechanism"] == mech and d["X"].shape[0] == 8 * len(lv)


def test_round_trip_to_session():
    d = syn.generate("input", syn.CUE_AMPLITUDE_LEVELS, n_trials=6, seed=3, bin_s=0.05)
    s = from_synthetic(d)
    assert s.X.shape == d["X"].shape and s.group == "synthetic"
    assert "latent" in s.meta and s.meta["latent"].shape[-1] == 2
    assert "cue_amp" in s.trials and len(s.trials["cue_amp"]) == s.n_trials


def test_dynamics_independent_of_synth():
    """neuralgeom.dynamics must not import neuralgeom.synth (real and synthetic data
    reach the estimators through the same Session interface, never through the
    generator). Checked statically on the source, since the unified package's
    __init__ imports both submodules."""
    dyn_dir = Path(__file__).resolve().parents[1] / "neuralgeom" / "dynamics"
    for py in dyn_dir.glob("*.py"):
        src = py.read_text()
        assert "synth" not in src.replace("synthetic", ""), \
            f"{py.name} references neuralgeom.synth"
    dat_dir = Path(__file__).resolve().parents[1] / "neuralgeom" / "data"
    for py in dat_dir.glob("*.py"):
        src = py.read_text()
        assert "from ..synth" not in src and "neuralgeom.synth import" not in src, \
            f"{py.name} imports the generator (adapters must read dicts by key only)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"{len(fns)}/{len(fns)} passed")
