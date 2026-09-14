"""
check_env.py — audit the current Python environment for neuralgeom.
===================================================================

Run inside any candidate environment to see, package by package, whether it is
usable as-is, needs an update, or is missing something — plus a verdict per
feature group. No neuralgeom import required, so it works before install.

    python examples/check_env.py            # imports + functional smoke tests
    python examples/check_env.py --quick    # imports and versions only

The functional tests matter: a wheel built against the wrong numpy C ABI
imports cleanly and fails later. Only exercising the compiled paths catches it.

Environment facts this script encodes (verified 2026-08-26):

  * numpy MUST be < 2. The sole reason is geomstats: 2.8.0 (latest) does
    `from numpy import (... trapz ...)` in geomstats/_backend/numpy/__init__.py,
    and numpy 2.0 removed np.trapz (renamed trapezoid). geomstats declares
    `numpy>=1.18.1` with NO upper bound, so a resolver will install a broken
    combination without complaint. ripser and persim work fine on numpy 2 —
    they are not the reason, despite what earlier comments in this repo said.

  * neurogym must NOT be installed. It requires `numpy==2.2.*`, unsatisfiable
    against the pin above, so it has never been usable here. Task environments
    are built directly on gymnasium instead.
"""
from __future__ import annotations

import importlib
import sys
import traceback

# (display name, import name, min version or None, extra group, note)
CORE = [
    ("numpy", "numpy", (1, 23), "core", "MUST be < 2.0 (geomstats imports np.trapz)"),
    ("scipy", "scipy", (1, 10), "core", ""),
    ("scikit-learn", "sklearn", (1, 3), "core", ""),
    ("matplotlib", "matplotlib", (3, 6), "core", ""),
    ("h5py", "h5py", (3, 7), "core", ""),
    ("torch", "torch", (2, 0), "core", "pullback metric / dynamics"),
]
OPTIONAL = [
    ("geomstats", "geomstats", (2, 7), "geom", "Grassmannian geometry (subspace analyses)"),
    ("ripser", "ripser", None, "topology", "persistent homology"),
    ("persim", "persim", None, "topology", "bottleneck distances"),
    ("gymnasium", "gymnasium", (0, 26), "tasks", "timing task environment"),
    ("dxtr", "dxtr", None, "dec", "discrete exterior calculus (NOT pydec)"),
    ("reportlab", "reportlab", None, "report", "PDF reports"),
    ("pillow", "PIL", None, "report", ""),
    ("pypdf", "pypdf", None, "report", ""),
]
# Packages whose PRESENCE is a fault.
CONFLICTS = [
    ("neurogym", "neurogym", "requires numpy==2.2.* — remove it (pip uninstall neurogym)"),
]


def _ver_tuple(v):
    out = []
    for part in str(v).split(".")[:3]:
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def _check(imp, minv):
    try:
        m = importlib.import_module(imp)
    except Exception as e:                     # noqa: BLE001
        return "MISSING", "", str(e).split("\n")[0][:44]
    ver = getattr(m, "__version__", "?")
    if minv and ver != "?" and _ver_tuple(ver) < minv:
        return "UPDATE", ver, f"need >= {'.'.join(map(str, minv))}"
    return "OK", ver, ""


# --------------------------------------------------------------------------- #
# Functional tests — exercise compiled paths, not just imports
# --------------------------------------------------------------------------- #
def _fixture():
    import numpy as np
    return np.random.default_rng(0), np.random.default_rng(0).normal(size=(120, 6))


def f_ripser():
    from ripser import ripser
    _, X = _fixture()
    d = ripser(X, maxdim=1, coeff=2)["dgms"]
    return f"H0={len(d[0])} H1={len(d[1])}"


def f_persim():
    from ripser import ripser
    import persim
    _, X = _fixture()
    a = ripser(X, maxdim=1)["dgms"][1]
    b = ripser(X + 0.1, maxdim=1)["dgms"][1]
    return f"bottleneck={persim.bottleneck(a, b):.4f}"


def f_geomstats():
    import numpy as np
    import geomstats.backend as gs
    from geomstats.geometry.grassmannian import Grassmannian
    rng, _ = _fixture()
    A = np.linalg.qr(rng.normal(size=(6, 2)))[0]
    B = np.linalg.qr(rng.normal(size=(6, 2)))[0]
    d = Grassmannian(6, 2).metric.dist(gs.array(A @ A.T), gs.array(B @ B.T))
    return f"grassmann_dist={float(d):.4f}"


def f_torch():
    import numpy as np
    import torch
    x = torch.randn(4, 10, 3, requires_grad=True)
    out, _ = torch.nn.RNN(3, 16, batch_first=True)(x)
    out.sum().backward()
    torch.from_numpy(np.zeros((3, 3)))
    return f"RNN fwd+bwd {tuple(out.shape)}, numpy interop ok"


def f_torch_func():
    import torch
    from torch.func import jacrev
    f = lambda h: torch.tanh(h @ torch.eye(5))
    return f"jacrev{tuple(jacrev(f)(torch.randn(5)).shape)}  (pullback needs this)"


def f_gymnasium():
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces

    class _E(gym.Env):
        def __init__(self):
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(4, np.float32), {}

        def step(self, a):
            return np.zeros(4, np.float32), 0.0, False, False, {}

    e = _E()
    e.reset(seed=0)
    obs, rew, term, trunc, info = e.step(1)
    return f"5-tuple step cycle ok, obs{obs.shape}"


def f_h5py():
    import os, tempfile
    import h5py
    _, X = _fixture()
    p = os.path.join(tempfile.mkdtemp(), "t.h5")
    with h5py.File(p, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
    with h5py.File(p, "r") as f:
        return f"gzip roundtrip {f['X'].shape}"


def f_sklearn():
    from sklearn.decomposition import PCA
    _, X = _fixture()
    return f"PCA evr={PCA(3).fit(X).explained_variance_ratio_.sum():.3f}"


FUNCTIONAL = [
    ("sklearn", "sklearn", f_sklearn), ("h5py", "h5py", f_h5py),
    ("torch", "torch", f_torch), ("torch.func", "torch", f_torch_func),
    ("geomstats", "geomstats", f_geomstats),
    ("ripser", "ripser", f_ripser), ("persim", "persim", f_persim),
    ("gymnasium", "gymnasium", f_gymnasium),
]


def main():
    quick = "--quick" in sys.argv
    print(f"Python {sys.version.split()[0]}   ({sys.executable})\n")

    numpy_bad = False
    present, rows = set(), []
    for label, group in (("CORE", CORE), ("OPTIONAL", OPTIONAL)):
        rows.append((label, "", "", ""))
        for name, imp, minv, extra, note in group:
            status, ver, why = _check(imp, minv)
            if status == "OK":
                present.add(extra)
            if name == "numpy" and status == "OK" and _ver_tuple(ver)[0] >= 2:
                status, why, numpy_bad = "UPDATE", "MUST be < 2.0", True
            tag = "" if extra == "core" else f"[{extra}]"
            rows.append((f"  {name} {tag}", status, ver, why or note))

    rows.append(("MUST BE ABSENT", "", "", ""))
    conflicts = []
    for name, imp, why in CONFLICTS:
        found = importlib.util.find_spec(imp) is not None
        if found:
            conflicts.append(name)
        rows.append((f"  {name}", "CONFLICT" if found else "absent", "", why if found else "correct"))

    w = max(len(r[0]) for r in rows)
    for a, b, c, d in rows:
        if b == "" and c == "" and not d:
            print(f"\n{a}")
            continue
        mark = {"OK": "+", "absent": "+", "UPDATE": "!", "MISSING": "x", "CONFLICT": "X"}.get(b, " ")
        print(f"  {mark} {a:<{w}}  {b:<9} {c:<12} {d}")

    fn_fail = []
    if not quick:
        print("\nFUNCTIONAL (compiled paths — an ABI mismatch imports fine and fails here)")
        for name, imp, fn in FUNCTIONAL:
            if importlib.util.find_spec(imp) is None:
                print(f"  - {name:<12} skipped (not installed)")
                continue
            try:
                print(f"  + {name:<12} {fn()}")
            except Exception as e:                 # noqa: BLE001
                fn_fail.append(name)
                print(f"  X {name:<12} {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)

    core_ok = all(_check(i, mv)[0] == "OK" for _, i, mv, *_ in CORE) and not numpy_bad
    geom_ok = core_ok and {"geom", "topology"} <= present and not fn_fail
    print("\nVerdict:")
    print(f"  Pullback metric / dynamics : {'READY' if core_ok and not fn_fail else 'NOT READY (fix CORE above)'}")
    print(f"  Subspace / topology        : {'READY' if geom_ok else 'install geomstats + ripser + persim'}")
    print(f"  DEC layer                : {'READY' if 'dec' in present else 'optional — pip install dxtr (not pydec)'}")

    if numpy_bad:
        print("\n  !  numpy >= 2 — geomstats will raise ImportError on np.trapz. Pin numpy < 2.")
    if conflicts:
        print(f"\n  X  remove: {', '.join(conflicts)}")
    if fn_fail:
        print(f"\n  X  functional failures: {', '.join(fn_fail)} — installed but not working")

    return 1 if (numpy_bad or conflicts or fn_fail or not core_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
