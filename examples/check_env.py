"""
check_env.py — audit the current Python environment for neuralgeom.
===================================================================

Run this inside any candidate environment (a fresh venv, your PullbackMetric
env, or your ProjectiveSpaceModels `nsg` env) to see, package by package,
whether it is usable as-is, needs an update, or is missing something — plus a
per-lens verdict. No neuralgeom import required, so it works before install.

    python examples/check_env.py
"""
from __future__ import annotations

import importlib
import sys

# (module, import name, min version or None, extra group, human note)
CORE = [
    ("numpy", "numpy", (1, 23), "core", "MUST be < 2.0 (geomstats/ripser need numpy 1.x)"),
    ("scipy", "scipy", (1, 10), "core", ""),
    ("scikit-learn", "sklearn", (1, 3), "core", ""),
    ("matplotlib", "matplotlib", (3, 6), "core", ""),
    ("h5py", "h5py", (3, 7), "core", ""),
    ("torch", "torch", (2, 0), "core", "the pullback/dynamics engine"),
]
OPTIONAL = [
    ("geomstats", "geomstats", (2, 7), "geom", "subspace lens (Grassmannian geometry)"),
    ("ripser", "ripser", None, "topology", "persistent homology"),
    ("persim", "persim", None, "topology", "bottleneck distances"),
    ("dxtr", "dxtr", None, "dec", "discrete exterior calculus (NOT pydec)"),
    ("reportlab", "reportlab", None, "report", "PDF reports"),
    ("pillow", "PIL", None, "report", ""),
    ("pypdf", "pypdf", None, "report", ""),
    ("neurogym", "neurogym", None, "neurogym", "task adapter"),
]


def _ver_tuple(v):
    out = []
    for part in str(v).split(".")[:3]:
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def _check(mod, imp, minv):
    try:
        m = importlib.import_module(imp)
    except Exception as e:                     # noqa: BLE001
        return "MISSING", "", str(e).split("\n")[0][:40]
    ver = getattr(m, "__version__", "?")
    if minv and ver != "?" and _ver_tuple(ver) < minv:
        return "UPDATE", ver, f"need >= {'.'.join(map(str, minv))}"
    return "OK", ver, ""


def main():
    print(f"Python {sys.version.split()[0]}   ({sys.executable})\n")
    numpy_bad = False
    present = set()
    rows = []
    for label, group in (("CORE", CORE), ("OPTIONAL", OPTIONAL)):
        rows.append((label, "", "", ""))
        for name, imp, minv, extra, note in group:
            status, ver, why = _check(name, imp, minv)
            if status == "OK":
                present.add(extra)
            if name == "numpy" and status == "OK":
                try:
                    import numpy as _np
                    if _ver_tuple(_np.__version__)[0] >= 2:
                        status, why, numpy_bad = "UPDATE", "MUST be < 2.0", True
                except Exception:
                    pass
            tag = "" if extra == "core" else f"[{extra}]"
            rows.append((f"  {name} {tag}", status, ver, why or note))

    w = max(len(r[0]) for r in rows)
    for a, b, c, d in rows:
        if b == "" and c == "" and not d:
            print(f"\n{a}")
            continue
        mark = {"OK": "✓", "UPDATE": "!", "MISSING": "×"}.get(b, " ")
        print(f"  {mark} {a:<{w}}  {b:<8} {c:<12} {d}")

    # per-lens verdict
    print("\nVerdict:")
    core_ok = all(_check(n, i, mv)[0] == "OK" for n, i, mv, *_ in CORE) and not numpy_bad
    pull = "READY" if core_ok else "NOT READY (fix CORE above)"
    sub = ("READY" if core_ok and "geom" in present and "topology" in present
           else "install geomstats + ripser + persim  (pip install 'neuralgeom[geom,topology]')")
    print(f"  Pullback / dynamics lens : {pull}")
    print(f"  Subspace / topology lens : {sub}")
    if numpy_bad:
        print("\n  ⚠  numpy >= 2 detected — geomstats/ripser will fail. "
              "Pin numpy < 2 in this env.")
    print("\n  DEC layer:", "READY" if "dec" in present else "optional — pip install dxtr (not pydec)")


if __name__ == "__main__":
    main()
