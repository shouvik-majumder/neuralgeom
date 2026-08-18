"""
Regenerate the rolling analysis dashboard (outputs/pdf/neural_dashboard.pdf).

Cheap: assembles whatever figures currently exist under outputs/figures/ into
one PDF with a status page and per-figure captions. Re-run any time.

    python scripts/neural/build_dashboard.py
"""
# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neuralgeom.viz.dashboard import build  # noqa: E402

if __name__ == "__main__":
    build()
