"""
neuralgeom.viz — shared plotting style, metric-field renderings and PDF reports.
===============================================================================

    style.py      one set of rcParams and one savefig helper
    report.py     the Report builder used by every write-up
    fields.py     2-D scalar/ellipse renderings of a PullbackMetric field
"""
from .style import (C_EARLY, C_REW, CHOICE_COLORS, apply_style, note, savefig)
from .fields import (scalar_metric_field, plot_scalar_metric_field,
                     metric_ellipse_polygon, plot_ellipse_field)

# ``Report`` depends on the optional [report] extra (reportlab/pillow/pypdf).
# Import it lazily so that ``import neuralgeom`` works without those packages;
# accessing ``neuralgeom.viz.Report`` without them raises a clear error.
try:  # pragma: no cover - exercised only when the extra is installed
    from .report import Report
except ImportError as _report_err:  # pragma: no cover
    _REPORT_ERR = _report_err

    def Report(*_args, **_kwargs):  # type: ignore
        raise ImportError(
            "neuralgeom.viz.Report needs the optional 'report' extra: "
            "pip install 'neuralgeom[report]' (reportlab, pillow, pypdf)."
        ) from _REPORT_ERR

__all__ = ["apply_style", "savefig", "note", "Report", "CHOICE_COLORS",
           "C_EARLY", "C_REW", "scalar_metric_field",
           "plot_scalar_metric_field", "metric_ellipse_polygon",
           "plot_ellipse_field"]
