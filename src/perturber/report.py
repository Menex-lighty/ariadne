"""Persist run metrics as JSON next to the figures, so the headline numbers are
reproducible machine-readable outputs rather than only prose in the docs.

Usage (scripts run with PYTHONPATH=src):
    from perturber.report import save_metrics
    save_metrics(outdir, {"scale_height_km": 48.2, ...})
"""
import json
from pathlib import Path


def _enc(o):
    import numpy as np
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_metrics(outdir, metrics, name="metrics.json"):
    """Write `metrics` (a dict) to outdir/name as indented JSON. numpy scalars
    and arrays are coerced. Returns the path."""
    p = Path(outdir) / name
    p.write_text(json.dumps(metrics, indent=2, default=_enc))
    print(f"metrics -> {p}")
    return p


if __name__ == "__main__":
    import numpy as np
    import tempfile
    d = tempfile.mkdtemp()
    p = save_metrics(d, {"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1.0, 2.0])})
    got = json.loads(Path(p).read_text())
    assert got == {"a": 1.5, "b": 3, "c": [1.0, 2.0]}
    print("[report] self-check passed")
