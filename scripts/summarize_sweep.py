"""Summarize a completed (or in-progress) threshold sweep from its cell JSONs.

Prints a per-cell table (aggregated over seeds) and regenerates both figures:
the log-mass-error heatmap and the detection-boundary plot.

Usage: python scripts/summarize_sweep.py [--dir results/threshold]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np  # noqa: E402
from perturber.plots import plot_threshold, plot_detection_boundary  # noqa: E402


def load_cells(d):
    cells = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json") and fn != "threshold_meta.json":
            with open(os.path.join(d, fn)) as f:
                obj = json.load(f)
            if "mass" in obj:
                cells.append(obj)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).parents[1] / "results" / "threshold"))
    args = ap.parse_args()

    cells = load_cells(args.dir)
    if not cells:
        print(f"No cell JSONs in {args.dir}"); return
    n_seeds = len({c["seed"] for c in cells})
    print(f"{len(cells)} cells loaded from {args.dir}  ({n_seeds} seed(s))\n")

    # Aggregate over seeds
    keys = sorted({(c["mass"], c["sigma"], c["n_periods"]) for c in cells},
                  key=lambda k: (k[2], -k[1], -k[0]))
    print(f"{'arc':>5} {'sigma':>8} {'mass':>8} {'logerr(med)':>12} "
          f"{'det':>5} {'2dlogL(med)':>12} {'n':>3}")
    print("-" * 60)
    n_det = 0
    for (m, s, a) in keys:
        sel = [c for c in cells if c["mass"] == m and c["sigma"] == s
               and c["n_periods"] == a]
        le = np.median([c["log_mass_error"] for c in sel])
        det = np.median([c["detected"] for c in sel]) >= 0.5
        stat = np.median([c["loglike_ratio_stat"] for c in sel])
        n_det += det
        print(f"{a:>5g} {s:>8.0e} {m:>8.0e} {le:>12.3f} "
              f"{'YES' if det else ' no':>5} {stat:>12.1f} {len(sel):>3}")
    print(f"\n{n_det}/{len(keys)} (mass,sigma,arc) points detected")

    h = plot_threshold(cells, os.path.join(args.dir, "threshold.png"))
    b = plot_detection_boundary(cells, os.path.join(args.dir, "boundary.png"))
    print(f"\nfigures: {h}\n         {b}")


if __name__ == "__main__":
    main()
