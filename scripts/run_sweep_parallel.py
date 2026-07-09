"""Run the detection-threshold sweep across CPU cores (one process per cell).

The fit is a Python-loop-bound N-body integration — CPU beats the local GPU and
cells are independent, so the sweep parallelizes near-linearly over cores. Each
worker is pinned to 1 torch thread to avoid oversubscription.

Usage:
  python scripts/run_sweep_parallel.py --grid core --workers 20
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from perturber.runner import sweep_cell_specs, execute_cell, _collect_and_plot  # noqa: E402
from perturber.plots import ensure_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="kaggle", choices=["smoke", "kaggle"],
                    help="fit fidelity (kaggle = reduced sweep fidelity)")
    ap.add_argument("--grid", default="core", choices=["smoke", "core", "kaggle"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads per worker (keep at 1 under a pool)")
    ap.add_argument("--max-restarts", type=int, default=6,
                    help="times to rebuild the pool if a worker crash breaks it")
    args = ap.parse_args()

    outdir = args.outdir or str(Path(__file__).parents[1] / "results" / "threshold")
    ensure_dir(outdir)
    specs, n_total = sweep_cell_specs(args.preset, outdir, grid=args.grid)
    pending = [s for s in specs if not os.path.exists(s["cell_path"])]
    print(f"[parallel] grid '{args.grid}': {n_total} cells, {len(pending)} to run, "
          f"{args.workers} workers", flush=True)

    # A single worker crash breaks the whole pool (BrokenProcessPool). Cells are
    # resumable (JSON skip), so on a broken pool we just rebuild it over the
    # still-pending cells. Bounded retries guard against a genuinely poisoned cell.
    attempt = 0
    while pending and attempt < args.max_restarts:
        attempt += 1
        done = 0
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(execute_cell, s, False, args.threads): s
                           for s in pending}
                for fut in as_completed(futures):
                    done += 1
                    try:
                        c = fut.result()
                        print(f"[parallel a{attempt} {done}/{len(pending)}] "
                              f"m={c['mass']:.0e} sigma={c['sigma']:.0e} "
                              f"arc={c['n_periods']:g} seed={c['seed']} -> "
                              f"logerr {c['log_mass_error']:.3f} "
                              f"{'DET' if c['detected'] else '---'}", flush=True)
                    except Exception as e:  # one bad future shouldn't sink the run
                        s = futures[fut]
                        print(f"[parallel a{attempt} {done}/{len(pending)}] "
                              f"FAILED {s['key']}: {type(e).__name__}: {e}", flush=True)
        except BrokenProcessPool as e:
            print(f"[parallel] pool broke (attempt {attempt}): {e} — restarting "
                  f"on remaining cells", flush=True)
        pending = [s for s in specs if not os.path.exists(s["cell_path"])]

    cells = _collect_and_plot(outdir)
    print(f"[parallel] {len(cells)}/{n_total} cells done, {len(pending)} unfinished "
          f"-> {outdir}/threshold.png", flush=True)


if __name__ == "__main__":
    main()
