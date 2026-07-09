"""Run the detection-threshold sweep (resumable; safe to interrupt and rerun).

Usage: python scripts/run_threshold_study.py --preset smoke|local|kaggle
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from perturber.runner import run_threshold_study  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke", choices=["smoke", "local", "kaggle"])
    ap.add_argument("--grid", default=None, choices=["smoke", "core", "kaggle"],
                    help="grid size (default: match preset; 'core' = 48-cell feasible map)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--shard", type=int, default=0, help="this shard's index")
    ap.add_argument("--n-shards", type=int, default=1,
                    help="split the grid into N shards for parallel sessions/machines")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir or str(Path(__file__).parents[1] / "results" / "threshold")
    run_threshold_study(args.preset, outdir, verbose=args.verbose,
                        shard=args.shard, n_shards=args.n_shards, grid=args.grid)


if __name__ == "__main__":
    main()
