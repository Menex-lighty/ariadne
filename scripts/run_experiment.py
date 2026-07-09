"""Run one hidden-perturber experiment end-to-end.

Usage: python scripts/run_experiment.py --preset smoke|local|kaggle
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from perturber.config import get_preset  # noqa: E402
from perturber.runner import run_single_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke", choices=["smoke", "local", "kaggle"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    sys_cfg, fit_cfg = get_preset(args.preset)
    outdir = args.outdir or str(Path(__file__).parents[1] / "results" / args.preset)
    out = run_single_experiment(sys_cfg, fit_cfg, outdir)

    if args.preset == "smoke":
        # The smoke run is a real test: easy case must be recovered and detected.
        lme = out["perturber"]["log_mass_error"]
        assert lme < 0.5, f"SMOKE FAIL: log-mass error {lme:.3f} >= 0.5"
        assert out["comparison"]["detected"], "SMOKE FAIL: perturber not detected"
        assert out["perturber"]["chi2_test"] < out["null"]["chi2_test"], \
            "SMOKE FAIL: perturber model did not beat null on held-out data"
        print("[smoke] all assertions passed")


if __name__ == "__main__":
    main()
