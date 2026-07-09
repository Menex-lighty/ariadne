"""Milestone 3 (VLEO) — density-field identifiability: one satellite vs a
constellation.

A satellite's drag samples the thermospheric density rho(altitude, local solar
time) only along its own track, so a single near-circular satellite pins one
altitude and cannot constrain the field's altitude structure. Adding satellites
at different altitudes / local times conditions the field design matrix and the
field becomes recoverable — the multi-object joint constraint that is the whole
point of M3, quantified with the same condition-number tooling as the force-form
and Kepler-9 results.

Produces results/vleo/vleo_restoration.png and a summary table. Best-case
(direct along-track rho samples) analysis, as for the force-form sweep.

Usage: python scripts/run_vleo_identifiability.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from perturber.density import (design_condition, satellite_track,   # noqa: E402
                               recover_field)
from perturber.plots import ensure_dir   # noqa: E402


def log_rho_true(h, lst):
    """A synthetic thermospheric field in log-density: exponential altitude decay
    (a scale-height trend) plus a diurnal bulge peaking mid-afternoon."""
    return -0.9 * h + 0.4 * np.cos(2 * np.pi * (lst - 0.6))


def constellation(n_sat, n_per=150, seed=0):
    """n_sat satellites spanning the altitude band, each sweeping local time."""
    alts = np.linspace(-2.0, 2.0, n_sat) if n_sat > 1 else np.array([0.0])
    hs, ls = [], []
    for j, a in enumerate(alts):
        h, l = satellite_track(a, n=n_per, seed=seed + j)
        hs.append(h); ls.append(l)
    return np.concatenate(hs), np.concatenate(ls)


NOISES = [0.02, 0.05, 0.10]


def main():
    outdir = ensure_dir(str(Path(__file__).parents[1] / "results" / "vleo"))
    n_sats = list(range(1, 8))

    print(f"{'n_sat':>5} {'cond#':>9} " +
          "  ".join(f"rec@{int(nz*100)}%" for nz in NOISES))
    conds = []
    recs = {nz: [] for nz in NOISES}
    for n in n_sats:
        h, l = constellation(n)
        cond = design_condition(h, l)
        conds.append(cond)
        row = f"{n:>5} {cond:>9.1e} "
        for nz in NOISES:
            r = recover_field(h, l, log_rho_true, rel_noise=nz, n_trials=40)
            recs[nz].append(r["support_recovery_rate"])
            row += f"   {r['support_recovery_rate']:>4.0%}"
        print(row, flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].semilogy(n_sats, conds, "o-", color="#7d3c98")
    ax[0].set_xlabel("satellites in the constellation")
    ax[0].set_ylabel("density-field design-matrix condition number")
    ax[0].set_title("One satellite can't constrain the field;\na constellation conditions it")
    ax[0].grid(alpha=0.3, which="both")

    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(NOISES)))
    for nz, c in zip(NOISES, colors):
        ax[1].plot(n_sats, recs[nz], "o-", color=c, label=f"drag noise {nz:.0%}")
    ax[1].set_xlabel("satellites in the constellation")
    ax[1].set_ylabel("field-recovery rate")
    ax[1].set_title("Recovering rho(altitude, local time)\nreturns with more satellites")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    plt.tight_layout()
    p = Path(outdir) / "vleo_restoration.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {p}")

    from perturber.report import save_metrics
    save_metrics(outdir, {"n_satellites": n_sats, "condition_numbers": [float(c) for c in conds],
                          "recovery_by_noise": {f"noise_{nz}": recs[nz] for nz in NOISES}})
    assert conds[0] > 10 * conds[-1], "constellation should drop the condition number"
    assert recs[0.05][-1] > recs[0.05][0], "more satellites should recover the field better"
    print("[vleo] identifiability demo complete")


if __name__ == "__main__":
    main()
