"""Force-form identifiability: the degeneracy metric and what restores it.

Sweeps observed radial coverage two ways — a single planet of increasing
eccentricity, and an increasing number of bodies spanning a range of
semi-major axes (the M3 mechanism) — and measures, at several force-noise
levels:

  * the analytic predictor: condition number of the force-law design matrix
    over the sampled radii;
  * the outcome: how often sparse regression recovers the correct functional
    form (support-recovery rate) from noisy force samples.

Produces results/identifiability/restoration.png and a summary table. Pure
numpy (best-case, no ODE fit) — the companion run_identifiability.py confirms
the same wall through the full differentiable trajectory inversion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from perturber.identifiability import (  # noqa: E402
    condition_number, recover_form, orbit_radii, multi_body_radii)
from perturber.plots import ensure_dir   # noqa: E402


def g_true(r):
    return 2e-3 / r ** 4                  # injected 1/r^4 law


NOISES = [0.02, 0.05, 0.10]


def single_planet_series(eccs, noise):
    rows = []
    for e in eccs:
        r = orbit_radii(1.0, e)
        out = recover_form(r, g_true, rel_noise=noise, n_trials=60)
        rows.append((r.max() / r.min(), out["condition_number"],
                     out["support_recovery_rate"]))
    return np.array(rows)


def multibody_series(n_bodies, noise):
    # bodies spread over a in [0.6, 3.0], mild eccentricity each
    rows = []
    for nb in n_bodies:
        a = np.linspace(0.6, 3.0, nb)
        e = np.full(nb, 0.2)
        r = multi_body_radii(a, e)
        out = recover_form(r, g_true, rel_noise=noise, n_trials=60)
        rows.append((r.max() / r.min(), out["condition_number"],
                     out["support_recovery_rate"], nb))
    return np.array(rows)


def main():
    outdir = ensure_dir(str(Path(__file__).parents[1] / "results" / "identifiability"))
    eccs = [0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6]
    n_bodies = [1, 2, 3, 4, 5, 6]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(NOISES)))

    print(f"{'noise':>6} {'coverage':>9} {'cond#':>10} {'form-recovery':>14}")
    all_cond, all_rec = [], []
    for noise, col in zip(NOISES, colors):
        sp = single_planet_series(eccs, noise)
        mb = multibody_series(n_bodies, noise)
        ax[0].plot(sp[:, 0], sp[:, 1], "o-", color=col, label=f"σ_f={noise:.0%}")
        ax[0].plot(mb[:, 0], mb[:, 1], "s--", color=col, alpha=0.7)
        ax[1].plot(sp[:, 0], sp[:, 2], "o-", color=col, label=f"σ_f={noise:.0%} (1 planet, vary e)")
        ax[1].plot(mb[:, 0], mb[:, 2], "s--", color=col, alpha=0.7,
                   label=f"σ_f={noise:.0%} (N bodies)")
        all_cond.append(np.concatenate([sp[:, 1], mb[:, 1]]))
        all_rec.append(np.concatenate([sp[:, 2], mb[:, 2]]))
        for row in mb:
            print(f"{noise:>6.0%} {row[0]:>9.1f} {row[1]:>10.1e} "
                  f"{row[2]:>13.0%}  (N={int(row[3])} bodies)")

    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("radial coverage  r_max / r_min")
    ax[0].set_ylabel("design-matrix condition number")
    ax[0].set_title("Degeneracy metric falls as coverage widens\n"
                    "(circle=1 planet vary e,  square=N bodies)")
    ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8)

    ax[1].set_xscale("log")
    ax[1].set_xlabel("radial coverage  r_max / r_min")
    ax[1].set_ylabel("form-recovery rate")
    ax[1].set_title("Recovering the FORM returns with coverage")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)

    cond = np.concatenate(all_cond); rec = np.concatenate(all_rec)
    ax[2].scatter(cond, rec, c="#c0392b", alpha=0.7, edgecolor="k", linewidth=0.4)
    ax[2].set_xscale("log")
    ax[2].axvspan(1e5, cond.max() * 2, alpha=0.08, color="red")
    ax[2].set_xlabel("design-matrix condition number")
    ax[2].set_ylabel("form-recovery rate")
    ax[2].set_title("The predictor: recovery collapses above\n"
                    "cond# ~ 1e5 (shaded = degenerate regime)")
    ax[2].grid(alpha=0.3, which="both")

    plt.tight_layout()
    p = Path(outdir) / "restoration.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {p}")

    # headline claims, asserted
    r_circ = orbit_radii(1.0, 0.03)
    r_multi = multi_body_radii(np.linspace(0.6, 3.0, 6), np.full(6, 0.2))
    assert condition_number(r_circ) > 1e8, "near-circular should be degenerate"
    assert condition_number(r_multi) < 1e5, "multi-body should be well-conditioned"
    assert recover_form(r_multi, g_true, rel_noise=0.05)["support_recovery_rate"] > 0.5
    print("[identifiability-study] assertions passed")


if __name__ == "__main__":
    main()
