"""Kepler-9 3-D photodynamical fit — does out-of-plane (inclination/node) freedom,
and better transit numerics, pull the 2-D chi2/dof (~600) toward 1?

Extends the 2-D fit (scripts/run_kepler9_identifiability.py) with inclinations
i_b, i_c and the relative node dOmega, using the 3-D N-body + parabolic transit
finder in perturber.threed. Checkpoint: at i=90 deg, dOmega=0 the model must
reproduce the 2-D transit times, so any chi2 change there is numerics, and any
further improvement with inclination is genuine 3-D structure.

Usage: python scripts/run_kepler9_3d.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from scipy.integrate import solve_ivp    # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from perturber.threed import (elements_to_state_3d, ode_rhs_3d, flat_3d,   # noqa: E402
                              find_transit_times_3d)
from perturber.plots import ensure_dir   # noqa: E402

DAY_PER_TU = 365.25 / (2 * np.pi)
M_EARTH = 3.003e-6
DATA = Path(__file__).parents[1] / "data" / "kepler9"
P_B = 19.24628649

# theta: m_b, m_c, h_b, k_b, h_c, k_c, P_c, phi_b, phi_c, i_b, i_c, dOmega
NAMES = ["m_b", "m_c", "h_b", "k_b", "h_c", "k_c", "P_c", "phi_b", "phi_c",
         "i_b", "i_c", "dOmega"]
# start = 2-D best fit, edge-on and coplanar (i=90 deg, dOmega=0)
THETA0 = np.array([35.3744, 25.2023, 0.0374, 0.0065, -0.0913, 0.0219, 38.9474,
                   0.0438, -0.0643, np.pi / 2, np.pi / 2, 0.0])
LO = [1, 1, -.4, -.4, -.4, -.4, 38.0, -np.pi, -np.pi, np.radians(75), np.radians(75), -0.7]
HI = [200, 200, .4, .4, .4, .4, 40.0, np.pi, np.pi, np.radians(105), np.radians(105), 0.7]
XS = np.array([40, 30, .05, .05, .05, .05, .1, .3, .3, .05, .05, .1])


def load_observed():
    out = {"b": {"N": [], "obs_t": [], "sig": []}, "c": {"N": [], "obs_t": [], "sig": []}}
    with open(DATA / "holczer2016_ttv.csv") as f:
        for row in csv.DictReader(f):
            if not row.get("KOI"):
                continue
            key = "b" if row["KOI"].strip() == "377.01" else "c"
            out[key]["N"].append(int(row["N"]))
            out[key]["obs_t"].append(float(row["tn"]) + float(row["O-C"]) / 1440.0)
            out[key]["sig"].append(float(row["e_O-C"]) / 1440.0)
    for k in out:
        for f_ in ("N", "obs_t", "sig"):
            out[k][f_] = np.array(out[k][f_], dtype=float)
        out[k]["N"] = out[k]["N"].astype(int)
        out[k]["oc"] = _detrend(out[k]["obs_t"], out[k]["N"])
    return out


def _detrend(times, N):
    A = np.vstack([N, np.ones_like(N, dtype=float)]).T
    coef, *_ = np.linalg.lstsq(A, times, rcond=None)
    return times - A @ coef


def setup(theta):
    (m_b, m_c, h_b, k_b, h_c, k_c, P_c, phi_b, phi_c, i_b, i_c, dOm) = theta
    masses = np.array([1.0, m_b * M_EARTH, m_c * M_EARTH])
    rows = [np.zeros(6)]
    specs = [(P_B, h_b, k_b, m_b * M_EARTH, phi_b, i_b, 0.0),
             (P_c, h_c, k_c, m_c * M_EARTH, phi_c, i_c, dOm)]
    for (P, h, k, mp, phi, inc, Om) in specs:
        e = float(np.hypot(h, k)); om = float(np.arctan2(k, h))
        a = ((1.0 + mp) * ((P / DAY_PER_TU) / (2 * np.pi)) ** 2) ** (1.0 / 3.0)
        nu = np.pi / 2 - om + phi        # near transit (planet in front) at t=0
        pos, vel = elements_to_state_3d(a, e, inc, Om, om, nu, 1.0 + mp)
        r = np.zeros(6); r[:3] = pos; r[3:] = vel
        rows.append(r)
    state = np.stack(rows)
    com = (masses[:, None] * state).sum(0) / masses.sum()
    state = state - com[None, :]
    return masses, flat_3d(state)


def model_transits(theta, n_max, npts_per=400):
    masses, s0 = setup(theta)
    n_periods = n_max + 4
    t_end = n_periods * (P_B / DAY_PER_TU)
    t = np.linspace(0, t_end, int(n_periods * npts_per))
    sol = solve_ivp(ode_rhs_3d, (0, t_end), s0, t_eval=t, args=(masses,),
                    method="DOP853", rtol=1e-11, atol=1e-13)
    posT = np.moveaxis(sol.y[:9].reshape(3, 3, -1), -1, 0)   # (T,3,3)
    tb = find_transit_times_3d(t, posT[:, 1, :], posT[:, 0, :]) * DAY_PER_TU
    tc = find_transit_times_3d(t, posT[:, 2, :], posT[:, 0, :]) * DAY_PER_TU
    return tb, tc


def model_oc_at(theta, obs, npts_per=400):
    tb, tc = model_transits(theta, obs["b"]["N"].max(), npts_per)
    if len(tb) <= obs["b"]["N"].max() or len(tc) <= obs["c"]["N"].max():
        return None, None
    return (_detrend(tb[obs["b"]["N"]], obs["b"]["N"]),
            _detrend(tc[obs["c"]["N"]], obs["c"]["N"]))


def residuals(theta, obs, npts_per=400):
    mb, mc = model_oc_at(theta, obs, npts_per)
    if mb is None:
        return np.full(len(obs["b"]["N"]) + len(obs["c"]["N"]), 1e3)
    return np.concatenate([(mb - obs["b"]["oc"]) / obs["b"]["sig"],
                           (mc - obs["c"]["oc"]) / obs["c"]["sig"]])


def chi2dof(theta, obs, npts_per=400):
    r = residuals(theta, obs, npts_per)
    return (r ** 2).sum() / (len(r) - len(theta))


def fit_free(obs, theta0, free_idx, max_nfev=200):
    """Least-squares fit varying only theta[free_idx], others held at theta0."""
    def resid(x):
        th = theta0.copy(); th[free_idx] = x
        return residuals(th, obs)
    res = least_squares(resid, theta0[free_idx],
                        bounds=([LO[i] for i in free_idx], [HI[i] for i in free_idx]),
                        x_scale=[XS[i] for i in free_idx], diff_step=0.02, max_nfev=max_nfev)
    th = theta0.copy(); th[free_idx] = res.x
    c2 = (res.fun ** 2).sum() / (len(res.fun) - len(free_idx))
    return th, c2, res.nfev


def main():
    outdir = ensure_dir(str(Path(__file__).parents[1] / "results" / "kepler9"))
    obs = load_observed()
    print(f"observed transits: b {len(obs['b']['N'])}, c {len(obs['c']['N'])}")

    # --- Fit A: COPLANAR (inclinations fixed edge-on) — reproduces the 2-D
    # physics with the 3-D machinery, and is the baseline to beat. ---
    print("\n[A] coplanar fit (i fixed 90 deg)...", flush=True)
    theta_cop, c2_cop, n1 = fit_free(obs, THETA0, list(range(9)), max_nfev=250)
    print(f"    coplanar chi2/dof: {c2_cop:.2f} (nfev {n1}); "
          f"m_b {theta_cop[0]:.1f}, m_c {theta_cop[1]:.1f}, ratio {theta_cop[0]/theta_cop[1]:.3f}")

    # --- Fit B: FULL 3-D (free inclinations + node), warm-started from A ---
    print("\n[B] 3-D fit (free i_b, i_c, dOmega)...", flush=True)
    theta, c2, n2 = fit_free(obs, theta_cop, list(range(12)), max_nfev=300)
    mut_incl = np.degrees(abs(theta[9] - theta[10]))
    print(f"    3-D chi2/dof: {c2:.2f} (nfev {n2})")
    print(f"    i_b {np.degrees(theta[9]):.2f}, i_c {np.degrees(theta[10]):.2f}, "
          f"dOmega {np.degrees(theta[11]):.2f} deg -> mutual inclination ~ {mut_incl:.2f} deg")
    print(f"    masses: m_b {theta[0]:.1f}, m_c {theta[1]:.1f} M_earth, ratio {theta[0]/theta[1]:.3f}")
    print(f"\n[verdict] coplanar {c2_cop:.1f} -> 3-D {c2:.1f}  "
          f"({'3-D improves' if c2 < 0.8 * c2_cop else 'no meaningful 3-D gain'})")

    mb, mc = model_oc_at(theta, obs)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(obs["b"]["N"], obs["b"]["oc"] * 1440, yerr=obs["b"]["sig"] * 1440,
                fmt="o", ms=3, color="#c0392b", alpha=0.5, label="b obs")
    ax.plot(obs["b"]["N"], mb * 1440, "-", color="#c0392b", label="b 3-D fit")
    ax.errorbar(obs["c"]["N"], obs["c"]["oc"] * 1440, yerr=obs["c"]["sig"] * 1440,
                fmt="s", ms=3, color="#2471a3", alpha=0.5, label="c obs")
    ax.plot(obs["c"]["N"], mc * 1440, "-", color="#2471a3", label="c 3-D fit")
    ax.set_xlabel("transit number N"); ax.set_ylabel("O-C (min)")
    ax.set_title(f"Kepler-9 3-D fit (chi2/dof {c2:.1f}, mutual incl {mut_incl:.1f} deg)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = Path(outdir) / "kepler9_3d_fit.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"figure -> {p}")
    print("[kepler9-3d] done")


if __name__ == "__main__":
    main()
