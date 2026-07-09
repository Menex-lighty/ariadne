"""Mass identifiability on the REAL Kepler-9 system, at a fitted configuration.

The "Illusory Precision" paper (ApJ, 10.3847/1538-4357/ae74c9) shows Kepler-9's
TTV *masses* are degenerate. We test whether a pre-fit conditioning number
predicts that — the mass-space analog of the force-form condition number in
perturber.identifiability:

  (1) least-squares fit a 3-body model (star + b + c) to the observed O-C from
      Holczer et al. 2016 (fitting masses, eccentricity vectors, the c period,
      and the two phases);
  (2) at the best fit, form the TTV Jacobian d(transit time)/d(mass, ecc-vector)
      weighted by the real per-transit timing errors;
  (3) read off its conditioning — an ill-conditioned singular direction is the
      degenerate parameter combination.

Local Fisher/Jacobian conditioning captures correlations near the best fit; the
paper's "hidden solutions" are a global, multimodal degeneracy this understates.

Usage: python scripts/run_kepler9_identifiability.py
"""
import argparse
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

from perturber.data import kepler_state          # noqa: E402
from perturber.dynamics import ode_rhs           # noqa: E402
from perturber.transits import find_transit_times  # noqa: E402
from perturber.plots import ensure_dir           # noqa: E402

DAY_PER_TU = 365.25 / (2 * np.pi)
M_EARTH = 3.003e-6
DATA = Path(__file__).parents[1] / "data" / "kepler9"
P_B = 19.24628649                         # planet-b period (days), fixed timescale

# fit vector: [m_b, m_c (M_earth), h_b, k_b, h_c, k_c, P_c (d), phi_b, phi_c]
FIT_NAMES = ["m_b", "m_c", "h_b", "k_b", "h_c", "k_c", "P_c", "phi_b", "phi_c"]
THETA0 = np.array([43.0, 30.0, 0.06, 0.0, -0.07, 0.0, 38.94980851, 0.0, 0.0])
PHYS = ["m_b", "m_c", "h_b", "k_b", "h_c", "k_c"]      # params for the conditioning Jacobian
LO = [1, 1, -0.4, -0.4, -0.4, -0.4, 38.0, -np.pi, -np.pi]
HI = [200, 200, 0.4, 0.4, 0.4, 0.4, 40.0, np.pi, np.pi]
XS = np.array([40, 30, 0.05, 0.05, 0.05, 0.05, 0.1, 0.3, 0.3])

# Transit node ("in-front" sign, see perturber.transits.find_transit_times).
# best_fit() sets this to the node the data prefers; downstream analysis (scan,
# conditioning, prior demo) then uses it automatically. See docs/kepler9_node_postmortem.md.
FRONT = 1.0


def load_observed():
    """Per planet: epochs N, observed transit times (days), timing errors sigma
    (days), and observed O-C (days) detrended with our own linear ephemeris.

    In Holczer table3, `tn` is the *calculated* (linear-ephemeris) time and `O-C`
    is the timing residual in minutes, so the actual observed time is
    tn + O-C/1440. We rebuild the absolute time and detrend it the same way the
    model O-C is detrended, for a consistent comparison."""
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
    m_b, m_c, h_b, k_b, h_c, k_c, P_c, phi_b, phi_c = theta
    masses = np.array([1.0, m_b * M_EARTH, m_c * M_EARTH])
    bodies = [np.zeros(4)]
    for (P, h, k, mp, phi) in [(P_B, h_b, k_b, m_b * M_EARTH, phi_b),
                               (P_c, h_c, k_c, m_c * M_EARTH, phi_c)]:
        e = float(np.hypot(h, k))
        omega = float(np.arctan2(k, h))
        a = ((1.0 + mp) * ((P / DAY_PER_TU) / (2 * np.pi)) ** 2) ** (1.0 / 3.0)
        nu = -np.pi / 2 - omega + phi
        pos, vel = kepler_state(a, e, nu, omega, 1.0 + mp)
        bodies.append(np.concatenate([pos, vel]))
    state = np.stack(bodies)
    com = (masses[:, None] * state).sum(0) / masses.sum()
    state = state - com[None, :]
    return masses, np.concatenate([state[:, :2].ravel(), state[:, 2:].ravel()])


def model_transits(theta, n_max, front=None):
    if front is None:
        front = FRONT
    masses, s0 = setup(theta)
    n_periods = n_max + 4
    t_end = n_periods * (P_B / DAY_PER_TU)
    t = np.linspace(0, t_end, int(n_periods * 300))
    sol = solve_ivp(ode_rhs, (0, t_end), s0, t_eval=t, args=(masses,),
                    method="DOP853", rtol=1e-11, atol=1e-13)
    posT = np.moveaxis(sol.y[:6].reshape(3, 2, -1), -1, 0)
    tt_b = find_transit_times(t, posT[:, 1, :], posT[:, 0, :], front) * DAY_PER_TU
    tt_c = find_transit_times(t, posT[:, 2, :], posT[:, 0, :], front) * DAY_PER_TU
    return tt_b, tt_c


def model_oc_at(theta, obs, front=None):
    """Model O-C (days) at observed epochs, or None if too few transits."""
    tt_b, tt_c = model_transits(theta, obs["b"]["N"].max(), front)
    if len(tt_b) <= obs["b"]["N"].max() or len(tt_c) <= obs["c"]["N"].max():
        return None, None
    mb = _detrend(tt_b[obs["b"]["N"]], obs["b"]["N"])
    mc = _detrend(tt_c[obs["c"]["N"]], obs["c"]["N"])
    return mb, mc


def residuals(theta, obs, front=None):
    mb, mc = model_oc_at(theta, obs, front)
    if mb is None:
        return np.full(len(obs["b"]["N"]) + len(obs["c"]["N"]), 1e3)
    rb = (mb - obs["b"]["oc"]) / obs["b"]["sig"]
    rc = (mc - obs["c"]["oc"]) / obs["c"]["sig"]
    return np.concatenate([rb, rc])


def conditioning_at(theta, obs):
    """Mass-space TTV Jacobian conditioning at config theta. Perturbs only the
    physical params (masses + ecc vectors); periods/phases held at the fit."""
    sig = np.concatenate([obs["b"]["sig"], obs["c"]["sig"]])
    steps = np.array([0.05 * theta[0], 0.05 * theta[1], 0.01, 0.01, 0.01, 0.01])
    J = np.zeros((len(sig), 6))
    for k in range(6):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += steps[k]; tm[k] -= steps[k]
        mbp, mcp = model_oc_at(tp, obs)
        mbm, mcm = model_oc_at(tm, obs)
        J[:, k] = (np.concatenate([mbp, mcp]) - np.concatenate([mbm, mcm])) / (2 * steps[k])
    Jw = (J / sig[:, None]) * steps[None, :6]
    U, S, Vt = np.linalg.svd(Jw, full_matrices=False)
    return S, Vt, J, sig


def refit_fixed(obs, theta_start, fixed_idx, fixed_vals):
    """Least-squares refit holding some parameters fixed. Returns (chi2/dof,
    full theta). Warm-started from theta_start for speed."""
    free = [i for i in range(len(theta_start)) if i not in fixed_idx]

    def build(x):
        th = theta_start.copy()
        for j, i in enumerate(free):
            th[i] = x[j]
        for i, v in zip(fixed_idx, fixed_vals):
            th[i] = v
        return th

    res = least_squares(lambda x: residuals(build(x), obs), theta_start[free],
                        bounds=([LO[i] for i in free], [HI[i] for i in free]),
                        x_scale=[XS[i] for i in free], diff_step=0.03, max_nfev=60)
    th = build(res.x)
    return (res.fun ** 2).sum() / (len(res.fun) - len(free)), th


def degeneracy_scan(theta_fit, obs, outdir):
    """Profile chi2 while fixing m_b across a wide range, refitting all else by
    *continuation* (each step warm-started from the previous solution, so every
    refit is a small reliable move). Flat chi2 + m_c tracking m_b at fixed ratio
    = the hidden-solutions family; a genuine chi2 rise = the edge of the valley."""
    mb_vals = np.linspace(15.0, 75.0, 13)
    chi2s = np.full(len(mb_vals), np.nan)
    mc_fit = np.full(len(mb_vals), np.nan)
    i0 = int(np.argmin(np.abs(mb_vals - theta_fit[0])))
    print("\n[scan] profiling m_b by continuation from the best fit...")
    for direction in (range(i0, len(mb_vals)), range(i0 - 1, -1, -1)):
        tp = theta_fit.copy()
        for i in direction:
            c2, tp = refit_fixed(obs, tp, [0], [mb_vals[i]])
            chi2s[i] = c2; mc_fit[i] = tp[1]
            print(f"   m_b={mb_vals[i]:5.1f} -> chi2/dof {c2:8.1f}, "
                  f"m_c={tp[1]:5.1f}, ratio {mb_vals[i] / tp[1]:.3f}", flush=True)
    c2min = np.nanmin(chi2s)
    valley = chi2s < 2 * c2min                              # "acceptable fit" band

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].semilogy(mb_vals, chi2s, "o-", color="#7d3c98")
    ax[0].semilogy(mb_vals[valley], chi2s[valley], "o", color="#27ae60", ms=9,
                   label="within 2x best chi2")
    ax[0].axhline(2 * c2min, color="#c0392b", ls="--", lw=1)
    ax[0].set_xlabel("planet-b mass held fixed (M_earth)")
    ax[0].set_ylabel("refit chi2/dof (log)")
    ax[0].set_title("Fit worsens only gradually with m_b\n"
                    "(the 2x-chi2 band is ~200 sigma, not a 1-sigma valley)")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")

    ax[1].plot(mb_vals[valley], mc_fit[valley], "o", color="#27ae60", ms=8, label="valley fits")
    ax[1].plot(mb_vals, mc_fit, "-", color="#2471a3", alpha=0.5)
    ax[1].plot(mb_vals, mb_vals / (theta_fit[0] / theta_fit[1]), "--",
               color="#999", label=f"constant ratio {theta_fit[0]/theta_fit[1]:.2f}")
    ax[1].set_xlabel("planet-b mass held fixed (M_earth)")
    ax[1].set_ylabel("refit planet-c mass (M_earth)")
    ax[1].set_title("...but the ratio is constrained:\nm_c tracks m_b along a fixed ratio")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    p = Path(outdir) / "kepler9_degeneracy_scan.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    mb_lo, mb_hi = mb_vals[valley].min(), mb_vals[valley].max()
    ratio_spread = np.nanstd((mb_vals / mc_fit)[valley]) / np.nanmean((mb_vals / mc_fit)[valley])
    print(f"[scan] degenerate m_b range (chi2 < 2x best): {mb_lo:.0f}..{mb_hi:.0f} M_earth; "
          f"ratio scatter within valley {ratio_spread:.1%}")
    print(f"[scan] figure -> {p}")


def truncate_obs(obs, frac):
    """Keep the earliest `frac` of each planet's transits (a shorter baseline)."""
    o2 = {}
    for k in ("b", "c"):
        n = max(4, int(round(frac * len(obs[k]["N"]))))
        idx = np.argsort(obs[k]["N"])[:n]
        o2[k] = {f: obs[k][f][idx] for f in ("N", "obs_t", "sig")}
        o2[k]["oc"] = _detrend(o2[k]["obs_t"], o2[k]["N"])
    return o2


def prior_demo(theta, obs, outdir):
    """Where the prior actually helps: as the observed baseline shrinks, the TTVs
    constrain the mass scale ever more weakly, until one external mass prior
    dominates. Shows the data-vs-prior crossover — the prescriptive payoff."""
    from perturber.identifiability import (fisher_information, marginal_sigma,
                                           prior_precision)
    fracs = np.linspace(0.2, 1.0, 9)
    prior_sig = 0.20 * theta[0]                        # a 20% external m_b prior
    n_b, sig_d, sig_p = [], [], []
    print("\n[prior-demo] mass constraint vs observed baseline "
          "(model-optimistic Fisher; relative trend is the point):")
    for fr in fracs:
        o2 = truncate_obs(obs, fr)
        _, _, J, sg = conditioning_at(theta, o2)
        F = fisher_information(J / sg[:, None])
        sd = marginal_sigma(F)[0]
        sp = marginal_sigma(F, prior_precision(6, 0, prior_sig))[0]
        n_b.append(len(o2["b"]["N"])); sig_d.append(sd); sig_p.append(sp)
        print(f"   {fr:.0%} baseline ({len(o2['b']['N']):2d} b-transits): "
              f"m_b sigma  data-only {sd:8.2f}  |  +20% prior {sp:6.2f}  M_earth")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.semilogy(n_b, sig_d, "o-", color="#c0392b", label="TTV data only")
    ax.semilogy(n_b, sig_p, "s-", color="#27ae60", label="TTV data + 20% external m_b prior")
    ax.axhline(prior_sig, color="#999", ls="--", lw=1, label="the prior alone (20%)")
    ax.set_xlabel("planet-b transits observed (baseline)")
    ax.set_ylabel("planet-b mass 1-sigma (M_earth, log)")
    ax.set_title("Where a prior earns its keep:\nsparse early data is mass-degenerate; one prior rescues it")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    p = Path(outdir) / "kepler9_prior_demo.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[prior-demo] figure -> {p}")


MULTISTARTS = [
    np.array([43., 30., 0.06, 0.0, -0.07, 0.0, 38.9498, 0.0, 0.0]),
    np.array([31., 22., 0.04, 0.006, -0.09, 0.02, 38.947, 0.04, -0.06]),
    np.array([25., 18., 0.03, 0.02, -0.05, -0.02, 38.95, 0.5, -0.5]),
    np.array([55., 40., 0.02, -0.02, -0.10, 0.0, 38.94, -0.5, 0.5]),
]


def best_fit(obs, verbose=True):
    """Multistart least-squares over BOTH transit nodes (front = +/-1). Kepler-9
    is exactly edge-on, so both nodes are geometrically valid transits; the data
    decides which is Earth's geometry via which fits the O-C. Returns
    (theta, chi2/dof, front) and sets the module-level FRONT to the winner.
    (The earlier single cold-start fit locked to the worse node — see
    docs/kepler9_node_postmortem.md.)"""
    global FRONT
    overall = None
    for front in (1.0, -1.0):
        node = None
        for s0 in MULTISTARTS:
            res = least_squares(lambda x: residuals(x, obs, front), s0, bounds=(LO, HI),
                                x_scale=XS, diff_step=0.03, max_nfev=150)
            c2 = (res.fun ** 2).sum() / (len(res.fun) - len(s0))
            if node is None or c2 < node[1]:
                node = (res.x, c2)
        if verbose:
            print(f"   node front={front:+.0f}: best chi2/dof {node[1]:8.1f}  "
                  f"(m_b {node[0][0]:.1f}, m_c {node[0][1]:.1f}, ratio {node[0][0]/node[0][1]:.3f})",
                  flush=True)
        if overall is None or node[1] < overall[1]:
            overall = (node[0], node[1], front)
    FRONT = overall[2]
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true",
                    help="also run the global mass-degeneracy profile scan")
    ap.add_argument("--prior-demo", action="store_true",
                    help="also run the baseline-vs-prior crossover demo")
    args = ap.parse_args()
    outdir = ensure_dir(str(Path(__file__).parents[1] / "results" / "kepler9"))
    obs = load_observed()
    print(f"observed: b {len(obs['b']['N'])} transits, c {len(obs['c']['N'])}; "
          f"O-C ptp (min): b {obs['b']['oc'].ptp()*1440:.0f}, c {obs['c']['oc'].ptp()*1440:.0f}")

    print("multistart fit over both transit nodes...", flush=True)
    theta, chi2, front = best_fit(obs)
    print(f"best fit: chi2/dof {chi2:.2f} at node front={front:+.0f}")
    print("fitted parameters:")
    for name, v in zip(FIT_NAMES, theta):
        unit = " M_earth" if name in ("m_b", "m_c") else (" d" if name == "P_c" else "")
        print(f"   {name:6s} {v:+.4f}{unit}")
    e_b, e_c = np.hypot(theta[2], theta[3]), np.hypot(theta[4], theta[5])
    print(f"   -> e_b {e_b:.3f}, e_c {e_c:.3f}, mass ratio m_b/m_c {theta[0]/theta[1]:.3f}")

    S, Vt, Jphys, sig = conditioning_at(theta, obs)
    cond = S[0] / S[-1]
    print(f"\nsingular values at fit: {np.array2string(S, precision=2)}")
    print(f"condition number: {cond:.1e}")
    print("least-constrained (degenerate) direction:")
    for name, w in sorted(zip(PHYS, Vt[-1]), key=lambda x: -abs(x[1])):
        print(f"   {name:5s} {w:+.3f}")

    # --- prior-aware analysis: what does an external prior buy? ---
    from perturber.identifiability import (fisher_information, marginal_sigma,
                                           prior_precision, prescribe_prior)
    Jwphys = Jphys / sig[:, None]                       # physical, noise-weighted
    F = fisher_information(Jwphys)
    base = marginal_sigma(F)
    p_mb = marginal_sigma(F, prior_precision(6, 0, 0.20 * theta[0]))   # 20% prior on m_b
    print("\nprior-aware: individual-mass 1-sigma (M_earth), data pins the ratio not the scale")
    print(f"   TTV data only:          m_b +-{base[0]:8.1f}, m_c +-{base[1]:8.1f}")
    print(f"   + 20% external m_b prior: m_b +-{p_mb[0]:8.2f}, m_c +-{p_mb[1]:8.2f}  "
          f"(e.g. one RV or mass-radius measurement)")
    _, rows = prescribe_prior(F, PHYS, theta[:6], rel_sigma=0.20)
    gains = sorted(((n, base[0] / s[0]) for n, s in rows), key=lambda x: -x[1])
    print("   prior that most tightens m_b (gain factor): " +
          ", ".join(f"{n} {g:.0f}x" for n, g in gains[:3]))

    # ── figures ──
    mb, mc = model_oc_at(theta, obs)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].errorbar(obs["b"]["N"], obs["b"]["oc"] * 1440, yerr=obs["b"]["sig"] * 1440,
                   fmt="o", ms=3, color="#c0392b", alpha=0.6, label="b observed")
    ax[0].plot(obs["b"]["N"], mb * 1440, "-", color="#c0392b", label="b model (fit)")
    ax[0].errorbar(obs["c"]["N"], obs["c"]["oc"] * 1440, yerr=obs["c"]["sig"] * 1440,
                   fmt="s", ms=3, color="#2471a3", alpha=0.6, label="c observed")
    ax[0].plot(obs["c"]["N"], mc * 1440, "-", color="#2471a3", label="c model (fit)")
    ax[0].set_xlabel("transit number N"); ax[0].set_ylabel("O-C (min)")
    ax[0].set_title(f"Fit to real Kepler-9 TTVs (chi2/dof {chi2:.1f})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].semilogy(range(1, len(S) + 1), S, "o-", color="#7d3c98")
    ax[1].set_xlabel("singular value index"); ax[1].set_ylabel("singular value")
    ax[1].set_title(f"TTV Jacobian at best fit — cond# {cond:.0e}")
    ax[1].grid(alpha=0.3, which="both")

    x = np.arange(6)
    ax[2].bar(x - 0.2, Vt[0], 0.4, color="#27ae60", label=f"best-constrained (s={S[0]:.0f})")
    ax[2].bar(x + 0.2, Vt[-1], 0.4, color="#c0392b", label=f"degenerate (s={S[-1]:.1e})")
    ax[2].set_xticks(x, PHYS); ax[2].axhline(0, color="#999", lw=0.8)
    ax[2].set_ylabel("parameter weight")
    ax[2].set_title("Constrained vs degenerate directions"); ax[2].legend(fontsize=8)
    plt.tight_layout()
    p = Path(outdir) / "kepler9_identifiability.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {p}")

    assert chi2 < 400, "multistart both-node fit should reach the better node (~180)"
    assert cond > 10, "expected a non-trivial conditioning number"
    print("[kepler9] fitted analysis complete")

    if args.scan:
        degeneracy_scan(theta, obs, outdir)
    if args.prior_demo:
        prior_demo(theta, obs, outdir)


if __name__ == "__main__":
    main()
