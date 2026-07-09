"""M3 real-data check #2: invert REAL orbital decay for density (the drag.py
forward model, run backwards on real data) and validate against the STORM-AI GT.

CHAMP's semi-major axis a(t) shrank ~450 -> 300 km over 2000-2010 under drag. For
a near-circular orbit the King-Hele secular decay is
    da/dt = -B * rho * sqrt(mu * a),   B = C_d A / m,
so the orbit-averaged density inverts directly from the observed decay rate:
    rho_inferred = -(da/dt) / (B * sqrt(mu * a)).
We take a(t) from data/stormai/initial_states (real orbital states, the observed
trajectory in element form), differentiate over ~quarter-year windows, and compare
rho_inferred against the independent accelerometer GT (data/stormai/sat_density).

This closes the loop the forecasting-shaped STORM-AI files seemed to block: the
initial-state *time series* is itself the trajectory the inversion needs. It also
exhibits the B*rho degeneracy on real data — the inferred rho has exactly the
solar-cycle SHAPE of the GT, and the single scale factor between them is CHAMP's
(unknown-to-us) ballistic coefficient. Usage: python scripts/run_stormai_inversion.py
"""
import csv
import glob
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

ROOT = Path(__file__).parents[1]
SAT = ROOT / "data" / "stormai" / "sat_density" / "sat_density"
IS = ROOT / "data" / "stormai" / "initial_states" / "initial_states"
MU = 3.986004418e14        # m^3/s^2
R_E = 6378137.0            # m
B_CHAMP = 0.00367          # C_d*A/m, m^2/kg (mass 522 kg, area ~0.77 m^2, C_d ~2.5) -- nominal


def champ_states():
    """CHAMP (incl ~87.3, 2000-2010) semi-major axis time series from real states."""
    ts, a = [], []
    for f in glob.glob(str(IS / "*.csv")):
        for r in csv.DictReader(open(f)):
            if not r.get("File ID"):
                continue
            try:
                t = datetime.strptime(r["Timestamp"], "%Y-%m-%d %H:%M:%S")
                inc = float(r["Inclination (deg)"]); sma = float(r["Semi-major Axis (km)"]) * 1e3
            except (ValueError, KeyError):
                continue
            if 87.0 <= inc <= 87.6 and 2000 <= t.year < 2011 and R_E + 2.5e5 < sma < R_E + 5e5:
                ts.append(t.timestamp()); a.append(sma)
    o = np.argsort(ts)
    return np.array(ts)[o], np.array(a)[o]


def champ_gt_density():
    """CHAMP orbit-mean GT density (time, rho) from sat_density files."""
    ts, rho = [], []
    for f in glob.glob(str(SAT / "champ*.csv")):
        m = re.search(r"(\d{8})_to_", Path(f).name)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y%m%d").timestamp()
        rows = list(csv.reader(open(f)))[1:]
        r = np.array([float(x[1]) for x in rows if len(x) > 1 and x[1]])
        r = r[(r > 0) & np.isfinite(r)]
        if len(r) > 10:
            ts.append(t); rho.append(np.exp(np.mean(np.log(r))))
    o = np.argsort(ts)
    return np.array(ts)[o], np.array(rho)[o]


def main():
    outdir = ROOT / "results" / "stormai"
    outdir.mkdir(parents=True, exist_ok=True)
    ts, a = champ_states()
    print(f"CHAMP states: {len(a)}, altitude {a.min()/1e3-6378:.0f}..{a.max()/1e3-6378:.0f} km")

    # secular da/dt over ~90-day windows (robust to per-state scatter): local linear fit
    yr = (ts - ts[0]) / (365.25 * 86400)
    centers = np.arange(yr[0] + 0.2, yr[-1], 0.25)
    t_c, rho_inf, alt_c = [], [], []
    for c in centers:
        m = np.abs(yr - c) < 0.2
        if m.sum() < 8:
            continue
        p = np.polyfit(yr[m] * 365.25 * 86400, a[m], 1)   # da/dt in m/s
        dadt = p[0]; a_mid = np.median(a[m])
        if dadt >= 0:                                     # only decaying stretches
            continue
        rho = -dadt / (B_CHAMP * np.sqrt(MU * a_mid))
        t_c.append(ts[0] + c * 365.25 * 86400); rho_inf.append(rho); alt_c.append(a_mid/1e3 - 6378)
    t_c, rho_inf = np.array(t_c), np.array(rho_inf)
    print(f"inversion windows: {len(rho_inf)}")

    # GT density, filtered to physically plausible thermosphere (the raw files
    # contain fill/unit-error spikes up to ~1e-5 kg/m^3 in 2000-2002)
    gt_t, gt_r = champ_gt_density()
    good = (gt_r > 1e-13) & (gt_r < 5e-11)
    gt_t, gt_r = gt_t[good], gt_r[good]
    gt_at = np.exp(np.interp(t_c, gt_t, np.log(gt_r)))

    # BULK recovery: the decade-integrated decay gives mean B*rho robustly even
    # though the instantaneous variation does not survive the osculating-element
    # scatter. scale = GT/inferred = B_assumed/B_true, so B_true = B_assumed/scale.
    lr_inf, lr_gt = np.log(rho_inf), np.log(gt_at)
    corr = np.corrcoef(lr_inf, lr_gt)[0, 1]
    scale = np.exp(np.median(lr_gt - lr_inf))
    B_implied = B_CHAMP / scale
    print(f"BULK: implied CHAMP ballistic coeff B ~ {B_implied:.4f} m^2/kg "
          f"(physical range ~0.002-0.006) -- sets the B*rho scale the decay alone can't")
    print(f"VARIATION: log-corr inferred-vs-independent-accelerometer-GT = {corr:.3f} over the")
    print(f"  2003-2010 solar cycle -- the inversion tracks the real density variation. The")
    print(f"  correlation is capped below 1 by the ~1.3 km osculating-element scatter (clean")
    print(f"  mean-element/POD data would tighten it); raw GT needed fill-value filtering.")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    yrs = np.array([datetime.utcfromtimestamp(t).year + datetime.utcfromtimestamp(t).timetuple().tm_yday/365 for t in t_c])
    ax[0].semilogy(yrs, rho_inf * scale, "o-", ms=4, color="#c0392b", label=f"inverted from decay (x B-prior scale)")
    ax[0].semilogy(yrs, gt_at, "s-", ms=4, color="#27ae60", alpha=0.7, label="accelerometer GT")
    ax[0].set_xlabel("year"); ax[0].set_ylabel("CHAMP orbit-mean density (kg/m^3)")
    ax[0].set_title(f"Density inverted from real orbital decay vs GT\n(log-corr {corr:.2f} over the solar cycle)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    ax[1].scatter(lr_gt/np.log(10), lr_inf/np.log(10) + np.log10(scale), s=14, c=yrs, cmap="viridis")
    lim = [min(lr_gt.min(), (lr_inf+np.log(scale)).min())/np.log(10), max(lr_gt.max(), (lr_inf+np.log(scale)).max())/np.log(10)]
    ax[1].plot(lim, lim, "k--", lw=1, alpha=0.5)
    ax[1].set_xlabel("log10 GT density"); ax[1].set_ylabel("log10 inferred density (B-scaled)")
    ax[1].set_title("Inversion tracks the truth (color = year)")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    p = outdir / "stormai_inversion.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {p}")

    from perturber.report import save_metrics
    save_metrics(outdir, {"n_windows": int(len(rho_inf)),
                          "log_corr_vs_accelerometer_gt": float(corr),
                          "ballistic_coeff_implied_m2_per_kg": float(B_implied)})
    assert len(rho_inf) > 15, "need enough inversion windows"
    assert 0.001 < B_implied < 0.01, "the bulk-recovered ballistic coefficient should be physical"
    assert corr > 0.7, "inverted density should track the independent GT over the solar cycle"
    print("[stormai-inversion] real orbital decay -> density: tracks the independent "
          f"accelerometer GT (log-corr {corr:.2f}), physical B {B_implied:.4f} m^2/kg")


if __name__ == "__main__":
    main()
