"""M3: identifiability -> prediction. Forecast a satellite's FUTURE orbital decay
from PAST data only, and show what bounds the horizon.

The drag decay obeys da/dt = -B*rho*sqrt(mu*a). A single satellite measures only
the PRODUCT B*rho (density and ballistic coefficient are degenerate) -- but that
product is exactly what its own future decay depends on. So self-prediction needs
only the identifiable quantity; you never have to separate rho from B.

Method: at each epoch t0 estimate B*rho from the trailing `train` days of CHAMP's
real altitude (data/stormai/initial_states), then integrate the decay forward and
compare to what actually happened. What limits the horizon is not identifiability
(B*rho is measured fine) but that B*rho DRIFTS with unforecast solar activity --
the driver-forecasting wall. We show the forecast beats naive baselines at short
range and degrades as the solar-driven density changes.

Usage: python scripts/run_stormai_prediction.py
"""
import csv
import glob
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

ROOT = Path(__file__).parents[1]
IS = ROOT / "data" / "stormai" / "initial_states" / "initial_states"
MU = 3.986004418e14
R_E = 6378137.0
DAY = 86400.0


def champ_monthly():
    """CHAMP semi-major axis, monthly medians (m) vs time (s). Monthly medians
    beat down the ~1.3 km osculating-element scatter to expose the secular decay."""
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
    ts, a = np.array(ts), np.array(a)
    o = np.argsort(ts); ts, a = ts[o], a[o]
    month = np.floor((ts - ts[0]) / (30.4 * DAY)).astype(int)
    mt, ma = [], []
    for m in sorted(set(month)):
        sel = month == m
        if sel.sum() >= 5:
            mt.append(np.median(ts[sel])); ma.append(np.median(a[sel]))
    return np.array(mt), np.array(ma)


def integrate_decay(a0, Brho, t_days):
    """Integrate da/dt = -Brho*sqrt(mu*a) forward. Returns a at each t_days (RK4)."""
    a = a0; out = []; prev = 0.0
    for td in t_days:
        dt = (td - prev) * DAY
        n = max(1, int(dt / DAY))
        h = dt / n
        for _ in range(n):
            k1 = -Brho * np.sqrt(MU * a)
            k2 = -Brho * np.sqrt(MU * max(a + 0.5*h*k1, 1.0))
            k3 = -Brho * np.sqrt(MU * max(a + 0.5*h*k2, 1.0))
            k4 = -Brho * np.sqrt(MU * max(a + h*k3, 1.0))
            a = a + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        out.append(a); prev = td
    return np.array(out)


def main():
    outdir = ROOT / "results" / "stormai"; outdir.mkdir(parents=True, exist_ok=True)
    mt, ma = champ_monthly()
    yr = (mt - mt[0]) / (365.25 * DAY)
    print(f"CHAMP monthly points: {len(ma)}, altitude {ma.min()/1e3-6378:.0f}-{ma.max()/1e3-6378:.0f} km")

    train_m, horizons = 6, [1, 2, 3, 6]           # months
    err_model = {h: [] for h in horizons}
    err_frozen = {h: [] for h in horizons}        # naive: altitude unchanged
    example = None
    for i in range(len(mt)):
        t0 = mt[i]
        tr = (mt >= t0 - train_m*30.4*DAY) & (mt <= t0)
        if tr.sum() < 4:
            continue
        # estimate B*rho from the training decay rate
        p = np.polyfit(mt[tr], ma[tr], 1); dadt = p[0]; a0 = ma[tr][-1]
        if dadt >= 0:
            continue
        Brho = -dadt / np.sqrt(MU * a0)
        for h in horizons:
            tf = t0 + h*30.4*DAY
            j = np.argmin(np.abs(mt - tf))
            if abs(mt[j] - tf) > 20*DAY or mt[j] <= t0:
                continue
            a_pred = integrate_decay(a0, Brho, [(mt[j]-t0)/DAY])[0]
            err_model[h].append(abs(a_pred - ma[j]) / 1e3)
            err_frozen[h].append(abs(a0 - ma[j]) / 1e3)
        if example is None and yr[i] > 3 and (mt <= t0 + 6*30.4*DAY).any():
            fut = (mt > t0) & (mt <= t0 + 7*30.4*DAY)
            if fut.sum() >= 3:
                tdays = (mt[fut] - t0) / DAY
                example = (yr[i], (mt[fut]-mt[0])/(365.25*DAY), ma[fut]/1e3-6378,
                           integrate_decay(a0, Brho, tdays)/1e3-6378, a0/1e3-6378, yr[tr])

    print("\nforecast error (km altitude) vs horizon -- drag-model vs frozen-altitude:")
    for h in horizons:
        em, ef = np.median(err_model[h]), np.median(err_frozen[h])
        print(f"   {h} month(s): model {em:5.2f} km   frozen {ef:6.2f} km   (n={len(err_model[h])})")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    if example:
        y0, tf, af, ap, a_start, _ = example
        ax[0].plot(tf, af, "o-", color="#2471a3", label="actual (held-out)")
        ax[0].plot(tf, ap, "s--", color="#c0392b", label="forecast from past B*rho")
        ax[0].axhline(a_start, color="#999", ls=":", lw=1, label="frozen-altitude naive")
        ax[0].set_xlabel("year"); ax[0].set_ylabel("altitude (km)")
        ax[0].set_title(f"CHAMP decay forecast (train {train_m} mo, from year {y0:.1f})")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    hs = horizons
    ax[1].plot(hs, [np.median(err_model[h]) for h in hs], "o-", color="#c0392b", label="drag model (identifiable B*rho)")
    ax[1].plot(hs, [np.median(err_frozen[h]) for h in hs], "s--", color="#999", label="frozen altitude")
    ax[1].set_xlabel("forecast horizon (months)"); ax[1].set_ylabel("median altitude error (km)")
    ax[1].set_title("Horizon is bounded by driver drift, not identifiability")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3); ax[1].set_xticks(hs)
    plt.tight_layout()
    p = outdir / "stormai_prediction.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {p}")

    from perturber.report import save_metrics
    save_metrics(outdir, {"horizons_months": horizons,
                          "median_err_km_model": {str(h): float(np.median(err_model[h])) for h in horizons},
                          "median_err_km_frozen": {str(h): float(np.median(err_frozen[h])) for h in horizons}})
    assert np.median(err_model[3]) < np.median(err_frozen[3]), "drag model should beat frozen altitude"
    assert np.median(err_model[1]) < 3.0, "1-month decay forecast should be within a few km"
    print("[stormai-prediction] self-decay forecast from identifiable B*rho; "
          "horizon limited by unforecast solar activity, not identifiability")


if __name__ == "__main__":
    main()
