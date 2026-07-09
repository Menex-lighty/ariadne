"""How much of real thermospheric density variation is explained by public solar
indices? The crux of whether a solar-data-driven VLEO drag tool is viable.

Regress real GRACE orbit-mean density (log10) on F10.7 (solar radio flux) and Ap
(geomagnetic index) from the GFZ daily series, controlling for the satellite's
altitude. The incremental R^2 says how much a model with only free, public solar
inputs could capture -- and what is left (local time, storm transients, model
error) for the physics/inverse layer.

Data: data/spaceweather/kp_ap_f107.txt (GFZ, CC BY 4.0), data/stormai/sat_density
(GRACE orbit-mean density), data/stormai/initial_states (altitude). All gitignored.

Usage: python scripts/run_solar_density_variance.py
"""
import csv
import glob
import re
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
GFZ = ROOT / "data" / "spaceweather" / "kp_ap_f107.txt"
SAT = ROOT / "data" / "stormai" / "sat_density" / "sat_density"
IS = ROOT / "data" / "stormai" / "initial_states" / "initial_states"
R_E = 6378.137


def solar_indices():
    """date-ordinal -> (F10.7, F10.7_81day, Ap). F10.7 filtered to the physical
    60-300 sfu (raw file carries fill/error spikes up to ~939); F10.7_81 is the
    81-day trailing average (the standard slow-EUV density driver)."""
    from datetime import date
    daily = {}
    for line in open(GFZ):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 27:
            continue
        try:
            y, m, d = int(p[0]), int(p[1]), int(p[2])
            Ap, F107 = float(p[23]), float(p[25])
        except ValueError:
            continue
        if 60 <= F107 <= 300 and 0 <= Ap <= 400:
            daily[date(y, m, d).toordinal()] = (F107, Ap)
    out = {}
    for o in daily:
        win = [daily[oo][0] for oo in range(o - 80, o + 1) if oo in daily]
        out[o] = (daily[o][0], float(np.mean(win)) if win else daily[o][0], daily[o][1])
    return out


def grace_altitude_curve():
    """(sorted timestamps, altitudes km) for GRACE from real initial states."""
    ts, al = [], []
    for f in glob.glob(str(IS / "*.csv")):
        for r in csv.DictReader(open(f)):
            if not r.get("File ID"):
                continue
            try:
                t = datetime.strptime(r["Timestamp"], "%Y-%m-%d %H:%M:%S")
                inc = float(r["Inclination (deg)"]); sma = float(r["Semi-major Axis (km)"])
            except (ValueError, KeyError):
                continue
            if 88.5 <= inc <= 89.6 and 2002 <= t.year < 2018:
                ts.append(t.timestamp()); al.append(sma - R_E)
    o = np.argsort(ts)
    return np.array(ts)[o], np.array(al)[o]


def r2(cols, y):
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    return 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    sw = solar_indices()
    at, av = grace_altitude_curve()
    print(f"solar days: {len(sw)}, GRACE altitude states: {len(av)}")

    from datetime import date
    rows = []
    for f in glob.glob(str(SAT / "grace*.csv")):
        m = re.search(r"(\d{4})(\d{2})(\d{2})_to_", Path(f).name)
        if not m:
            continue
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        o = date(y, mo, d).toordinal()
        if o not in sw:
            continue
        rr = np.array([float(x[1]) for x in list(csv.reader(open(f)))[1:] if len(x) > 1 and x[1]])
        rr = rr[(rr > 1e-13) & (rr < 5e-11)]                 # physical thermosphere
        if len(rr) < 10:
            continue
        alt = float(np.interp(datetime(y, mo, d).timestamp(), at, av))
        F107, F81, Ap = sw[o]
        rows.append((np.log10(np.exp(np.mean(np.log(rr)))), F107, F81, Ap, alt,
                     datetime(y, mo, d).timetuple().tm_yday))
    rows = np.array(rows)
    y = rows[:, 0]; F107 = rows[:, 1]; F81 = rows[:, 2]; Ap = rows[:, 3]; alt = rows[:, 4]; doy = rows[:, 5]
    seas = [np.cos(2*np.pi*doy/365), np.sin(2*np.pi*doy/365)]
    print(f"aligned GRACE density days: {len(y)}  (2002-2016)")
    print(f"  F10.7 {F107.min():.0f}-{F107.max():.0f} sfu, Ap {Ap.min():.0f}-{Ap.max():.0f}, "
          f"alt {alt.min():.0f}-{alt.max():.0f} km")

    print("\nfraction of log-density variance explained (incremental R^2):")
    print(f"   F10.7 (daily) alone            : {r2([F107], y):.3f}")
    print(f"   F10.7 daily + 81-day average   : {r2([F107, F81], y):.3f}")
    print(f"   + Ap (geomagnetic)             : {r2([F107, F81, Ap], y):.3f}")
    print(f"   + altitude                     : {r2([F107, F81, Ap, alt], y):.3f}")
    full = r2([F107, F81, Ap, alt] + seas, y)
    print(f"   + season                       : {full:.3f}")
    print(f"\n=> public solar+geomagnetic indices (+alt/season) explain ~{full:.0%} of the")
    print(f"   orbit-mean density variation. The residual ~{1-full:.0%} (local-time sampling,")
    print(f"   storm transients, composition, model error) is where a physics/inverse layer")
    print(f"   -- and along-track (not orbit-mean) data -- must earn its keep.")

    from perturber.report import save_metrics
    _out = ROOT / "results" / "stormai"; _out.mkdir(parents=True, exist_ok=True)
    save_metrics(_out, {"n_days": int(len(y)), "r2_f107": float(r2([F107], y)),
                        "r2_f107_plus_81day": float(r2([F107, F81], y)),
                        "r2_full_incl_alt_season": float(full)}, name="solar_variance_metrics.json")
    assert r2([F107, F81], y) > 0.5, "solar flux should explain the bulk of density variance"
    assert full > r2([F107], y), "adding drivers should help"
    print("\n[solar-variance] done")


if __name__ == "__main__":
    main()
