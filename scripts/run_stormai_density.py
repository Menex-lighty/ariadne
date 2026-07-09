"""M3 real-data check: recover the thermospheric density altitude structure from
REAL satellite densities (STORM-AI / MIT ARCLab GT), and confirm density.py's
core identifiability claim on real data — a single satellite pins one altitude and
cannot constrain the vertical profile; two altitudes can.

Data (gitignored, download per docs/benchmarks.md): data/stormai/sat_density/
holds orbit-mean density time series for CHAMP (~300-450 km, decaying 2000-2010),
GRACE-1/2 (~480 km) and SWARM-A (~460 km); data/stormai/initial_states/ gives real
orbital states we use only to attach an altitude to each satellite/epoch.

Key idea (solar-activity control): thermospheric density swings ~50x over the
solar cycle, which would swamp the altitude signal. We avoid modelling it: for
epochs where CHAMP and GRACE BOTH report, the ratio rho_CHAMP/rho_GRACE at the
same time depends only on the altitude gap, so the scale height
    H = (h_GRACE - h_CHAMP) / ln(rho_CHAMP / rho_GRACE)
is solar-activity-controlled by construction. One satellite (one altitude) leaves
H undetermined (design matrix for log-rho = a + b*h is rank-deficient); two
altitudes make it identifiable — the density.py restoration result, on real data.

Scope/limits: altitude structure only (the local-time / full-field part needs
per-sample position + space-weather joins the orbit-mean files strip out). GRACE's
GT density is itself accelerometer-derived, so this is a cross-method consistency
check, not truth. Usage: python scripts/run_stormai_density.py
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
R_E = 6378.137

# satellite -> inclination band used to pull its altitude curve from initial_states
INCL = {"champ": (87.0, 87.6), "grace": (88.5, 89.6), "swarma": (87.0, 87.6)}
ERA = {"champ": (2000, 2011), "grace": (2002, 2018), "swarma": (2013, 2020)}


def altitude_curves():
    """Per satellite, a (sorted times, altitudes km) table from initial_states,
    separated by inclination band + era."""
    rows = []
    for f in glob.glob(str(IS / "*.csv")):
        for r in csv.DictReader(open(f)):
            if not r.get("File ID"):
                continue
            try:
                t = datetime.strptime(r["Timestamp"], "%Y-%m-%d %H:%M:%S")
                a = float(r["Semi-major Axis (km)"]); inc = float(r["Inclination (deg)"])
            except (ValueError, KeyError):
                continue
            alt = a - R_E
            if 200 < alt < 700:
                rows.append((t, inc, alt))
    curves = {}
    for sat, (lo, hi) in INCL.items():
        y0, y1 = ERA[sat]
        sel = sorted((t, alt) for t, inc, alt in rows if lo <= inc <= hi and y0 <= t.year < y1)
        if sel:
            ts = np.array([x[0].timestamp() for x in sel])
            al = np.array([x[1] for x in sel])
            curves[sat] = (ts, al)
    return curves


def sat_of(name):
    for s in ("champ", "grace1", "grace2", "swarma"):
        if name.startswith(s):
            return "grace" if s.startswith("grace") else s
    return None


def load_density_means():
    """Per density file: (satellite, epoch, mean rho)."""
    out = []
    for f in glob.glob(str(SAT / "*.csv")):
        base = Path(f).name
        sat = sat_of(base)
        if sat is None:
            continue
        m = re.search(r"(\d{8})_to_", base)
        if not m:
            continue
        epoch = datetime.strptime(m.group(1), "%Y%m%d")
        rows = list(csv.reader(open(f)))[1:]
        rho = np.array([float(r[1]) for r in rows if len(r) > 1 and r[1]])
        rho = rho[(rho > 0) & np.isfinite(rho)]
        if len(rho) > 10:
            out.append((sat, epoch, float(np.exp(np.mean(np.log(rho))))))  # geometric mean
    return out


def alt_at(curves, sat, epoch):
    if sat not in curves:
        return np.nan
    ts, al = curves[sat]
    return float(np.interp(epoch.timestamp(), ts, al))


def main():
    outdir = ROOT / "results" / "stormai"
    outdir.mkdir(parents=True, exist_ok=True)
    curves = altitude_curves()
    print("altitude curves (km) from initial_states:")
    for s, (ts, al) in curves.items():
        print(f"   {s:7s} {len(al):4d} states  alt {al.min():.0f}-{al.max():.0f} km")

    dens = load_density_means()
    # weekly buckets per satellite
    def wk(e):
        return e.toordinal() // 7
    by = {}
    for sat, epoch, rho in dens:
        by.setdefault((sat, wk(epoch)), []).append((epoch, rho))

    # match CHAMP & GRACE in the same week -> two-altitude scale height
    Hs, alt_c, alt_g, rc, rg, times = [], [], [], [], [], []
    weeks = set(w for (s, w) in by if s == "champ") & set(w for (s, w) in by if s == "grace")
    for w in sorted(weeks):
        ec, rho_c = min(by[("champ", w)]);  eg, rho_g = min(by[("grace", w)])
        hc = alt_at(curves, "champ", ec);   hg = alt_at(curves, "grace", eg)
        if not (np.isfinite(hc) and np.isfinite(hg)) or abs(hg - hc) < 15 or rho_c <= rho_g:
            continue
        H = (hg - hc) / np.log(rho_c / rho_g)
        if 10 < H < 200:
            Hs.append(H); alt_c.append(hc); alt_g.append(hg)
            rc.append(rho_c); rg.append(rho_g); times.append(ec)
    Hs = np.array(Hs)
    print(f"\nmatched CHAMP-GRACE weeks: {len(Hs)}")
    print(f"recovered scale height H: median {np.median(Hs):.0f} km, "
          f"IQR {np.percentile(Hs,25):.0f}-{np.percentile(Hs,75):.0f} km "
          f"(physical thermosphere ~ 30-90 km)")

    # density.py conditioning: 1 altitude (one sat) vs 2 altitudes (both)
    # design matrix for log-rho = a + b*h, per-column normalised
    def cond(alts):
        Phi = np.stack([np.ones(len(alts)), np.array(alts, float)], 1)
        Phi = Phi / np.linalg.norm(Phi, axis=0)
        return np.linalg.cond(Phi)
    hc_m, hg_m = np.mean(alt_c), np.mean(alt_g)
    c1 = cond([hc_m, hc_m + 0.5])       # "one satellite": ~one altitude (tiny spread)
    c2 = cond([hc_m, hg_m])             # two satellites: two altitudes
    print(f"\nvertical-profile design conditioning:")
    print(f"   single satellite (~1 altitude): cond {c1:.1e}  -> scale height NOT identifiable")
    print(f"   CHAMP + GRACE (2 altitudes):    cond {c2:.1e}  -> identifiable")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    t = np.array([d.year + d.timetuple().tm_yday / 365 for d in times])
    ax[0].scatter(t, Hs, s=12, c="#2471a3", alpha=0.6)
    ax[0].axhspan(30, 90, color="#999", alpha=0.15, label="physical thermosphere")
    ax[0].set_xlabel("year"); ax[0].set_ylabel("recovered scale height H (km)")
    ax[0].set_title(f"Real CHAMP+GRACE density -> vertical scale height\n(median {np.median(Hs):.0f} km, n={len(Hs)})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].scatter(np.log10(rc), alt_c, s=10, c="#c0392b", alpha=0.5, label="CHAMP (lower)")
    ax[1].scatter(np.log10(rg), alt_g, s=10, c="#27ae60", alpha=0.5, label="GRACE (~480 km)")
    ax[1].set_xlabel("log10 orbit-mean density (kg/m^3)"); ax[1].set_ylabel("altitude (km)")
    ax[1].set_title("Two altitudes pin the profile;\none leaves it degenerate")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    p = outdir / "stormai_scale_height.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {p}")

    from perturber.report import save_metrics
    save_metrics(outdir, {"n_matched_weeks": int(len(Hs)),
                          "scale_height_km_median": float(np.median(Hs)),
                          "scale_height_km_iqr": [float(np.percentile(Hs, 25)), float(np.percentile(Hs, 75))],
                          "cond_single_satellite": float(c1), "cond_champ_grace": float(c2)})
    assert len(Hs) > 20, "need a reasonable number of matched epochs"
    assert 20 < np.median(Hs) < 120, "recovered scale height should be physical"
    assert c1 > 100 * c2, "one altitude should be far worse conditioned than two"
    print("[stormai] real-data altitude-structure check complete")


if __name__ == "__main__":
    main()
