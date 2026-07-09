"""Detailed storm survey: harden (and probe) the VLEO storm-density niche across
multiple superstorms, two satellites, and the along-track structure.

Data (TU Delft, CC BY 4.0, gitignored): CHAMP storm-months 2001-03 (~480 km),
2001-11 (~426), 2003-10/11 (~393), 2004-11 (~400); GRACE 2003-10, 2004-11 (~486).
GFZ F10.7/Ap. Storm density enhancement is measured vs a quiet-day baseline
log rho ~ F10.7_81 + altitude (81-day flux avoids flare contamination, so the
enhancement is purely geomagnetic).

Four probes:
  1  multi-storm  : peak enhancement vs peak Ap and altitude (robustness + scaling)
  2  altitude     : same storm, CHAMP (~400 km) vs GRACE (~486 km)
  3  latitude     : within-orbit density / orbit-mean vs |lat|, storm vs quiet
                    (does the storm deposit at high latitude?)
  4  lag          : cross-correlation of enhancement vs ap (response delay)

Usage: python scripts/run_storm_survey.py
"""
import glob
from datetime import datetime, date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

ROOT = Path(__file__).parents[1]
GFZ = ROOT / "data" / "spaceweather" / "kp_ap_f107.txt"
TD = ROOT / "data" / "tudelft"


def solar():
    daily, ah, av = {}, [], []
    for line in open(GFZ):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 27:
            continue
        try:
            o = date(int(p[0]), int(p[1]), int(p[2])).toordinal()
            daily[o] = (float(p[25]), float(p[23]))
            for k in range(8):
                a = float(p[15 + k])
                if a >= 0:
                    ah.append(o * 24 + k * 3 + 1.5); av.append(a)
        except ValueError:
            continue
    f81 = {}
    for o in daily:
        f81[o] = np.mean([daily[oo][0] for oo in range(o - 80, o + 1) if oo in daily])
    return daily, f81, np.array(ah), np.array(av)


def load(files, step=30):
    """(datetime, alt_km, lst, lat, rho_inst, rho_orb) from along-track files."""
    T, A, L, LA, RI, RO = [], [], [], [], [], []
    for f in files:
        n = 0
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            n += 1
            if n % step:
                continue
            p = line.split()
            try:
                ri, ro = float(p[8]), float(p[9])
                if ri <= 0 or ro <= 0 or float(p[10]) or float(p[11]):
                    continue
                dt = datetime.strptime(p[0] + " " + p[1].split(".")[0], "%Y-%m-%d %H:%M:%S")
                T.append(dt); A.append(float(p[3])/1e3); L.append(float(p[6]))
                LA.append(float(p[5])); RI.append(ri); RO.append(ro)
            except (ValueError, IndexError):
                continue
    return dict(t=np.array(T), alt=np.array(A), lst=np.array(L), lat=np.array(LA),
                ri=np.array(RI), ro=np.array(RO))


def attach(d, daily, f81):
    o = np.array([x.toordinal() for x in d["t"]])
    d["F81"] = np.array([f81.get(oo, np.nan) for oo in o])
    d["Ap"] = np.array([daily.get(oo, (np.nan, np.nan))[1] for oo in o])
    d["he"] = np.array([x.toordinal()*24 + x.hour + x.minute/60 for x in d["t"]])
    return d


def enhancement(d):
    """orbit-mean density / quiet-baseline(F81, altitude)."""
    q = d["Ap"] < 15
    X = np.column_stack([np.ones(q.sum()), d["F81"][q], d["alt"][q]])
    b, *_ = np.linalg.lstsq(X, np.log(d["ro"][q]), rcond=None)
    pred = np.exp(b[0] + b[1]*d["F81"] + b[2]*d["alt"])
    return d["ro"] / pred


def main():
    daily, f81, ah, av = solar()
    champ = {"2001-03": ["CH_DNS_ACC_2001-03_v02.txt"], "2001-11": ["CH_DNS_ACC_2001-11_v02.txt"],
             "2003-10": ["CH_DNS_ACC_2003-10_v02.txt"], "2003-11": ["CH_DNS_ACC_2003-11_v02.txt"],
             "2004-11": ["CH_DNS_ACC_2004-11_v02.txt"]}
    sets = {}
    for k, fs in champ.items():
        for base in ("champ2003", "champ_storms"):
            p = [str(TD/base/x) for x in fs if (TD/base/x).exists()]
            if p:
                sets["CHAMP "+k] = attach(load(p), daily, f81); break
    for k, fn in {"2003-10": "GA_DNS_ACC_2003_10_v02.txt", "2004-11": "GA_DNS_ACC_2004_11_v02.txt"}.items():
        p = TD/"grace_storms"/fn
        if p.exists():
            sets["GRACE "+k] = attach(load([str(p)]), daily, f81)

    # ---- Probe 1: multi-storm enhancement ----
    print("Probe 1 -- storm density enhancement (orbit-mean / quiet baseline):")
    pts = []
    for name, d in sets.items():
        if len(d["t"]) < 100:
            continue
        e = enhancement(d); storm = d["Ap"] >= 80
        if storm.sum() < 5:
            continue
        peak, apk, alt = np.percentile(e[storm], 98), np.nanmax(d["Ap"]), np.median(d["alt"])
        pts.append((name, peak, apk, alt))
        print(f"   {name:14s} alt {alt:3.0f} km  peakAp {apk:3.0f}  ->  {peak:.1f}x enhancement")
    pts = pts

    # ---- Probe 2: altitude scaling, same storm ----
    print("\nProbe 2 -- altitude scaling (same storm, CHAMP vs GRACE):")
    for st in ("2003-10", "2004-11"):
        c, g = sets.get("CHAMP "+st), sets.get("GRACE "+st)
        if c and g:
            ec, eg = enhancement(c), enhancement(g)
            sc, sg = c["Ap"] >= 100, g["Ap"] >= 100
            if sc.sum() and sg.sum():
                pc, pg = np.percentile(ec[sc], 98), np.percentile(eg[sg], 98)
                print(f"   {st}: CHAMP {np.median(c['alt']):.0f} km -> {pc:.1f}x   |   "
                      f"GRACE {np.median(g['alt']):.0f} km -> {pg:.1f}x   (ratio {pc/pg:.1f})")

    # ---- Probe 3: latitude structure (Halloween, CHAMP) ----
    print("\nProbe 3 -- latitude structure of the storm (CHAMP Halloween):")
    d = sets["CHAMP 2003-10"]; r = d["ri"]/d["ro"]           # within-orbit / orbit-mean
    latb = np.arange(0, 90, 15)
    for lo in latb:
        m = (np.abs(d["lat"]) >= lo) & (np.abs(d["lat"]) < lo+15)
        q = m & (d["Ap"] < 20); s = m & (d["Ap"] >= 100)
        if q.sum() > 20 and s.sum() > 20:
            print(f"   |lat| {lo:2d}-{lo+15:2d}: quiet {np.median(r[q]):.2f}  storm {np.median(r[s]):.2f}  "
                  f"(+{100*(np.median(r[s])/np.median(r[q])-1):.0f}%)")

    # ---- Probe 4: response lag (CHAMP Halloween) ----
    d = sets["CHAMP 2003-10"]; e = enhancement(d)
    lo = np.searchsorted(ah, d["he"] - 0); ap_now = av[np.clip(lo-1, 0, len(av)-1)]
    best = None
    for lag in range(0, 25, 3):
        li = np.searchsorted(ah, d["he"] - lag); apl = av[np.clip(li-1, 0, len(av)-1)]
        c = np.corrcoef(np.log(e), np.log1p(apl))[0, 1]
        if best is None or c > best[1]:
            best = (lag, c)
    print(f"\nProbe 4 -- response lag: enhancement best correlates with ap lagged {best[0]} h "
          f"(corr {best[1]:.2f})")

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    nm = [x[0] for x in pts]; pk = [x[1] for x in pts]; apx = [x[2] for x in pts]; al = [x[3] for x in pts]
    sc = ax[0, 0].scatter(apx, pk, c=al, s=90, cmap="viridis_r", edgecolor="k")
    for n, x, yv in zip(nm, apx, pk):
        ax[0, 0].annotate(n.replace("CHAMP ", "C").replace("GRACE ", "G"), (x, yv), fontsize=7)
    ax[0, 0].set_xlabel("peak Ap"); ax[0, 0].set_ylabel("density enhancement (x baseline)")
    ax[0, 0].set_title("1. Large enhancements (2-6x) across storms, but noisy vs Ap")
    fig.colorbar(sc, ax=ax[0, 0], label="altitude (km)"); ax[0, 0].grid(alpha=0.3)
    # latitude structure
    d = sets["CHAMP 2003-10"]; r = d["ri"]/d["ro"]; centers = latb+7.5
    qp = [np.median(r[(np.abs(d["lat"])>=lo)&(np.abs(d["lat"])<lo+15)&(d["Ap"]<20)]) for lo in latb]
    sp = [np.median(r[(np.abs(d["lat"])>=lo)&(np.abs(d["lat"])<lo+15)&(d["Ap"]>=100)]) for lo in latb]
    ax[0, 1].plot(centers, qp, "o-", color="#2471a3", label="quiet (Ap<20)")
    ax[0, 1].plot(centers, sp, "s-", color="#c0392b", label="storm (Ap>=100)")
    ax[0, 1].set_xlabel("|latitude| (deg)"); ax[0, 1].set_ylabel("density / orbit-mean")
    ax[0, 1].set_title("2. No clean auroral signature (crude orbit-mean probe)"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)
    # lag scan
    d = sets["CHAMP 2003-10"]; e = enhancement(d); lags = list(range(0, 25, 3)); cc = []
    for lag in lags:
        li = np.searchsorted(ah, d["he"]-lag); apl = av[np.clip(li-1, 0, len(av)-1)]
        cc.append(np.corrcoef(np.log(e), np.log1p(apl))[0, 1])
    ax[1, 0].plot(lags, cc, "o-", color="#7d3c98")
    ax[1, 0].set_xlabel("ap lag (hours)"); ax[1, 0].set_ylabel("corr(enhancement, ap)")
    ax[1, 0].set_title(f"3. No resolvable lag (best ~{best[0]} h; coarse data)"); ax[1, 0].grid(alpha=0.3)
    # altitude scaling
    for st, col in (("2003-10", "#e67e22"), ("2004-11", "#16a085")):
        c, g = sets.get("CHAMP "+st), sets.get("GRACE "+st)
        if c and g:
            ec, eg = enhancement(c), enhancement(g)
            aa = [np.median(c["alt"]), np.median(g["alt"])]
            ee = [np.percentile(ec[c["Ap"]>=100], 98), np.percentile(eg[g["Ap"]>=100], 98)]
            ax[1, 1].plot(aa, ee, "o-", color=col, label=st, ms=9)
    ax[1, 1].set_xlabel("altitude (km)"); ax[1, 1].set_ylabel("peak enhancement (x)")
    ax[1, 1].set_title("4. Same storm: RELATIVE enhancement grows with altitude"); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)
    plt.tight_layout()
    p = ROOT/"results"/"stormai"/"storm_survey.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {p}")

    from perturber.report import save_metrics
    save_metrics(ROOT/"results"/"stormai", {
        "storms": [{"name": n, "peak_enhancement_x": float(pp), "peak_ap": float(a), "altitude_km": float(al)}
                   for n, pp, a, al in pts],
        "response_lag_h_best": int(best[0]), "corr_at_best_lag": float(best[1])},
        name="storm_survey_metrics.json")
    assert max(pk) > 2.5, "at least one storm should show a large enhancement"
    assert best[1] > 0.4, "enhancement should track the lagged driver"
    print("\n[storm-survey] multi-storm / multi-altitude / along-track structure characterised")


if __name__ == "__main__":
    main()
