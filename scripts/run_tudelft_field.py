"""M3 real-data capstone: recover the coupled density field rho(altitude, local
solar time) from THREE real satellites, and draw the real-data version of the
density.py restoration curve — field conditioning improving with satellite count.

Why 3 (not 2): a single scale height is a slope (2 altitudes pin it), but the real
field has altitude *curvature* and altitude x local-time *coupling* (the diurnal
bulge grows with altitude). That is the synthetic 7-term field
[1, h, h^2, cos, sin, h*cos, h*sin]; resolving it needs 3-4 distinct altitudes.

Data (TU Delft, CC BY 4.0, gitignored), 2016, with altitude + local solar time:
GRACE-A (~370 km), Swarm-A (~450 km), Swarm-B (~515 km).

Physical fit uses per-day fixed effects to absorb the solar/geomagnetic trend
(shared across satellites each day); the altitude and local-time terms then give
the scale height, the diurnal bulge, and the coupling. The identifiability curve
reuses perturber.density on the REAL (altitude, local-time) coverage.

Usage: python scripts/run_tudelft_field.py
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from perturber.density import density_features, design_condition   # noqa: E402

ROOT = Path(__file__).parents[1]
SATS = {"GRACE-A": (ROOT/"data"/"tudelft"/"ga2016", 20),
        "Swarm-A": (ROOT/"data"/"tudelft"/"sa2016", 6),
        "Swarm-B": (ROOT/"data"/"tudelft"/"sb2016", 6)}


def load(dirp, step):
    doy, alt, lst, lrho = [], [], [], []
    for f in sorted(glob.glob(str(dirp/"*.txt"))):
        n = 0
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            n += 1
            if n % step:
                continue
            p = line.split()
            try:
                rho = float(p[8])
                if rho <= 0:
                    continue
                _, m, d = p[0].split("-")
                doy.append((int(m)-1)*31 + int(d)); alt.append(float(p[3])/1e3)
                lst.append(float(p[6])); lrho.append(np.log(rho))
            except (ValueError, IndexError):
                continue
    return np.array(doy), np.array(alt), np.array(lst), np.array(lrho)


def main():
    outdir = ROOT/"results"/"tudelft"; outdir.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, (d, step) in SATS.items():
        data[name] = load(d, step)
        a = data[name][1]
        print(f"{name}: {len(a)} samples, alt {a.min():.0f}-{a.max():.0f} km")
    order = ["GRACE-A", "Swarm-A", "Swarm-B"]
    amean = np.concatenate([data[n][1] for n in order]).mean()

    # ---- real-data restoration curve: field conditioning vs satellite count ----
    print("\ndensity-field conditioning over REAL (altitude, local-time) coverage:")
    conds = []
    for k in (1, 2, 3):
        subs = order[:k]
        alt = np.concatenate([data[n][1] for n in subs])
        lst = np.concatenate([data[n][2] for n in subs])
        h = (alt - amean) / 50.0                                # scaled altitude
        c = design_condition(h, lst/24.0)
        conds.append(c)
        print(f"   {k} satellite(s) [{', '.join(subs)}]: cond {c:.1e}")

    # ---- physical fit (all 3), per-day fixed effects ----
    doy = np.concatenate([data[n][0] for n in order]); alt = np.concatenate([data[n][1] for n in order])
    lst = np.concatenate([data[n][2] for n in order]); lrho = np.concatenate([data[n][3] for n in order])
    h = (alt - amean)/50.0; ang = 2*np.pi*lst/24
    feats = np.stack([h, h**2, np.cos(ang), np.sin(ang), h*np.cos(ang), h*np.sin(ang),
                      np.cos(2*ang), np.sin(2*ang)], 1)
    days = sorted(set(doy)); di = {d: i for i, d in enumerate(days)}
    D = np.zeros((len(h), len(days)))
    for i, d in enumerate(doy):
        D[i, di[d]] = 1.0
    X = np.concatenate([feats, D], 1)
    beta, *_ = np.linalg.lstsq(X, lrho, rcond=None)
    b = beta[:8]
    # scale height at mean altitude: d(log rho)/d(alt) = (b0 + 2 b1 h)/50 at h=0 -> b0/50
    H = -50.0/b[0]
    tt = np.linspace(0, 24, 200); aa = 2*np.pi*tt/24
    def bulge(hval):
        return (b[2]*np.cos(aa)+b[3]*np.sin(aa)+hval*(b[4]*np.cos(aa)+b[5]*np.sin(aa))
                + b[6]*np.cos(2*aa)+b[7]*np.sin(2*aa))
    g_lo, g_hi = bulge(-1.0), bulge(+1.0)      # low vs high altitude diurnal shape
    peak = tt[np.argmax(bulge(0))]; amp = np.exp(bulge(0).max()-bulge(0).min())
    amp_lo, amp_hi = np.exp(g_lo.max()-g_lo.min()), np.exp(g_hi.max()-g_hi.min())
    print(f"\nrecovered coupled field (3 satellites):")
    print(f"   scale height H = {H:.0f} km (physical ~40-60)")
    print(f"   diurnal bulge peak LST {peak:.0f} h, day/night {amp:.1f}x")
    print(f"   coupling: day/night contrast {amp_lo:.1f}x at ~370 km -> {amp_hi:.1f}x at ~515 km "
          f"({'grows' if amp_hi > amp_lo else 'shrinks'} with altitude)")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].semilogy([1, 2, 3], conds, "o-", color="#7d3c98", ms=9)
    ax[0].set_xticks([1, 2, 3]); ax[0].set_xlabel("satellites (distinct altitudes)")
    ax[0].set_ylabel("coupled-field design conditioning")
    ax[0].set_title("Real-data restoration: 1 altitude can't pin\nthe coupled field; 3 altitudes condition it")
    ax[0].grid(alpha=0.3, which="both")
    HH, LL = np.meshgrid(np.linspace(360, 520, 50), np.linspace(0, 24, 90))
    hh = (HH-amean)/50.0; an = 2*np.pi*LL/24
    field = (b[0]*hh + b[1]*hh**2 + b[2]*np.cos(an)+b[3]*np.sin(an)
             + hh*(b[4]*np.cos(an)+b[5]*np.sin(an)) + b[6]*np.cos(2*an)+b[7]*np.sin(2*an))
    im = ax[1].contourf(LL, HH, np.exp(field-field.mean()), 20, cmap="inferno")
    ax[1].set_xlabel("local solar time (h)"); ax[1].set_ylabel("altitude (km)")
    ax[1].set_title(f"Recovered rho(altitude, LST) from 3 satellites\nH={H:.0f} km, bulge {peak:.0f}h")
    ax[1].set_xticks(range(0, 25, 6)); fig.colorbar(im, ax=ax[1], label="rho / mean")
    plt.tight_layout()
    p = outdir/"tudelft_joint_field.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {p}")

    from perturber.report import save_metrics
    save_metrics(outdir, {"scale_height_km": float(H), "diurnal_peak_lst_h": float(peak),
                          "day_night_factor_x": float(amp), "conditioning_by_nsats": [float(c) for c in conds],
                          "coupling_daynight_low_alt": float(amp_lo), "coupling_daynight_high_alt": float(amp_hi)})
    assert conds[0] > 10*conds[-1], "conditioning should improve markedly with satellite count"
    assert 25 < H < 90, "scale height should be physical"
    assert 11 <= peak <= 17 and amp > 1.5, "diurnal bulge should peak afternoon, ~2-3x"
    print("[tudelft-field] coupled rho(altitude, LST) recovered from 3 real satellites; "
          "restoration curve confirmed")


if __name__ == "__main__":
    main()
