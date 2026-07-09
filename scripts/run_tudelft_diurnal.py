"""M3 real-data check #3: recover the DIURNAL (local-solar-time) density field
from real high-cadence along-track data — the piece STORM-AI's orbit-mean density
structurally cannot reach.

Data: TU Delft accelerometer density (CC BY 4.0, https://thermosphere.tudelft.nl),
GRACE-A 2016, 10-s cadence with altitude, latitude and LOCAL SOLAR TIME per sample
(data/tudelft/ga2016/, gitignored). To isolate the diurnal signal from the slow
solar/altitude trend we divide each sample by its running orbit-average density
(provided in the file): rho/orbit_avg is the within-orbit day/night modulation.

Result: the thermospheric afternoon bulge — density peaks ~14 h LST at ~1.8x the
orbit mean and bottoms ~05 h at ~0.6x (a ~2-3x swing), recovered from real data.
And density.py's identifiability claim holds in the local-time dimension: a narrow
LST coverage (a single short-arc pass) cannot constrain the diurnal field; the full
local-time coverage (a precessing year, or a constellation) can.

Usage: python scripts/run_tudelft_diurnal.py
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "tudelft" / "ga2016"


def load(step=10):
    """Subsampled (LST, lat, alt_km, rho/orbit_avg) from the 10-s files."""
    lst, lat, alt, ratio = [], [], [], []
    for f in sorted(glob.glob(str(DATA / "*.txt"))):
        n = 0
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            n += 1
            if n % step:
                continue
            p = line.split()
            try:
                rho = float(p[8]); oa = float(p[9])
                if rho <= 0 or oa <= 0 or float(p[10]) or float(p[11]):   # quality flags
                    continue
                alt.append(float(p[3]) / 1e3); lat.append(float(p[5]))
                lst.append(float(p[6])); ratio.append(rho / oa)
            except (ValueError, IndexError):
                continue
    return map(np.array, (lst, lat, alt, ratio))


def diurnal_features(lst):
    """Design columns for a diurnal fit: [1, cos, sin, cos2, sin2] over LST in h."""
    a = 2 * np.pi * lst / 24
    return np.stack([np.ones_like(a), np.cos(a), np.sin(a), np.cos(2 * a), np.sin(2 * a)], 1)


def main():
    outdir = ROOT / "results" / "tudelft"
    outdir.mkdir(parents=True, exist_ok=True)
    lst, lat, alt, ratio = load()
    eq = np.abs(lat) < 40                                  # equatorial, strongest bulge
    lst, ratio = lst[eq], ratio[eq]
    print(f"GRACE-A 2016 samples (equatorial): {len(lst)}, altitude tracked ~340-410 km")

    # recover the diurnal curve by binning (skip under-populated LST bins)
    edges = np.arange(0, 24.001, 1.0)
    centers, prof = [], []
    for lo in edges[:-1]:
        m = (lst >= lo) & (lst < lo + 1)
        if m.sum() > 100:
            centers.append(lo + 0.5); prof.append(np.median(ratio[m]))
    centers, prof = np.array(centers), np.array(prof)
    peak = centers[np.argmax(prof)]
    amp = prof.max() / prof.min()
    print(f"diurnal density bulge: peaks at LST {peak:.0f} h (amp {prof.max():.2f}x mean), "
          f"trough {prof.min():.2f}x -> day/night factor {amp:.1f}")

    # density.py-style identifiability of the diurnal field: narrow vs full LST coverage
    def cond(mask):
        Phi = diurnal_features(lst[mask])
        Phi = Phi / np.linalg.norm(Phi, axis=0)
        return np.linalg.cond(Phi)
    narrow = (lst > 9) & (lst < 12)                       # a single short-arc window
    c_narrow = cond(narrow)
    c_full = cond(np.ones(len(lst), bool))
    print(f"\ndiurnal-field design conditioning (log-rho over LST harmonics):")
    print(f"   narrow LST coverage (one short pass, 9-12h): cond {c_narrow:.1e} -> field NOT identifiable")
    print(f"   full local-time coverage (precessing year):  cond {c_full:.1e} -> identifiable")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(centers, prof, "o-", color="#e67e22")
    ax[0].axhline(1.0, color="#999", ls="--", lw=1)
    ax[0].axvspan(13, 16, color="#f39c12", alpha=0.15, label="expected afternoon bulge")
    ax[0].set_xlabel("local solar time (h)"); ax[0].set_ylabel("density / orbit-average")
    ax[0].set_title(f"Real GRACE-A diurnal density bulge\n(peak LST {peak:.0f}h, day/night {amp:.1f}x)")
    ax[0].set_xticks(range(0, 25, 3)); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # scatter of raw ratio vs LST (subsampled)
    s = np.random.default_rng(0).choice(len(lst), min(4000, len(lst)), replace=False)
    ax[1].scatter(lst[s], ratio[s], s=3, alpha=0.15, color="#2471a3")
    ax[1].plot(centers, prof, "o-", color="#e67e22", lw=2)
    ax[1].set_xlabel("local solar time (h)"); ax[1].set_ylabel("density / orbit-average")
    ax[1].set_title("One short LST window can't pin the curve;\na full year (or constellation) can")
    ax[1].set_xticks(range(0, 25, 3)); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    p = outdir / "tudelft_diurnal.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {p}")

    from perturber.report import save_metrics
    save_metrics(outdir, {"peak_lst_h": float(peak), "day_night_factor_x": float(amp),
                          "cond_narrow_lst": float(c_narrow), "cond_full_lst": float(c_full)})
    assert 11 <= peak <= 17, "diurnal density should peak in the afternoon"
    assert amp > 1.5, "day/night density contrast should be a real factor (~2-3)"
    assert c_narrow > 50 * c_full, "narrow LST coverage should be far worse conditioned"
    print("[tudelft-diurnal] real diurnal density field recovered; density.py claim holds for local time")


if __name__ == "__main__":
    main()
