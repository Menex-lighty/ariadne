# Kepler-9 data (for the identifiability analysis)

Real observational data for the Kepler-9 two-planet system (KOI-377), the system
whose TTV *mass* degeneracy is documented in *"The Illusory Precision of Transit
Timing Variation Masses"* ([ApJ, 10.3847/1538-4357/ae74c9](https://iopscience.iop.org/article/10.3847/1538-4357/ae74c9)).
This is the real-data anchor for the identifiability angle (see
`../../docs/identifiability.md`): does a pre-fit conditioning number predict the
known mass degeneracy on a real system?

## Files

- **`holczer2016_ttv.csv`** — individual transit times, from Holczer et al. 2016
  (ApJS 225, 9), VizieR `J/ApJS/225/9/table3`, pulled via the VizieR TAP service.
  107 transits: **71 for planet b (KOI 377.01)**, **36 for planet c (377.02)**.
  Columns: `N` (transit number), `tn` (mid-transit time, BJD − 2,454,900, days),
  `O-C` (timing residual vs a linear ephemeris) and `e_O-C` (its uncertainty) —
  both in **minutes** (Holczer convention; verify against the catalog ReadMe
  before quantitative use), plus duration/depth-variation columns and flags.
- **`holczer2016_ephemerides.csv`** — per-planet linear ephemeris from the same
  catalog (`table2`): period, T0, duration, depth, S/N. b: P = 19.2463 d;
  c: P = 38.9498 d (ratio ≈ 2.023, just outside 2:1).
- **`nasa_exoplanet_archive_params.csv`** — all published system parameters from
  the NASA Exoplanet Archive `ps` table (one row per publication). **The mass
  degeneracy is visible directly here:** planet-b masses span ~26–80 M⊕
  (0.08–0.25 M_Jup) and c ~27–54 M⊕ across analyses — the "illusory precision."
  Stellar mass ≈ 1.0 M☉.

## What the analysis uses

The design-matrix / Fisher conditioning analysis needs the **sampling** (the
transit epochs `N`/`tn` — when each planet was observed) and the **noise**
(`e_O-C`, ~1.0–1.2 min timing precision). The TTV *signal* is regenerated from a
forward N-body model (`perturber` differentiable integrator), so the exact `O-C`
unit/reference convention does not gate the conditioning result.

## Reproduce the pull

VizieR TAP (ADQL), e.g.:
`SELECT * FROM "J/ApJS/225/9/table3" WHERE KOI>=377 AND KOI<378 ORDER BY KOI,N`
at `https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync`. Params: NASA Exoplanet
Archive TAP, `select * from ps where hostname='Kepler-9'`.
