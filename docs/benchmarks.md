# Public benchmarks, datasets, and baselines

What "good" means for each direction — the public data, standard baselines, and
evaluation metrics a serious version of this work would measure against.

## M3 (VLEO density) — a turnkey benchmark exists

The strongest reason to prioritise M3: you don't have to invent an evaluation.

- **MIT ARCLab STORM-AI Challenge (2025)** — a public competition to forecast
  orbit-averaged thermospheric density. Released
  [dev-kit](https://github.com/ARCLab-MIT/STORM-AI-devkit-2025), training data
  (8,118 samples: initial orbital params + 60 days of space-weather indices +
  GOES X-ray + OMNI2 solar wind → 3-day density target), a hidden test set, a
  **Codabench leaderboard**, and a defined metric — **OD-RMSE (orbital-density
  RMSE), benchmarked against MSIS**.
  [Challenge](https://aeroastro.mit.edu/arclab/aichallenge/) ·
  [dataset docs](https://2025-ai-challenge.readthedocs.io/en/latest/dataset.html).
- **Reference density database:** [SET HASDM](https://www.researchgate.net/publication/349894929_The_SET_HASDM_density_database)
  — 20 years (2000–2019), 3-hourly, gridded 175–825 km. In-situ ground truth:
  CHAMP, GRACE / GRACE-FO, GOCE, Swarm-A accelerometer-derived densities.
- **Baselines to beat:** NRLMSISE-00, JB2008,
  [DTM2020](https://www.swsc-journal.org/articles/swsc/full_html/2021/01/swsc210039/swsc210039.html),
  NRLMSIS 2.0. Published ML models improve on these by ~20–60% MAPE.

**Takeaway:** an M3 entry has data, metric, baselines, and a leaderboard ready —
a rare, concrete target for a *measurable* contribution.

### Update (2026-07): what we actually did, and why the niche is crowded

We chose **Path B** (infer the density field + quantify identifiability) over the
STORM-AI *forecasting* leaderboard, and validated it on real CHAMP/GRACE/Swarm data
(see [m3_plan.md](m3_plan.md)). We then probed a **storm-time VLEO drag nowcast**
([vleo_nowcast_gate.md](vleo_nowcast_gate.md)). The prior-art search closed that as
an open niche — it is a mature field:

- **HASDM Dynamic Calibration Atmosphere** (operational, US Space Force): assimilate
  radar drag from ~80–90 calibration satellites to correct JB2008 every 3 h.
- **Gondelach & Linares 2021** ([Space Weather](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020SW002620)):
  reduced-order density model (POD + DMDc) + **Kalman filter** assimilating radar/GPS/TLE
  tracking → real-time global density. This *is* "infer the field from tracking."
- **Licata et al.** ([ML-HASDM + UQ](https://arxiv.org/pdf/2109.07651)); 2024
  physics-informed SINDy reduced-order assimilation.

A differentiable-simulation assimilation would be a **method variant** in this
space, not an open niche.

## Force-form / identifiability — a methodology benchmark and a real-data anchor

- **Evaluation standard:** [SRBench](https://arxiv.org/pdf/2206.10540) — the
  community symbolic-regression benchmark (Feynman 119 physics equations +
  Strogatz + Black-box) with ground truth and a sympy-based **solution rate**.
  Physics-focused successors: Phy-SRBench (NeurIPS 2025 ML4PS), LLM-SRBench.
  Report form-recovery in this solution-rate language.
- **Real-data anchor & closest prior art:** *"The Illusory Precision of Transit
  Timing Variation Masses: Hidden Solutions Behind Kepler-9's Tight Mass Ratio"*
  ([ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/ae74c9)) — shows
  TTV *mass* determination has hidden degenerate solutions on the **real Kepler-9
  system** even when it looks precise. This is essentially the identifiability
  finding, on real data. It means the niche is not empty — but Kepler-9 is the
  ideal system on which to test our specific claim (condition number as an
  a-priori predictor of *form/mass* recoverability).

## M1 (hidden mass / TTV) — real data to move past synthetic

Public Kepler TTV catalogs (Ford et al. 2012; Holczer et al.; DR25 TTVs) with
[1,791 multi-planet systems](https://iopscience.iop.org/article/10.3847/1538-3881/ab0d91)
(masses determined for ~88). The real-data upgrade path for M1.

## Implication for direction (honest, as of 2026-07)

Both VLEO paths are populated: **forecasting** (STORM-AI, ML-MSIS) and
**assimilation** (HASDM, MIT reduced-order + Kalman). So is force discovery (Lemos)
and dark matter (M4). There is **no open research niche** in what we've touched —
the honest contribution is *method + rigor*, not novelty. The pieces that remain
defensible as a portfolio/methods artifact:

- **The identifiability thread** — the condition-number predictor validated across
  force-form (synthetic), **Kepler-9 masses** (real, a documented degeneracy), and
  the **VLEO density field** (real, constellation restoration). One method, three
  domains, real-data-anchored.
- **The verification record** — the twice-corrected Kepler-9 fit and the killed
  pitches (M4, storm-niche-at-GRACE) as a candid case study in not fooling yourself
  in computational inverse problems.

If a *measurable* entry is ever wanted, STORM-AI (forecasting) or a differentiable
re-implementation benchmarked against reduced-order+Kalman (assimilation) are the
targets — as re-implementation exercises, understood as such.
