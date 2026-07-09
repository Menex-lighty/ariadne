# Where Ariadne sits in the literature

A deliberate prior-art check (July 2026). Short version: the differentiable-
simulation / learn-the-unknown-force approach is **sound and validated** — good
groups independently chose it — but the headline ideas on every milestone are
**already published**, some almost exactly. This reframes Ariadne from "novel
research" toward "rebuild validated methods + contribute on the under-served
identifiability angle."

## Closest prior work — nearly M1 + M2 combined

- **Lemos, Jeffrey, Cranmer, Ho, Battaglia (2022), "Rediscovering orbital
  mechanics with machine learning"** — [arXiv:2202.02306](https://arxiv.org/abs/2202.02306),
  *Mach. Learn.: Sci. Technol.* (2023). A GNN trained on 30 years of real Solar
  System trajectories; symbolic regression then recovers the **force law**
  (Newtonian gravity) and the **masses are inferred without being assumed**. That
  is Ariadne's M1 (infer hidden masses) and M2 (discover the force form) together,
  on real data.
- **Cranmer et al. (2020), "Discovering Symbolic Models from Deep Learning with
  Inductive Biases"** — [NeurIPS](https://proceedings.neurips.cc/paper/2020/file/c9f2f917078bd2db12f23c3b413d9cba-Paper.pdf).
  The GNN + symbolic-regression recipe underlying the above; the method M2 was
  reinventing.

## Milestone by milestone

- **M1 — hidden mass from visible trajectories.** Mature as the **transit-timing
  variation (TTV)** inverse problem, and now via **differentiable N-body**:
  a [differentiable N-body for transit timing (MNRAS 2024/25)](https://academic.oup.com/mnras/article/540/1/106/8125473),
  and [ODISSEO, differentiable N-body for galactic-dynamics inference (2025)](https://arxiv.org/pdf/2511.22468).
  The TTV literature explicitly calls the perturber inverse problem *"complex and
  possibly highly degenerate"* — the same wall our sweep hit, already named. Most
  pointedly, [*"The Illusory Precision of TTV Masses" (Kepler-9)*](https://iopscience.iop.org/article/10.3847/1538-4357/ae74c9)
  demonstrates hidden degenerate mass solutions on a *real* system — the closest
  prior work to our identifiability angle, and the ideal system to test the
  condition-number predictor on. See [benchmarks.md](benchmarks.md).
- **M2 — discover the force form.** Done (Lemos 2022), plus ongoing
  [neural-ODE + symbolic regression](https://arxiv.org/abs/2601.20637) and
  [Hamiltonian-GNN symbolic-law-from-trajectory](https://arxiv.org/html/2307.05299) work.
- **M3 — VLEO shared density field.** Very active *now*:
  [MSIS-UN fuses multi-satellite CHAMP/GRACE/SWARM density with uncertainty (2024)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023SW003652),
  ensemble ML beats NRLMSISE by ~61%, ConvLSTM global forecasting. Multi-satellite
  joint density is being published through 2024–2026.
- **M4 — dark matter subhalos from streams.** Crowded and well-resourced:
  [GNNs (2025)](https://arxiv.org/html/2502.03522v1),
  [neural simulation-based inference (2020)](https://arxiv.org/abs/2011.14923),
  GD-1 perturber searches, differentiable stream simulators.

## What this means, and the chosen angle

Not a race for novelty — the core ideas are taken, often by the groups who will
keep owning them. But two things remain genuinely useful:

1. **Skill/portfolio value is high and real.** Independently arriving at the
   Lemos-2022 architecture, plus a correct detection-vs-characterization sweep,
   is a strong on-ramp into an active field.
2. **The under-served angle: quantify identifiability.** The field *asserts* these
   inverse problems are "degenerate" and shows it on individual real systems
   (Kepler-9 above); what is under-served is a *predictive* handle — a quantity
   computable *before* fitting that says whether the form/mass is recoverable.
   Ariadne proposes the design-matrix condition number for exactly this, and
   already embodies the measurement:
   - **M1** produced a detection-vs-characterization frontier — presence is
     recoverable far below where mass becomes measurable (see
     [milestone1_results.md](milestone1_results.md)).
   - **M2** yielded a clean, quantified force-form degeneracy: with a *perfect*
     initial state the fit reproduces the trajectory to the noise floor and
     recovers the radial **profile** g(r), yet the **symbolic form** is
     unrecoverable — the power-law basis is near-degenerate over the r-range a few
     orbits sample (see `scripts/run_identifiability.py`).

**Direction:** lean into identifiability quantification, and toward **M3**, where
the multi-object joint constraint is precisely what *breaks* the single-object
degeneracy — the one place the method's value and its identifiability are both
highest, on real data with real demand.
