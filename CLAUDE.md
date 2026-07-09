# Ariadne — Unknown-Force Discovery from Trajectories

## Vision

Infer **unseen force structure** from observed trajectories. Hidden-companion /
dark-matter localization and VLEO drag prediction are the same inverse problem:
model acceleration as `a = known_gravity + unknown`, put the unknown inside a
**differentiable simulation**, and fit integrated trajectories to noisy
observations.

This replaced the earlier trajectory-smoothing CD-PINN (`legacy/`): the smoother
interpolated but could not extrapolate (ISS test RMSE 7,276 km vs SGP4's
1,416 km), its free-form drag network absorbed all model error (learned β̄≈0.74
vs physical ~1.4e-5), and its refactored derivative code crashed on 3-D outputs.
A Neural-ODE-style model structurally obeys the dynamics it learns, extrapolates
by integration, and its unknown term is directly the object of discovery.

**Positioning (see [docs/related_work.md](docs/related_work.md)):** the core
approach is validated but not novel — Lemos et al. 2022 already recovered forces
*and* masses from real Solar-System trajectories, and every milestone has active
published work. Ariadne's distinctive contribution is **quantifying
identifiability** — measuring *where* these inverse problems become degenerate,
which the field asserts but rarely maps. **M3 (VLEO) is done (Path B, below);
current direction is a storm-time VLEO drag nowcast** (see
[docs/vleo_nowcast_gate.md](docs/vleo_nowcast_gate.md)) — a real, important problem
but a **crowded** one (HASDM is operational; MIT's reduced-order + Kalman-filter
assimilation, Gondelach & Linares 2021, does exactly this), so any contribution is
*method* (differentiable-simulation assimilation), not an open niche.

## Milestone ladder

1. **M1 (done): hidden-perturber detection** — synthetic star + 2 planets + one
   invisible body; recover its mass and orbit from noisy visible tracks. The
   216-cell detection-threshold study (72 mass×noise×arc cells × 3 seeds;
   the `kaggle` grid in the commands below is the 135-cell science subset),
   null-gated on held-out data, is complete: **detection is near-universal down to 3e-6; mass
   characterization is the real limit**. See [docs/milestone1_results.md](docs/milestone1_results.md).
2. **M2 (explored → identifiability result): force-form discovery.** Inject a
   non-Newtonian central force (`extra_force` hook), fit it through the
   differentiable sim over a power-law basis (`residual.py`), prune via STLSQ
   (`symbolic.py`). **Finding:** with perfect state the trajectory and radial
   *profile* g(r) are recovered, but the *symbolic form* is not — the basis is
   near-degenerate over the r-range a few orbits sample. Quantified in
   `scripts/run_identifiability.py`. This degeneracy (also documented in the TTV
   literature) is the pivot point, not a bug to fix.
3. **M3 (done — Path B): VLEO density-field identifiability + real-data anchors.**
   Chose *inference + identifiability* over the NRLMSISE-benchmark forecasting path.
   `drag.py` (orbit-with-drag forward model; the B·ρ ballistic-coefficient
   degeneracy via Fisher), `density.py` (ρ(altitude, local-solar-time) field; the
   multi-satellite joint constraint restores it), plus synthetic + **real-data
   anchors** — STORM-AI (48 km scale height, decay→density 0.80 solar-cycle corr,
   decay prediction) and TU Delft accelerometer densities (14 h diurnal bulge
   ≈2.5×, coupled 3-satellite ρ(altitude,LST) field). See
   [docs/m3_plan.md](docs/m3_plan.md). **Current direction = storm-time VLEO drag
   nowcast** ([docs/vleo_nowcast_gate.md](docs/vleo_nowcast_gate.md)): the gate +
   5-storm survey confirmed the signal (2–6× model-missed enhancement, not
   index-predictable). Caveat: all three M3 modules are **numpy** Fisher/STLSQ
   analyses — none wired into the differentiable torch pipeline yet; the nowcast is
   where torch would finally be used.
4. **M4 (shelved, crowded): dark matter** — same machinery on stellar-stream
   data; a well-resourced field (GD-1 perturber searches, SBI, GNNs). Deprioritised
   as the least-winnable arena for a general toolkit.

## Repo layout

```
src/perturber/    source of truth. M1: config, dynamics, data, integrators,
                  model, fit, evaluate, plots, runner. M2/identifiability:
                  forces, residual, symbolic, identifiability, transits.
                  M3 (numpy, not torch): drag (orbit+drag, B·ρ degeneracy),
                  density (ρ(h,LST) field), threed (3-D N-body + transits, Kepler-9).
scripts/          thin CLIs. M1: run_experiment, run_threshold_study,
                  run_sweep_parallel, summarize_sweep, build_notebook,
                  run_identifiability(_study), run_kepler9_identifiability.
                  M3/VLEO: run_vleo_identifiability, run_kepler9_3d,
                  run_stormai_{density,inversion,prediction},
                  run_tudelft_{diurnal,field}, run_solar_density_variance,
                  run_residual_decompose, run_champ_storm, run_storm_survey.
data/kepler9/     real Kepler-9 TTVs + params (small) for the identifiability
                  anchor. data/{stormai,tudelft,spaceweather}/ = real VLEO datasets
                  (~1.1 GB, gitignored — download; see docs/m3_plan.md, vleo_nowcast_gate.md).
docs/             milestone1_results, identifiability, related_work, benchmarks,
                  m3_plan, vleo_nowcast_gate, kepler9_node_postmortem,
                  figures/ (result figures).
notebooks/kaggle_perturber.ipynb   BUILD ARTIFACT — never edit; rebuild from src/
                  (M1 sweep only; identifiability/Kepler-9 are script-based).
legacy/           frozen CD-PINN notebook, reference only.
results/          gitignored run outputs (figures, metrics.json, sweep cells).
```

## Environment & commands

Python 3.10, `pip install -r requirements.txt` (developed on torch 2.5.1). A GPU
is **not** required — the fit is a Python-loop N-body integration over tiny
tensors, so it is CPU-bound (a GPU is actually slower here). Keep code
3.10-compatible.

```powershell
python scripts/run_experiment.py --preset smoke      # ~2 min gate: asserts recovery + detection
python scripts/run_experiment.py --preset local      # full single experiment (~85 min on CPU)
# Detection-threshold sweep — CPU-bound, cells independent, so parallelize over cores:
python scripts/run_sweep_parallel.py --grid core --workers 20   # local, all cores (recommended)
python scripts/run_sweep_parallel.py --grid kaggle --workers 20 # full 135-cell science grid
python scripts/run_threshold_study.py --preset kaggle --shard 0 --n-shards 4  # sequential/sharded (Kaggle)
python scripts/build_notebook.py                     # rebuild the Kaggle notebook
# verify the built notebook standalone (smoke mode):
$env:PERTURBER_SMOKE = "1"
python -m jupyter nbconvert --to notebook --execute notebooks/kaggle_perturber.ipynb --output tmp_exec.ipynb --ExecutePreprocessor.timeout=600
```

**Compute note (important):** the fit is a Python-loop N-body integration over
tiny tensors, so it is **CPU-bound — the local T400 GPU is ~3× *slower* than CPU**
(kernel-launch overhead dominates). Run the sweep on CPU across cores. Cells are
independent and resumable (one JSON each, skipped if present), so they shard
cleanly across processes (`run_sweep_parallel.py`), sessions, or machines
(`--shard/--n-shards`). Grids: `smoke` (2), `core` (48, feasible), `kaggle`
(135, full). Per-cell CPU cost ≈ 7 min (8-period) to 38 min (40-period) at the
reduced sweep fidelity (12 restarts, trimmed curriculum, noise-scaled substeps).

Module self-checks (set `PYTHONPATH=src` first): `python -m perturber.dynamics`
(also `data`, `integrators`, `model`, `forces`, `transits`, `identifiability`;
`residual` and `symbolic` self-check via a short ODE fit, ~minutes).

## Conventions

- Units G = 1, star mass = 1, planet-1 semi-major axis = 1 (period 2π). 2-D state
  `(..., N, 4)` = (x, y, vx, vy); hidden body is always the **last** body.
- **float64 everywhere** in fitting — noise is 1e-4 in O(1) coordinates; float32
  rounding over thousands of RK4 steps eats the signal.
- Everything seeded (`SystemConfig.seed`, `FitConfig.seed`); configs are dumped
  to JSON next to every result.
- Import rule (notebook buildability): internal imports only as
  `from perturber.x import name` — no `import perturber.x as y`. New modules must
  be added to `ORDER` in `scripts/build_notebook.py` in dependency order.
- Presets (`smoke`/`local`/`kaggle`) in `config.get_preset` size single
  experiments; the sweep uses its own reduced fidelity in `runner.sweep_fit_config`
  (12 restarts, trimmed curriculum, `sweep_substeps(sigma)`). Grids are separate
  from presets (`runner.threshold_grid`: `smoke`/`core`/`kaggle`). Code paths are
  otherwise identical.
- The **null model fits 1 restart** (it has no randomized parameters — all
  restarts are identical); the perturber uses the full multi-start. Both get the
  same curriculum budget so the held-out detection comparison stays fair.
- The smoke preset is a real test: `run_experiment.py --preset smoke` asserts
  log-mass error < 0.5 and detection. Run it before committing changes to fit
  logic.

## Known risks / gotchas

- **Nonconvex search**: hidden-orbit recovery has local minima. Trust the
  multi-start spread (top-3 restart agreement is reported), never a single fit.
- **Mass–distance degeneracy**: short arcs can't separate a heavy-far from a
  light-near perturber; this is physics, quantified by the arc-length sweep.
- **Sweep-boundary interpretation**: a non-detection at the reduced sweep
  fidelity is only a *physical* threshold if the restarts disagree (genuine
  non-identifiability). If restarts agree but the fit is poor, suspect
  under-optimization — bump restarts/curriculum before claiming the boundary.
- **Ultra-low-noise optimization wall**: at sigma=1e-5 the fit cannot reach the
  noise floor in a practical step budget (first-order optimization converges too
  slowly to a ~1e-5 tolerance; a fidelity sweep confirmed even 800 curriculum
  steps stall at ~1e-2 residual). This yields *spurious* non-detections, not a
  physical boundary, so the study grid uses realistic noise only (1e-4..1e-3).
  Revisiting sigma<=1e-5 would need a second-order/LBFGS polish stage.
- **Detection claims** must beat the null model on the *held-out* window
  (`evaluate.compare`); train-window fit quality alone is the legacy failure mode.
- **Multiple shooting is OFF by default** (`FitConfig.ms_steps = 0`). On the
  near-Keplerian M1 systems the single-shooting curriculum recovers log-mass to
  ~0.01 on its own; the current MS tuning destabilizes an already-good fit
  (observed: curriculum reaches the right mass, then MS blows chi² up ~1000×).
  It's kept as an opt-in for later milestones' longer/chaotic arcs and will need
  retuning (lr on segment states, `lambda_cont` schedule) before it helps.
  If enabling: train intervals must divide `FitConfig.n_segments` — arc lengths
  in presets/sweeps assume 0.75·arc·40 intervals divide by 8.
- Gradient clipping and the short-arc curriculum are load-bearing — chaos-amplified
  gradients through long integrations blow up without them.
