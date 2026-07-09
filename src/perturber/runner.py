"""Experiment orchestration: generate -> fit (perturber + null) -> evaluate -> plots.

Shared by the CLI scripts and the built Kaggle notebook.
"""
import hashlib
import json
import os
from dataclasses import asdict, replace

import numpy as np

from perturber.config import get_preset, dump_configs
from perturber.data import generate
from perturber.evaluate import compare, evaluate
from perturber.fit import fit, make_model
from perturber.plots import ensure_dir, plot_experiment, plot_threshold


def run_single_experiment(sys_cfg, fit_cfg, outdir, make_plot=True, verbose=True):
    """One full experiment. Returns the combined metrics dict."""
    ensure_dir(outdir)
    ds = generate(sys_cfg)
    if verbose:
        print(f"[run] arc {sys_cfg.n_periods:g} periods, {len(ds.t)} obs, "
              f"sigma {sys_cfg.sigma:.0e}, hidden mass {sys_cfg.hidden_mass:.0e}")

    results = {}
    for kind in ("perturber", "null"):
        # The null model has no randomized parameters — every restart is
        # identical, so one is enough. Only the perturber benefits from
        # multi-start over the nonconvex hidden-orbit landscape.
        cfg_k = fit_cfg if kind == "perturber" else replace(fit_cfg, n_restarts=1)
        if verbose:
            print(f"[fit] {kind} model, {cfg_k.n_restarts} restarts")
        model = make_model(kind, ds, cfg_k)
        res = fit(model, ds, cfg_k, verbose=verbose)
        metrics, traj = evaluate(model, res, ds, cfg_k)
        results[kind] = (metrics, traj)

    metrics_pert, traj_pert = results["perturber"]
    metrics_null, traj_null = results["null"]
    cmp = compare(metrics_pert, metrics_null)

    out = {"perturber": metrics_pert, "null": metrics_null, "comparison": cmp,
           "system": asdict(sys_cfg)}
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    dump_configs(sys_cfg, fit_cfg, os.path.join(outdir, "config.json"))
    if make_plot:
        p = plot_experiment(ds, traj_pert, traj_null, metrics_pert, metrics_null,
                            cmp, os.path.join(outdir, "experiment.png"))
        if verbose:
            print(f"[run] figure -> {p}")

    if verbose:
        print(f"[run] mass true {metrics_pert['mass_true']:.2e} "
              f"recovered {metrics_pert['mass_recovered']:.2e} "
              f"(log err {metrics_pert['log_mass_error']:.3f}) | "
              f"2dlogL {cmp['loglike_ratio_stat']:.1f} "
              f"{'DETECTED' if cmp['detected'] else 'not detected'}")
    return out


def threshold_grid(preset):
    """(masses, sigmas, arcs, seeds) for the sweep.

    'kaggle' is the full science grid (5x3x3x3 = 135 cells). 'core' drops the
    unrealistically-low-noise row and one seed for a ~feasible 4x2x3x2 = 48-cell
    map. 'smoke' is a 2-cell sanity grid.
    """
    if preset == "smoke":
        return [1e-2, 3e-3], [1e-4], [4.0], [0]
    if preset == "core":
        return ([1e-2, 1e-3, 3e-4, 1e-4], [1e-4, 1e-3], [8.0, 16.0, 40.0], [0, 1])
    # full grid — realistic noise only. sigma=1e-5 (10 ppm astrometry) is excluded:
    # it is physically unrealistic AND a first-order fit can't reach its noise floor
    # in a practical step budget, producing spurious non-detections rather than a
    # physical boundary (confirmed by a fidelity sweep on m=3e-3/arc=8).
    # Mass range extends below the visible planets (1e-4, 3e-5) down to 3e-6 to
    # locate the actual detection floor (detection is robust down to ~1e-4).
    return ([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6],
            [1e-4, 3e-4, 1e-3],
            [8.0, 16.0, 40.0],
            [0, 1, 2])


def sweep_substeps(sigma):
    """RK4 substeps chosen so integration truncation error stays well below the
    noise floor (RK4 error ~ substeps^-4; empirically ~6e-5 at substeps=2)."""
    if sigma <= 1e-5:
        return 6
    if sigma <= 3e-4:      # covers 1e-4 and 3e-4: integration error ~2.6e-6 << noise
        return 4
    return 2               # sigma >= 1e-3


def sweep_fit_config(base_fit, sigma, seed):
    """Reduced-fidelity fit for sweep cells: CPU (3x faster than the T400 GPU on
    this Python-loop-bound workload), 8 restarts, trimmed curriculum, no MS,
    integration fidelity scaled to the noise."""
    return replace(base_fit, seed=seed, device="cpu", n_restarts=12, ms_steps=0,
                   substeps=sweep_substeps(sigma),
                   curriculum=((0.25, 120), (0.5, 120), (1.0, 250)))


def _collect_and_plot(outdir):
    """Render the heatmap from every cell JSON present in outdir (robust to
    sharding and cross-session resume — always plots the full merged set)."""
    cells = []
    for fn in sorted(os.listdir(outdir)):
        if fn.endswith(".json") and fn != "threshold_meta.json":
            with open(os.path.join(outdir, fn)) as f:
                obj = json.load(f)
            if "mass" in obj:  # a cell record, not some other artifact
                cells.append(obj)
    if cells:
        plot_threshold(cells, os.path.join(outdir, "threshold.png"))
    return cells


def sweep_cell_specs(preset, outdir, shard=0, n_shards=1, grid=None):
    """Build the (picklable) work list for the sweep. Each spec carries the
    fully-resolved sys/fit configs and the cell's output path — enough for a
    worker process to run it with no shared state."""
    grid = grid or ("smoke" if preset == "smoke" else preset)
    base_sys, base_fit = get_preset("smoke" if preset == "smoke" else "kaggle")
    masses, sigmas, arcs, seeds = threshold_grid(grid)
    todo = [(m, s, a, sd) for m in masses for s in sigmas for a in arcs for sd in seeds]

    specs = []
    for i, (m, s, a, sd) in enumerate(todo):
        if i % n_shards != shard:
            continue
        sys_cfg = replace(base_sys, hidden_mass=m, sigma=s, n_periods=a, seed=sd)
        fit_cfg = replace(base_fit, seed=sd) if preset == "smoke" \
            else sweep_fit_config(base_fit, s, sd)
        key = hashlib.md5(json.dumps(
            {**asdict(sys_cfg), "fit": asdict(fit_cfg)}, sort_keys=True
        ).encode()).hexdigest()[:12]
        specs.append({"sys": sys_cfg, "fit": fit_cfg, "key": key,
                      "cell_path": os.path.join(outdir, f"{key}.json"),
                      "run_dir": os.path.join(outdir, f"run_{key}")})
    return specs, len(todo)


def execute_cell(spec, verbose=False, threads=None):
    """Run one sweep cell (skips if its JSON already exists). Module-level and
    self-contained so it can be dispatched to a worker process. Returns the
    cell dict."""
    if os.path.exists(spec["cell_path"]):
        with open(spec["cell_path"]) as f:
            return json.load(f)
    if threads is not None:
        import torch
        torch.set_num_threads(threads)   # avoid oversubscription under a process pool
    sys_cfg, fit_cfg = spec["sys"], spec["fit"]
    print(f"[cell] m={sys_cfg.hidden_mass:.0e} sigma={sys_cfg.sigma:.0e} "
          f"arc={sys_cfg.n_periods:g} seed={sys_cfg.seed} "
          f"substeps={fit_cfg.substeps}", flush=True)
    r = run_single_experiment(sys_cfg, fit_cfg, spec["run_dir"],
                              make_plot=False, verbose=verbose)
    cell = {"mass": sys_cfg.hidden_mass, "sigma": sys_cfg.sigma,
            "n_periods": sys_cfg.n_periods, "seed": sys_cfg.seed,
            "log_mass_error": r["perturber"]["log_mass_error"],
            "detected": r["comparison"]["detected"],
            "loglike_ratio_stat": r["comparison"]["loglike_ratio_stat"],
            "delta_rmse_test": r["comparison"]["delta_rmse_test"]}
    with open(spec["cell_path"], "w") as f:
        json.dump(cell, f, indent=2)
    return cell


def run_threshold_study(preset, outdir, verbose=False, shard=0, n_shards=1,
                        grid=None):
    """Resumable, sharded detection-threshold sweep (sequential).

    One JSON per grid cell; existing cells are skipped, so the sweep resumes
    after any interruption and can be split across sessions/machines via
    (shard, n_shards): each shard runs cell i where i % n_shards == shard.
    Copy prior sessions' cell JSONs into outdir before starting to resume them.
    For local multi-core runs use scripts/run_sweep_parallel.py instead.
    """
    ensure_dir(outdir)
    specs, n_total = sweep_cell_specs(preset, outdir, shard, n_shards, grid)
    print(f"[sweep] {n_total} cells total, shard {shard}/{n_shards} "
          f"handles {len(specs)}", flush=True)
    for j, spec in enumerate(specs):
        print(f"[sweep {j + 1}/{len(specs)}]", flush=True)
        execute_cell(spec, verbose=verbose)
    cells = _collect_and_plot(outdir)
    print(f"[sweep] {len(cells)} cells present in {outdir} -> threshold.png")
    return cells
