"""Experiment configuration and presets.

Units: G = 1, star mass = 1, planet-1 semi-major axis = 1.
Planet-1 period is therefore 2*pi time units.
"""
from dataclasses import dataclass, asdict, replace  # noqa: F401  (replace used by callers)
import json

import numpy as np
import torch


@dataclass
class SystemConfig:
    # Visible bodies: index 0 is the star, then planets.
    masses_visible: tuple = (1.0, 1e-4, 3e-5)
    planet_a: tuple = (1.0, 1.9)
    planet_e: tuple = (0.0, 0.05)
    planet_phase: tuple = (0.0, 2.1)   # true anomaly at t0 [rad]
    planet_omega: tuple = (0.0, 0.8)   # argument of periapsis [rad]
    # Hidden perturber (ground truth; never shown to the model)
    hidden_mass: float = 1e-3
    hidden_a: float = 3.2
    hidden_e: float = 0.1
    hidden_phase: float = 1.0
    hidden_omega: float = 0.5
    # Observations
    n_periods: float = 16.0            # arc length in planet-1 periods
    obs_per_period: int = 40
    sigma: float = 1e-4                # position noise, absolute units
    train_frac: float = 0.75
    seed: int = 0
    # M2 only: inject a non-Newtonian central force to discover, e.g.
    # {"kind": "power_law", "params": {"alpha": 2e-3, "n": 4.0}}. None => pure M1.
    m2_force: dict = None

    @property
    def n_visible(self):
        return len(self.masses_visible)


@dataclass
class FitConfig:
    n_restarts: int = 16
    # (train-arc fraction, adam steps) single-shooting curriculum phases
    curriculum: tuple = ((0.25, 200), (0.5, 200), (1.0, 400))
    # Multiple shooting is OFF by default: on the near-Keplerian M1 systems the
    # single-shooting curriculum converges cleanly (recovers log-mass to ~0.01),
    # and the current MS tuning destabilizes an already-good fit. Retained as an
    # opt-in (set ms_steps>0) for the longer/chaotic arcs of later milestones,
    # where it will need retuning. See CLAUDE.md "Known risks".
    ms_steps: int = 0                  # multiple-shooting steps on full train arc (0 = skip)
    n_segments: int = 8                # multiple-shooting segment count
    lambda_cont: float = 1e2           # continuity penalty weight (annealed up 10x mid-phase)
    lr_elements: float = 1e-2
    lr_mass: float = 3e-2
    clip: float = 1.0
    substeps: int = 4                  # RK4 steps between consecutive observations
    softening: float = 1e-2            # distance clamp in the fitted dynamics only
    delta_prior: float = 1e-3          # weight on visible initial-state delta penalty
    mass_init: float = 1e-3
    log_a_range: tuple = (0.4, 1.8)    # restart prior: ln(a) uniform -> a in [1.5, 6.0]
    seed: int = 0
    device: str = "auto"               # "auto" | "cpu" | "cuda"


def resolve_device(spec):
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def get_preset(name):
    """Return (SystemConfig, FitConfig) for a named preset."""
    if name == "smoke":
        # Easy, fast case: big perturber, short arc, few restarts. ~1-2 min on CPU.
        sys_cfg = SystemConfig(n_periods=4.0, hidden_mass=1e-2, sigma=1e-4)
        fit_cfg = FitConfig(n_restarts=4, curriculum=((0.5, 80), (1.0, 120)),
                            ms_steps=0, n_segments=1, device="cpu")
    elif name == "local":
        sys_cfg = SystemConfig()
        fit_cfg = FitConfig()
    elif name == "kaggle":
        sys_cfg = SystemConfig()
        fit_cfg = FitConfig(device="auto")
    else:
        raise ValueError(f"unknown preset '{name}'")
    return sys_cfg, fit_cfg


def dump_configs(sys_cfg, fit_cfg, path):
    with open(path, "w") as f:
        json.dump({"system": asdict(sys_cfg), "fit": asdict(fit_cfg)}, f, indent=2)


def planet1_period(sys_cfg):
    mu = sys_cfg.masses_visible[0] + sys_cfg.masses_visible[1]
    return 2 * np.pi * np.sqrt(sys_cfg.planet_a[0] ** 3 / mu)
