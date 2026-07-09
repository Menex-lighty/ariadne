"""Batched differentiable fixed-step RK4 integrator (pure torch)."""
import torch
from torch.utils.checkpoint import checkpoint

from perturber.dynamics import rhs_torch


def _interval(s, h, masses, substeps, softening, extra_force):
    """Advance one observation interval with `substeps` RK4 steps."""
    for _ in range(substeps):
        k1 = rhs_torch(s, masses, softening=softening, extra_force=extra_force)
        k2 = rhs_torch(s + 0.5 * h * k1, masses, softening=softening, extra_force=extra_force)
        k3 = rhs_torch(s + 0.5 * h * k2, masses, softening=softening, extra_force=extra_force)
        k4 = rhs_torch(s + h * k3, masses, softening=softening, extra_force=extra_force)
        s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def rk4_integrate(state0, masses, t_grid, substeps=4, softening=0.0,
                  extra_force=None, checkpointing=False):
    """Integrate N-body dynamics on a fixed time grid.

    state0   (..., N, 4)   initial state (arbitrary batch dims)
    masses   (..., N)      broadcastable against state0's batch dims
    t_grid   (T,)          observation times (need not be uniform)
    substeps               RK4 steps between consecutive grid points

    Returns trajectory (T, ..., N, 4) — time axis first, gradients flow to
    state0 and masses through the unrolled loop.

    With `checkpointing`, the inner sub-steps of each interval are recomputed
    during backward instead of stored, bounding peak memory to ~one interval's
    graph regardless of arc length (essential for long arcs / many parallel
    fits). Costs one extra forward pass (~1.5x wall). Skipped automatically when
    the graph isn't needed (no grad).
    """
    out = [state0]
    s = state0
    use_ckpt = checkpointing and torch.is_grad_enabled() and state0.requires_grad
    for i in range(len(t_grid) - 1):
        h = (t_grid[i + 1] - t_grid[i]) / substeps
        if use_ckpt:
            s = checkpoint(_interval, s, h, masses, substeps, softening,
                           extra_force, use_reentrant=False)
        else:
            s = _interval(s, h, masses, substeps, softening, extra_force)
        out.append(s)
    return torch.stack(out, dim=0)


if __name__ == "__main__":
    # Verify RK4 against the scipy ground truth on the smoke system: the
    # integration error at the chosen substeps must sit far below the noise.
    import numpy as np
    from perturber.config import get_preset
    from perturber.data import generate

    cfg, fit_cfg = get_preset("smoke")
    ds = generate(cfg)
    masses = torch.tensor(np.array(list(cfg.masses_visible) + [cfg.hidden_mass]))
    s0 = torch.tensor(ds.truth[0])
    t = torch.tensor(ds.t)
    for sub in (2, 4):
        traj = rk4_integrate(s0, masses, t, substeps=sub)
        err = np.abs(traj.numpy()[:, :, :2] - ds.truth[:, :, :2]).max()
        print(f"[integrators] substeps={sub}: max position error {err:.2e} "
              f"(noise sigma {cfg.sigma:.0e})")
        if sub == fit_cfg.substeps:
            assert err < 0.1 * cfg.sigma, "RK4 error not far below noise floor"
    print("[integrators] self-checks passed")
