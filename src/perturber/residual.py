"""Milestone 2 — discover an unknown central force law from trajectories.

The unknown extra force is modelled as a central radial profile over a fixed
library of power-law terms:

    g(r) = sum_k c_k * r^(p_k)          a_extra(body i) = -g(r_i) * rhat_i

The coefficients c_k are fit *through the differentiable RK4 integrator* (so the
model structurally obeys the dynamics), with an L1 penalty that pushes most
coefficients to zero. Sequential thresholding (see symbolic.py) then prunes the
library to the few surviving terms — the discovered functional form. This is
symbolic regression through the simulation: the library is deliberately broad
(sub- to super-Newtonian powers), so selecting the right term is nontrivial.
"""
import numpy as np
import torch
import torch.nn as nn

from perturber.data import Dataset, estimate_visible_state0
from perturber.forces import radial_apply_torch
from perturber.integrators import rk4_integrate

# Default library: powers spanning steep-attractive to mild-repulsive, excluding
# p=-2 would hide a mass-degenerate term — keep it in so the fit must reject it.
DEFAULT_EXPONENTS = (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0)


class RadialBasisForce(nn.Module):
    """Learnable central force g(r) = sum_k c_k r^(p_k), star at body `star_idx`.

    Coefficients are scaled by `coef_scale` so they and the initial-state deltas
    sit at comparable magnitudes for a single optimizer/lr.
    """

    def __init__(self, exponents=DEFAULT_EXPONENTS, star_idx=0, coef_scale=1e-3,
                 active=None):
        super().__init__()
        self.exponents = tuple(exponents)
        self.star_idx = star_idx
        self.coef_scale = coef_scale
        self.register_buffer("p", torch.tensor(self.exponents, dtype=torch.float64))
        # active mask lets STLSQ freeze pruned terms at exactly zero
        if active is None:
            active = [True] * len(self.exponents)
        self.register_buffer("active",
                             torch.tensor(active, dtype=torch.float64))
        self.raw = nn.Parameter(torch.zeros(len(self.exponents), dtype=torch.float64))

    def coeffs(self):
        return self.raw * self.coef_scale * self.active

    def g_of_r(self, r):
        # r: (...,) -> g: (...,).  g(r) = sum_k c_k r^p_k
        c = self.coeffs()                                  # (K,)
        rp = r.unsqueeze(-1) ** self.p                     # (..., K)
        return (rp * c).sum(-1)

    def extra_force(self, pos):
        return radial_apply_torch(pos, self.star_idx, self.g_of_r)


class StateDeltas(nn.Module):
    """Small learnable corrections to the observation-estimated initial state."""

    def __init__(self, state0, sigma, dt_obs):
        super().__init__()
        self.register_buffer("base", state0)               # (N,4)
        scale = torch.tensor([5 * sigma, 5 * sigma,
                              10 * sigma / dt_obs, 10 * sigma / dt_obs],
                             dtype=torch.float64)
        self.register_buffer("scale", scale)
        self.delta = nn.Parameter(torch.zeros_like(state0))

    def state0(self):
        return self.base + self.delta * self.scale

    def penalty(self):
        return (self.delta ** 2).mean()


def fit_force(ds: Dataset, exponents=DEFAULT_EXPONENTS, active=None,
              curriculum=((0.3, 250), (0.6, 250), (1.0, 400)), lr=3e-2, l1=1e-3,
              substeps=4, device="cpu", fit_state=True, state0_true=None,
              verbose=False):
    """Fit the basis force (and initial-state deltas) through the integrator,
    with a short-arc curriculum (long precessing orbits are nonconvex — the
    curriculum walks the basin, as in M1).

    All bodies are observed in M2 (q_obs holds every body). If state0_true is
    given it is used as the fixed initial state (isolates force discovery from
    state estimation). Returns (model, state_module, history)."""
    dev = torch.device(device)
    masses = torch.tensor(np.array(ds.sys.masses_visible), dtype=torch.float64, device=dev)
    t = torch.tensor(ds.t, dtype=torch.float64, device=dev)
    q = torch.tensor(ds.q_obs, dtype=torch.float64, device=dev)          # (T,N,2)
    n_train = int(ds.train_mask.sum())
    dt_obs = float(ds.t[1] - ds.t[0])
    sigma = ds.sys.sigma

    if state0_true is not None:
        fixed_s0 = torch.tensor(state0_true, dtype=torch.float64, device=dev)
        fit_state = False
    else:
        fixed_s0 = None
    est = torch.tensor(estimate_visible_state0(ds), dtype=torch.float64, device=dev)
    model = RadialBasisForce(exponents, active=active).to(dev)
    state = StateDeltas(est, sigma, dt_obs).to(dev)

    params = list(model.parameters()) + (list(state.parameters()) if fit_state else [])
    hist = []
    for ph, (frac, steps) in enumerate(curriculum):
        n_sub = max(2, int(round(frac * (n_train - 1))) + 1)
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for it in range(steps):
            opt.zero_grad()
            s0 = fixed_s0 if fixed_s0 is not None else \
                (state.state0() if fit_state else est)
            traj = rk4_integrate(s0, masses, t[:n_sub], substeps=substeps,
                                 extra_force=model.extra_force)
            data = ((traj[:, :, :2] - q[:n_sub]) / sigma).pow(2).mean()
            reg = l1 * model.coeffs().abs().sum() / model.coef_scale
            loss = data + reg + (10.0 * state.penalty() if fit_state else 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            hist.append(float(data))
        if verbose:
            with torch.no_grad():
                c = model.coeffs().cpu().numpy()
            print(f"  [ph{ph} frac{frac}] data {data.item():.2f} "
                  f"| c={np.array2string(c, precision=2, suppress_small=True)}",
                  flush=True)
    return model, state, hist


if __name__ == "__main__":
    # With a perfect initial state the fit reproduces the trajectory to the noise
    # floor and recovers the radial PROFILE g(r) — but the coefficients spread
    # across the library rather than the sparse injected r^-4. That force-form
    # degeneracy is the M2 finding, quantified in perturber.identifiability; the
    # self-check therefore verifies the fit works, NOT exact form recovery.
    from perturber.config import SystemConfig
    from perturber.data import generate

    cfg = SystemConfig(masses_visible=(1.0, 1e-4, 3e-5), hidden_mass=0.0,
                       planet_a=(1.0, 1.9), planet_e=(0.25, 0.3), n_periods=8.0,
                       sigma=1e-4, seed=0,
                       m2_force={"kind": "power_law", "params": {"alpha": 2e-3, "n": 4.0}})
    ds = generate(cfg)
    s0 = ds.truth[0][:ds.n_visible]
    model, state, hist = fit_force(ds, state0_true=s0,
                                   curriculum=((0.5, 300), (1.0, 400)), verbose=True)
    c = model.coeffs().detach().numpy()
    n_eff = int((np.abs(c) > 0.05 * np.abs(c).max()).sum())
    print(f"  trajectory misfit {hist[-1]:.2f} (noise floor ~1); "
          f"{n_eff} of {len(c)} basis terms active (injected form has 1)")
    for p, ck in zip(model.exponents, c):
        print(f"  r^{int(p):+d}: {ck:+.2e}")
    assert hist[-1] < 20.0, "with true state the fit should reach near the noise floor"
    print("[residual] self-check passed (form degeneracy quantified in identifiability.py)")
