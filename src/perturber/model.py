"""Learnable models: hidden perturber (mass + orbital elements) and the null model.

Both expose the same interface, batched over B restart candidates:
    masses()        -> (B, N)
    initial_state() -> (B, N, 4)
    delta_penalty() -> (B,)   weak prior pulling visible-state deltas to zero
"""
import math

import torch
import torch.nn as nn


class NullModel(nn.Module):
    """No hidden body — only small corrections to the observed initial states.

    This is the baseline any detection claim must beat on held-out data.
    """

    def __init__(self, n_restarts, vis_state0, masses_visible, sigma, dt_obs):
        super().__init__()
        b = n_restarts
        self.register_buffer("base", vis_state0.unsqueeze(0).repeat(b, 1, 1))  # (B, Nv, 4)
        self.register_buffer("m_vis", masses_visible.unsqueeze(0).repeat(b, 1))
        # Raw deltas are unitless; scales convert to physical units so one lr fits all.
        dt = vis_state0.dtype                             # float64 in fitting (project rule)
        scale = torch.tensor([5 * sigma, 5 * sigma,
                              10 * sigma / dt_obs, 10 * sigma / dt_obs], dtype=dt)
        self.register_buffer("delta_scale", scale)
        self.vis_delta = nn.Parameter(torch.zeros(b, vis_state0.shape[0], 4, dtype=dt))

    def visible_state0(self):
        return self.base + self.vis_delta * self.delta_scale

    def masses(self):
        return self.m_vis

    def initial_state(self):
        return self.visible_state0()

    def delta_penalty(self):
        return (self.vis_delta ** 2).mean(dim=(1, 2))


class PerturberModel(NullModel):
    """Null model + one hidden body: learnable log10-mass and 2-D orbital
    elements (log-a, bounded e, true anomaly, argument of periapsis)."""

    def __init__(self, n_restarts, vis_state0, masses_visible, sigma, dt_obs,
                 mass_init=1e-3, log_a_range=(0.4, 1.8), generator=None):
        super().__init__(n_restarts, vis_state0, masses_visible, sigma, dt_obs)
        b = n_restarts
        g = generator
        lo, hi = log_a_range
        dt = vis_state0.dtype                                  # float64 in fitting (project rule)
        self.log10_mass = nn.Parameter(
            torch.full((b,), math.log10(mass_init), dtype=dt))
        self.log_a = nn.Parameter(lo + (hi - lo) * torch.rand(b, generator=g, dtype=dt))
        self.raw_e = nn.Parameter(torch.full((b,), -2.0, dtype=dt))   # sigmoid/2 -> e ~ 0.06
        self.phase = nn.Parameter(2 * math.pi * torch.rand(b, generator=g, dtype=dt))
        self.omega = nn.Parameter(2 * math.pi * torch.rand(b, generator=g, dtype=dt))

    def hidden_mass(self):
        return 10.0 ** self.log10_mass                        # (B,)

    def hidden_elements(self):
        return dict(a=torch.exp(self.log_a),
                    e=0.5 * torch.sigmoid(self.raw_e),
                    nu=self.phase, omega=self.omega)

    def masses(self):
        return torch.cat([self.m_vis, self.hidden_mass().unsqueeze(-1)], dim=-1)

    def initial_state(self):
        vis = self.visible_state0()                           # (B, Nv, 4)
        el = self.hidden_elements()
        a, e, nu, om = el["a"], el["e"], el["nu"], el["omega"]
        mu = self.m_vis[:, 0] + self.hidden_mass()

        p = a * (1.0 - e * e)
        r = p / (1.0 + e * torch.cos(nu))
        pos_o = torch.stack([r * torch.cos(nu), r * torch.sin(nu)], dim=-1)
        vf = torch.sqrt(mu / p)
        vel_o = vf.unsqueeze(-1) * torch.stack([-torch.sin(nu), e + torch.cos(nu)], dim=-1)
        c, s = torch.cos(om), torch.sin(om)
        rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], dim=-2)
        pos = (rot @ pos_o.unsqueeze(-1)).squeeze(-1) + vis[:, 0, :2]   # orbit the star
        vel = (rot @ vel_o.unsqueeze(-1)).squeeze(-1) + vis[:, 0, 2:]

        hidden = torch.cat([pos, vel], dim=-1).unsqueeze(1)   # (B, 1, 4)
        return torch.cat([vis, hidden], dim=1)


if __name__ == "__main__":
    # Self-check: element -> state round-trip against the numpy version in data.py
    import numpy as np
    from perturber.data import kepler_state

    vis0 = torch.zeros(3, 4, dtype=torch.float64)             # star at rest at origin
    m_vis = torch.tensor([1.0, 1e-4, 3e-5], dtype=torch.float64)
    m = PerturberModel(2, vis0, m_vis, sigma=1e-4, dt_obs=0.157).double()
    with torch.no_grad():
        m.log10_mass.fill_(-3.0)
        m.log_a.fill_(np.log(3.2))
        m.raw_e.fill_(-2.0)
        m.phase.fill_(1.0)
        m.omega.fill_(0.5)
    st = m.initial_state()
    e_val = float(0.5 * torch.sigmoid(torch.tensor(-2.0)))
    pos_np, vel_np = kepler_state(3.2, e_val, 1.0, 0.5, mu=1.0 + 1e-3)
    err = max(np.abs(st[0, 3, :2].detach().numpy() - pos_np).max(),
              np.abs(st[0, 3, 2:].detach().numpy() - vel_np).max())
    # params now built in vis0's float64 dtype, so agreement is to ~1e-12
    assert err < 1e-7, f"element->state mismatch vs numpy: {err}"
    assert st.shape == (2, 4, 4)
    assert m.masses().shape == (2, 4)
    print("[model] self-checks passed")
