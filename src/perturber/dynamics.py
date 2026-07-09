"""N-body gravitational dynamics: numpy (ground truth) and torch (differentiable).

State layout: (..., N, 4) with [:2] = position (x, y) and [2:] = velocity (vx, vy).
G = 1 everywhere.
"""
import numpy as np
import torch

G = 1.0


# ── numpy side (ground-truth integration via scipy) ──────────────────────────

def accel_numpy(pos, masses, extra_force=None):
    """pos (N, 2), masses (N,) -> acceleration (N, 2). Pure gravity, no softening.

    extra_force(pos) -> (N, 2) adds a non-Newtonian term (M2 injected law)."""
    diff = pos[None, :, :] - pos[:, None, :]          # diff[i, j] = r_j - r_i
    d2 = (diff ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    inv_d3 = d2 ** -1.5
    acc = G * (diff * (masses[None, :, None] * inv_d3[:, :, None])).sum(axis=1)
    if extra_force is not None:
        acc = acc + extra_force(pos)
    return acc


def ode_rhs(t, y, masses, extra_force=None):
    """Flat RHS for scipy.integrate.solve_ivp. y = [pos.ravel(), vel.ravel()]."""
    n = len(masses)
    pos = y[: 2 * n].reshape(n, 2)
    vel = y[2 * n:].reshape(n, 2)
    return np.concatenate([vel.ravel(),
                           accel_numpy(pos, masses, extra_force=extra_force).ravel()])


# ── torch side (differentiable, batched) ──────────────────────────────────────

def accel_torch(pos, masses, softening=0.0, extra_force=None):
    """pos (..., N, 2), masses (..., N) or (N,) -> acceleration (..., N, 2).

    softening clamps pairwise distance^2 (fitted dynamics only; truth uses 0).
    extra_force(pos, vel=None) is the milestone-2 hook for a neural residual force.
    """
    diff = pos.unsqueeze(-3) - pos.unsqueeze(-2)       # (..., N, N, 2), diff[i, j] = r_j - r_i
    d2 = (diff ** 2).sum(-1)                           # (..., N, N)
    n = pos.shape[-2]
    eye = torch.eye(n, dtype=pos.dtype, device=pos.device)
    d2 = d2.clamp(min=softening ** 2) + eye            # +eye avoids 0^-1.5 on the diagonal
    inv_d3 = d2.pow(-1.5) * (1.0 - eye)                # zero self-interaction
    m_j = masses.unsqueeze(-2)                         # (..., 1, N)
    acc = G * (diff * (m_j * inv_d3).unsqueeze(-1)).sum(-2)
    if extra_force is not None:
        acc = acc + extra_force(pos)
    return acc


def rhs_torch(state, masses, softening=0.0, extra_force=None):
    """state (..., N, 4) -> d(state)/dt (..., N, 4)."""
    pos, vel = state[..., :2], state[..., 2:]
    acc = accel_torch(pos, masses, softening=softening, extra_force=extra_force)
    return torch.cat([vel, acc], dim=-1)


def total_energy(state, masses):
    """Diagnostic: kinetic + potential energy. state (N, 4) numpy, masses (N,)."""
    pos, vel = state[:, :2], state[:, 2:]
    ke = 0.5 * (masses * (vel ** 2).sum(-1)).sum()
    diff = pos[None, :, :] - pos[:, None, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pe = -0.5 * G * (masses[:, None] * masses[None, :] / d).sum()
    return ke + pe


if __name__ == "__main__":
    # Self-check: torch acceleration matches numpy on random states.
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(4, 2))
    m = np.abs(rng.normal(size=4)) + 0.1
    a_np = accel_numpy(pos, m)
    a_t = accel_torch(torch.tensor(pos), torch.tensor(m)).numpy()
    err = np.abs(a_np - a_t).max()
    assert err < 1e-12, f"torch/numpy acceleration mismatch: {err}"

    # Batched shape check
    posb = torch.randn(3, 5, 4, 2, dtype=torch.float64)
    mb = torch.rand(5, 4, dtype=torch.float64) + 0.1
    out = accel_torch(posb, mb)
    assert out.shape == (3, 5, 4, 2)
    print("[dynamics] self-checks passed")
