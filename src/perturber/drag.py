"""Milestone 3 — orbit-with-drag forward model and the density-ballistic-
coefficient degeneracy.

A LEO satellite feels central gravity plus atmospheric drag:

    a = -mu * r/|r|^3  -  0.5 * B * rho(r) * |v| * v

with B = C_d A / m the ballistic coefficient and rho the local density. The orbit
decays; crucially the drag term depends only on the PRODUCT B*rho, so from one
satellite's trajectory density and ballistic coefficient are perfectly
degenerate — you measure B*rho, never B or rho alone. An external prior on B (from
the spacecraft's known area/mass) breaks it and yields rho; this is the drag-
domain instance of the identifiability + prior-aware story (cf. Kepler-9 mass
scale, force-form recovery). Scaled units: mu = 1, initial circular orbit radius
= 1 (period 2*pi).
"""
import numpy as np

MU = 1.0


def _rho_exp(r, rho0=1.0, H=0.5, r0=1.0):
    """Exponential atmosphere in radius: rho = rho0 * exp(-(r-r0)/H). A gentle
    scale height (H=0.5 of the orbit radius) keeps the decaying orbit stable —
    a tiny H makes rho blow up as r drops and the orbit crashes numerically."""
    return rho0 * np.exp(-(r - r0) / H)


def accel(state, B, rho0, rho_fn=_rho_exp):
    """state (4,) = x,y,vx,vy -> acceleration (2,). Central gravity + drag."""
    pos, vel = state[:2], state[2:]
    r = np.hypot(*pos)
    a_grav = -MU * pos / r ** 3
    v = np.hypot(*vel)
    a_drag = -0.5 * B * rho_fn(r, rho0) * v * vel
    return np.concatenate([vel, a_grav + a_drag])


def simulate(B, rho0, n_orbits=40, steps_per_orbit=200, e0=0.0, rho_fn=_rho_exp):
    """Integrate a decaying orbit (fixed-step RK4). Returns (t, positions (T,2))."""
    # start on a mildly eccentric orbit at r=1
    r0 = 1.0
    v_circ = np.sqrt(MU / r0)
    s = np.array([r0, 0.0, 0.0, v_circ * np.sqrt(1 + e0)])
    T = n_orbits * 2 * np.pi
    n = n_orbits * steps_per_orbit
    dt = T / n
    out = [s[:2].copy()]
    ts = [0.0]
    for i in range(n):
        k1 = accel(s, B, rho0, rho_fn)
        k2 = accel(s + 0.5 * dt * k1, B, rho0, rho_fn)
        k3 = accel(s + 0.5 * dt * k2, B, rho0, rho_fn)
        k4 = accel(s + dt * k3, B, rho0, rho_fn)
        s = s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(s[:2].copy())
        ts.append((i + 1) * dt)
    return np.array(ts), np.array(out)


def decay_jacobian(B0, rho0, sigma, n_orbits=40, sample_every=50):
    """Finite-difference Jacobian of the (sub-sampled) observed positions w.r.t.
    (B, rho0), weighted by 1/sigma. Rows = 2*observations, cols = [B, rho0].
    Its conditioning shows the B*rho degeneracy (one near-zero singular value)."""
    def obs(B, rho0):
        _, p = simulate(B, rho0, n_orbits)
        return p[::sample_every].ravel()
    base = obs(B0, rho0)
    dB, dr = 0.02 * B0, 0.02 * rho0
    JB = (obs(B0 + dB, rho0) - obs(B0 - dB, rho0)) / (2 * dB)
    Jr = (obs(B0, rho0 + dr) - obs(B0, rho0 - dr)) / (2 * dr)
    J = np.stack([JB, Jr], axis=1) / sigma          # (2*obs, 2)
    return J, base


if __name__ == "__main__":
    from perturber.identifiability import fisher_information, marginal_sigma, prior_precision

    B0, rho0, sigma = 0.1, 0.02, 1e-3          # B*rho ~ 2e-3: weak, stable decay
    # sanity: the orbit decays (semi-major axis / radius shrinks), stably
    _, p = simulate(B0, rho0, n_orbits=40)
    r = np.hypot(p[:, 0], p[:, 1])
    r_start, r_end = r[:200].max(), r[-200:].max()   # apoapsis proxy, avoid phase
    print(f"orbit radius {r_start:.4f} -> {r_end:.4f} over 40 orbits (drag decay)")
    assert np.isfinite(r).all() and r.max() < 2.0, "orbit should stay bounded (stable)"
    assert r_end < r_start - 0.003, "drag should shrink the orbit"

    # the B*rho degeneracy: drag depends only on the PRODUCT, so the Fisher is
    # rank-deficient — the product is constrained, the B/rho ratio is not.
    J, _ = decay_jacobian(B0, rho0, sigma)
    F = fisher_information(J)
    ev, evec = np.linalg.eigh(F)                       # ascending
    cond = ev[-1] / ev[0] if ev[0] > 0 else np.inf
    # eigenvector of the degenerate (small) eigenvalue: B and rho move oppositely
    degen = evec[:, 0] / np.abs(evec[:, 0]).max()
    s_prior = marginal_sigma(F, prior_precision(2, 0, 0.10 * B0))   # 10% prior on B
    print(f"Fisher eigenvalues {ev[0]:.1e}, {ev[-1]:.1e}  -> cond {cond:.1e}")
    print(f"  degenerate direction (dB/B, d(rho)/rho) ~ [{degen[0]:+.2f}, {degen[1]:+.2f}]"
          f"  (B up <-> rho down: only the product B*rho is measured)")
    print(f"  + 10% B prior:  sigma_rho/rho {s_prior[1]/rho0:.3f}"
          f"  (a prior on B transfers straight to rho)")
    assert cond > 1e6, "B and rho are degenerate from one orbit (Fisher rank-deficient)"
    assert ev[0] / ev[-1] < 1e-6, "the B/rho ratio direction is unconstrained by drag data"
    assert degen[0] * degen[1] < 0, "the degenerate direction moves B and rho oppositely"
    assert s_prior[1] / rho0 < 0.2, "a B prior should recover rho to ~prior precision"
    print("[drag] self-check passed (exact B*rho degeneracy; a B prior recovers rho)")
