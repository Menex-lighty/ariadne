"""Synthetic dataset generation: ground truth, noisy observations, splits."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from perturber.config import SystemConfig, planet1_period
from perturber.dynamics import ode_rhs, total_energy


def kepler_state(a, e, nu, omega, mu):
    """2-D orbital elements -> (pos(2,), vel(2,)) relative to the central body."""
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(nu))
    pos_o = np.array([r * np.cos(nu), r * np.sin(nu)])
    vf = np.sqrt(mu / p)
    vel_o = vf * np.array([-np.sin(nu), e + np.cos(nu)])
    c, s = np.cos(omega), np.sin(omega)
    rot = np.array([[c, -s], [s, c]])
    return rot @ pos_o, rot @ vel_o


def initial_state(cfg: SystemConfig):
    """Full-system initial state (N, 4) in the centre-of-momentum frame.

    N = n_visible + 1; the hidden perturber is the last body.
    """
    m_star = cfg.masses_visible[0]
    bodies = [np.zeros(4)]  # star at origin, at rest (pre-COM correction)
    for i in range(1, cfg.n_visible):
        pos, vel = kepler_state(cfg.planet_a[i - 1], cfg.planet_e[i - 1],
                                cfg.planet_phase[i - 1], cfg.planet_omega[i - 1],
                                mu=m_star + cfg.masses_visible[i])
        bodies.append(np.concatenate([pos, vel]))
    pos, vel = kepler_state(cfg.hidden_a, cfg.hidden_e, cfg.hidden_phase,
                            cfg.hidden_omega, mu=m_star + cfg.hidden_mass)
    bodies.append(np.concatenate([pos, vel]))
    state = np.stack(bodies)                                  # (N, 4)

    masses = np.array(list(cfg.masses_visible) + [cfg.hidden_mass])
    com = (masses[:, None] * state).sum(0) / masses.sum()     # COM pos and momentum/M
    return state - com[None, :], masses


@dataclass
class Dataset:
    t: np.ndarray            # (T,) observation times
    truth: np.ndarray        # (T, N, 4) all bodies, noiseless (hidden body last)
    q_obs: np.ndarray        # (T, Nv, 2) noisy visible positions
    train_mask: np.ndarray   # (T,) bool — first train_frac of the arc
    test_mask: np.ndarray    # (T,) bool — held-out future window
    sys: SystemConfig

    @property
    def n_visible(self):
        return self.sys.n_visible


def generate(cfg: SystemConfig) -> Dataset:
    rng = np.random.default_rng(cfg.seed)
    state0, masses = initial_state(cfg)

    p1 = planet1_period(cfg)
    t_end = cfg.n_periods * p1
    n_obs = int(round(cfg.n_periods * cfg.obs_per_period)) + 1
    t = np.linspace(0.0, t_end, n_obs)

    extra = None
    if cfg.m2_force is not None:
        from perturber.forces import make_injected
        extra = make_injected(cfg.m2_force["kind"], cfg.m2_force["params"],
                              backend="numpy")

    sol = solve_ivp(ode_rhs, (0.0, t_end), _flat(state0),
                    t_eval=t, args=(masses, extra), method="DOP853",
                    rtol=1e-11, atol=1e-12)
    assert sol.success, sol.message
    n = len(masses)
    pos_t = np.moveaxis(sol.y[: 2 * n].reshape(n, 2, -1), -1, 0)   # (T, N, 2)
    vel_t = np.moveaxis(sol.y[2 * n:].reshape(n, 2, -1), -1, 0)
    truth = np.concatenate([pos_t, vel_t], axis=-1)                # (T, N, 4)

    nv = cfg.n_visible
    q_obs = truth[:, :nv, :2] + cfg.sigma * rng.standard_normal((len(t), nv, 2))

    n_train = int(round(cfg.train_frac * len(t)))
    train_mask = np.zeros(len(t), dtype=bool)
    train_mask[:n_train] = True

    # Energy-conservation sanity on the ground truth (pure-gravity only; the M2
    # injected force adds an unaccounted potential, so skip the check there).
    if cfg.m2_force is None:
        e0 = total_energy(truth[0], masses)
        e1 = total_energy(truth[-1], masses)
        assert abs(e1 - e0) / abs(e0) < 1e-8, "ground-truth integration drifted"

    return Dataset(t=t, truth=truth, q_obs=q_obs,
                   train_mask=train_mask, test_mask=~train_mask, sys=cfg)


def _flat(state):
    """(N, 4) -> flat [pos.ravel(), vel.ravel()] as ode_rhs expects."""
    return np.concatenate([state[:, :2].ravel(), state[:, 2:].ravel()])


def estimate_visible_state0(ds: Dataset, n_fit=11, deg=4):
    """Estimate visible bodies' initial positions/velocities from the first
    few noisy observations (quartic fit per coordinate — the higher degree
    keeps orbital-curvature truncation bias below the noise). Returns (Nv, 4)."""
    t = ds.t[:n_fit] - ds.t[0]
    out = np.zeros((ds.n_visible, 4))
    for b in range(ds.n_visible):
        for c in range(2):
            coef = np.polyfit(t, ds.q_obs[:n_fit, b, c], deg=deg)
            out[b, c] = coef[-1]           # value at t0
            out[b, 2 + c] = coef[-2]       # slope at t0
    return out


if __name__ == "__main__":
    from perturber.config import get_preset
    cfg, _ = get_preset("smoke")
    ds = generate(cfg)
    assert ds.truth.shape == (161, 4, 4)
    assert ds.q_obs.shape == (161, 3, 2)
    est = estimate_visible_state0(ds)
    err_p = np.abs(est[:, :2] - ds.truth[0, :3, :2]).max()
    err_v = np.abs(est[:, 2:] - ds.truth[0, :3, 2:]).max()
    assert err_p < 5 * cfg.sigma, f"position estimate too far off: {err_p}"
    assert err_v < 100 * cfg.sigma, f"velocity estimate too far off: {err_v}"
    print(f"[data] self-checks passed  (q0 err {err_p:.1e}, v0 err {err_v:.1e})")
