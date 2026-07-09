"""Transit-time extraction from a 2-D N-body trajectory.

Edge-on geometry: the observer sits at y = -infinity looking along +y, so the
sky plane is the x-axis. A planet transits its star when the sky-projected
separation x_planet - x_star crosses zero with the planet in front of the star
(y_planet < y_star). Transit mid-times are the linearly-interpolated zero
crossings — accurate to well below a minute at the dense sampling used here, and
(more importantly) consistent under parameter perturbations, which is what the
finite-difference TTV Jacobian needs.
"""
import numpy as np


def find_transit_times(t, planet_xy, star_xy, front=1.0):
    """t (T,), planet_xy (T,2), star_xy (T,2) -> transit times (1-D array).

    A transit is a zero crossing of dx = x_planet - x_star with the planet in
    front. `front` sets which side the observer is on: with front=+1 the transit
    node is where dy = y_planet - y_star < 0 (observer at y=-inf); front=-1 picks
    the opposite node (observer at y=+inf). For an exactly edge-on model both are
    geometrically valid transits, so the correct `front` is the one whose TTV
    pattern matches the data — see scripts/run_kepler9_identifiability.py, which
    fits both and keeps the better."""
    dx = planet_xy[:, 0] - star_xy[:, 0]
    dy = planet_xy[:, 1] - star_xy[:, 1]
    times = []
    for i in range(len(t) - 1):
        a, b = dx[i], dx[i + 1]
        if a == 0.0:
            if front * dy[i] < 0.0:
                times.append(float(t[i]))
            continue
        if a * b < 0.0:                                  # sign change -> crossing
            frac = -a / (b - a)                          # in [0,1]
            y_cross = dy[i] + frac * (dy[i + 1] - dy[i])
            if front * y_cross < 0.0:                    # planet in front of star
                times.append(float(t[i] + frac * (t[i + 1] - t[i])))
    return np.array(times)


def match_epochs(model_times, n_obs):
    """Map the first n_obs+1 model transits to transit numbers 0..n_obs.
    Returns model transit times indexed by epoch (length n_obs+1), or fewer if
    the integration was too short."""
    return model_times[:n_obs + 1]


if __name__ == "__main__":
    # A single planet on a circular orbit (a=1) around a fixed heavy star should
    # transit once per orbit, spaced by the orbital period (2*pi in G=1 units).
    from perturber.dynamics import ode_rhs
    from perturber.data import kepler_state
    from scipy.integrate import solve_ivp

    m = np.array([1.0, 1e-6])                            # star + test planet
    pos, vel = kepler_state(a=1.0, e=0.0, nu=0.0, omega=0.0, mu=1.0)
    state0 = np.concatenate([[0.0, 0.0], pos, [0.0, 0.0], vel])
    t_end = 6 * 2 * np.pi
    t = np.linspace(0, t_end, 6 * 500)
    sol = solve_ivp(ode_rhs, (0, t_end), state0, t_eval=t, args=(m,),
                    method="DOP853", rtol=1e-10, atol=1e-12)
    n = len(m)
    posT = np.moveaxis(sol.y[:2 * n].reshape(n, 2, -1), -1, 0)   # (T,n,2)
    tt = find_transit_times(t, posT[:, 1, :], posT[:, 0, :])
    periods = np.diff(tt)
    print(f"found {len(tt)} transits; spacing {periods.mean():.4f} "
          f"(expected {2 * np.pi:.4f})")
    assert 5 <= len(tt) <= 6, f"expected ~6 transits, got {len(tt)}"
    assert np.allclose(periods, 2 * np.pi, rtol=1e-3), "transit spacing != period"
    print("[transits] self-check passed")
