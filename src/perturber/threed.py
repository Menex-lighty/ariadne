"""3-D N-body dynamics and transit timing, for the Kepler-9 photodynamical fit.

The 2-D `perturber` core (state (N,4)) is kept intact; this module adds the
3-D pieces (state (N,6) = x,y,z,vx,vy,vz) needed to include orbital inclination
and node, which a real TTV system has and a 2-D model cannot represent.

Geometry: observer at Z = +infinity, line of sight along -Z, sky plane = (X,Y).
A transit is a local minimum of the sky-projected star-planet separation with the
planet in front (Z_planet > Z_star). i = 90 deg is edge-on (transiting).
"""
import numpy as np

G = 1.0


def elements_to_state_3d(a, e, inc, Omega, omega, nu, mu):
    """Classical orbital elements -> (pos(3,), vel(3,)) in the inertial frame.
    inc, Omega, omega, nu in radians. mu = G(M_center + m)."""
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(nu))
    # perifocal frame
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    vf = np.sqrt(mu / p)
    v_pf = np.array([-vf * np.sin(nu), vf * (e + np.cos(nu)), 0.0])
    # rotation perifocal -> inertial: Rz(Omega) Rx(inc) Rz(omega)
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(omega), np.sin(omega)
    Rz_O = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])
    Rx_i = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    Rz_w = np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
    R = Rz_O @ Rx_i @ Rz_w
    return R @ r_pf, R @ v_pf


def accel_numpy_3d(pos, masses):
    """pos (N,3), masses (N,) -> acceleration (N,3). Pure Newtonian gravity."""
    diff = pos[None, :, :] - pos[:, None, :]
    d2 = (diff ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    inv_d3 = d2 ** -1.5
    return G * (diff * (masses[None, :, None] * inv_d3[:, :, None])).sum(axis=1)


def ode_rhs_3d(t, y, masses):
    n = len(masses)
    pos = y[:3 * n].reshape(n, 3)
    vel = y[3 * n:].reshape(n, 3)
    return np.concatenate([vel.ravel(), accel_numpy_3d(pos, masses).ravel()])


def flat_3d(state):
    """(N,6) -> [pos.ravel(), vel.ravel()] for ode_rhs_3d."""
    return np.concatenate([state[:, :3].ravel(), state[:, 3:].ravel()])


def find_transit_times_3d(t, planet_xyz, star_xyz):
    """Local minima of the *squared* sky separation dX^2+dY^2 with the planet in
    front (dZ = Z_planet - Z_star > 0), refined by parabolic interpolation.

    Squared separation is used (not sqrt): at exactly edge-on the sky separation
    |dX| is V-shaped (a kink), which breaks a parabolic vertex fit; the squared
    separation is a smooth parabola near the minimum, so the vertex — the
    mid-transit time — is recovered accurately."""
    d = planet_xyz - star_xyz
    sky2 = d[:, 0] ** 2 + d[:, 1] ** 2
    dz = d[:, 2]
    times = []
    for i in range(1, len(t) - 1):
        if sky2[i] < sky2[i - 1] and sky2[i] < sky2[i + 1] and dz[i] > 0.0:
            y0, y1, y2 = sky2[i - 1], sky2[i], sky2[i + 1]
            denom = (y0 - 2 * y1 + y2)
            shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            dt = t[i + 1] - t[i]
            times.append(float(t[i] + shift * dt))
    return np.array(times)


if __name__ == "__main__":
    # Sanity: an edge-on (i=90) 3-D orbit must give the same transit period as a
    # 2-D orbit of the same (a,e) — the physics is orientation-independent.
    from scipy.integrate import solve_ivp

    m = np.array([1.0, 1e-6])
    pos, vel = elements_to_state_3d(a=1.0, e=0.0, inc=np.pi / 2, Omega=0.0,
                                    omega=0.0, nu=-np.pi / 2, mu=1.0)
    state0 = np.zeros((2, 6)); state0[1, :3] = pos; state0[1, 3:] = vel
    t_end = 6 * 2 * np.pi
    t = np.linspace(0, t_end, 6 * 600)
    sol = solve_ivp(ode_rhs_3d, (0, t_end), flat_3d(state0), t_eval=t, args=(m,),
                    method="DOP853", rtol=1e-10, atol=1e-12)
    posT = np.moveaxis(sol.y[:6].reshape(2, 3, -1), -1, 0)   # (T,2,3)
    tt = find_transit_times_3d(t, posT[:, 1, :], posT[:, 0, :])
    per = np.diff(tt)
    # an unperturbed 2-body orbit transits exactly periodically -> O-C ~ 0.
    # This is the strict check that catches mid-time errors (e.g. the edge-on
    # kink): a coarse period match can pass while individual times are wrong.
    Nn = np.arange(len(tt), dtype=float)
    A = np.vstack([Nn, np.ones_like(Nn)]).T
    oc = tt - A @ np.linalg.lstsq(A, tt, rcond=None)[0]
    print(f"3-D edge-on: {len(tt)} transits, spacing {per.mean():.4f} "
          f"(expected {2 * np.pi:.4f}), max |O-C| {np.abs(oc).max():.2e}")
    assert 5 <= len(tt) <= 6, f"expected ~6 transits, got {len(tt)}"
    assert np.allclose(per, 2 * np.pi, rtol=1e-4), "transit period should match orbit"
    assert np.abs(oc).max() < 1e-3, "unperturbed 2-body O-C should be ~0 (mid-time accuracy)"

    # inclined orbit (i=80) should still transit (grazing) or not, but must run
    pos2, vel2 = elements_to_state_3d(1.0, 0.0, np.radians(80), 0.0, 0.0, -np.pi / 2, 1.0)
    assert np.isfinite(pos2).all() and np.isfinite(vel2).all()
    print("[threed] self-check passed")
