"""Milestone 2 — non-Newtonian force laws to inject and discover.

A central extra force anchored on the dominant mass (the star, body 0):

    a_extra(body i) = -g(r_i) * rhat_i ,   r_i = |pos_i - pos_star|

with rhat_i the unit vector from star to body i (so positive g is *attractive*,
an extra inward pull). The discovery target in M2 is the radial profile g(r).

Injected truth uses a power law g(r) = alpha / r^n (n != 2 is genuinely
non-Newtonian; n = 2 would be degenerate with the star's mass). The star itself
feels no extra force here (a perturbative external-field approximation — valid
because the star dominates the mass and the term is small; momentum is not
exactly conserved, which is acceptable for a synthetic discovery target).
"""
import numpy as np
import torch


def _radial_apply_np(pos, star_idx, g_of_r):
    """pos (N,2) -> extra accel (N,2). g_of_r: array r -> array g."""
    d = pos - pos[star_idx][None, :]          # (N,2) from star
    r = np.linalg.norm(d, axis=1)             # (N,)
    out = np.zeros_like(pos)
    mask = np.arange(len(pos)) != star_idx
    rr = r[mask]
    rhat = d[mask] / rr[:, None]
    out[mask] = -(g_of_r(rr)[:, None]) * rhat
    return out


def radial_apply_torch(pos, star_idx, g_of_r):
    """pos (...,N,2) -> extra accel (...,N,2). g_of_r: tensor r -> tensor g."""
    d = pos - pos[..., star_idx:star_idx + 1, :]
    r = torch.linalg.norm(d, dim=-1).clamp(min=1e-9)      # (...,N)
    rhat = d / r.unsqueeze(-1)
    g = g_of_r(r)                                         # (...,N)
    out = -g.unsqueeze(-1) * rhat
    # zero the star's own entry
    n = pos.shape[-2]
    keep = torch.ones(n, dtype=pos.dtype, device=pos.device)
    keep[star_idx] = 0.0
    return out * keep.unsqueeze(-1)


def power_law(alpha=1e-3, n=4.0, star_idx=0, backend="numpy"):
    """Injected truth: g(r) = alpha / r^n. Returns an extra_force(pos) callable."""
    if backend == "numpy":
        return lambda pos: _radial_apply_np(pos, star_idx, lambda r: alpha / r ** n)
    return lambda pos: radial_apply_torch(pos, star_idx, lambda r: alpha / r ** n)


def yukawa(alpha=1e-3, lam=1.0, star_idx=0, backend="numpy"):
    """Injected truth: g(r) = alpha * exp(-r/lam) / r^2 (screened gravity)."""
    if backend == "numpy":
        return lambda pos: _radial_apply_np(
            pos, star_idx, lambda r: alpha * np.exp(-r / lam) / r ** 2)
    return lambda pos: radial_apply_torch(
        pos, star_idx, lambda r: alpha * torch.exp(-r / lam) / r ** 2)


def make_injected(kind, params, star_idx=0, backend="numpy"):
    if kind == "power_law":
        return power_law(star_idx=star_idx, backend=backend, **params)
    if kind == "yukawa":
        return yukawa(star_idx=star_idx, backend=backend, **params)
    raise ValueError(kind)


if __name__ == "__main__":
    # torch and numpy applicators must agree on random positions
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(4, 2)) * np.array([1.0, 1.0]) + np.array([0.0, 0.0])
    pos[0] = [0.05, -0.03]  # star near origin
    for kind, params in [("power_law", dict(alpha=2e-3, n=4.0)),
                         ("yukawa", dict(alpha=2e-3, lam=1.5))]:
        f_np = make_injected(kind, params, backend="numpy")
        f_t = make_injected(kind, params, backend="torch")
        a_np = f_np(pos)
        a_t = f_t(torch.tensor(pos)).numpy()
        err = np.abs(a_np - a_t).max()
        assert err < 1e-12, f"{kind}: numpy/torch mismatch {err}"
        assert np.allclose(a_np[0], 0.0), "star should feel no extra force"
    print("[forces] self-checks passed")
