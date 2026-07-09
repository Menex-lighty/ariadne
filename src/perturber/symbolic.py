"""Milestone 2 — prune the fitted force library to its functional form (STLSQ).

Sequential thresholded least squares, but each "least squares" is a full
differentiable fit through the simulation (fit_force): fit all terms, drop the
terms whose coefficient is negligible relative to the largest, refit the
survivors, repeat until the active set is stable. sympy renders the result.

This is the *through-the-simulation* form-pruning tool; the cheaper, analytic
best-case version (sparse regression on directly sampled force values) and the
identifiability metric that predicts when pruning can succeed live in
`perturber.identifiability`. Over narrow radial coverage the library is
degenerate and pruning does not recover the injected term — see that module.
"""
import numpy as np
import sympy as sp

from perturber.residual import DEFAULT_EXPONENTS, fit_force


def stlsq(ds, exponents=DEFAULT_EXPONENTS, rel_threshold=0.05, max_iter=4,
          curriculum=((0.3, 200), (0.6, 200), (1.0, 300)), l1=1e-3, device="cpu",
          state0_true=None, verbose=False):
    """Discover the force form. Returns dict with active exponents, coefficients,
    the sympy expression, and the final data misfit."""
    active = [True] * len(exponents)
    result = None
    for it in range(max_iter):
        # No L1 once the set is small — get clean coefficients on the survivors.
        use_l1 = l1 if sum(active) > 2 else 0.0
        model, state, hist = fit_force(ds, exponents, active=active,
                                       curriculum=curriculum, l1=use_l1,
                                       device=device, state0_true=state0_true)
        c = model.coeffs().detach().cpu().numpy()
        result = (c, float(hist[-1]))
        cmax = np.abs(c).max()
        new_active = [bool(a and abs(ck) >= rel_threshold * cmax)
                      for a, ck in zip(active, c)]
        if verbose:
            terms = [f"r^{p:+.0f}:{ck:+.2e}" for p, ck, a in
                     zip(exponents, c, active) if a]
            print(f"  [stlsq it{it}] misfit {hist[-1]:.2f} active={sum(active)} "
                  f"| {'  '.join(terms)}", flush=True)
        if new_active == active:
            break
        active = new_active

    c, misfit = result
    kept = [(p, ck) for p, ck, a in zip(exponents, c, active) if a and abs(ck) > 0]
    return {"exponents": exponents, "coeffs": c.tolist(), "active": active,
            "kept": kept, "expr": render(kept), "misfit": misfit}


def render(kept):
    """kept: list of (exponent, coeff) -> a tidy sympy string for g(r)."""
    r = sp.Symbol("r", positive=True)
    expr = sum(sp.Float(round(ck, 6)) * r ** sp.Rational(int(p)) for p, ck in kept)
    return str(sp.nsimplify(expr, rational=False)) if kept else "0"


if __name__ == "__main__":
    # Smoke-check that the through-the-simulation STLSQ machinery runs and fits
    # the trajectory. It does NOT assert exact form recovery: over the limited
    # radial coverage of a couple of orbits the library is degenerate, so the
    # surviving term need not be the injected r^-4 (that boundary is the point,
    # measured in perturber.identifiability).
    from perturber.config import SystemConfig
    from perturber.data import generate

    cfg = SystemConfig(masses_visible=(1.0, 1e-4, 3e-5), hidden_mass=0.0,
                       planet_a=(1.0, 1.9), planet_e=(0.25, 0.3), n_periods=8.0,
                       sigma=1e-4, seed=0,
                       m2_force={"kind": "power_law", "params": {"alpha": 2e-3, "n": 4.0}})
    ds = generate(cfg)
    out = stlsq(ds, state0_true=ds.truth[0][:ds.n_visible], max_iter=2,
                curriculum=((0.5, 200), (1.0, 250)), verbose=True)
    print(f"discovered g(r) = {out['expr']}   (misfit {out['misfit']:.2f})")
    assert out["misfit"] < 20.0, "STLSQ fit should reach near the noise floor"
    print("[symbolic] self-check passed (form recovery is coverage-limited; "
          "see identifiability.py)")
