"""Milestone 3 — VLEO thermospheric-density-field identifiability.

A satellite's drag measures the density rho along its ground track: at each
epoch the along-track deceleration ~ (C_d A/m) * rho(h, lst, ...) * v^2, so with
a known ballistic coefficient the trajectory samples rho at the satellite's
altitude h and local solar time lst. Recovering the *field* rho(h, lst) from
those along-track samples is a design-matrix problem identical in structure to
the force-form recovery in `identifiability`:

    log rho(h, lst) = sum_k c_k phi_k(h, lst)

A single near-circular satellite samples essentially one altitude, so the
altitude structure of the field is unconstrained (the h, h^2 columns are
degenerate); a constellation spanning altitudes and local times conditions the
design matrix and the field becomes recoverable. This is the multi-object joint
constraint (the reason M3 is on the ladder), quantified with the same
condition-number tooling.
"""
import numpy as np

from perturber.identifiability import stlsq_solve

FEATURE_NAMES = ["1", "h", "h^2", "cos(lst)", "sin(lst)", "h*cos(lst)", "h*sin(lst)"]


def density_features(h, lst):
    """Design features for log-density over altitude h (scaled) and local solar
    time lst in [0,1). Physical building blocks: an altitude trend (scale
    height), diurnal harmonics, and altitude-local-time coupling."""
    h = np.asarray(h, float)
    lst = np.asarray(lst, float)
    w = 2 * np.pi * lst
    return np.stack([np.ones_like(h), h, h ** 2,
                     np.cos(w), np.sin(w), h * np.cos(w), h * np.sin(w)], axis=-1)


def design_condition(h, lst):
    """Condition number of the density design matrix over the sampled (h, lst) —
    the pre-fit predictor of field recoverability.

    Unlike the force-form case (where the degeneracy is *collinearity* over a
    limited radius range, so columns are normalised), the field degeneracy here
    is that a satellite may not *sample* a dimension at all — a near-circular
    satellite barely varies its altitude, leaving the h/h^2 columns near-zero.
    That is captured by the un-normalised condition number (those columns get
    tiny singular values); features are already in comparable scaled units."""
    return float(np.linalg.cond(density_features(h, lst)))


def satellite_track(altitude, n=200, lst_coverage=1.0, alt_scatter=0.05, seed=0):
    """(h, lst) a near-circular satellite samples: ~constant altitude (small
    scatter — a circular orbit barely changes altitude), local time sweeping over
    the mission as the orbit precesses. `altitude`/`alt_scatter` in scaled units.
    A single such satellite pins one altitude, so the field's altitude structure
    is unconstrained; only satellites at *different* altitudes constrain it."""
    rng = np.random.default_rng(seed)
    h = altitude + alt_scatter * rng.standard_normal(n)
    lst = rng.uniform(0.0, lst_coverage, n) % 1.0
    return h, lst


def recover_field(h, lst, rho_true_fn, rel_noise=0.05, n_trials=30, seed=0,
                  threshold=0.1):
    """Best-case field recovery: sample log-rho_true at (h,lst) with relative
    noise, STLSQ-regress onto the feature library, and measure how often the true
    active features are selected. Returns support-recovery rate, profile error,
    and the condition number."""
    Phi = density_features(h, lst)
    y0 = rho_true_fn(h, lst)
    c_clean = stlsq_solve(Phi, y0, threshold=threshold)
    true_support = tuple(np.where(np.abs(c_clean) >
                         threshold * np.abs(c_clean).max())[0])
    rng = np.random.default_rng(seed)
    hits, errs = 0, []
    for _ in range(n_trials):
        y = y0 + rel_noise * np.abs(y0).mean() * rng.standard_normal(len(y0))
        c = stlsq_solve(Phi, y, threshold=threshold)
        sup = tuple(np.where(np.abs(c) > threshold * np.abs(c).max())[0])
        hits += (sup == true_support)
        errs.append(np.sqrt(np.mean((Phi @ c - y0) ** 2)) / np.sqrt(np.mean(y0 ** 2)))
    return {"support_recovery_rate": hits / n_trials,
            "profile_rel_error": float(np.median(errs)),
            "condition_number": design_condition(h, lst)}


if __name__ == "__main__":
    # A single near-circular satellite samples ~one altitude -> the field's
    # altitude structure is degenerate; a constellation across altitudes fixes it.
    def log_rho_true(h, lst):
        # exponential-ish altitude decay + diurnal bulge (in log space)
        return -0.9 * h + 0.4 * np.cos(2 * np.pi * (lst - 0.6))

    h1, l1 = satellite_track(0.0, n=300, seed=1)                      # one altitude
    hs, ls = [], []
    for j, a in enumerate([-1.5, -0.5, 0.5, 1.5]):                    # 4 altitudes
        hh, ll = satellite_track(a, n=120, seed=10 + j)
        hs.append(hh); ls.append(ll)
    hc, lc = np.concatenate(hs), np.concatenate(ls)

    c1 = design_condition(h1, l1)
    cc = design_condition(hc, lc)
    r1 = recover_field(h1, l1, log_rho_true)
    rc = recover_field(hc, lc, log_rho_true)
    print(f"single satellite:  cond {c1:.1e}  field-recovery {r1['support_recovery_rate']:.0%}")
    print(f"4-sat constellation: cond {cc:.1e}  field-recovery {rc['support_recovery_rate']:.0%}")
    assert cc < c1, "constellation should condition the field design matrix"
    assert rc["support_recovery_rate"] > r1["support_recovery_rate"], \
        "constellation should recover the field better than one satellite"
    print("[density] self-check passed")
