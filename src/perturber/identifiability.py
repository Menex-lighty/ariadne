"""Quantify force-form identifiability — the project's distinctive contribution.

The field asserts that force/perturber inverse problems are "degenerate"; this
module *measures* it. The key object is the design matrix of the candidate
force-law library evaluated at the radii an orbit actually samples:

    Phi[i, k] = r_i ^ p_k

Whether the functional FORM g(r) = sum_k c_k r^p_k can be recovered is governed
by the conditioning of Phi (after per-column normalisation, so it reflects
collinearity of the basis functions, not their scale). This is a *best-case*
statement: if the form cannot be recovered even from directly sampled, noisy
force values, it certainly cannot be recovered through the harder trajectory
inversion. So it upper-bounds identifiability and is cheap (an SVD), letting us
sweep coverage and noise without a full differentiable fit per point.

`recover_form` runs the direct experiment (sparse regression on noisy g samples);
`coverage_condition` is the analytic predictor. The restoration result: widening
the sampled r-range (more eccentric orbits, or more bodies at different a) drops
the condition number and returns form-recovery — which is exactly the multi-body
joint constraint that motivates M3.
"""
import numpy as np

DEFAULT_EXPONENTS = (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0)


def design_matrix(r, exponents=DEFAULT_EXPONENTS, normalize=True):
    """Phi[i,k] = r_i^p_k. With normalize, each column is unit-norm so the
    condition number measures basis collinearity over the sampled radii."""
    r = np.asarray(r, dtype=float)
    phi = r[:, None] ** np.asarray(exponents)[None, :]
    if normalize:
        phi = phi / np.linalg.norm(phi, axis=0, keepdims=True)
    return phi


def condition_number(r, exponents=DEFAULT_EXPONENTS):
    """Condition number of the normalised design matrix — the analytic predictor
    of form-identifiability. Large => the library is degenerate over these radii."""
    return float(np.linalg.cond(design_matrix(r, exponents)))


def effective_rank(r, exponents=DEFAULT_EXPONENTS, tol=1e-3):
    """Number of singular values above tol * (largest) — how many basis
    directions the sampled radii actually constrain."""
    s = np.linalg.svd(design_matrix(r, exponents), compute_uv=False)
    return int((s > tol * s[0]).sum())


def stlsq_solve(phi, y, threshold=0.1, iters=10):
    """Sequential thresholded least squares (Brunton-style). phi columns should
    be raw (un-normalised) so coefficients are in physical units."""
    c = np.linalg.lstsq(phi, y, rcond=None)[0]
    for _ in range(iters):
        big = np.abs(c) >= threshold * np.abs(c).max()
        if not big.any():
            break
        c_new = np.zeros_like(c)
        c_new[big] = np.linalg.lstsq(phi[:, big], y, rcond=None)[0]
        if np.array_equal(big, np.abs(c_new) >= threshold * np.abs(c_new).max()):
            c = c_new
            break
        c = c_new
    return c


def recover_form(r, g_true_fn, exponents=DEFAULT_EXPONENTS, rel_noise=0.05,
                 n_trials=40, seed=0, threshold=0.15):
    """Direct best-case form recovery: sample g_true at r with relative noise,
    STLSQ-regress onto the raw library, and measure how often the correct single
    term is selected. Returns dict with support-recovery rate, profile error,
    and the (physical, un-normalised) recovered coefficients averaged over trials.
    """
    r = np.asarray(r, float)
    exps = list(exponents)
    phi_raw = r[:, None] ** np.asarray(exps)[None, :]
    g0 = g_true_fn(r)
    rng = np.random.default_rng(seed)

    # the true support: the exponent(s) g_true_fn is built from (probe by fitting
    # clean g and taking the dominant term)
    c_clean = stlsq_solve(phi_raw, g0, threshold=threshold)
    true_support = tuple(sorted(int(exps[k]) for k in np.where(np.abs(c_clean) >
                          threshold * np.abs(c_clean).max())[0]))

    hits = 0
    prof_errs = []
    coeffs = []
    for tr in range(n_trials):
        g = g0 * (1.0 + rel_noise * rng.standard_normal(len(r)))
        c = stlsq_solve(phi_raw, g, threshold=threshold)
        coeffs.append(c)
        support = tuple(sorted(int(exps[k]) for k in np.where(np.abs(c) >
                        threshold * np.abs(c).max())[0]))
        hits += (support == true_support)
        prof_errs.append(np.sqrt(np.mean((phi_raw @ c - g0) ** 2)) /
                         np.sqrt(np.mean(g0 ** 2)))
    return {
        "support_recovery_rate": hits / n_trials,
        "true_support": true_support,
        "profile_rel_error": float(np.median(prof_errs)),
        "mean_coeffs": np.mean(coeffs, axis=0).tolist(),
        "exponents": exps,
        "condition_number": condition_number(r, exponents),
        "effective_rank": effective_rank(r, exponents),
    }


def fisher_information(jac_weighted):
    """Fisher information F = J^T J from a noise-weighted Jacobian (rows = data
    points i, columns = parameters k, entries (1/sigma_i) d(model_i)/d(theta_k))."""
    return jac_weighted.T @ jac_weighted


def marginal_sigma(F, prior_precision=None, ridge_frac=1e-10):
    """1-sigma marginal parameter uncertainties = sqrt(diag((F + prior)^-1)).

    A degenerate direction gives a near-zero Fisher eigenvalue and hence a *huge*
    marginal sigma; a prior (its precision matrix, inverse of the prior
    covariance) adds to F and lifts exactly those directions. A weak ridge
    (a very-weak implicit prior) keeps degenerate directions large-but-finite
    rather than singular — using pinv here would wrongly report zero variance in
    the unconstrained direction."""
    M = (F.copy() if prior_precision is None else F + prior_precision)
    M = M + (ridge_frac * np.trace(M) / M.shape[0]) * np.eye(M.shape[0])
    cov = np.linalg.inv(M)
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def prior_precision(n, index, sigma):
    """Precision matrix for one independent Gaussian prior: sigma on parameter
    `index` (e.g. an external RV or mass-radius measurement)."""
    P = np.zeros((n, n))
    P[index, index] = 1.0 / sigma ** 2
    return P


def prescribe_prior(F, names, fit_values, rel_sigma=0.2):
    """For each parameter, add a modest independent prior (rel_sigma of its
    value) and report how much every parameter's uncertainty shrinks. Returns
    (data_only_sigma, [(name, posterior_sigma_vector), ...]) — the prior that
    most collapses the *degenerate* parameters is the one worth measuring."""
    n = len(names)
    base = marginal_sigma(F)
    rows = []
    for k in range(n):
        P = prior_precision(n, k, rel_sigma * abs(fit_values[k]))
        rows.append((names[k], marginal_sigma(F, P)))
    return base, rows


def orbit_radii(a, e, n=200, time_weighted=True):
    """Radii an orbit of (a, e) samples. time_weighted spends more samples near
    apoapsis (as a real orbit does), via the eccentric anomaly parameterisation."""
    if time_weighted:
        E = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return a * (1 - e * np.cos(E))
    nu = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return a * (1 - e * e) / (1 + e * np.cos(nu))


def multi_body_radii(a_list, e_list, n_each=200):
    return np.concatenate([orbit_radii(a, e, n_each)
                           for a, e in zip(a_list, e_list)])


if __name__ == "__main__":
    def g_true(r):
        return 2e-3 / r ** 4

    # Narrow coverage (near-circular single planet) -> degenerate
    r_narrow = orbit_radii(1.0, 0.05)
    # Wide coverage (eccentric single planet)
    r_ecc = orbit_radii(1.0, 0.5)
    # Multi-body spanning a range of a -> widest coverage (the M3 mechanism)
    r_multi = multi_body_radii([0.6, 1.0, 1.7, 2.6], [0.2, 0.3, 0.25, 0.2])

    for tag, r in [("near-circular", r_narrow), ("eccentric", r_ecc),
                   ("multi-body", r_multi)]:
        out = recover_form(r, g_true)
        print(f"{tag:14s} r in [{r.min():.2f},{r.max():.2f}] "
              f"cond={out['condition_number']:.1e} rank={out['effective_rank']} "
              f"-> form-recovery {out['support_recovery_rate']:.0%} "
              f"(profile err {out['profile_rel_error']:.1%})")

    # Sanity: condition number must fall and recovery must rise with coverage
    c_narrow = condition_number(r_narrow)
    c_multi = condition_number(r_multi)
    assert c_multi < c_narrow, "wider coverage should reduce condition number"
    assert recover_form(r_multi, g_true)["support_recovery_rate"] > \
        recover_form(r_narrow, g_true)["support_recovery_rate"], \
        "wider coverage should improve form recovery"

    # Prior-aware: a degenerate 2-parameter problem (near-collinear columns, so
    # only the sum is constrained). Data alone leaves both parameters loose; an
    # independent prior on one collapses BOTH (the constrained sum then fixes the
    # other) — the mechanism behind combining TTVs (ratio) with RV (scale).
    Jw = np.array([[1.0, 1.0], [1.0, 1.0001]])
    F = fisher_information(Jw)
    base = marginal_sigma(F)
    post = marginal_sigma(F, prior_precision(2, 0, 0.1))
    print(f"[prior] data-only sigma {np.array2string(base, precision=1)}; "
          f"with a prior on p0: {np.array2string(post, precision=3)}")
    assert base.min() > 10, "degenerate params should be loosely constrained by data alone"
    assert post[0] < 0.11 and post[1] < 1.0, "a prior on p0 should pin both params"
    assert base.max() / post.max() > 50, "prior should collapse the degenerate direction"
    print("[identifiability] self-checks passed")
