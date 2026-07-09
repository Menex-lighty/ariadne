"""Force-form identifiability demonstration (the project's distinctive angle).

With a *perfect* initial state (so state-estimation error is removed), fit an
injected central force g_true(r) = alpha/r^n through the differentiable
simulation, over a broad power-law library. Then ask three questions:

  1. Does the fit reproduce the trajectory?          (yes — to the noise floor)
  2. Is the radial PROFILE g(r) recovered?           (yes — over the sampled r)
  3. Is the symbolic FORM recovered?                 (no — coefficients degenerate)

This isolates and quantifies the identifiability wall that the literature calls
out but rarely measures: over the limited r-range a few orbits sample, the
force-law basis is nearly degenerate, so the profile is constrained while the
functional form is not.

Usage: python scripts/run_identifiability.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402

torch.set_num_threads(1)  # small tensors: threading only adds contention here

from perturber.config import SystemConfig      # noqa: E402
from perturber.data import generate            # noqa: E402
from perturber.residual import fit_force, DEFAULT_EXPONENTS  # noqa: E402
from perturber.plots import ensure_dir         # noqa: E402


def g_true(r, alpha=2e-3, n=4.0):
    return alpha / r ** n


def observed_r_range(ds):
    nv = ds.n_visible
    d = ds.truth[:, 1:nv, :2] - ds.truth[:, :1, :2]     # planets rel. to star
    r = np.linalg.norm(d, axis=-1)
    return float(r.min()), float(r.max())


def profile(model, r):
    with torch.no_grad():
        g = model.g_of_r(torch.tensor(r, dtype=torch.float64)).numpy()
    return g


def main():
    outdir = ensure_dir(str(Path(__file__).parents[1] / "results" / "identifiability"))
    cfg = SystemConfig(masses_visible=(1.0, 1e-4, 3e-5), hidden_mass=0.0,
                       planet_a=(1.0, 1.9), planet_e=(0.25, 0.3), n_periods=8.0,
                       sigma=1e-4, seed=0,
                       m2_force={"kind": "power_law", "params": {"alpha": 2e-3, "n": 4.0}})
    ds = generate(cfg)
    s0 = ds.truth[0][:ds.n_visible]
    rlo, rhi = observed_r_range(ds)
    print(f"observed r-range: [{rlo:.2f}, {rhi:.2f}]")

    curr = ((0.5, 300), (1.0, 400))
    # (a) broad library
    m_full, _, h_full = fit_force(ds, state0_true=s0, curriculum=curr, l1=1e-3)
    # (b) the 'correct' sparse model: only r^-4 active
    active = [p == -4.0 for p in DEFAULT_EXPONENTS]
    m_sparse, _, h_sparse = fit_force(ds, state0_true=s0, active=active,
                                      curriculum=curr, l1=0.0)

    c_full = m_full.coeffs().detach().numpy()
    c4 = m_sparse.coeffs().detach().numpy()[DEFAULT_EXPONENTS.index(-4.0)]
    n_eff = int((np.abs(c_full) > 0.05 * np.abs(c_full).max()).sum())

    rg = np.linspace(rlo, rhi, 200)
    gt = g_true(rg)
    g_full = profile(m_full, rg)
    g_sparse = profile(m_sparse, rg)
    prof_err_full = np.sqrt(np.mean((g_full - gt) ** 2)) / np.sqrt(np.mean(gt ** 2))
    prof_err_sparse = np.sqrt(np.mean((g_sparse - gt) ** 2)) / np.sqrt(np.mean(gt ** 2))

    print(f"\ntrajectory misfit (chi2/pt): broad {h_full[-1]:.2f} | "
          f"sparse-true-form {h_sparse[-1]:.2f}   (noise floor ~1)")
    print(f"g(r) profile rel-RMS error:  broad {prof_err_full:.1%} | "
          f"sparse {prof_err_sparse:.1%}")
    print(f"recovered r^-4 coeff (sparse model): {c4:.2e}  (true 2.00e-03)")
    print(f"broad-library coefficients (should be sparse, are not):")
    for p, ck in zip(DEFAULT_EXPONENTS, c_full):
        print(f"   r^{int(p):+d}: {ck:+.2e}")
    print(f"effective non-zero terms: {n_eff} of {len(DEFAULT_EXPONENTS)}")

    # ── figure ──
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(rg, gt, "k-", lw=3, alpha=0.5, label="true  g(r)=2e-3/r⁴")
    ax[0].plot(rg, g_full, "--", color="#c0392b", lw=1.8,
               label=f"broad-library fit ({n_eff} terms)")
    ax[0].plot(rg, g_sparse, ":", color="#2471a3", lw=1.8, label="r⁻⁴-only fit")
    ax[0].set_xlabel("r  (star–planet distance)"); ax[0].set_ylabel("g(r)")
    ax[0].set_title(f"Profile IS recovered — both fits overlap truth\n"
                    f"(rel-RMS {prof_err_full:.0%} broad, {prof_err_sparse:.0%} sparse)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    xs = np.arange(len(DEFAULT_EXPONENTS))
    ax[1].axhline(0, color="#999", lw=0.8)
    ax[1].bar(xs, c_full, color="#c0392b", alpha=0.8, label="broad-library coeffs")
    ax[1].bar([DEFAULT_EXPONENTS.index(-4.0)], [2e-3], width=0.4,
              color="k", alpha=0.6, label="true (r⁻⁴ = 2e-3)")
    ax[1].set_xticks(xs, [f"r^{int(p):+d}" for p in DEFAULT_EXPONENTS])
    ax[1].set_ylabel("coefficient")
    ax[1].set_title("Form is NOT recovered — coefficients spread,\n"
                    "not the sparse truth (basis degenerate over sampled r)")
    ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    p = Path(outdir) / "identifiability.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {p}")

    # The headline claim, asserted: trajectory + profile recovered, form is not.
    assert h_full[-1] < 20, "broad fit should reach near the noise floor"
    assert prof_err_full < 0.25, "profile should be recovered over sampled r"
    assert n_eff >= 4, "coefficients should be non-sparse (degenerate), not the true 1 term"
    print("[identifiability] demonstration assertions passed")


if __name__ == "__main__":
    main()
