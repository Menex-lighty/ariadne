"""Figures: single-experiment diagnostic panel and threshold-study heatmaps."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_experiment(ds, traj_pert, traj_null, metrics_pert, metrics_null, cmp, path):
    """Six-panel diagnostic figure for one experiment."""
    t = ds.t
    tr, te = ds.train_mask, ds.test_mask
    nv = ds.n_visible
    t_split = t[tr][-1]
    p_per = t / t[-1] * ds.sys.n_periods

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    det = "DETECTED" if cmp["detected"] else "not detected"
    fig.suptitle(
        f"Hidden-perturber discovery | m_true={ds.sys.hidden_mass:.1e} "
        f"m_hat={metrics_pert['mass_recovered']:.2e} | "
        f"sigma={ds.sys.sigma:.0e} | arc={ds.sys.n_periods:g} periods | {det}",
        fontweight="bold")

    # (0,0) orbits in XY
    ax = axes[0, 0]
    colors = ["#f39c12", "#2980b9", "#27ae60"]
    for b in range(nv):
        ax.plot(ds.truth[:, b, 0], ds.truth[:, b, 1], color=colors[b], lw=2,
                alpha=0.4, label=f"truth body {b}")
        ax.plot(traj_pert[:, b, 0], traj_pert[:, b, 1], color=colors[b], lw=1, ls="--")
    ax.plot(ds.truth[:, -1, 0], ds.truth[:, -1, 1], "k-", lw=2, alpha=0.4,
            label="hidden (truth)")
    ax.plot(traj_pert[:, -1, 0], traj_pert[:, -1, 1], "r--", lw=1.2,
            label="hidden (recovered)")
    ax.set_aspect("equal"); ax.legend(fontsize=7); ax.set_title("Orbits (XY)")

    # (0,1) planet-2 residuals vs time, both models
    ax = axes[0, 1]
    b = nv - 1
    for traj, lbl, c in [(traj_null, "null model", "gray"),
                         (traj_pert, "perturber model", "tab:blue")]:
        r = np.linalg.norm(traj[:, b, :2] - ds.truth[:, b, :2], axis=1)
        ax.semilogy(p_per, np.maximum(r, 1e-12), color=c, lw=1.2, label=lbl)
    ax.axhline(ds.sys.sigma, color="k", ls=":", lw=1, label="noise sigma")
    ax.axvline(p_per[tr][-1], color="red", ls=":", lw=1)
    ax.set_xlabel("planet-1 periods"); ax.set_title(f"Body {b} error vs truth")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (0,2) star wobble: the mass signal
    ax = axes[0, 2]
    ax.plot(p_per, ds.q_obs[:, 0, 0], ".", ms=2, alpha=0.3, color="gray", label="obs")
    ax.plot(p_per, ds.truth[:, 0, 0], "g-", lw=1.5, alpha=0.7, label="truth")
    ax.plot(p_per, traj_pert[:, 0, 0], "b--", lw=1, label="perturber fit")
    ax.plot(p_per, traj_null[:, 0, 0], "-", color="orange", lw=1, label="null fit")
    ax.axvline(p_per[tr][-1], color="red", ls=":", lw=1)
    ax.set_xlabel("planet-1 periods"); ax.set_title("Star x (reflex wobble)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,0) held-out forward prediction error, all visible bodies
    ax = axes[1, 0]
    for traj, lbl, c in [(traj_null, "null", "gray"), (traj_pert, "perturber", "tab:blue")]:
        r = np.linalg.norm(traj[:, :nv, :2] - ds.truth[:, :nv, :2], axis=2).mean(1)
        ax.semilogy(p_per, np.maximum(r, 1e-12), color=c, lw=1.2, label=lbl)
    ax.axvspan(p_per[te][0], p_per[-1], alpha=0.08, color="red", label="held-out")
    ax.axhline(ds.sys.sigma, color="k", ls=":", lw=1)
    ax.set_xlabel("planet-1 periods"); ax.set_title("Mean visible-body error")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,1) restart agreement
    ax = axes[1, 1]
    chi2s = np.array(metrics_pert["restart_train_chi2"])
    masses = np.array(metrics_pert["top3_log10_mass"])
    ax.plot(np.sort(chi2s), "o-", color="tab:blue")
    ax.set_yscale("log"); ax.set_xlabel("restart (sorted)")
    ax.set_title(f"Restart train chi2 | top-3 log10(m): "
                 f"{np.array2string(masses, precision=2)}")
    ax.grid(alpha=0.3)

    # (1,2) summary text
    ax = axes[1, 2]; ax.axis("off")
    lines = [
        f"log-mass error      {metrics_pert['log_mass_error']:.3f}",
        f"a: true {metrics_pert['a_true']:.2f}  rec {metrics_pert['a_recovered']:.2f}",
        f"hidden pos err mid  {metrics_pert['hidden_pos_error_midarc']:.3f}",
        "",
        f"chi2 test  pert {metrics_pert['chi2_test']:.2f}  null {metrics_null['chi2_test']:.2f}",
        f"rmse test  pert {metrics_pert['rmse_truth_test']:.2e}  null {metrics_null['rmse_truth_test']:.2e}",
        f"2*dlogL = {cmp['loglike_ratio_stat']:.1f}  ->  {det}",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return path


def plot_threshold(cells, path):
    """Threshold-study summary. cells: list of result dicts, each with keys
    mass, sigma, n_periods, seed, log_mass_error, detected."""
    arcs = sorted({c["n_periods"] for c in cells})
    masses = sorted({c["mass"] for c in cells})
    sigmas = sorted({c["sigma"] for c in cells})

    fig, axes = plt.subplots(1, len(arcs), figsize=(5.5 * len(arcs), 4.5),
                             squeeze=False)
    for ai, arc in enumerate(arcs):
        ax = axes[0, ai]
        grid = np.full((len(masses), len(sigmas)), np.nan)
        det = np.zeros_like(grid, dtype=bool)
        for mi, m in enumerate(masses):
            for si, s in enumerate(sigmas):
                sel = [c for c in cells
                       if c["mass"] == m and c["sigma"] == s and c["n_periods"] == arc]
                if sel:
                    grid[mi, si] = np.median([c["log_mass_error"] for c in sel])
                    det[mi, si] = np.median([c["detected"] for c in sel]) >= 0.5
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r",
                       vmin=0, vmax=2)
        for mi in range(len(masses)):
            for si in range(len(sigmas)):
                if np.isnan(grid[mi, si]):
                    continue
                mark = "" if det[mi, si] else "  X"
                ax.text(si, mi, f"{grid[mi, si]:.2f}{mark}", ha="center",
                        va="center", fontsize=9,
                        color="white" if grid[mi, si] > 1 else "black")
        ax.set_xticks(range(len(sigmas)), [f"{s:.0e}" for s in sigmas])
        ax.set_yticks(range(len(masses)), [f"{m:.0e}" for m in masses])
        ax.set_xlabel("noise sigma"); ax.set_ylabel("true hidden mass")
        ax.set_title(f"arc = {arc:g} periods  (X = not detected)")
    fig.colorbar(im, ax=axes[0, -1], label="median |log10(m_hat/m_true)|")
    fig.suptitle("Detection threshold: median log-mass error", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return path


def _floor(cells, arc, s, masses, predicate):
    """Smallest true mass at (arc, sigma) whose seed-median satisfies predicate."""
    ok = []
    for m in masses:
        sel = [c for c in cells if c["mass"] == m and c["sigma"] == s
               and c["n_periods"] == arc]
        if sel and predicate(sel):
            ok.append(m)
    return min(ok) if ok else None


def plot_detection_boundary(cells, path, char_thresh=0.3):
    """Two sensitivity floors vs noise, one colour per arc length:

    - solid  = detection floor: smallest mass the perturber model still beats the
      null on held-out data (median over seeds).
    - dashed = characterization floor: smallest mass recovered to within
      |log10(m_hat/m_true)| < char_thresh (~2x) — the practically useful limit.

    Lower = more sensitive. A floor at the smallest grid mass means the true
    floor is below the grid (not yet located)."""
    arcs = sorted({c["n_periods"] for c in cells})
    sigmas = sorted({c["sigma"] for c in cells})
    masses = sorted({c["mass"] for c in cells})
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(arcs)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for arc, col in zip(arcs, colors):
        det_x, det_y, char_x, char_y = [], [], [], []
        for s in sigmas:
            d = _floor(cells, arc, s, masses,
                       lambda sel: np.median([c["detected"] for c in sel]) >= 0.5)
            c_ = _floor(cells, arc, s, masses,
                        lambda sel: np.median([c["log_mass_error"] for c in sel]) < char_thresh)
            if d:
                det_x.append(s); det_y.append(d)
            if c_:
                char_x.append(s); char_y.append(c_)
        if det_x:
            ax.plot(det_x, det_y, "o-", color=col, lw=2, label=f"{arc:g}p detect")
        if char_x:
            ax.plot(char_x, char_y, "s--", color=col, lw=1.5, alpha=0.8,
                    label=f"{arc:g}p charac.")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("observation noise sigma")
    ax.set_ylabel("min hidden mass")
    ax.set_title(f"Sensitivity floors (solid=detect, dashed=recover to <{char_thresh} dex)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, ncol=len(arcs))
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return path


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
