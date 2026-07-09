"""Evaluation: parameter recovery, forward prediction, null-model comparison."""
import numpy as np
import torch

from perturber.fit import chi2_per_candidate
from perturber.integrators import rk4_integrate
from perturber.model import PerturberModel


def integrate_best(model, result, ds, cfg):
    """Single-shot integrate the best restart over the FULL arc (train + held-out).

    Returns trajectory (T, N, 4) numpy. No softening here — evaluation uses the
    same pure dynamics as the ground truth.
    """
    device = next(model.parameters()).device
    t_all = torch.tensor(ds.t, dtype=torch.float64, device=device)
    with torch.no_grad():
        traj = rk4_integrate(model.initial_state(), model.masses(), t_all,
                             substeps=cfg.substeps, softening=0.0)
    return traj[:, result.best_idx].cpu().numpy()


def evaluate(model, result, ds, cfg):
    """Metrics dict for one fitted model (perturber or null)."""
    traj = integrate_best(model, result, ds, cfg)
    nv = ds.n_visible
    tr, te = ds.train_mask, ds.test_mask
    sigma = ds.sys.sigma

    def chi2(mask):
        r = (traj[mask][:, :nv, :2] - ds.q_obs[mask]) / sigma
        return float((r ** 2).mean())

    def rmse_truth(mask):
        d = traj[mask][:, :nv, :2] - ds.truth[mask][:, :nv, :2]
        return float(np.sqrt((d ** 2).sum(-1).mean()))

    metrics = {
        "chi2_train": chi2(tr),
        "chi2_test": chi2(te),
        "rmse_truth_train": rmse_truth(tr),
        "rmse_truth_test": rmse_truth(te),
        "n_obs_test": int(te.sum()) * nv * 2,
        "wall_time_s": result.wall_time,
        "restart_train_chi2": result.train_chi2.tolist(),
    }

    if isinstance(model, PerturberModel):
        i = result.best_idx
        with torch.no_grad():
            m_hat = float(model.hidden_mass()[i])
            a_hat = float(model.hidden_elements()["a"][i])
        m_true, a_true = ds.sys.hidden_mass, ds.sys.hidden_a
        mid = len(ds.t) // 2
        pos_err = float(np.linalg.norm(traj[mid, -1, :2] - ds.truth[mid, -1, :2]))
        # Restart agreement: log-mass spread of the top-3 candidates
        order = np.argsort(result.train_chi2)[:3]
        with torch.no_grad():
            top_masses = model.hidden_mass().cpu().numpy()[order]
        metrics.update({
            "mass_true": m_true,
            "mass_recovered": m_hat,
            "log_mass_error": abs(np.log10(m_hat / m_true)),
            "a_true": a_true,
            "a_recovered": a_hat,
            "a_rel_error": abs(a_hat - a_true) / a_true,
            "hidden_pos_error_midarc": pos_err,
            "top3_log10_mass": np.log10(top_masses).tolist(),
        })
    return metrics, traj


def compare(metrics_pert, metrics_null):
    """Detection statistic: does the perturber model beat the null on held-out
    data? Under the Gaussian noise model, 2*delta(logL) = n_test * delta(chi2)."""
    d_chi2 = metrics_null["chi2_test"] - metrics_pert["chi2_test"]
    stat = metrics_pert["n_obs_test"] * d_chi2
    return {
        "delta_chi2_test": d_chi2,
        "delta_rmse_test": metrics_null["rmse_truth_test"] - metrics_pert["rmse_truth_test"],
        "loglike_ratio_stat": stat,       # ~2*(logL_pert - logL_null)
        "detected": bool(stat > 25.0),    # ≈5-sigma-equivalent threshold, 1 extra dof family
    }
