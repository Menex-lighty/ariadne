"""Fitting engine: batched multi-start, short-arc curriculum, multiple shooting.

Everything runs in float64 — observation noise is 1e-4 in O(1) coordinates and
float32 rounding accumulated over thousands of RK4 steps would eat the signal.
"""
import time
from dataclasses import dataclass

import numpy as np
import torch

from perturber.config import FitConfig, resolve_device
from perturber.data import Dataset, estimate_visible_state0
from perturber.integrators import rk4_integrate
from perturber.model import NullModel, PerturberModel


def make_model(kind, ds: Dataset, cfg: FitConfig):
    device = resolve_device(cfg.device)
    vis0 = torch.tensor(estimate_visible_state0(ds), dtype=torch.float64)
    m_vis = torch.tensor(np.array(ds.sys.masses_visible), dtype=torch.float64)
    dt_obs = float(ds.t[1] - ds.t[0])
    gen = torch.Generator().manual_seed(cfg.seed)
    if kind == "perturber":
        model = PerturberModel(cfg.n_restarts, vis0, m_vis, ds.sys.sigma, dt_obs,
                               mass_init=cfg.mass_init, log_a_range=cfg.log_a_range,
                               generator=gen)
    elif kind == "null":
        model = NullModel(cfg.n_restarts, vis0, m_vis, ds.sys.sigma, dt_obs)
    else:
        raise ValueError(kind)
    return model.double().to(device)


def chi2_per_candidate(traj, q_obs, sigma):
    """traj (T, B, N, 4), q_obs (T, Nv, 2) -> mean normalized sq residual (B,)."""
    nv = q_obs.shape[1]
    res = (traj[:, :, :nv, :2] - q_obs.unsqueeze(1)) / sigma
    return (res ** 2).mean(dim=(0, 2, 3))


@dataclass
class FitResult:
    best_idx: int
    train_chi2: np.ndarray      # (B,) final single-shot train chi2 per candidate
    history: list               # (phase, step, best loss, [mass spread]) tuples
    wall_time: float


def _make_optimizer(model, cfg, extra_params=()):
    fast, slow = [], list(extra_params)
    for name, p in model.named_parameters():
        (fast if name == "log10_mass" else slow).append(p)
    groups = [{"params": slow, "lr": cfg.lr_elements}]
    if fast:
        groups.append({"params": fast, "lr": cfg.lr_mass})
    return torch.optim.Adam(groups)


def _clamp_(model, cfg):
    if isinstance(model, PerturberModel):
        with torch.no_grad():
            model.log10_mass.clamp_(-6.0, -0.5)
            lo, hi = cfg.log_a_range
            model.log_a.clamp_(lo - 0.3, hi + 0.3)


def _log_line(model, tag, step, loss_b):
    best = loss_b.min().item()
    msg = f"  [{tag}] step {step:4d} | best chi2 {best:9.3f}"
    if isinstance(model, PerturberModel):
        m = model.hidden_mass().detach().cpu().numpy()
        msg += f" | log10 m: {np.log10(m).min():.2f}..{np.log10(m).max():.2f}"
    print(msg, flush=True)


def fit(model, ds: Dataset, cfg: FitConfig, verbose=True):
    device = next(model.parameters()).device
    t_all = torch.tensor(ds.t, dtype=torch.float64, device=device)
    q_all = torch.tensor(ds.q_obs, dtype=torch.float64, device=device)
    n_train = int(ds.train_mask.sum())
    sigma = ds.sys.sigma
    history = []
    t_start = time.time()

    # ── Phase A: single-shooting curriculum on growing train arcs ────────────
    opt = _make_optimizer(model, cfg)
    for phase_i, (frac, steps) in enumerate(cfg.curriculum):
        n_sub = int(round(frac * (n_train - 1))) + 1
        t_sub, q_sub = t_all[:n_sub], q_all[:n_sub]
        for step in range(steps):
            opt.zero_grad()
            traj = rk4_integrate(model.initial_state(), model.masses(), t_sub,
                                 substeps=cfg.substeps, softening=cfg.softening)
            loss_b = chi2_per_candidate(traj, q_sub, sigma) \
                + cfg.delta_prior * model.delta_penalty()
            loss_b.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            _clamp_(model, cfg)
            if verbose and step % max(1, steps // 4) == 0:
                _log_line(model, f"phase {phase_i} frac {frac}", step, loss_b)
        history.append(("curriculum", frac, float(loss_b.min())))

    # ── Phase B: multiple shooting on the full train arc ─────────────────────
    if cfg.ms_steps > 0 and cfg.n_segments > 1:
        k = cfg.n_segments
        assert (n_train - 1) % k == 0, \
            f"train intervals {n_train - 1} not divisible by n_segments {k}"
        seg_len = (n_train - 1) // k
        b = model.vis_delta.shape[0]
        n_bodies = model.masses().shape[-1]

        # Initialize interior segment starts from the current best integration.
        with torch.no_grad():
            traj = rk4_integrate(model.initial_state(), model.masses(),
                                 t_all[:n_train], substeps=cfg.substeps,
                                 softening=cfg.softening)
        seg_states = torch.nn.Parameter(
            traj[[i * seg_len for i in range(1, k)]].permute(1, 0, 2, 3)
            .contiguous())                                   # (B, K-1, N, 4)

        # State-space scales for the continuity penalty
        dt_obs = float(ds.t[1] - ds.t[0])
        cont_scale = torch.tensor([sigma, sigma, sigma / dt_obs, sigma / dt_obs],
                                  dtype=torch.float64, device=device)
        t_seg = t_all[: seg_len + 1]                         # uniform grid: shared offsets
        q_segs = torch.stack([q_all[i * seg_len: i * seg_len + seg_len + 1]
                              for i in range(k)])            # (K, T_seg, Nv, 2)

        opt = _make_optimizer(model, cfg, extra_params=[seg_states])
        for step in range(cfg.ms_steps):
            opt.zero_grad()
            starts = torch.cat([model.initial_state().unsqueeze(1), seg_states],
                               dim=1)                        # (B, K, N, 4)
            flat = starts.reshape(b * k, n_bodies, 4)
            m_flat = model.masses().unsqueeze(1).expand(b, k, n_bodies) \
                                   .reshape(b * k, n_bodies)
            traj = rk4_integrate(flat, m_flat, t_seg, substeps=cfg.substeps,
                                 softening=cfg.softening)    # (T_seg, B*K, N, 4)
            traj = traj.reshape(-1, b, k, n_bodies, 4)

            nv = q_all.shape[1]
            res = (traj[:, :, :, :nv, :2]
                   - q_segs.permute(1, 0, 2, 3).unsqueeze(1)) / sigma
            data_b = (res ** 2).mean(dim=(0, 2, 3, 4))       # (B,)

            gap = (traj[-1, :, :-1] - seg_states) / cont_scale
            lam = cfg.lambda_cont * (10.0 if step > cfg.ms_steps // 2 else 1.0)
            cont_b = (gap ** 2).mean(dim=(1, 2, 3))

            loss_b = data_b + lam * cont_b + cfg.delta_prior * model.delta_penalty()
            loss_b.sum().backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + [seg_states], cfg.clip)
            opt.step()
            _clamp_(model, cfg)
            if verbose and step % max(1, cfg.ms_steps // 4) == 0:
                _log_line(model, "mshoot", step, data_b)
        history.append(("mshoot", 1.0, float(data_b.min())))

    # ── Final honest selection: single-shot from t0 over the full train arc ──
    with torch.no_grad():
        traj = rk4_integrate(model.initial_state(), model.masses(),
                             t_all[:n_train], substeps=cfg.substeps,
                             softening=cfg.softening)
        train_chi2 = chi2_per_candidate(traj, q_all[:n_train], sigma) \
            .cpu().numpy()
    best_idx = int(train_chi2.argmin())
    if verbose:
        order = np.argsort(train_chi2)
        print(f"  [select] train chi2 per restart (sorted): "
              f"{np.array2string(train_chi2[order][:5], precision=2)}")
    return FitResult(best_idx=best_idx, train_chi2=train_chi2,
                     history=history, wall_time=time.time() - t_start)
