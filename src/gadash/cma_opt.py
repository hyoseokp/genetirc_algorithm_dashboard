"""CMA-ES optimizer — self-contained (numpy only, no pycma dependency).

Uses the same evaluation pipeline and progress format as ga_opt.run_ga()
so the dashboard can display CMA-ES runs identically.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import AppConfig
from .generator import build_generator
from .ga_opt import (
    _torch_device,
    _seed01_from_raw,
    build_surrogate,
    build_eval_fn,
    save_topk_snapshot,
    write_run_meta,
    write_run_meta_resume,
    load_resume_state,
    setup_fdtd_scheduler,
)
from .optimization_logger import OptimizationLogManager
from .progress_logger import ProgressLogger


# ---------------------------------------------------------------------------
# CMA-ES internals (Hansen 2016 reference implementation, simplified)
# ---------------------------------------------------------------------------

def _cma_weights(lam: int):
    """Log-linear recombination weights for the top mu individuals."""
    mu = lam // 2
    raw = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
    weights = raw / raw.sum()
    mu_eff = 1.0 / float(np.sum(weights ** 2))
    return mu, weights, mu_eff


def _cma_params(dim: int, mu_eff: float):
    """Derive CMA-ES learning rates and damping from dimension."""
    n = float(dim)
    # Step-size control (CSA)
    c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
    # Covariance matrix adaptation
    c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
    c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
    return c_sigma, d_sigma, c_c, c_1, c_mu, chi_n


def _cma_sample(mean: np.ndarray, sigma: float, C: np.ndarray, lam: int, rng: np.random.Generator):
    """Sample *lam* solutions from N(mean, sigma^2 * C) using eigendecomposition."""
    dim = mean.shape[0]
    eigvals, B = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-20)
    D = np.sqrt(eigvals)
    # z_i ~ N(0, I),  x_i = mean + sigma * B * D * z_i
    Z = rng.standard_normal((lam, dim))
    X = mean[None, :] + sigma * (Z @ np.diag(D) @ B.T)
    return X, B, D


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cmaes(
    cfg: AppConfig,
    progress_dir: str | Path,
    device: str = "cuda",
    dry_run: bool = False,
    fdtd_verify: bool = False,
    fdtd_every: int = 0,
    fdtd_k: int | None = None,
    fdtd_config: str | Path = "configs/fdtd.yaml",
    paths_yaml: str | Path = "configs/paths.yaml",
    resume: bool = False,
    optimization_log_dir: str | Path | None = None,
) -> dict[str, Any]:
    start_time = time.time()
    dev = _torch_device(device)

    opt_log_manager = None
    success = False
    error_msg = None

    if optimization_log_dir:
        try:
            opt_log_manager = OptimizationLogManager(
                Path(optimization_log_dir),
                optimizer_type="CMA-ES",
                seed=int(getattr(cfg.ga, "seed", 0) or 0),
                start_time=start_time,
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize optimization logger: {e}", flush=True)

    surrogate = build_surrogate(cfg, dev, dry_run=dry_run)
    gen = build_generator(cfg.design, cfg.generator)

    logger = ProgressLogger(progress_dir, engine="cmaes")

    topk = int(cfg.io.topk)
    S = int(cfg.design.seed_size)
    dim = S * S  # 256 for 16x16

    chunk = int(getattr(cfg.ga, "chunk_size", 0) or 0)
    if chunk <= 0:
        chunk = int(cfg.ga.population)
    if dev.type == "cpu":
        chunk = min(int(chunk), 16)
    else:
        chunk = min(int(chunk), 32)

    eval_losses = build_eval_fn(cfg, gen, surrogate, dev, initial_chunk=chunk)

    gens = int(cfg.ga.generations)
    if dry_run:
        gens = 2

    fdtd_sched, fdtd_every_n = setup_fdtd_scheduler(
        fdtd_verify=fdtd_verify, dry_run=dry_run, fdtd_every=fdtd_every,
        fdtd_k=fdtd_k, fdtd_config=fdtd_config, paths_yaml=paths_yaml,
        progress_dir=progress_dir,
    )

    lam = int(cfg.ga.population)
    sigma = float(cfg.ga.cma_sigma0)

    # CMA-ES state
    rng = np.random.default_rng(int(cfg.ga.seed))
    mean = rng.standard_normal(dim).astype(np.float64) * 0.5
    C = np.eye(dim, dtype=np.float64)
    p_sigma = np.zeros(dim, dtype=np.float64)
    p_c = np.zeros(dim, dtype=np.float64)

    mu, weights, mu_eff = _cma_weights(lam)
    c_sigma, d_sigma, c_c, c_1, c_mu, chi_n = _cma_params(dim, mu_eff)

    best_araw = None
    best_loss_vec = None
    start_step = 0

    progress_dir_p = Path(progress_dir)

    if resume:
        state = load_resume_state(progress_dir_p, dev)
        if state is not None:
            start_step = state["last_step"] + 1
            print(f"[RESUME] resuming CMA-ES from step {start_step} (warm restart)", flush=True)
            best_araw = state["a_raw"][:topk]
            best_loss_vec = state["best_loss"][:topk] if state["best_loss"] is not None else None
            # Warm restart: set mean from best K logits, reset C/p_sigma/p_c
            raw_flat = state["a_raw"][:, 0].cpu().numpy().reshape(-1, dim)
            mean = raw_flat.mean(axis=0).astype(np.float64)
            C = np.eye(dim, dtype=np.float64)
            p_sigma = np.zeros(dim, dtype=np.float64)
            p_c = np.zeros(dim, dtype=np.float64)
            sigma = float(cfg.ga.cma_sigma0)
            write_run_meta_resume(cfg, logger, dev, engine="cmaes")
        else:
            print("[RESUME] no checkpoint found, starting fresh", flush=True)
            write_run_meta(cfg, logger, dev, engine="cmaes")
    else:
        write_run_meta(cfg, logger, dev, engine="cmaes")

    try:
        for gidx in range(start_step, gens):
            # --- Sample ---
            solutions, B, D = _cma_sample(mean, sigma, C, lam, rng)

            # Convert to torch tensor (B,1,S,S) for evaluation
            pop = torch.tensor(solutions, dtype=torch.float32, device=dev).reshape(lam, 1, S, S)

            # --- Evaluate ---
            ev = eval_losses(pop)
            loss_total = ev["loss_total"]

            # --- Sort ---
            order = torch.argsort(loss_total)
            loss_sorted = loss_total[order]
            solutions_sorted = solutions[order.cpu().numpy()]

            # --- Update best-so-far pool ---
            if best_araw is None:
                best_araw = pop[order[: min(topk, lam)]].detach().clone()
                best_loss_vec = loss_sorted[: min(topk, lam)].detach().clone()
            else:
                cand_araw = torch.cat([best_araw, pop], dim=0)
                cand_loss = torch.cat([best_loss_vec, loss_total], dim=0)
                o2 = torch.argsort(cand_loss)[: min(topk, cand_loss.numel())]
                best_araw = cand_araw[o2].detach().clone()
                best_loss_vec = cand_loss[o2].detach().clone()

            # --- Save snapshots ---
            a_cur = pop[order[:topk]]
            save_topk_snapshot(a_cur, loss_sorted[:topk], gen, "cur_loss", progress_dir_p / f"topk_cur_step-{gidx}.npz")

            assert best_araw is not None and best_loss_vec is not None
            best_npz = progress_dir_p / f"topk_step-{gidx}.npz"
            save_topk_snapshot(best_araw[:topk], best_loss_vec[:topk], gen, "best_loss", best_npz)

            # --- Metrics ---
            m = {
                "ts": logger.now_iso(),
                "step": int(gidx),
                "loss_total": float(loss_total.mean().item()),
                "loss_spec": float(ev["loss_spec"].mean().item()),
                "loss_reg": float(ev["loss_reg"].mean().item()),
                "loss_purity": float(ev["loss_purity"].mean().item()),
                "loss_fill": float(ev["loss_fill"].mean().item()),
                "fill": float(ev["fill"].mean().item()),
            }
            logger.append_metrics(m)

            if cfg.io.print_every > 0 and (gidx % int(cfg.io.print_every) == 0 or gidx == gens - 1):
                print(
                    f"[CMA {gidx}/{gens-1}] loss_mean={m['loss_total']:.4f} fill={m['fill']:.3f} sigma={sigma:.4f}"
                )

            # Periodic FDTD verification (non-blocking).
            if fdtd_sched is not None:
                if (gidx % fdtd_every_n == 0) or (gidx == gens - 1):
                    try:
                        if best_npz.exists():
                            fdtd_sched.request(step=int(gidx), topk_npz=str(best_npz))
                    except Exception:
                        pass

            # --- CMA-ES update ---
            # Weighted mean of top-mu solutions
            top_mu = solutions_sorted[:mu]  # (mu, dim)
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * top_mu, axis=0)

            # Cumulation: step-size (p_sigma)
            invsqrtC = B @ np.diag(1.0 / D) @ B.T
            delta_mean = (mean - mean_old) / sigma
            p_sigma = (1.0 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (invsqrtC @ delta_mean)

            # Heaviside function (hsig)
            norm_ps = float(np.linalg.norm(p_sigma))
            hsig_threshold = (1.4 + 2.0 / (dim + 1.0)) * chi_n * math.sqrt(1.0 - (1.0 - c_sigma) ** (2 * (gidx + 1)))
            hsig = 1.0 if norm_ps < hsig_threshold else 0.0

            # Cumulation: covariance (p_c)
            p_c = (1.0 - c_c) * p_c + hsig * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * delta_mean

            # Covariance matrix update (rank-one + rank-mu)
            diffs = (top_mu - mean_old[None, :]) / sigma  # (mu, dim)
            rank_mu_update = np.zeros_like(C)
            for i in range(mu):
                rank_mu_update += weights[i] * np.outer(diffs[i], diffs[i])

            C = (
                (1.0 - c_1 - c_mu) * C
                + c_1 * (np.outer(p_c, p_c) + (1.0 - hsig) * c_c * (2.0 - c_c) * C)
                + c_mu * rank_mu_update
            )

            # Enforce symmetry
            C = 0.5 * (C + C.T)

            # Step-size update (CSA)
            sigma = sigma * math.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1.0))

            # Clamp sigma to avoid degenerate behaviour
            sigma = max(1e-12, min(sigma, 1e6))

        if fdtd_sched is not None:
            fdtd_sched.drain()

        success = True
        return {"progress_dir": str(progress_dir), "gens": gens}

    except KeyboardInterrupt:
        error_msg = "User interrupted"
        success = False
        print("[CMAES] Optimization interrupted by user", flush=True)

    except Exception as e:
        error_msg = str(e)
        success = False
        print(f"[CMAES] Error during optimization: {e}", flush=True)

    finally:
        if opt_log_manager is not None:
            try:
                opt_log_manager.finalize(
                    progress_dir=Path(progress_dir),
                    engine="cmaes",
                    success=success,
                    error_msg=error_msg,
                )
                print(f"[LOG] Optimization logging finalized", flush=True)
            except Exception as e:
                print(f"[WARN] Error during finalization: {e}", flush=True)