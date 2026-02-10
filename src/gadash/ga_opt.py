from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import AppConfig
from .generator import build_generator
from .losses import loss_from_pred
from .progress_logger import ProgressLogger
from .spectral import merge_rggb_to_rgb
from .surrogate_interface import CRReconSurrogate, MockSurrogate


def _torch_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _init_population(cfg: AppConfig, device: torch.device) -> torch.Tensor:
    # population of raw logits -> seed01 via sigmoid
    B = cfg.ga.population
    S = cfg.design.seed_size
    g = torch.Generator(device="cpu")
    g.manual_seed(int(cfg.ga.seed))
    a = torch.randn((B, 1, S, S), generator=g, device=device, dtype=torch.float32)
    return a


def _seed01_from_raw(a_raw: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(a_raw)


def _tournament_select(losses: torch.Tensor, k: int) -> int:
    # losses: (N,) lower is better
    n = int(losses.numel())
    idx = torch.randint(0, n, (k,), device=losses.device)
    best = idx[losses[idx].argmin()]
    return int(best.item())


def _make_child(p1: torch.Tensor, p2: torch.Tensor, alpha: float) -> torch.Tensor:
    # blend crossover
    return alpha * p1 + (1.0 - alpha) * p2


def run_ga(
    cfg: AppConfig,
    progress_dir: str | Path,
    device: str = "cpu",
    dry_run: bool = False,
) -> dict[str, Any]:
    dev = _torch_device(device)

    # Surrogate
    if dry_run:
        surrogate = MockSurrogate(n_channels=int(cfg.spectra.channels))
    else:
        if (not cfg.paths.forward_model_root) or (not cfg.paths.forward_checkpoint) or (not cfg.paths.forward_config_yaml):
            raise ValueError(
                "missing paths config. Create configs/paths.yaml with "
                "forward_model_root / forward_config_yaml / forward_checkpoint."
            )
        surrogate = CRReconSurrogate(
            forward_model_root=Path(cfg.paths.forward_model_root),
            checkpoint_path=Path(cfg.paths.forward_checkpoint),
            config_yaml=Path(cfg.paths.forward_config_yaml),
            device=dev,
        )

    gen = build_generator(cfg.design, cfg.generator)

    logger = ProgressLogger(progress_dir)
    # Write meta in the same shape as Inverse_design_CR dashboard expects.
    logger.write_meta(
        {
            "ts_start": logger.now_iso(),
            "engine": "ga",
            "n_start": int(cfg.ga.population),
            "n_steps": int(cfg.ga.generations),
            "topk": int(cfg.io.topk),
            "seed_size": int(cfg.design.seed_size),
            "struct_size": int(cfg.design.struct_size),
            "device": str(dev),
            "ga": asdict(cfg.ga),
            "generator": asdict(cfg.generator),
            "spectra": {"n_channels": int(cfg.spectra.channels), "rgb_weights": dict(cfg.spectra.rgb_weights or {})},
            "loss": asdict(cfg.loss),
        }
    )

    topk = int(cfg.io.topk)
    pop = _init_population(cfg, dev)

    # helper: evaluate
    def eval_pop(a_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        seed01 = _seed01_from_raw(a_raw)
        struct01 = gen(seed01)

        # Surrogate: RGGB -> RGB
        t = surrogate.predict(struct01[:, 0])
        pred = merge_rggb_to_rgb(t)
        losses = loss_from_pred(pred, struct01, cfg.spectra, cfg.loss)
        return {
            "seed01": seed01,
            "struct01": struct01,
            "pred": pred,
            **{k: v for k, v in losses.items() if k != "A"},
        }

    best_so_far = None
    best_loss = None

    gens = int(cfg.ga.generations)
    elite = int(cfg.ga.elite)
    t_k = int(cfg.ga.tournament_k)
    alpha = float(cfg.ga.crossover_alpha)
    mut_sigma = float(cfg.ga.mutation_sigma)
    mut_p = float(cfg.ga.mutation_p)

    if dry_run:
        gens = 2

    for gidx in range(gens):
        ev = eval_pop(pop)
        loss_total = ev["loss_total"]

        # sort by fitness
        order = torch.argsort(loss_total)
        pop_sorted = pop[order]
        loss_sorted = loss_total[order]

        # track best
        if best_loss is None or float(loss_sorted[0].item()) < float(best_loss):
            best_loss = float(loss_sorted[0].item())
            best_so_far = pop_sorted[0:topk].detach().clone()

        # pack + save snapshots in the same format as Inverse_design_CR dashboard expects.
        def _pack(seed01_top: torch.Tensor, struct01_top: torch.Tensor, metric_name: str, metric_vals: torch.Tensor, path: Path) -> None:
            seed16 = seed01_top[:, 0].detach().cpu().numpy().astype(np.float32)  # (K,16,16)
            struct_u8 = (struct01_top[:, 0].detach().cpu().numpy() >= 0.5).astype(np.uint8)  # (K,128,128)
            np.savez_compressed(
                path,
                seed16_topk=seed16,
                struct128_topk=struct_u8,
                **{f"metric_{metric_name}": metric_vals.detach().cpu().numpy().astype(np.float32)},
            )

        progress_dir_p = Path(progress_dir)
        # current-gen topk
        seed01_cur = ev["seed01"][order[:topk]]
        struct01_cur = ev["struct01"][order[:topk]]
        _pack(seed01_cur, struct01_cur, "cur_loss", loss_sorted[:topk], progress_dir_p / f"topk_cur_step-{gidx}.npz")

        # best-so-far topk
        if best_so_far is not None:
            with torch.no_grad():
                seed01_best = _seed01_from_raw(best_so_far[:topk])
                struct01_best = gen(seed01_best)
            # best losses are tracked approximately via best_loss scalar; store per-item current estimate.
            # For dashboard, metric is optional; keep a vector shaped (K,)
            best_metric = torch.zeros((seed01_best.shape[0],), device=loss_sorted.device, dtype=loss_sorted.dtype)
            best_metric[0] = float(best_loss)
            _pack(seed01_best, struct01_best, "best_loss", best_metric, progress_dir_p / f"topk_step-{gidx}.npz")

        # metrics
        m = {
            "ts": logger.now_iso(),
            "step": int(gidx),
            "loss_total": float(loss_sorted.mean().item()),
            "loss_spec": float(ev["loss_spec"].mean().item()),
            "loss_reg": float(ev["loss_reg"].mean().item()),
            "loss_purity": float(ev["loss_purity"].mean().item()),
            "loss_fill": float(ev["loss_fill"].mean().item()),
            "fill": float(ev["fill"].mean().item()),
        }
        logger.append_metrics(m)

        if cfg.io.print_every > 0 and (gidx % int(cfg.io.print_every) == 0 or gidx == gens - 1):
            print(
                f"[GEN {gidx}/{gens-1}] loss_mean={m['loss_total']:.4f} fill={m['fill']:.3f}"
            )

        if dry_run:
            continue

        # build next generation
        next_pop = []

        # elites
        next_pop.append(pop_sorted[:elite])

        # rest by tournament + crossover + mutation
        N = int(cfg.ga.population)
        while sum(x.shape[0] for x in next_pop) < N:
            i1 = _tournament_select(loss_total, t_k)
            i2 = _tournament_select(loss_total, t_k)
            p1 = pop[i1 : i1 + 1]
            p2 = pop[i2 : i2 + 1]
            child = _make_child(p1, p2, alpha)

            if random.random() < mut_p:
                child = child + mut_sigma * torch.randn_like(child)

            next_pop.append(child)

        pop = torch.cat(next_pop, dim=0)[:N]

    return {"progress_dir": str(progress_dir), "gens": gens}
