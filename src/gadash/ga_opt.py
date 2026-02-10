from __future__ import annotations

import json
import math
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
from .spectral import wavelengths_nm
from .surrogate_interface import CRReconSurrogate, SurrogateError


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
    if not cfg.paths.checkpoint_path or not cfg.paths.cr_recon_root:
        raise SurrogateError(
            "missing paths config. Create configs/paths.yaml with checkpoint_path and cr_recon_root."
        )

    surrogate = CRReconSurrogate(
        cr_recon_root=cfg.paths.cr_recon_root,
        checkpoint_path=cfg.paths.checkpoint_path,
        device=str(dev),
    )

    gen = build_generator(cfg.design, cfg.generator)

    logger = ProgressLogger(progress_dir)
    meta = {
        "app": "gadash",
        "device": str(dev),
        "config": {
            "ga": asdict(cfg.ga),
            "design": asdict(cfg.design),
            "generator": asdict(cfg.generator),
            "robustness": asdict(cfg.robustness),
            "spectra": asdict(cfg.spectra),
            "loss": asdict(cfg.loss),
            "io": asdict(cfg.io),
        },
        "wavelengths_nm": wavelengths_nm(cfg.spectra.channels, cfg.spectra.wavelength_min_nm, cfg.spectra.wavelength_max_nm).tolist(),
    }
    logger.write_meta(meta)

    topk = int(cfg.io.topk)
    pop = _init_population(cfg, dev)

    # helper: evaluate
    def eval_pop(a_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        seed01 = _seed01_from_raw(a_raw)
        struct01 = gen(seed01)

        # robustness: average over repeated forward runs (placeholder: same surrogate, no noise)
        # hook: add dropout/noise in future
        pred = surrogate.predict(struct01).pred_rgbc
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

        # pack + save best-so-far topk and current-gen topk
        def _pack(a_raw_top: torch.Tensor, path: Path) -> None:
            with torch.no_grad():
                seed01 = _seed01_from_raw(a_raw_top)
                struct01 = gen(seed01)
                pred = surrogate.predict(struct01).pred_rgbc
            np.savez_compressed(
                path,
                a_raw=a_raw_top.detach().cpu().numpy(),
                seed01=seed01.detach().cpu().numpy(),
                struct01=struct01.detach().cpu().numpy(),
                pred=pred.detach().cpu().numpy(),
            )

        progress_dir_p = Path(progress_dir)
        _pack(pop_sorted[:topk], progress_dir_p / f"topk_cur_step-{gidx}.npz")
        if best_so_far is not None:
            _pack(best_so_far[:topk], progress_dir_p / f"topk_step-{gidx}.npz")

        # metrics
        m = {
            "ts": logger.now_iso(),
            "gen": int(gidx),
            "loss_total": float(loss_sorted.mean().item()),
            "loss_best": float(loss_sorted[0].item()),
            "loss_spec": float(ev["loss_spec"].mean().item()),
            "loss_reg": float(ev["loss_reg"].mean().item()),
            "fill": float(ev["fill"].mean().item()),
        }
        logger.append_metrics(m)

        if cfg.io.print_every > 0 and (gidx % int(cfg.io.print_every) == 0 or gidx == gens - 1):
            print(
                f"[GEN {gidx}/{gens-1}] loss_best={m['loss_best']:.4f} loss_mean={m['loss_total']:.4f} fill={m['fill']:.3f}"
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
