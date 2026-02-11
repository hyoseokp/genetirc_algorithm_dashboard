from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import threading
import shutil

from .config import AppConfig
from .generator import build_generator
from .losses import loss_from_pred
from .progress_logger import ProgressLogger
from .spectral import merge_rggb_to_rgb
from .surrogate_interface import CRReconSurrogate, MockSurrogate
from .fdtd_verify import resolve_fdtd_cfg, verify_topk_with_fdtd


def _torch_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _init_population(cfg: AppConfig, device: torch.device) -> torch.Tensor:
    # population of raw logits -> seed01 via sigmoid
    B = cfg.ga.population
    S = cfg.design.seed_size
    # Match generator device to tensor device (CUDA requires a CUDA generator).
    g = torch.Generator(device=device.type)
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

def _tournament_select_batch(losses: torch.Tensor, k: int, n_select: int) -> torch.Tensor:
    """Vectorized tournament selection. Returns indices (n_select,) on losses.device."""
    n = int(losses.numel())
    if n <= 0:
        raise ValueError("empty population")
    k = max(1, int(k))
    n_select = max(0, int(n_select))
    if n_select == 0:
        return torch.empty((0,), device=losses.device, dtype=torch.long)
    idx = torch.randint(0, n, (n_select, k), device=losses.device)
    best_in_tourn = losses[idx].argmin(dim=1, keepdim=True)
    sel = idx.gather(1, best_in_tourn).squeeze(1)
    return sel.to(torch.long)


def _make_child(p1: torch.Tensor, p2: torch.Tensor, alpha: float) -> torch.Tensor:
    # blend crossover
    return alpha * p1 + (1.0 - alpha) * p2


def run_ga(
    cfg: AppConfig,
    progress_dir: str | Path,
    device: str = "cpu",
    dry_run: bool = False,
    fdtd_verify: bool = False,
    fdtd_every: int = 0,
    fdtd_k: int | None = None,
    fdtd_config: str | Path = "configs/fdtd.yaml",
    paths_yaml: str | Path = "configs/paths.yaml",
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

    chunk = int(getattr(cfg.ga, "chunk_size", 0) or 0)
    if chunk <= 0:
        chunk = int(cfg.ga.population)

    # helper: evaluate losses only (chunked to cap memory)
    def eval_losses(a_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        B = a_raw.shape[0]
        loss_total = torch.empty((B,), device=dev, dtype=torch.float32)
        loss_spec = torch.empty_like(loss_total)
        loss_reg = torch.empty_like(loss_total)
        loss_purity = torch.empty_like(loss_total)
        loss_fill = torch.empty_like(loss_total)
        fill = torch.empty_like(loss_total)

        for i0 in range(0, B, chunk):
            i1 = min(B, i0 + chunk)
            seed01 = _seed01_from_raw(a_raw[i0:i1])
            struct01 = gen(seed01)
            t = surrogate.predict(struct01[:, 0])
            pred = merge_rggb_to_rgb(t)
            losses = loss_from_pred(pred, struct01, cfg.spectra, cfg.loss)
            loss_total[i0:i1] = losses["loss_total"].detach()
            loss_spec[i0:i1] = losses["loss_spec"].detach()
            loss_reg[i0:i1] = losses["loss_reg"].detach()
            loss_purity[i0:i1] = losses["loss_purity"].detach()
            loss_fill[i0:i1] = losses["loss_fill"].detach()
            fill[i0:i1] = losses["fill"].detach()

        return {
            "loss_total": loss_total,
            "loss_spec": loss_spec,
            "loss_reg": loss_reg,
            "loss_purity": loss_purity,
            "loss_fill": loss_fill,
            "fill": fill,
        }

    def _struct_for(a_raw_sel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seed01 = _seed01_from_raw(a_raw_sel)
        struct01 = gen(seed01)
        return seed01, struct01

    best_araw = None
    best_loss_vec = None

    gens = int(cfg.ga.generations)
    t_k = int(cfg.ga.tournament_k)
    alpha = float(cfg.ga.crossover_alpha)
    mut_sigma = float(cfg.ga.mutation_sigma)
    mut_p = float(cfg.ga.mutation_p)

    if dry_run:
        gens = 2  # still evolve; dry-run only swaps surrogate to MockSurrogate

    # Optional periodic FDTD verification scheduler.
    do_fdtd = bool(fdtd_verify) and (not dry_run)
    every = int(fdtd_every or 0)
    if every < 0:
        raise ValueError("fdtd_every must be >= 0")

    fdtd_cfg = None
    if do_fdtd:
        try:
            fdtd_cfg = resolve_fdtd_cfg(fdtd_yaml=fdtd_config, paths_yaml=paths_yaml)
        except Exception as e:
            print(f"[WARN] FDTD config not ready; disabling FDTD verify: {e}", flush=True)
            do_fdtd = False

    class _FDTDScheduler:
        def __init__(self):
            self.lock = threading.Lock()
            self.thread: threading.Thread | None = None
            self.pending: tuple[int, str] | None = None
            self.verified: set[int] = set()

        def request(self, *, step: int, topk_npz: str) -> None:
            with self.lock:
                if int(step) in self.verified:
                    return
                self.pending = (int(step), str(topk_npz))
                self._maybe_start_locked()

        def _maybe_start_locked(self) -> None:
            if self.thread is not None and self.thread.is_alive():
                return
            if self.pending is None:
                return
            step, topk_npz = self.pending
            self.pending = None

            def _worker() -> None:
                try:
                    assert fdtd_cfg is not None
                    v = verify_topk_with_fdtd(
                        topk_npz=topk_npz,
                        fdtd_cfg=fdtd_cfg,
                        out_dir=os.environ.get("GADASH_FDTD_OUT", r"C:\gadash_fdtd_results"),
                        k=fdtd_k,
                    )
                    pdir = Path(progress_dir)
                    pdir.mkdir(parents=True, exist_ok=True)
                    dst = pdir / f"fdtd_rggb_step-{int(step)}.npy"
                    shutil.copyfile(v.fdtd_rggb_path, dst)
                    meta = {
                        "step": int(step),
                        "k": int(fdtd_k) if fdtd_k is not None else None,
                        "out_dir": str(v.out_dir),
                    }
                    (pdir / "fdtd_meta.json").write_text(
                        json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
                    )
                    print(f"[OK] fdtd-verify saved: {dst}", flush=True)
                    with self.lock:
                        self.verified.add(int(step))
                except Exception as e:
                    print(f"[WARN] fdtd-verify failed (step={step}): {e}", flush=True)
                finally:
                    with self.lock:
                        self.thread = None
                        self._maybe_start_locked()

            self.thread = threading.Thread(target=_worker, daemon=False)
            self.thread.start()

        def drain(self) -> None:
            while True:
                with self.lock:
                    th = self.thread
                    pending = self.pending
                if th is None and pending is None:
                    return
                if th is not None:
                    th.join(timeout=0.5)

    fdtd_sched = _FDTDScheduler() if (do_fdtd and every > 0 and fdtd_cfg is not None) else None

    for gidx in range(gens):
        ev = eval_losses(pop)
        loss_total = ev["loss_total"]

        # sort by fitness
        order = torch.argsort(loss_total)
        pop_sorted = pop[order]
        loss_sorted = loss_total[order]

        # update best-so-far pool
        if best_araw is None:
            best_araw = pop_sorted[: min(topk, pop.shape[0])].detach().clone()
            best_loss_vec = loss_sorted[: min(topk, pop.shape[0])].detach().clone()
        else:
            cand_araw = torch.cat([best_araw, pop], dim=0)
            cand_loss = torch.cat([best_loss_vec, loss_total], dim=0)
            o2 = torch.argsort(cand_loss)[: min(topk, cand_loss.numel())]
            best_araw = cand_araw[o2].detach().clone()
            best_loss_vec = cand_loss[o2].detach().clone()

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
        a_cur = pop[order[:topk]]
        seed01_cur, struct01_cur = _struct_for(a_cur)
        _pack(seed01_cur, struct01_cur, "cur_loss", loss_sorted[:topk], progress_dir_p / f"topk_cur_step-{gidx}.npz")

        # best-so-far topk
        assert best_araw is not None and best_loss_vec is not None
        seed01_best, struct01_best = _struct_for(best_araw[:topk])
        best_npz = progress_dir_p / f"topk_step-{gidx}.npz"
        _pack(seed01_best, struct01_best, "best_loss", best_loss_vec[:topk], best_npz)

        # metrics
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
                f"[GEN {gidx}/{gens-1}] loss_mean={m['loss_total']:.4f} fill={m['fill']:.3f}"
            )

        # Periodic FDTD verification (non-blocking).
        if fdtd_sched is not None:
            if (gidx % every == 0) or (gidx == gens - 1):
                try:
                    if best_npz.exists():
                        fdtd_sched.request(step=int(gidx), topk_npz=str(best_npz))
                except Exception:
                    pass

        # build next generation
        N = int(cfg.ga.population)
        keep_k = min(int(cfg.io.topk), N)
        keep = pop_sorted[:keep_k]

        # Elite cloning (micro Gaussian noise in raw-logit space).
        clone_k = int(getattr(cfg.ga, "topk_clone_k", 0) or 0)
        clone_m = int(getattr(cfg.ga, "topk_clone_m", 0) or 0)
        clone_k = max(0, min(clone_k, keep_k))
        clone_m = max(0, clone_m)
        sig_min = float(getattr(cfg.ga, "topk_clone_sigma_min", 0.02) or 0.02)
        sig_max = float(getattr(cfg.ga, "topk_clone_sigma_max", 0.08) or 0.08)
        if sig_max < sig_min:
            sig_min, sig_max = sig_max, sig_min

        clones_list: list[torch.Tensor] = []
        cap_after_keep = max(0, N - keep.shape[0])
        total_clone_target = clone_k * clone_m
        total_clone = min(total_clone_target, cap_after_keep)
        if clone_k > 0 and clone_m > 0 and total_clone > 0:
            src = pop_sorted[:clone_k]
            if clone_m == 1:
                sigmas = [sig_min]
            else:
                sigmas = [sig_min + (sig_max - sig_min) * (j / (clone_m - 1)) for j in range(clone_m)]

            remaining = total_clone
            for sigma in sigmas:
                if remaining <= 0:
                    break
                c = src + float(sigma) * torch.randn_like(src)
                if c.shape[0] > remaining:
                    c = c[:remaining]
                clones_list.append(c)
                remaining -= int(c.shape[0])

        clones = torch.cat(clones_list, dim=0) if clones_list else torch.empty((0,) + keep.shape[1:], device=dev)

        # Children: tournament + blend crossover + mutation.
        n_child = max(0, N - int(keep.shape[0]) - int(clones.shape[0]))
        if n_child > 0:
            sel1 = _tournament_select_batch(loss_total, t_k, n_child)
            sel2 = _tournament_select_batch(loss_total, t_k, n_child)
            p1 = pop[sel1]
            p2 = pop[sel2]
            child = _make_child(p1, p2, alpha)
            if mut_p > 0.0 and mut_sigma > 0.0:
                msk = (torch.rand((n_child,), device=dev) < float(mut_p))
                if bool(msk.any()):
                    child[msk] = child[msk] + float(mut_sigma) * torch.randn_like(child[msk])
        else:
            child = torch.empty((0,) + keep.shape[1:], device=dev)

        pop = torch.cat([keep, clones, child], dim=0)[:N]

    if fdtd_sched is not None:
        fdtd_sched.drain()

    return {"progress_dir": str(progress_dir), "gens": gens}
