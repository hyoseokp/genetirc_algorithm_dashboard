from __future__ import annotations

from dataclasses import asdict
import json
import os
import gc
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


# ---------------------------------------------------------------------------
# Fine-tuning dataset accumulation (FDTD verified data)
# ---------------------------------------------------------------------------

def _load_npz_struct_topk(path: Path) -> np.ndarray:
    """Load struct128_topk (K,128,128) from topk .npz file."""
    z = np.load(path, allow_pickle=False)
    if "struct128_topk" not in z.files:
        raise KeyError(f"struct128_topk missing in {path}")
    arr = np.asarray(z["struct128_topk"])
    if arr.ndim != 3:
        raise ValueError(f"expected struct128_topk (K,128,128), got {arr.shape}")
    return arr


def load_finetune_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load existing fine-tuning dataset.

    Returns:
        (struct_list, spectrum_fdtd) where:
            struct_list: (N, 128, 128) uint8 - structure patterns
            spectrum_fdtd: (N, 2, 2, C) float32 - RGGB spectra
        or None if not found
    """
    if not dataset_path.exists():
        return None
    try:
        z = np.load(dataset_path, allow_pickle=False)
        if "struct_list" not in z.files or "spectrum_fdtd" not in z.files:
            return None
        struct_list = np.asarray(z["struct_list"], dtype=np.uint8)
        spectrum_fdtd = np.asarray(z["spectrum_fdtd"], dtype=np.float32)
        # Ensure spectrum_fdtd is (N, 2, 2, C)
        if spectrum_fdtd.ndim != 4:
            print(f"[WARN] spectrum_fdtd has unexpected shape {spectrum_fdtd.shape}, expected (N,2,2,C)", flush=True)
            return None
        return struct_list, spectrum_fdtd
    except Exception as e:
        print(f"[WARN] failed to load fine-tuning dataset {dataset_path}: {e}", flush=True)
        return None


def save_finetune_data(
    topk_npz: Path,
    fdtd_rggb_npy: Path,
    dataset_path: Path,
    step: int,
) -> None:
    """Accumulate structure + FDTD spectrum pairs into fine-tuning dataset.

    Args:
        topk_npz: path to topk_step-{N}.npz containing struct128_topk (K,128,128)
        fdtd_rggb_npy: path to fdtd_rggb.npy (K,2,2,C) containing FDTD-verified spectra
        dataset_path: output dataset path (data/finetune_dataset.npz)
        step: current optimization step (for logging)

    Dataset format:
        struct_list: (N, 128, 128) uint8 - structure patterns
        spectrum_fdtd: (N, 2, 2, C) float32 - RGGB spectra from FDTD simulation
    """
    try:
        # Load structures from topk snapshot
        struct_topk = _load_npz_struct_topk(topk_npz)  # (K,128,128)

        # Load FDTD spectrum
        fdtd_rggb = np.load(fdtd_rggb_npy, allow_pickle=False)  # (K,2,2,C)
        if fdtd_rggb.ndim != 4 or fdtd_rggb.shape[1:3] != (2, 2):
            raise ValueError(f"expected fdtd_rggb (K,2,2,C), got {fdtd_rggb.shape}")
        K = struct_topk.shape[0]

        # Load existing dataset if available
        existing = load_finetune_dataset(dataset_path)
        if existing is not None:
            struct_list_existing, spectrum_existing = existing
            struct_list = np.concatenate([struct_list_existing, struct_topk], axis=0)
            spectrum_fdtd = np.concatenate([spectrum_existing, fdtd_rggb], axis=0)
        else:
            struct_list = struct_topk
            spectrum_fdtd = fdtd_rggb

        # Save cumulative dataset
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_path,
            struct_list=struct_list.astype(np.uint8),
            spectrum_fdtd=spectrum_fdtd.astype(np.float32),
        )

        total_count = struct_list.shape[0]
        print(f"[OK] fine-tuning dataset updated (step {step}): total {total_count} samples -> {dataset_path}", flush=True)
    except Exception as e:
        print(f"[WARN] failed to save fine-tuning data (step {step}): {e}", flush=True)


# ---------------------------------------------------------------------------
# Shared helpers reusable by both GA and CMA-ES optimizers
# ---------------------------------------------------------------------------

def build_surrogate(cfg: AppConfig, dev: torch.device, dry_run: bool = False):
    """Build and return a surrogate model (or mock for dry_run)."""
    if dry_run:
        return MockSurrogate(n_channels=int(cfg.spectra.channels))
    if (not cfg.paths.forward_model_root) or (not cfg.paths.forward_checkpoint) or (not cfg.paths.forward_config_yaml):
        raise ValueError(
            "missing paths config. Create configs/paths.yaml with "
            "forward_model_root / forward_config_yaml / forward_checkpoint."
        )
    return CRReconSurrogate(
        forward_model_root=Path(cfg.paths.forward_model_root),
        checkpoint_path=Path(cfg.paths.forward_checkpoint),
        config_yaml=Path(cfg.paths.forward_config_yaml),
        device=dev,
    )


def build_eval_fn(cfg: AppConfig, gen, surrogate, dev: torch.device, initial_chunk: int = 32):
    """Return an eval_losses(a_raw) function and the mutable chunk state.

    The returned callable accepts a_raw (B,1,S,S) raw logit tensor and returns
    a dict of per-sample loss tensors.
    """
    chunk_cur = int(max(1, initial_chunk))
    chunk_state = {"chunk_cur": chunk_cur}

    def _is_memory_error(e: Exception) -> bool:
        msg = str(e).lower()
        if isinstance(e, MemoryError):
            return True
        return (
            ("out of memory" in msg)
            or ("unable to allocate" in msg)
            or ("arraymemoryerror" in msg)
            or ("cuda out of memory" in msg)
        )

    def eval_losses(a_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        B = a_raw.shape[0]
        loss_total = torch.empty((B,), device=dev, dtype=torch.float32)
        loss_spec = torch.empty_like(loss_total)
        loss_reg = torch.empty_like(loss_total)
        loss_purity = torch.empty_like(loss_total)
        loss_fill = torch.empty_like(loss_total)
        fill = torch.empty_like(loss_total)

        i0 = 0
        while i0 < B:
            i1 = min(B, i0 + int(chunk_state["chunk_cur"]))
            try:
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
                i0 = i1
            except Exception as e:
                if _is_memory_error(e) and int(chunk_state["chunk_cur"]) > 1:
                    new_chunk = max(1, int(chunk_state["chunk_cur"]) // 2)
                    print(
                        f"[WARN] memory pressure in eval chunk={chunk_state['chunk_cur']}; retry with chunk={new_chunk}",
                        flush=True,
                    )
                    chunk_state["chunk_cur"] = new_chunk
                    if dev.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    gc.collect()
                    continue
                raise

        return {
            "loss_total": loss_total,
            "loss_spec": loss_spec,
            "loss_reg": loss_reg,
            "loss_purity": loss_purity,
            "loss_fill": loss_fill,
            "fill": fill,
        }

    return eval_losses


def save_topk_snapshot(
    a_raw: torch.Tensor,
    loss_vals: torch.Tensor,
    gen,
    metric_name: str,
    path: Path,
) -> None:
    """Save a top-k snapshot npz (seed16 + struct128 + metric)."""
    seed01 = _seed01_from_raw(a_raw)
    struct01 = gen(seed01)
    seed16 = seed01[:, 0].detach().cpu().numpy().astype(np.float32)
    struct_u8 = (struct01[:, 0].detach().cpu().numpy() >= 0.5).astype(np.uint8)
    np.savez_compressed(
        path,
        seed16_topk=seed16,
        struct128_topk=struct_u8,
        **{f"metric_{metric_name}": loss_vals.detach().cpu().numpy().astype(np.float32)},
    )


def write_run_meta(cfg: AppConfig, logger: ProgressLogger, dev: torch.device, engine: str = "ga") -> None:
    """Write run_meta.json in the progress directory."""
    logger.write_meta(
        {
            "ts_start": logger.now_iso(),
            "engine": engine,
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


class FDTDScheduler:
    """Non-blocking periodic FDTD verification (shared by GA and CMA-ES)."""

    def __init__(self, fdtd_cfg, fdtd_k: int | None, progress_dir: str | Path):
        self.fdtd_cfg = fdtd_cfg
        self.fdtd_k = fdtd_k
        self.progress_dir = Path(progress_dir)
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

        fdtd_cfg = self.fdtd_cfg
        fdtd_k = self.fdtd_k
        pdir = self.progress_dir

        def _worker() -> None:
            try:
                assert fdtd_cfg is not None
                v = verify_topk_with_fdtd(
                    topk_npz=topk_npz,
                    fdtd_cfg=fdtd_cfg,
                    out_dir=os.environ.get("GADASH_FDTD_OUT", r"D:\gadash_fdtd_results"),
                    k=fdtd_k,
                )
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

                # Mark as verified (before attempting fine-tuning dataset save)
                with self.lock:
                    self.verified.add(int(step))

                # Accumulate fine-tuning dataset with verified data
                # Note: For multi-seed, finetune_dataset should be in a common location
                # pdir = data/progress/ (seed=0) or data/progress/seed_N/ (seed>0)
                # We want: data/finetune_dataset.npz for all seeds
                try:
                    if pdir.name.startswith("seed_"):
                        # Multi-seed: pdir = data/progress/seed_1000 -> common = data/
                        finetune_dataset_path = pdir.parent.parent / "finetune_dataset.npz"
                    else:
                        # Single seed: pdir = data/progress -> common = data/
                        finetune_dataset_path = pdir.parent / "finetune_dataset.npz"

                    save_finetune_data(
                        topk_npz=Path(topk_npz),
                        fdtd_rggb_npy=dst,
                        dataset_path=finetune_dataset_path,
                        step=int(step),
                    )
                except Exception as e:
                    # Fine-tuning dataset is optional; don't fail FDTD verification on this error
                    print(f"[WARN] failed to save finetune dataset (step={step}): {e}", flush=True)
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


def setup_fdtd_scheduler(
    fdtd_verify: bool,
    dry_run: bool,
    fdtd_every: int,
    fdtd_k: int | None,
    fdtd_config: str | Path,
    paths_yaml: str | Path,
    progress_dir: str | Path,
) -> tuple[FDTDScheduler | None, int]:
    """Resolve FDTD config and create scheduler. Returns (scheduler_or_None, every)."""
    do_fdtd = bool(fdtd_verify) and (not dry_run)
    every = int(fdtd_every or 0)
    if every < 0:
        raise ValueError("fdtd_every must be >= 0")

    if not do_fdtd or every <= 0:
        return None, every

    try:
        fdtd_cfg = resolve_fdtd_cfg(fdtd_yaml=fdtd_config, paths_yaml=paths_yaml)
    except Exception as e:
        print(f"[WARN] FDTD config not ready; disabling FDTD verify: {e}", flush=True)
        return None, every

    return FDTDScheduler(fdtd_cfg, fdtd_k, progress_dir), every


def load_resume_state(progress_dir: Path, device: torch.device) -> dict | None:
    """Load the last checkpoint from progress_dir for resuming optimization.

    Returns {"last_step": int, "a_raw": Tensor(K,1,S,S), "best_loss": Tensor(K,)}
    or None if no valid state is found.
    """
    progress_dir = Path(progress_dir)
    metrics_path = progress_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    # 1. Find last step from metrics.jsonl
    last_step = None
    with metrics_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "step" in obj:
                    step = int(obj["step"])
                    if last_step is None or step > last_step:
                        last_step = step
            except Exception:
                continue
    if last_step is None:
        return None

    # 2. Load topk_step-{last_step}.npz
    npz_path = progress_dir / f"topk_step-{last_step}.npz"
    if not npz_path.exists():
        # Try to find the largest available step <= last_step
        import re
        rx = re.compile(r"^topk_step-(\d+)\.npz$")
        best_step = None
        for p in progress_dir.iterdir():
            m = rx.match(p.name)
            if m:
                s = int(m.group(1))
                if s <= last_step and (best_step is None or s > best_step):
                    best_step = s
        if best_step is None:
            return None
        last_step = best_step
        npz_path = progress_dir / f"topk_step-{last_step}.npz"

    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception:
        return None

    # 3. seed16_topk (K, S, S) in [0,1] -> logit inverse -> a_raw (K, 1, S, S)
    if "seed16_topk" not in data:
        return None
    seed01 = data["seed16_topk"].astype(np.float32)  # (K, S, S)
    # Clamp to avoid log(0) or log(inf)
    eps = 1e-6
    seed01 = np.clip(seed01, eps, 1.0 - eps)
    a_raw_np = np.log(seed01 / (1.0 - seed01))  # inverse sigmoid
    a_raw = torch.tensor(a_raw_np, dtype=torch.float32, device=device).unsqueeze(1)  # (K, 1, S, S)

    # 4. metric_best_loss
    best_loss = None
    if "metric_best_loss" in data:
        best_loss = torch.tensor(
            data["metric_best_loss"].astype(np.float32), dtype=torch.float32, device=device
        )

    return {"last_step": int(last_step), "a_raw": a_raw, "best_loss": best_loss}


def write_run_meta_resume(cfg: AppConfig, logger: ProgressLogger, dev: torch.device, engine: str = "ga") -> None:
    """Update run_meta.json for resume: preserve ts_start, update n_steps."""
    meta_path = Path(logger.progress_dir) / "run_meta.json"
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing["n_steps"] = int(cfg.ga.generations)
                existing["engine"] = engine
                logger.write_meta(existing)
                return
        except Exception:
            pass
    # Fallback: write fresh meta
    write_run_meta(cfg, logger, dev, engine=engine)


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
    resume: bool = False,
    optimization_log_dir: str | Path | None = None,
) -> dict[str, Any]:
    import time
    from .optimization_logger import OptimizationLogManager

    start_time = time.time()
    dev = _torch_device(device)

    opt_log_manager = None
    success = False
    error_msg = None

    if optimization_log_dir:
        try:
            opt_log_manager = OptimizationLogManager(
                Path(optimization_log_dir),
                optimizer_type="GA",
                seed=int(getattr(cfg.ga, "seed", 0) or 0),
                start_time=start_time,
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize optimization logger: {e}", flush=True)

    surrogate = build_surrogate(cfg, dev, dry_run=dry_run)
    gen = build_generator(cfg.design, cfg.generator)

    logger = ProgressLogger(progress_dir, engine="ga")

    topk = int(cfg.io.topk)

    chunk = int(getattr(cfg.ga, "chunk_size", 0) or 0)
    if chunk <= 0:
        chunk = int(cfg.ga.population)
    if dev.type == "cpu":
        chunk = min(int(chunk), 16)
    else:
        chunk = min(int(chunk), 32)

    eval_losses = build_eval_fn(cfg, gen, surrogate, dev, initial_chunk=chunk)

    best_araw = None
    best_loss_vec = None
    start_step = 0

    gens = int(cfg.ga.generations)
    t_k = int(cfg.ga.tournament_k)
    alpha = float(cfg.ga.crossover_alpha)
    mut_sigma = float(cfg.ga.mutation_sigma)
    mut_p = float(cfg.ga.mutation_p)

    if dry_run:
        gens = 2  # still evolve; dry-run only swaps surrogate to MockSurrogate

    N = int(cfg.ga.population)

    if resume:
        state = load_resume_state(Path(progress_dir), dev)
        if state is not None:
            start_step = state["last_step"] + 1
            print(f"[RESUME] resuming GA from step {start_step}", flush=True)
            best_araw = state["a_raw"][:topk]
            best_loss_vec = state["best_loss"][:topk] if state["best_loss"] is not None else None
            # Initialize population from best K: keep as elite, fill rest with clone + noise
            K = best_araw.shape[0]
            if K >= N:
                pop = best_araw[:N].clone()
            else:
                noise_pop = best_araw.repeat((N // K) + 1, 1, 1, 1)[:N - K]
                noise_pop = noise_pop + 0.05 * torch.randn_like(noise_pop)
                pop = torch.cat([best_araw, noise_pop], dim=0)[:N]
            write_run_meta_resume(cfg, logger, dev, engine="ga")
        else:
            print("[RESUME] no checkpoint found, starting fresh", flush=True)
            pop = _init_population(cfg, dev)
            write_run_meta(cfg, logger, dev, engine="ga")
    else:
        pop = _init_population(cfg, dev)
        write_run_meta(cfg, logger, dev, engine="ga")

    fdtd_sched, every = setup_fdtd_scheduler(
        fdtd_verify=fdtd_verify, dry_run=dry_run, fdtd_every=fdtd_every,
        fdtd_k=fdtd_k, fdtd_config=fdtd_config, paths_yaml=paths_yaml,
        progress_dir=progress_dir,
    )

    try:
        for gidx in range(start_step, gens):
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

            progress_dir_p = Path(progress_dir)
            # current-gen topk
            a_cur = pop[order[:topk]]
            save_topk_snapshot(a_cur, loss_sorted[:topk], gen, "cur_loss", progress_dir_p / f"topk_cur_step-{gidx}.npz")

            # best-so-far topk
            assert best_araw is not None and best_loss_vec is not None
            best_npz = progress_dir_p / f"topk_step-{gidx}.npz"
            save_topk_snapshot(best_araw[:topk], best_loss_vec[:topk], gen, "best_loss", best_npz)

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
            # NOTE: FDTD scheduler uses progress_dir from its initialization.
            # For multi-seed: this should work as each seed has its own run context,
            # but finetune_dataset is saved to pdir.parent which may differ per seed.
            if fdtd_sched is not None:
                if (gidx % every == 0) or (gidx == gens - 1):
                    try:
                        if best_npz.exists():
                            fdtd_sched.request(step=int(gidx), topk_npz=str(best_npz))
                    except Exception as e:
                        print(f"[WARN] FDTD request failed (step={gidx}): {e}", flush=True)

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

    except KeyboardInterrupt:
        error_msg = "Interrupted by user"
        print("[GA] Optimization interrupted by user", flush=True)
    except Exception as e:
        error_msg = str(e)
        print(f"[GA] Error during optimization: {e}", flush=True)
    finally:
        if fdtd_sched is not None:
            fdtd_sched.drain()
        if opt_log_manager is not None:
            try:
                opt_log_manager.finalize(
                    progress_dir=Path(progress_dir),
                    engine="ga",
                    success=(error_msg is None),
                    error_msg=error_msg,
                )
                print(f"[LOG] Optimization logging finalized", flush=True)
            except Exception as e:
                print(f"[WARN] Error during finalization: {e}", flush=True)

    if error_msg is not None and error_msg != "Interrupted by user":
        # Only raise for real errors, not interrupts
        pass  # Continue to return below

    return {"progress_dir": str(progress_dir), "gens": gens}
