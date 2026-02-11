from __future__ import annotations

import io
import json
import math
import os
import re
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import shutil

import tempfile

import numpy as np
import torch
import yaml
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

def _merge_rggb_to_rgb(t: torch.Tensor) -> torch.Tensor:
    """Support either RGGB (B,2,2,C) or already-merged RGB (B,3,C)."""
    if not torch.is_tensor(t):
        raise TypeError("surrogate output must be a torch.Tensor")
    if t.ndim == 3 and t.shape[1] == 3:
        return t
    if t.ndim != 4 or tuple(t.shape[1:3]) != (2, 2):
        raise ValueError(f"unexpected surrogate output shape: {tuple(t.shape)}")
    r = t[:, 0, 0, :]
    g = 0.5 * (t[:, 0, 1, :] + t[:, 1, 0, :])
    b = t[:, 1, 1, :]
    return torch.stack([r, g, b], dim=1)

_TOPK_RE_BEST = re.compile(r"^topk_step-(\d+)\.npz$")
_TOPK_RE_CUR = re.compile(r"^topk_cur_step-(\d+)\.npz$")
_PURITY_RE = re.compile(r"^purity_step-(\d+)\.json$")
_FDTD_RE = re.compile(r"^fdtd_rggb_step-(\d+)\.npy$")


def _json_sanitize(obj: Any) -> Any:
    """Make payload safe for JSON.parse (replace NaN/Inf)."""
    if obj is None:
        return None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


def _tail_jsonl(path: Path, n: int) -> list[dict[str, Any]]:
    if n <= 0 or not path.exists():
        return []
    dq: deque[str] = deque(maxlen=n)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                dq.append(s)
    out: list[dict[str, Any]] = []
    for s in dq:
        try:
            o = json.loads(s)
            if isinstance(o, dict):
                out.append(o)
        except Exception:
            continue
    return out


def _read_meta(progress_dir: Path) -> dict[str, Any]:
    """Read the most recent run_meta file (prefer engine-specific versions)."""
    # Try engine-specific files first
    for engine in ["ga", "cmaes"]:
        p = progress_dir / f"run_meta_{engine}.json"
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    # Fallback to old unified file
    p = progress_dir / "run_meta.json"
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    return {}


def _ts_start_epoch(meta: dict[str, Any]) -> float | None:
    ts = meta.get("ts_start")
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


@dataclass
class TopKCache:
    key: tuple[str, int] | None = None
    npz: dict[str, np.ndarray] | None = None
    merged_key: str | None = None  # Cache key for merged topk
    merged_data: dict[str, Any] | None = None  # Cached merged topk result


@dataclass
class SpectrumCache:
    key: tuple[int, int, int | None] | None = None  # (step, idx, seed)
    rgb: np.ndarray | None = None


@dataclass
class RunProcState:
    proc: subprocess.Popen | None = None
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=400))
    started_ts: str | None = None
    last_exit_code: int | None = None


class DashboardFDTDScheduler:
    """Dashboard-level FDTD verification for multi-seed runs.

    Instead of each subprocess running FDTD independently, the dashboard
    manages a single centralized FDTD verification on the merged best
    structure (lowest loss across all seeds).
    """

    def __init__(self, progress_dir: Path, fdtd_every: int = 10, fdtd_k: int = 1):
        self.progress_dir = Path(progress_dir)
        self.fdtd_every = fdtd_every
        self.fdtd_k = fdtd_k
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.verified_steps: set[int] = set()
        self.last_checked_max_step: int = -1
        self.fdtd_cfg = None  # Set after resolve
        self.enabled = False

    def configure(self, fdtd_yaml: str | Path, paths_yaml: str | Path) -> bool:
        """Resolve FDTD config. Returns True if successful."""
        try:
            from .fdtd_verify import resolve_fdtd_cfg
            self.fdtd_cfg = resolve_fdtd_cfg(fdtd_yaml=fdtd_yaml, paths_yaml=paths_yaml)
            self.enabled = True
            print(f"[DASHBOARD-FDTD] Configured successfully", flush=True)
            return True
        except Exception as e:
            print(f"[DASHBOARD-FDTD] Config failed: {e}", flush=True)
            self.enabled = False
            return False

    def maybe_trigger(self, merged_data: dict[str, Any]) -> None:
        """Check if FDTD should be triggered based on merged topk data.

        Called from API endpoints during normal polling.
        """
        if not self.enabled or self.fdtd_cfg is None:
            return

        seed_steps = merged_data.get("seed_steps", {})
        if not seed_steps:
            return

        # Use max step across all seeds as the reference
        max_step = max(seed_steps.values())

        # Check if this step is eligible for FDTD (fdtd_every interval)
        if self.fdtd_every <= 0:
            return
        if max_step < self.fdtd_every:
            return
        # Find the nearest fdtd_every boundary
        trigger_step = (max_step // self.fdtd_every) * self.fdtd_every
        if trigger_step <= 0:
            return

        with self.lock:
            if trigger_step in self.verified_steps:
                return
            if self.thread is not None and self.thread.is_alive():
                return  # Already running

            struct = merged_data.get("struct128_topk")
            loss = merged_data.get("metric_best_loss")
            seed_origins = merged_data.get("seed_origins", [])
            if struct is None or not isinstance(struct, np.ndarray) or struct.ndim != 3:
                return
            if struct.shape[0] == 0:
                return

            # Take only the best fdtd_k structures
            k_use = min(self.fdtd_k, struct.shape[0])
            best_struct = struct[:k_use]
            best_loss = loss[:k_use] if isinstance(loss, np.ndarray) else None
            best_seeds = seed_origins[:k_use] if seed_origins else []

            print(f"[DASHBOARD-FDTD] Triggering FDTD for merged best at step={trigger_step}, "
                  f"k={k_use}, seeds={best_seeds}", flush=True)

            self.verified_steps.add(trigger_step)

            # Start worker thread
            self.thread = threading.Thread(
                target=self._worker,
                args=(trigger_step, best_struct, best_loss, best_seeds),
                daemon=False,
            )
            self.thread.start()

    def _worker(self, step: int, struct: np.ndarray, loss, seed_origins) -> None:
        """Run FDTD in background thread."""
        try:
            from .fdtd_verify import verify_topk_with_fdtd

            # Create temp NPZ with merged best
            tmp_dir = self.progress_dir / "_fdtd_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_npz = tmp_dir / f"merged_best_step-{step}.npz"
            save_dict = {"struct128_topk": struct}
            if loss is not None:
                save_dict["metric_best_loss"] = loss
            np.savez_compressed(str(tmp_npz), **save_dict)

            v = verify_topk_with_fdtd(
                topk_npz=str(tmp_npz),
                fdtd_cfg=self.fdtd_cfg,
                out_dir=os.environ.get("GADASH_FDTD_OUT", r"D:\gadash_fdtd_results"),
                k=struct.shape[0],
            )

            # Save result to base progress_dir (not seed-specific)
            dst = self.progress_dir / f"fdtd_rggb_step-{step}.npy"
            shutil.copyfile(v.fdtd_rggb_path, dst)

            # Save metadata including seed origins
            meta = {
                "step": int(step),
                "k": int(struct.shape[0]),
                "out_dir": str(v.out_dir),
                "seed_origins": [int(s) for s in seed_origins] if seed_origins else [],
                "source": "merged_best",
            }
            (self.progress_dir / "fdtd_meta.json").write_text(
                json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[DASHBOARD-FDTD] Saved: {dst}", flush=True)

            # Accumulate fine-tuning dataset
            try:
                from .ga_opt import save_finetune_data
                finetune_path = self.progress_dir.parent / "finetune_dataset.npz"
                save_finetune_data(
                    topk_npz=tmp_npz,
                    fdtd_rggb_npy=dst,
                    dataset_path=finetune_path,
                    step=step,
                )
            except Exception as e:
                print(f"[DASHBOARD-FDTD] finetune dataset save failed: {e}", flush=True)

            # Clean up temp
            try:
                tmp_npz.unlink()
            except Exception:
                pass

        except Exception as e:
            print(f"[DASHBOARD-FDTD] Failed (step={step}): {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            with self.lock:
                self.thread = None


def _get_progress_dir_for_seed(base_dir: Path, seed: int | None) -> Path:
    """Return the progress directory for a given seed.

    seed=None or seed=0 → base_dir (backward compat: single run writes to data/progress/).
    seed=N (N>0) → base_dir / f"seed_{N}".
    """
    if seed is None or seed == 0:
        return base_dir
    return base_dir / f"seed_{seed}"


def _init_seed_dir(base_dir: Path, seed: int | None) -> Path:
    """Create and return the seed-specific progress directory."""
    d = _get_progress_dir_for_seed(base_dir, seed)
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_app(*, progress_dir: Path, surrogate=None) -> FastAPI:
    progress_dir = Path(progress_dir).resolve()  # Convert to absolute path
    cache = TopKCache()
    scache = SpectrumCache()

    # Multi-seed state tracking
    rstate_dict: dict[int, RunProcState] = {}
    active_seeds: list[int] = []

    # Dashboard-level FDTD scheduler for multi-seed
    dash_fdtd = DashboardFDTDScheduler(progress_dir)

    app = FastAPI(title="GA Dashboard")

    static_dir = Path(__file__).parent / "dashboard_static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = static_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Dashboard UI missing</h1>", status_code=500)

    @app.get("/api/ping")
    def ping() -> JSONResponse:
        routes = sorted({getattr(r, "path", "") for r in app.router.routes if getattr(r, "path", "")})
        return JSONResponse({"ok": True, "routes": routes})

    @app.get("/api/debug")
    def debug() -> JSONResponse:
        """Minimal diagnostics to confirm which code is running (cwd/import path)."""
        try:
            import gadash  # local package
            gadash_file = getattr(gadash, "__file__", None)
        except Exception:
            gadash_file = None

        surrogate_status = "✓ Loaded" if surrogate is not None else "✗ Not available"

        return JSONResponse(
            _json_sanitize(
                {
                    "python": sys.executable,
                    "cwd": os.getcwd(),
                    "dashboard_app": str(Path(__file__).resolve()),
                    "gadash": gadash_file,
                    "repo_root_guess": str(Path(__file__).resolve().parents[2]),
                    "progress_dir": str(progress_dir),
                    "surrogate": surrogate_status,
                }
            )
        )

    @app.get("/api/meta")
    def meta() -> JSONResponse:
        return JSONResponse(_json_sanitize(_read_meta(progress_dir)))

    @app.get("/api/metrics")
    def metrics(
        tail: int = Query(default=2000, ge=1, le=20000),
        seed: int | None = Query(default=None),
    ) -> JSONResponse:
        """Read metrics from the most recent engine-specific file.

        seed=None and multiple seeds exist → merge from all seeds.
        seed=N → read from seed N's directory only.
        """
        has_multiple = len(_discover_seed_dirs()) > 1

        if seed is not None or not has_multiple:
            target_dir = _get_progress_dir_for_seed(progress_dir, seed)
            items = _read_metrics_from(target_dir, int(tail))
            return JSONResponse({"items": _json_sanitize(items), "seed": seed})

        # Multi-seed merge
        items = _merge_metrics_all_seeds(int(tail))
        return JSONResponse({"items": _json_sanitize(items), "merged": True})

    @app.get("/api/ls")
    def ls() -> JSONResponse:
        items = []
        if progress_dir.exists():
            for p in sorted(progress_dir.iterdir(), key=lambda x: x.name):
                try:
                    st = p.stat()
                    items.append({"name": p.name, "bytes": int(st.st_size), "is_dir": p.is_dir()})
                except Exception:
                    items.append({"name": p.name, "bytes": None, "is_dir": p.is_dir()})
        return JSONResponse({"progress_dir": str(progress_dir), "exists": progress_dir.exists(), "items": items})

    def _latest_topk_step_in(pdir: Path, *, mode: str) -> int | None:
        """Find latest topk step in a specific directory."""
        if not pdir.exists():
            return None
        mode = str(mode or "best").lower()
        rx = _TOPK_RE_BEST if mode != "cur" else _TOPK_RE_CUR
        m_meta = _read_meta(pdir)
        nsteps = m_meta.get("n_steps")
        max_step = None
        try:
            nsteps = int(nsteps) if nsteps is not None else None
            if nsteps is not None and nsteps > 0:
                max_step = nsteps - 1
        except Exception:
            max_step = None

        ts0 = _ts_start_epoch(m_meta)
        best = None
        for p in pdir.iterdir():
            m = rx.match(p.name)
            if not m:
                continue
            step = int(m.group(1))
            if max_step is not None and step > max_step:
                continue
            if ts0 is not None:
                try:
                    if p.stat().st_mtime < ts0:
                        continue
                except Exception:
                    pass
            if best is None or step > best:
                best = step
        return best

    def _latest_topk_step(*, mode: str) -> int | None:
        return _latest_topk_step_in(progress_dir, mode=mode)

    def _load_topk_from(pdir: Path, step: int, *, mode: str) -> dict[str, np.ndarray]:
        """Load topk npz from a specific directory."""
        mode = str(mode or "best").lower()
        prefix = "topk_cur" if mode == "cur" else "topk"
        p = Path(pdir).resolve() / f"{prefix}_step-{int(step)}.npz"
        if not p.exists():
            raise FileNotFoundError(f"TopK file not found: {p}")
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}

    def _load_topk(step: int, *, mode: str) -> dict[str, np.ndarray]:
        mode = str(mode or "best").lower()
        key = (("cur" if mode == "cur" else "best"), int(step))
        if cache.key == key and cache.npz is not None:
            return cache.npz
        data = _load_topk_from(progress_dir, step, mode=mode)
        cache.key = key
        cache.npz = data
        return data

    # ── Multi-seed aggregation helpers ──

    def _discover_seed_dirs() -> list[tuple[int, Path]]:
        """Return list of (seed, dir_path) for all active seed directories.

        Always includes the base progress_dir as seed=0 if it has topk/metrics files.
        Also scans for seed_*/ subdirectories.
        """
        results: list[tuple[int, Path]] = []
        # Base dir counts as seed=0
        if progress_dir.exists():
            has_data = any(
                _TOPK_RE_BEST.match(p.name) or p.name.startswith("metrics_")
                for p in progress_dir.iterdir() if p.is_file()
            )
            if has_data:
                results.append((0, progress_dir))
        # Scan seed_*/ subdirs
        if progress_dir.exists():
            seed_dir_re = re.compile(r"^seed_(\d+)$")
            for d in sorted(progress_dir.iterdir()):
                if not d.is_dir():
                    continue
                m = seed_dir_re.match(d.name)
                if m:
                    s = int(m.group(1))
                    results.append((s, d))
        return results

    def _scan_all_seeds_topk(*, mode: str) -> dict[int, int]:
        """Find latest topk step in each seed directory.

        Returns {seed: latest_step}.
        """
        result: dict[int, int] = {}
        for seed, pdir in _discover_seed_dirs():
            step = _latest_topk_step_in(pdir, mode=mode)
            if step is not None:
                result[seed] = step
        return result

    def _load_topk_all_seeds(*, mode: str, k: int = 10) -> dict[str, Any]:
        """Load topk from all seeds, merge by loss, return top-K overall.

        Returns dict with:
          struct128_topk: np.ndarray (K, 128, 128)
          metric_best_loss: np.ndarray (K,)
          seed_origins: list[int] (K,) — which seed each entry came from
          seed_steps: dict[int, int] — step used per seed
        """
        # Generate cache key including latest file mtimes for invalidation
        seed_steps = _scan_all_seeds_topk(mode=mode)
        mtime_sum = 0
        try:
            for seed, step in seed_steps.items():
                pdir = _get_progress_dir_for_seed(progress_dir, seed)
                prefix = "topk_cur" if mode == "cur" else "topk"
                p = pdir / f"{prefix}_step-{step}.npz"
                if p.exists():
                    mtime_sum += int(p.stat().st_mtime)
        except Exception:
            pass
        cache_key = f"merged_{mode}_{mtime_sum}"

        # Check cache first
        if cache.merged_key == cache_key and cache.merged_data is not None:
            return cache.merged_data

        if not seed_steps:
            result = {"struct128_topk": None, "seed_steps": {}, "seed_origins": []}
            cache.merged_key = cache_key
            cache.merged_data = result
            return result

        all_structs = []
        all_losses = []
        all_seeds = []
        all_metrics: dict[str, list] = {}

        for seed, step in seed_steps.items():
            pdir = _get_progress_dir_for_seed(progress_dir, seed)
            try:
                data = _load_topk_from(pdir, step, mode=mode)
            except Exception:
                continue
            struct = data.get("struct128_topk")
            if not isinstance(struct, np.ndarray) or struct.ndim != 3:
                continue
            n = struct.shape[0]
            all_structs.append(struct)
            all_seeds.extend([seed] * n)

            # Collect loss for sorting
            loss_key = "metric_best_loss" if mode != "cur" else "metric_cur_loss"
            loss_arr = data.get(loss_key)
            if isinstance(loss_arr, np.ndarray) and loss_arr.shape[0] == n:
                all_losses.append(loss_arr)
            else:
                all_losses.append(np.full(n, float("inf")))

            # Collect all metric arrays
            for mkey, marr in data.items():
                if mkey.startswith("metric_") and isinstance(marr, np.ndarray) and marr.shape[0] == n:
                    all_metrics.setdefault(mkey, []).append(marr)

        if not all_structs:
            return {"struct128_topk": None, "seed_steps": seed_steps, "seed_origins": []}

        combined_struct = np.concatenate(all_structs, axis=0)
        combined_loss = np.concatenate(all_losses, axis=0)
        combined_seeds = all_seeds

        # Sort by loss ascending, take top-K
        order = np.argsort(combined_loss)[:k]
        result_struct = combined_struct[order]
        result_loss = combined_loss[order]
        result_seeds = [combined_seeds[i] for i in order]

        result: dict[str, Any] = {
            "struct128_topk": result_struct,
            "metric_best_loss": result_loss,
            "seed_origins": result_seeds,
            "seed_steps": seed_steps,
        }

        # Also slice other metric arrays
        for mkey, marr_list in all_metrics.items():
            if mkey in ("metric_best_loss", "metric_cur_loss"):
                continue
            combined = np.concatenate(marr_list, axis=0)
            result[mkey] = combined[order]

        # Cache the merged result
        cache_key = f"merged_{mode}"
        cache.merged_key = cache_key
        cache.merged_data = result
        return result

    def _read_metrics_from(pdir: Path, tail: int) -> list[dict[str, Any]]:
        """Read metrics items from a specific directory."""
        metrics_files = []
        for engine in ["ga", "cmaes"]:
            p = pdir / f"metrics_{engine}.jsonl"
            if p.exists():
                metrics_files.append((p.stat().st_mtime, p))
        if metrics_files:
            metrics_files.sort(reverse=True)
            return _tail_jsonl(metrics_files[0][1], tail)
        fallback = pdir / "metrics.jsonl"
        if fallback.exists():
            return _tail_jsonl(fallback, tail)
        return []

    def _merge_metrics_all_seeds(tail: int) -> list[dict[str, Any]]:
        """Read and merge metrics from all seed directories.

        Each item gets a 'seed' field. Items sorted by step.
        """
        all_items: list[dict[str, Any]] = []
        for seed, pdir in _discover_seed_dirs():
            items = _read_metrics_from(pdir, tail)
            for it in items:
                it["seed"] = seed
            all_items.extend(items)
        # Sort by step
        all_items.sort(key=lambda x: (x.get("step", 0), x.get("seed", 0)))
        return all_items

    @app.get("/api/topk/latest")
    def topk_latest(
        mode: str = Query(default="best"),
        seed: int | None = Query(default=None),
    ) -> JSONResponse:
        has_multiple_seeds = len(_discover_seed_dirs()) > 1

        # If seed specified, or only single seed exists, use single-seed path
        if seed is not None or not has_multiple_seeds:
            target_dir = _get_progress_dir_for_seed(progress_dir, seed)
            step = _latest_topk_step_in(target_dir, mode=mode)
            if step is None:
                return JSONResponse({"step": None, "k": 0, "metrics": {}, "seed": seed})
            try:
                data = _load_topk_from(target_dir, step, mode=mode)
            except Exception as e:
                return JSONResponse({"step": int(step), "k": 0, "error": f"failed to load npz: {e}", "seed": seed}, status_code=500)
            struct = data.get("struct128_topk")
            k = int(struct.shape[0]) if isinstance(struct, np.ndarray) and struct.ndim == 3 else 0
            metrics_out: dict[str, Any] = {}
            for mkey, arr in data.items():
                if mkey.startswith("metric_"):
                    metrics_out[mkey] = arr.tolist()
            fill = []
            if isinstance(struct, np.ndarray) and struct.ndim == 3:
                fill = [float(struct[i].mean()) for i in range(struct.shape[0])]
            mode_str = "cur" if str(mode).lower() == "cur" else "best"
            seed_q = f"&seed={seed}" if seed is not None else ""
            return JSONResponse(
                _json_sanitize(
                    {
                        "step": int(step),
                        "k": k,
                        "mode": mode_str,
                        "seed": seed,
                        "images": [f"/api/topk/{int(step)}/{i}.png?mode={mode_str}{seed_q}" for i in range(k)],
                        "metrics": metrics_out,
                        "fill_frac": fill,
                    }
                )
            )

        # Multi-seed: merge best-K from all seeds
        merged = _load_topk_all_seeds(mode=mode)
        struct = merged.get("struct128_topk")
        if struct is None or not isinstance(struct, np.ndarray):
            return JSONResponse({"step": None, "k": 0, "metrics": {}, "merged": True, "seed_steps": merged.get("seed_steps", {})})

        # Trigger dashboard-level FDTD if configured
        if dash_fdtd.enabled and mode != "cur":
            dash_fdtd.maybe_trigger(merged)

        k = int(struct.shape[0])
        seed_steps = merged.get("seed_steps", {})
        seed_origins = merged.get("seed_origins", [])
        # Use max step across seeds for image URLs
        max_step = max(seed_steps.values()) if seed_steps else 0
        metrics_out = {}
        for mkey in merged:
            if mkey.startswith("metric_") and isinstance(merged[mkey], np.ndarray):
                metrics_out[mkey] = merged[mkey].tolist()
        fill = [float(struct[i].mean()) for i in range(k)]
        mode_str = "cur" if str(mode).lower() == "cur" else "best"
        return JSONResponse(
            _json_sanitize(
                {
                    "step": max_step,
                    "k": k,
                    "mode": mode_str,
                    "merged": True,
                    "seed_steps": seed_steps,
                    "seed_origins": seed_origins,
                    "images": [f"/api/topk/merged/{i}.png?mode={mode_str}" for i in range(k)],
                    "metrics": metrics_out,
                    "fill_frac": fill,
                }
            )
        )

    @app.get("/api/topk/merged/{idx}.png")
    def topk_merged_png(
        idx: int,
        invert: int = Query(default=1, ge=0, le=1),
        mode: str = Query(default="best"),
    ) -> Response:
        """Serve a single merged topk structure image from cache."""
        try:
            # Always use _load_topk_all_seeds which handles caching with mtime validation
            merged = _load_topk_all_seeds(mode=mode)

            struct = merged.get("struct128_topk")
            if struct is None or not isinstance(struct, np.ndarray):
                return Response(status_code=404)
            if idx < 0 or idx >= struct.shape[0]:
                return Response(status_code=404)
            s = struct[idx].astype(np.uint8)
            if int(invert) == 1:
                img = (1 - np.clip(s, 0, 1)) * 255
            else:
                img = np.clip(s, 0, 1) * 255
            im = Image.fromarray(img, mode="L")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return Response(
                content=buf.getvalue(),
                media_type="image/png",
                headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
            )
        except Exception:
            return Response(status_code=500)

    @app.get("/api/topk/{step}/{idx}.png")
    def topk_png(
        step: int,
        idx: int,
        invert: int = Query(default=1, ge=0, le=1),
        mode: str = Query(default="best"),
        seed: int | None = Query(default=None),
    ) -> Response:
        try:
            target_dir = _get_progress_dir_for_seed(progress_dir, seed)
            data = _load_topk_from(target_dir, int(step), mode=mode)
        except Exception:
            return Response(status_code=500)
        struct = data["struct128_topk"]
        if idx < 0 or idx >= struct.shape[0]:
            return Response(status_code=404)
        s = struct[idx].astype(np.uint8)
        if int(invert) == 1:
            img = (1 - np.clip(s, 0, 1)) * 255
        else:
            img = np.clip(s, 0, 1) * 255
        im = Image.fromarray(img, mode="L")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
        )

    @app.get("/api/topk/{step}/{idx}/spectrum")
    def topk_spectrum(
        step: int,
        idx: int,
        mode: str = Query(default="best"),
        seed: int | None = Query(default=None),
    ) -> JSONResponse:
        if surrogate is None:
            return JSONResponse({"error": "surrogate not configured"}, status_code=400)
        # Cache key includes seed to avoid collisions
        key = (int(step), int(idx), seed)
        if scache.key == key and scache.rgb is not None:
            rgb = scache.rgb
        else:
            target_dir = _get_progress_dir_for_seed(progress_dir, seed)
            data = _load_topk_from(target_dir, int(step), mode=mode)
            struct = data["struct128_topk"]
            if idx < 0 or idx >= struct.shape[0]:
                return JSONResponse({"error": "idx out of range"}, status_code=404)
            # Surrogate expects x_binary shaped [B,128,128] in [0,1].
            x = torch.from_numpy(struct[idx].astype(np.float32))[None, ...]  # (1,128,128)
            with torch.no_grad():
                y = surrogate.predict(x)
                if hasattr(y, "pred_rgbc"):
                    y = y.pred_rgbc
                rgb_t = _merge_rggb_to_rgb(y)[0]
            rgb = rgb_t.detach().cpu().numpy().astype(np.float32)
            scache.key = key
            scache.rgb = rgb
        C = int(rgb.shape[-1])
        return JSONResponse(
            _json_sanitize({"step": int(step), "idx": int(idx), "seed": seed, "n_channels": C, "rgb": rgb.tolist()}),
            headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
        )

    def _latest_purity() -> dict[str, Any] | None:
        if not progress_dir.exists():
            return None
        meta = _read_meta(progress_dir)
        nsteps = meta.get("n_steps")
        max_step = None
        try:
            nsteps = int(nsteps) if nsteps is not None else None
            if nsteps is not None and nsteps > 0:
                max_step = nsteps - 1
        except Exception:
            max_step = None

        ts0 = _ts_start_epoch(meta)
        best = None
        best_path = None
        for p in progress_dir.iterdir():
            m = _PURITY_RE.match(p.name)
            if not m:
                continue
            step = int(m.group(1))
            if max_step is not None and step > max_step:
                continue
            if ts0 is not None:
                try:
                    if p.stat().st_mtime < ts0:
                        continue
                except Exception:
                    pass
            if best is None or step > best:
                best = step
                best_path = p
        if best_path is None:
            return None
        try:
            obj = json.loads(best_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    @app.get("/api/purity/latest")
    def purity_latest() -> JSONResponse:
        obj = _latest_purity()
        if obj is None:
            return JSONResponse({"step": None})
        return JSONResponse(_json_sanitize(obj))

    @app.get("/api/topk/{step}/{idx}/fdtd_spectrum")
    def topk_fdtd_spectrum(
        step: int,
        idx: int,
        fallback: int = Query(default=1, ge=0, le=1),
        seed: int | None = Query(default=None),
    ) -> JSONResponse:
        """Return FDTD-verified RGB spectrum.

        If exact step is unavailable and fallback=1, use latest available step <= requested;
        if none, use the latest available step overall.

        For multi-seed (no seed param): also check base progress_dir for dashboard-level FDTD results.
        """
        req_step = int(step)
        target_dir = _get_progress_dir_for_seed(progress_dir, seed)
        avail: list[int] = []
        if target_dir.exists():
            for q in target_dir.iterdir():
                m = _FDTD_RE.match(q.name)
                if m:
                    avail.append(int(m.group(1)))
        # Also check base progress_dir for dashboard-level (merged) FDTD results
        if seed is not None and seed != 0:
            # For specific seed, also check base dir for merged FDTD
            if progress_dir.exists():
                for q in progress_dir.iterdir():
                    m = _FDTD_RE.match(q.name)
                    if m:
                        s_val = int(m.group(1))
                        if s_val not in avail:
                            avail.append(s_val)
        if not avail:
            return JSONResponse({"error": "fdtd spectrum not available", "requested_step": req_step}, status_code=404)

        avail_sorted = sorted(set(avail))
        candidates: list[int] = []
        if req_step in avail_sorted:
            candidates.append(req_step)
        if int(fallback) == 1:
            le = [s for s in avail_sorted if s <= req_step]
            for s in sorted(le, reverse=True):
                if s not in candidates:
                    candidates.append(s)
            for s in sorted(avail_sorted, reverse=True):
                if s not in candidates:
                    candidates.append(s)
        else:
            if req_step not in candidates:
                candidates.append(req_step)

        arr = None
        used_step = None
        last_err: Exception | None = None
        # Search in both target_dir and base progress_dir (for merged FDTD)
        search_dirs = [target_dir]
        if target_dir != progress_dir and progress_dir.exists():
            search_dirs.append(progress_dir)
        for s in candidates:
            found = False
            for sdir in search_dirs:
                p = sdir / f"fdtd_rggb_step-{s}.npy"
                if not p.exists():
                    continue
                try:
                    arr_try = np.load(p, allow_pickle=False)
                    arr = arr_try
                    used_step = int(s)
                    found = True
                    break
                except Exception as e:
                    last_err = e
                    continue
            if found:
                break

        if arr is None or used_step is None:
            return JSONResponse(
                {"error": f"fdtd spectrum not readable: {last_err}", "requested_step": req_step},
                status_code=500,
            )
        if arr.ndim != 4 or arr.shape[1:3] != (2, 2):
            return JSONResponse({"error": f"bad fdtd_rggb shape: {arr.shape}"}, status_code=500)
        if arr.shape[0] <= 0:
            return JSONResponse({"error": "empty fdtd_rggb"}, status_code=500)
        if int(idx) >= arr.shape[0]:
            return JSONResponse({"error": f"fdtd idx {idx} out of range (k={arr.shape[0]})", "requested_step": req_step}, status_code=404)
        used_idx = int(idx)
        t = torch.from_numpy(arr[used_idx : used_idx + 1].astype(np.float32))
        rgb = _merge_rggb_to_rgb(t)[0].detach().cpu().numpy().astype(np.float32)  # (3,C)
        return JSONResponse(
            _json_sanitize(
                {
                    "step": req_step,
                    "fdtd_step": int(used_step),
                    "idx": int(used_idx),
                    "seed": seed,
                    "n_channels": int(rgb.shape[-1]),
                    "rgb": rgb.tolist(),
                }
            ),
            headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
        )

    def _reader_thread(p: subprocess.Popen, seed: int) -> None:
        try:
            assert p.stdout is not None
            rs = rstate_dict.get(seed)
            if rs is None:
                return
            for line in p.stdout:
                s = line.rstrip("\n")
                if s:
                    if "lumapi.py" in s and "SyntaxWarning: invalid escape sequence" in s:
                        continue
                    if "message = re.sub('^(Error:)\\s(prompt line)\\s[0-9]+:'" in s:
                        continue
                    rs.lines.append(s)
        except Exception:
            pass

    @app.post("/api/run/start")
    def run_start(
        n_start: int = Query(default=200, ge=1),
        n_steps: int = Query(default=2000, ge=1),
        topk: int = Query(default=50, ge=1),
        dry_run: int = Query(default=0, ge=0, le=1),
        device: str = Query(default="cpu"),
        chunk_size: int = Query(default=32, ge=1),
        generator_backend: str | None = Query(default=None),
        fdtd_verify: int = Query(default=0, ge=0, le=1),
        fdtd_every: int = Query(default=10, ge=0, le=100000),
        fdtd_k: int | None = Query(default=1, ge=1),
        ga_mutation_p: float | None = Query(default=None),
        ga_mutation_sigma: float | None = Query(default=None),
        ga_topk_clone_k: int | None = Query(default=None, ge=0),
        ga_topk_clone_m: int | None = Query(default=None, ge=0),
        ga_topk_clone_sigma_min: float | None = Query(default=None, ge=0.0),
        ga_topk_clone_sigma_max: float | None = Query(default=None, ge=0.0),
        optimizer_type: str | None = Query(default=None),
        cma_sigma0: float | None = Query(default=None, ge=0.0),
        w_purity: float | None = Query(default=None),
        w_abs: float | None = Query(default=None),
        w_fill: float | None = Query(default=None),
        fill_min: float | None = Query(default=None),
        fill_max: float | None = Query(default=None),
        purity_dist_w: float | None = Query(default=None),
        rgb_weight_r: float | None = Query(default=None, ge=0.0),
        rgb_weight_g: float | None = Query(default=None, ge=0.0),
        rgb_weight_b: float | None = Query(default=None, ge=0.0),
        resume: int = Query(default=0, ge=0, le=1),
        seed: int | None = Query(default=None, ge=0),
    ) -> JSONResponse:
        """Start inverse optimization as a subprocess.

        Each seed runs in its own progress subdirectory (seed=0 or None → base dir).
        Multiple seeds can run in parallel without file conflicts.
        """
        effective_seed = seed if seed is not None else 0

        # Check if this specific seed is already running
        existing = rstate_dict.get(effective_seed)
        if existing is not None and existing.proc is not None and existing.proc.poll() is None:
            return JSONResponse(
                {"ok": False, "error": f"seed {effective_seed} already running (pid={existing.proc.pid})"},
                status_code=409,
            )

        # Create seed-specific progress directory
        seed_progress_dir = _init_seed_dir(progress_dir, effective_seed)

        # Clean up old topk/metrics files unless resuming
        if int(resume) != 1:
            try:
                for f in seed_progress_dir.glob("topk_*.npz"):
                    f.unlink()
                for f in seed_progress_dir.glob("topk_cur_*.npz"):
                    f.unlink()
                for f in seed_progress_dir.glob("metrics*.jsonl"):
                    f.unlink()
                for f in seed_progress_dir.glob("run_meta*.json"):
                    f.unlink()
            except Exception:
                pass  # Best effort cleanup
            # Also clear caches
            cache.key = None
            cache.npz = None
            cache.merged_key = None
            cache.merged_data = None
            scache.key = None
            scache.rgb = None

        # Always write a run-local config so dashboard inputs (n_start/n_steps/topk/ga knobs) apply.
        base_cfg_path = Path(__file__).resolve().parents[2] / "configs" / "ga.yaml"
        obj = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(obj, dict):
            obj = {}

        # Update GA/IO knobs from UI.
        ga = obj.get("ga") if isinstance(obj.get("ga"), dict) else {}
        ga["population"] = int(n_start)
        ga["generations"] = int(n_steps)
        ga["chunk_size"] = int(chunk_size)
        if ga_mutation_p is not None:
            ga["mutation_p"] = float(ga_mutation_p)
        if ga_mutation_sigma is not None:
            ga["mutation_sigma"] = float(ga_mutation_sigma)
        if ga_topk_clone_k is not None:
            ga["topk_clone_k"] = int(ga_topk_clone_k)
        if ga_topk_clone_m is not None:
            ga["topk_clone_m"] = int(ga_topk_clone_m)
        if ga_topk_clone_sigma_min is not None:
            ga["topk_clone_sigma_min"] = float(ga_topk_clone_sigma_min)
        if ga_topk_clone_sigma_max is not None:
            ga["topk_clone_sigma_max"] = float(ga_topk_clone_sigma_max)
        if optimizer_type is not None:
            ga["optimizer_type"] = str(optimizer_type)
        if cma_sigma0 is not None:
            ga["cma_sigma0"] = float(cma_sigma0)
        if seed is not None:
            ga["seed"] = int(seed)
        obj["ga"] = ga

        io_cfg = obj.get("io") if isinstance(obj.get("io"), dict) else {}
        io_cfg["topk"] = int(topk)
        obj["io"] = io_cfg

        # Optional generator backend override.
        if generator_backend is not None:
            gen_cfg = obj.get("generator") if isinstance(obj.get("generator"), dict) else {}
            gen_cfg["backend"] = str(generator_backend)
            obj["generator"] = gen_cfg

        # Optional loss overrides.
        loss = obj.get("loss") if isinstance(obj.get("loss"), dict) else {}
        if w_purity is not None:
            loss["w_purity"] = float(w_purity)
        if w_abs is not None:
            loss["w_abs"] = float(w_abs)
        if w_fill is not None:
            loss["w_fill"] = float(w_fill)
        if fill_min is not None:
            loss["fill_min"] = float(fill_min)
        if fill_max is not None:
            loss["fill_max"] = float(fill_max)
        if purity_dist_w is not None:
            loss["purity_dist_w"] = float(purity_dist_w)
        obj["loss"] = loss

        # Optional RGB weights override for purity matrix.
        if rgb_weight_r is not None or rgb_weight_g is not None or rgb_weight_b is not None:
            spectra = obj.get("spectra") if isinstance(obj.get("spectra"), dict) else {}
            rw = spectra.get("rgb_weights") if isinstance(spectra.get("rgb_weights"), dict) else {"R": 1.0, "G": 2.0, "B": 1.0}
            if rgb_weight_r is not None:
                rw["R"] = float(rgb_weight_r)
            if rgb_weight_g is not None:
                rw["G"] = float(rgb_weight_g)
            if rgb_weight_b is not None:
                rw["B"] = float(rgb_weight_b)
            spectra["rgb_weights"] = rw
            obj["spectra"] = spectra

        cfg_path = seed_progress_dir / "run_config.yaml"
        cfg_path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "gadash.run_ga",
            "--config",
            str(cfg_path),
            "--paths",
            str(Path(__file__).resolve().parents[2] / "configs" / "paths.yaml"),
            "--device",
            str(device),
            "--progress-dir",
            str(seed_progress_dir),
        ]
        if int(dry_run) == 1:
            cmd.append("--dry-run")
        if int(resume) == 1:
            cmd.append("--resume")
        # FDTD handling: multi-seed uses dashboard-level FDTD, single-seed uses subprocess
        is_multi_seed = len(active_seeds) > 0 or (effective_seed != 0)
        if is_multi_seed and int(fdtd_verify) == 1:
            # Multi-seed: disable per-subprocess FDTD, use dashboard-level instead
            cmd += ["--fdtd-verify", "off"]
            fdtd_yaml = str(Path(__file__).resolve().parents[2] / "configs" / "fdtd.yaml")
            paths_yaml = str(Path(__file__).resolve().parents[2] / "configs" / "paths.yaml")
            if not dash_fdtd.enabled:
                dash_fdtd.fdtd_every = int(fdtd_every) if int(fdtd_every) > 0 else 10
                dash_fdtd.fdtd_k = int(fdtd_k) if fdtd_k is not None else 1
                dash_fdtd.configure(fdtd_yaml, paths_yaml)
            print(f"[DASHBOARD-FDTD] Multi-seed: FDTD managed by dashboard (every={dash_fdtd.fdtd_every}, k={dash_fdtd.fdtd_k})", flush=True)
        else:
            # Single-seed: use subprocess FDTD as before
            cmd += [
                "--fdtd-verify",
                "on" if int(fdtd_verify) == 1 else "off",
                "--fdtd-config",
                str(Path(__file__).resolve().parents[2] / "configs" / "fdtd.yaml"),
            ]
            if int(fdtd_verify) == 1 and int(fdtd_every) > 0:
                cmd += ["--fdtd-every", str(int(fdtd_every))]
            if int(fdtd_verify) == 1 and fdtd_k is not None:
                cmd += ["--fdtd-k", str(int(fdtd_k))]

        # Initialize per-seed run state
        rs = RunProcState()
        rs.started_ts = datetime.now(timezone.utc).isoformat()
        rstate_dict[effective_seed] = rs
        if effective_seed not in active_seeds:
            active_seeds.append(effective_seed)

        p = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),  # repo root
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONWARNINGS": "ignore::SyntaxWarning"},
        )
        rs.proc = p
        t = threading.Thread(target=_reader_thread, args=(p, effective_seed), daemon=True)
        t.start()
        return JSONResponse({"ok": True, "pid": int(p.pid), "seed": effective_seed, "progress_dir": str(seed_progress_dir), "cmd": cmd})

    @app.get("/api/run/status")
    def run_status(seed: int | None = Query(default=None)) -> JSONResponse:
        """Return status of running processes.

        seed=None → return status of ALL seeds (multi-seed overview).
        seed=N → return status for a specific seed only.
        """
        if seed is not None:
            rs = rstate_dict.get(seed)
            if rs is None:
                return JSONResponse(_json_sanitize({
                    "running": False, "pid": None, "seed": seed,
                    "started_ts": None, "last_exit_code": None, "tail": [],
                }))
            p = rs.proc
            running = p is not None and p.poll() is None
            if p is not None and not running:
                rs.last_exit_code = p.poll()
            return JSONResponse(_json_sanitize({
                "running": bool(running),
                "pid": int(p.pid) if p is not None else None,
                "seed": seed,
                "started_ts": rs.started_ts,
                "last_exit_code": rs.last_exit_code,
                "tail": list(rs.lines)[-50:],
            }))

        # Multi-seed overview
        seeds_status: dict[str, Any] = {}
        num_running = 0
        combined_tail: list[str] = []
        for s in sorted(active_seeds):
            rs = rstate_dict.get(s)
            if rs is None:
                continue
            p = rs.proc
            running = p is not None and p.poll() is None
            if p is not None and not running:
                rs.last_exit_code = p.poll()
            if running:
                num_running += 1
            seeds_status[str(s)] = {
                "running": bool(running),
                "pid": int(p.pid) if p is not None else None,
                "started_ts": rs.started_ts,
                "last_exit_code": rs.last_exit_code,
            }
            # Collect last few lines from each seed for combined tail
            recent = list(rs.lines)[-10:]
            for line in recent:
                combined_tail.append(f"[seed={s}] {line}")

        # Backward compat: also include top-level running/tail fields
        any_running = num_running > 0
        return JSONResponse(_json_sanitize({
            "running": any_running,
            "num_running": num_running,
            "active_seeds": sorted(active_seeds),
            "seeds_status": seeds_status,
            "tail": combined_tail[-50:],
            # Legacy fields for backward compat
            "pid": None,
            "started_ts": None,
            "last_exit_code": None,
        }))

    @app.post("/api/run/stop")
    def run_stop(seed: int | None = Query(default=None)) -> JSONResponse:
        """Stop running process(es).

        seed=None → stop ALL running seeds.
        seed=N → stop only seed N.
        """
        if seed is not None:
            rs = rstate_dict.get(seed)
            if rs is None or rs.proc is None or rs.proc.poll() is not None:
                return JSONResponse({"ok": True, "stopped": False, "error": f"seed {seed}: no running process"})
            try:
                rs.lines.append(f"[dashboard] stopping seed {seed}...")
                rs.proc.terminate()
                try:
                    rs.proc.wait(timeout=3.0)
                except Exception:
                    pass
                if rs.proc.poll() is None:
                    rs.lines.append("[dashboard] terminate timed out; killing...")
                    rs.proc.kill()
                code = rs.proc.poll()
                rs.last_exit_code = code
                rs.lines.append(f"[dashboard] stopped (exit={code})")
                return JSONResponse({"ok": True, "stopped": True, "seed": seed, "exit_code": code})
            except Exception as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

        # Stop ALL seeds
        stopped_seeds = []
        for s in list(active_seeds):
            rs = rstate_dict.get(s)
            if rs is None or rs.proc is None or rs.proc.poll() is not None:
                continue
            try:
                rs.lines.append(f"[dashboard] stopping seed {s}...")
                rs.proc.terminate()
                try:
                    rs.proc.wait(timeout=3.0)
                except Exception:
                    pass
                if rs.proc.poll() is None:
                    rs.lines.append("[dashboard] terminate timed out; killing...")
                    rs.proc.kill()
                code = rs.proc.poll()
                rs.last_exit_code = code
                rs.lines.append(f"[dashboard] stopped (exit={code})")
                stopped_seeds.append(s)
            except Exception:
                pass

        if not stopped_seeds:
            return JSONResponse({"ok": True, "stopped": False, "error": "no running processes"})
        return JSONResponse({"ok": True, "stopped": True, "stopped_seeds": stopped_seeds})

    @app.post("/api/run/reset")
    def run_reset(seed: int | None = Query(default=None)) -> JSONResponse:
        """Archive progress_dir contents and reset.

        seed=None → reset ALL (base + seed_*/ dirs). Refuses if any running.
        seed=N → reset only seed N's directory.
        """
        # Check for running processes
        if seed is not None:
            rs = rstate_dict.get(seed)
            if rs is not None and rs.proc is not None and rs.proc.poll() is None:
                return JSONResponse({"ok": False, "error": f"seed {seed} still running; stop first"}, status_code=409)
        else:
            for s in active_seeds:
                rs = rstate_dict.get(s)
                if rs is not None and rs.proc is not None and rs.proc.poll() is None:
                    return JSONResponse({"ok": False, "error": f"seed {s} still running; stop all first"}, status_code=409)

        try:
            if not progress_dir.exists():
                progress_dir.mkdir(parents=True, exist_ok=True)
            archive_root = progress_dir.parent / "progress_archive"
            archive_root.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            if seed is not None:
                # Reset only a specific seed's directory
                seed_dir = _get_progress_dir_for_seed(progress_dir, seed)
                if not seed_dir.exists():
                    return JSONResponse({"ok": True, "moved": [], "info": f"seed {seed} dir does not exist"})
                dst = archive_root / f"seed_{seed}_{ts}"
                dst.mkdir(parents=True, exist_ok=True)
                moved = []
                for item in list(seed_dir.iterdir()):
                    target = dst / item.name
                    try:
                        shutil.move(str(item), str(target))
                        moved.append(item.name)
                    except Exception:
                        pass
                # Clean up rstate
                rs = rstate_dict.get(seed)
                if rs is not None:
                    rs.lines.clear()
                    rs.started_ts = None
                    rs.last_exit_code = None
                return JSONResponse({"ok": True, "archived_to": str(dst), "moved": moved, "seed": seed})

            # Reset ALL
            dst = archive_root / f"{progress_dir.name}_{ts}"
            dst.mkdir(parents=True, exist_ok=True)
            moved = []
            for item in list(progress_dir.iterdir()):
                target = dst / item.name
                try:
                    shutil.move(str(item), str(target))
                    moved.append(item.name)
                except Exception:
                    pass

            # Recreate expected subdir
            (progress_dir / "previews").mkdir(parents=True, exist_ok=True)
            # Clear all rstate entries
            for s in list(rstate_dict.keys()):
                rs = rstate_dict[s]
                rs.lines.clear()
                rs.started_ts = None
                rs.last_exit_code = None
            active_seeds.clear()
            rstate_dict.clear()
            return JSONResponse({"ok": True, "archived_to": str(dst), "moved": moved})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.get("/api/status")
    def status(window: str = Query(default="all")) -> JSONResponse:
        """Convenience endpoint for the Chart.js UI.

        window: all|50|200|1000
        Multi-seed: automatically merges metrics from all seeds.
        """
        # Check if multiple seeds exist
        seed_dirs = _discover_seed_dirs()
        has_multiple = len(seed_dirs) > 1

        if window == "all":
            tail = 20000
        else:
            try:
                tail = int(window)
            except Exception:
                tail = 200
        tail = max(1, min(20000, tail))

        # For multi-seed: merge all metrics, use max nsteps across seeds
        if has_multiple:
            items = _merge_metrics_all_seeds(tail)
            # Get max nsteps across all seed metas
            nsteps = 0
            for seed, pdir in seed_dirs:
                m = _read_meta(pdir)
                try:
                    ns = int(m.get("n_steps", 0) or 0)
                    nsteps = max(nsteps, ns)
                except Exception:
                    pass
            meta = _read_meta(progress_dir) if progress_dir.exists() else {}
            meta["n_steps"] = nsteps  # Override with max
            ts0 = None  # Don't filter by timestamp for merged data
        else:
            # Single seed: use base progress_dir (seed=0)
            meta = _read_meta(progress_dir)
            nsteps = int(meta.get("n_steps", 0) or 0)
            ts0 = _ts_start_epoch(meta)
            items = _read_metrics_from(progress_dir, tail)

        # Filter to this run only (if single seed with ts0).
        if ts0 is not None:
            filt = []
            for it in items:
                try:
                    t = datetime.fromisoformat(str(it.get("ts", "")).replace("Z", "+00:00")).timestamp()
                    if t < ts0:
                        continue
                except Exception:
                    continue
                filt.append(it)
            items = filt

        # Clamp steps.
        if nsteps > 0:
            tmp = []
            for it in items:
                try:
                    s = int(it.get("step"))
                except Exception:
                    continue
                if s <= nsteps - 1:
                    tmp.append(it)
            items = tmp

        # Build series preserving all seeds (for multi-seed visualization).
        # For multi-seed: keep all seed curves separate.
        # For single-seed: same as before.
        by_seed_step: dict[tuple[int, int], dict[str, Any]] = {}  # (seed, step) -> item
        all_seeds = set()

        for it in items:
            try:
                s = int(it.get("step"))
                seed = int(it.get("seed", 0))  # Default seed=0 if not specified
                all_seeds.add(seed)
                by_seed_step[(seed, s)] = it
            except Exception:
                continue

        # Get all unique steps
        steps_set = {step for seed, step in by_seed_step.keys()}
        steps_sorted = sorted(steps_set)

        def _f(v) -> float:
            try:
                x = float(v)
                return x if math.isfinite(x) else float("nan")
            except Exception:
                return float("nan")

        # For multi-seed: create separate series per seed
        if has_multiple and all_seeds:
            seed_loss: dict[str, dict[str, list]] = {}
            for seed in sorted(all_seeds):
                seed_str = str(seed)
                seed_loss[seed_str] = {
                    "loss_total": [],
                    "loss_spec": [],
                    "loss_reg": [],
                    "loss_purity": [],
                    "loss_fill": []
                }
                for step in steps_sorted:
                    item = by_seed_step.get((seed, step), {})
                    seed_loss[seed_str]["loss_total"].append(_f(item.get("loss_total")))
                    seed_loss[seed_str]["loss_spec"].append(_f(item.get("loss_spec")))
                    seed_loss[seed_str]["loss_reg"].append(_f(item.get("loss_reg")))
                    seed_loss[seed_str]["loss_purity"].append(_f(item.get("loss_purity")))
                    seed_loss[seed_str]["loss_fill"].append(_f(item.get("loss_fill")))

            # For latest, use the most recent step's best loss across all seeds
            latest = {}
            if steps_sorted:
                best_loss = float("inf")
                for seed in all_seeds:
                    item = by_seed_step.get((seed, steps_sorted[-1]))
                    if item:
                        loss = float(item.get("loss_total", float("inf")))
                        if loss < best_loss:
                            best_loss = loss
                            latest = item
        else:
            # Single seed: use old format (backward compat)
            by_step = {}
            for (seed, step), item in by_seed_step.items():
                if step not in by_step:
                    by_step[step] = item

            loss_total = [_f(by_step[s].get("loss_total")) for s in steps_sorted]
            loss_spec = [_f(by_step[s].get("loss_spec")) for s in steps_sorted]
            loss_reg = [_f(by_step[s].get("loss_reg")) for s in steps_sorted]
            loss_purity = [_f(by_step[s].get("loss_purity")) for s in steps_sorted]
            loss_fill = [_f(by_step[s].get("loss_fill")) for s in steps_sorted]
            seed_loss = None
            latest = by_step[steps_sorted[-1]] if steps_sorted else {}

        # Build response
        series_dict = {
            "steps": steps_sorted,
        }

        if has_multiple and seed_loss:
            # Multi-seed: include per-seed loss data
            # Format: {"seed_0": {"loss_total": [...], ...}, "seed_1000": {...}, ...}
            for seed_str, losses in seed_loss.items():
                for loss_key, values in losses.items():
                    if f'seed_{loss_key}' not in series_dict:
                        series_dict[f'seed_{loss_key}'] = {}
                    series_dict[f'seed_{loss_key}'][seed_str] = values

            # Calculate average loss across all seeds
            avg_loss: dict[str, list] = {}
            loss_keys = ["loss_total", "loss_spec", "loss_reg", "loss_purity", "loss_fill"]
            for loss_key in loss_keys:
                avg_loss[loss_key] = []
                for step_idx in range(len(steps_sorted)):
                    values_at_step = []
                    for seed_str in seed_loss:
                        val = seed_loss[seed_str][loss_key][step_idx]
                        if math.isfinite(val):
                            values_at_step.append(val)
                    if values_at_step:
                        avg_loss[loss_key].append(sum(values_at_step) / len(values_at_step))
                    else:
                        avg_loss[loss_key].append(float("nan"))

            # Add average data to series_dict
            for loss_key, avg_values in avg_loss.items():
                if f'seed_{loss_key}' not in series_dict:
                    series_dict[f'seed_{loss_key}'] = {}
                series_dict[f'seed_{loss_key}']['avg'] = avg_values
        else:
            # Single-seed: use old format for backward compatibility
            series_dict["loss_total"] = loss_total
            series_dict["loss_spec"] = loss_spec
            series_dict["loss_reg"] = loss_reg
            series_dict["loss_purity"] = loss_purity
            series_dict["loss_fill"] = loss_fill

        return JSONResponse(
            _json_sanitize(
                {
                    "meta": meta,
                    "series": series_dict,
                    "latest": latest,
                    "merged": has_multiple,
                }
            )
        )

    return app
