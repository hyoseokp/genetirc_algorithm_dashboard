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
    p = progress_dir / "run_meta.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
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


@dataclass
class SpectrumCache:
    key: tuple[int, int] | None = None
    rgb: np.ndarray | None = None


@dataclass
class RunProcState:
    proc: subprocess.Popen | None = None
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=400))
    started_ts: str | None = None
    last_exit_code: int | None = None


def create_app(*, progress_dir: Path, surrogate=None) -> FastAPI:
    progress_dir = Path(progress_dir)
    cache = TopKCache()
    scache = SpectrumCache()
    rstate = RunProcState()

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
        return JSONResponse(
            _json_sanitize(
                {
                    "python": sys.executable,
                    "cwd": os.getcwd(),
                    "dashboard_app": str(Path(__file__).resolve()),
                    "gadash": gadash_file,
                    "repo_root_guess": str(Path(__file__).resolve().parents[2]),
                    "progress_dir": str(progress_dir),
                }
            )
        )

    @app.get("/api/meta")
    def meta() -> JSONResponse:
        return JSONResponse(_json_sanitize(_read_meta(progress_dir)))

    @app.get("/api/metrics")
    def metrics(tail: int = Query(default=2000, ge=1, le=20000)) -> JSONResponse:
        items = _tail_jsonl(progress_dir / "metrics.jsonl", int(tail))
        return JSONResponse({"items": _json_sanitize(items)})

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

    def _latest_topk_step(*, mode: str) -> int | None:
        if not progress_dir.exists():
            return None
        mode = str(mode or "best").lower()
        rx = _TOPK_RE_BEST if mode != "cur" else _TOPK_RE_CUR
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
        for p in progress_dir.iterdir():
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

    def _load_topk(step: int, *, mode: str) -> dict[str, np.ndarray]:
        mode = str(mode or "best").lower()
        key = (("cur" if mode == "cur" else "best"), int(step))
        if cache.key == key and cache.npz is not None:
            return cache.npz
        prefix = "topk_cur" if mode == "cur" else "topk"
        p = progress_dir / f"{prefix}_step-{int(step)}.npz"
        z = np.load(p, allow_pickle=False)
        data = {k: z[k] for k in z.files}
        cache.key = key
        cache.npz = data
        return data

    @app.get("/api/topk/latest")
    def topk_latest(mode: str = Query(default="best")) -> JSONResponse:
        step = _latest_topk_step(mode=mode)
        if step is None:
            return JSONResponse({"step": None, "k": 0, "metrics": {}})
        try:
            data = _load_topk(step, mode=mode)
        except Exception as e:
            return JSONResponse({"step": int(step), "k": 0, "error": f"failed to load npz: {e}"}, status_code=500)
        struct = data.get("struct128_topk")
        k = int(struct.shape[0]) if isinstance(struct, np.ndarray) and struct.ndim == 3 else 0
        metrics: dict[str, Any] = {}
        for key, arr in data.items():
            if key.startswith("metric_"):
                metrics[key] = arr.tolist()
            if key == "metric_best_loss":
                metrics[key] = arr.tolist()
        fill = []
        if isinstance(struct, np.ndarray) and struct.ndim == 3:
            fill = [float(struct[i].mean()) for i in range(struct.shape[0])]
        return JSONResponse(
            _json_sanitize(
                {
                    "step": int(step),
                    "k": k,
                    "mode": ("cur" if str(mode).lower() == "cur" else "best"),
                    "images": [f"/api/topk/{int(step)}/{i}.png?mode={('cur' if str(mode).lower() == 'cur' else 'best')}" for i in range(k)],
                    "metrics": metrics,
                    "fill_frac": fill,
                }
            )
        )

    @app.get("/api/topk/{step}/{idx}.png")
    def topk_png(
        step: int,
        idx: int,
        invert: int = Query(default=1, ge=0, le=1),
        mode: str = Query(default="best"),
    ) -> Response:
        try:
            data = _load_topk(int(step), mode=mode)
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
    def topk_spectrum(step: int, idx: int, mode: str = Query(default="best")) -> JSONResponse:
        if surrogate is None:
            return JSONResponse({"error": "surrogate not configured"}, status_code=400)
        key = (int(step), int(idx))
        if scache.key == key and scache.rgb is not None:
            rgb = scache.rgb
        else:
            data = _load_topk(int(step), mode=mode)
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
        return JSONResponse(_json_sanitize({"step": int(step), "idx": int(idx), "n_channels": C, "rgb": rgb.tolist()}))

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
    def topk_fdtd_spectrum(step: int, idx: int) -> JSONResponse:
        """Return FDTD-verified RGB spectrum for the given topk step/index, if available."""
        p = progress_dir / f"fdtd_rggb_step-{int(step)}.npy"
        if not p.exists():
            return JSONResponse({"error": "fdtd spectrum not available"}, status_code=404)
        arr = np.load(p)
        if arr.ndim != 4 or arr.shape[1:3] != (2, 2):
            return JSONResponse({"error": f"bad fdtd_rggb shape: {arr.shape}"}, status_code=500)
        if idx < 0 or idx >= arr.shape[0]:
            return JSONResponse({"error": "idx out of range"}, status_code=404)
        t = torch.from_numpy(arr[idx : idx + 1].astype(np.float32))
        rgb = _merge_rggb_to_rgb(t)[0].detach().cpu().numpy().astype(np.float32)  # (3,C)
        return JSONResponse(_json_sanitize({"step": int(step), "idx": int(idx), "n_channels": int(rgb.shape[-1]), "rgb": rgb.tolist()}))

    def _reader_thread(p: subprocess.Popen) -> None:
        try:
            assert p.stdout is not None
            for line in p.stdout:
                s = line.rstrip("\n")
                if s:
                    rstate.lines.append(s)
        except Exception:
            pass

    @app.post("/api/run/start")
    def run_start(
        n_start: int = Query(default=200, ge=1),
        n_steps: int = Query(default=2000, ge=1),
        topk: int = Query(default=50, ge=1),
        dry_run: int = Query(default=0, ge=0, le=1),
        device: str = Query(default="cpu"),
        chunk_size: int = Query(default=64, ge=1),
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
        w_purity: float | None = Query(default=None),
        w_abs: float | None = Query(default=None),
        w_fill: float | None = Query(default=None),
        fill_min: float | None = Query(default=None),
        fill_max: float | None = Query(default=None),
    ) -> JSONResponse:
        """Start inverse optimization as a subprocess."""
        if rstate.proc is not None and rstate.proc.poll() is None:
            return JSONResponse({"ok": False, "error": "run already in progress"}, status_code=409)

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
        obj["loss"] = loss

        cfg_path = Path(progress_dir) / "run_config.yaml"
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
            str(progress_dir),
        ]
        if int(dry_run) == 1:
            cmd.append("--dry-run")
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
        rstate.lines.clear()
        rstate.started_ts = datetime.now(timezone.utc).isoformat()
        rstate.last_exit_code = None

        p = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),  # repo root
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        rstate.proc = p
        t = threading.Thread(target=_reader_thread, args=(p,), daemon=True)
        t.start()
        return JSONResponse({"ok": True, "pid": int(p.pid), "cmd": cmd})

    @app.get("/api/run/status")
    def run_status() -> JSONResponse:
        p = rstate.proc
        running = p is not None and p.poll() is None
        code = None
        if p is not None and not running:
            code = p.poll()
            rstate.last_exit_code = code
        return JSONResponse(
            _json_sanitize(
                {
                    "running": bool(running),
                    "pid": int(p.pid) if p is not None else None,
                    "started_ts": rstate.started_ts,
                    "last_exit_code": rstate.last_exit_code,
                    "tail": list(rstate.lines)[-50:],
                }
            )
        )

    @app.post("/api/run/stop")
    def run_stop() -> JSONResponse:
        p = rstate.proc
        if p is None or p.poll() is not None:
            return JSONResponse({"ok": True, "stopped": False, "error": "no running process"})
        try:
            rstate.lines.append("[dashboard] stopping run...")
            p.terminate()
            try:
                p.wait(timeout=3.0)
            except Exception:
                pass
            if p.poll() is None:
                rstate.lines.append("[dashboard] terminate timed out; killing...")
                p.kill()
            code = p.poll()
            rstate.last_exit_code = code
            rstate.lines.append(f"[dashboard] stopped (exit={code})")
            return JSONResponse({"ok": True, "stopped": True, "exit_code": code})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.post("/api/run/reset")
    def run_reset() -> JSONResponse:
        """Archive current progress_dir contents and reset it to empty."""
        # Refuse to reset while a run is active.
        p = rstate.proc
        if p is not None and p.poll() is None:
            return JSONResponse({"ok": False, "error": "run in progress; stop first"}, status_code=409)

        try:
            if not progress_dir.exists():
                progress_dir.mkdir(parents=True, exist_ok=True)
            archive_root = progress_dir.parent / "progress_archive"
            archive_root.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            dst = archive_root / f"{progress_dir.name}_{ts}"
            dst.mkdir(parents=True, exist_ok=True)

            moved = []
            for item in list(progress_dir.iterdir()):
                # Keep the directory itself; move everything inside.
                target = dst / item.name
                try:
                    shutil.move(str(item), str(target))
                    moved.append(item.name)
                except Exception:
                    # Best-effort; ignore file locking issues.
                    pass

            # Recreate expected subdir.
            (progress_dir / "previews").mkdir(parents=True, exist_ok=True)
            rstate.lines.clear()
            rstate.started_ts = None
            rstate.last_exit_code = None
            return JSONResponse({"ok": True, "archived_to": str(dst), "moved": moved})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.get("/api/status")
    def status(window: str = Query(default="all")) -> JSONResponse:
        """Convenience endpoint for the Chart.js UI.

        window: all|50|200|1000
        """
        meta = _read_meta(progress_dir)
        nsteps = int(meta.get("n_steps", 0) or 0)
        ts0 = _ts_start_epoch(meta)
        if window == "all":
            tail = 20000
        else:
            try:
                tail = int(window)
            except Exception:
                tail = 200
        tail = max(1, min(20000, tail))
        items = _tail_jsonl(progress_dir / "metrics.jsonl", tail)
        # Filter to this run only.
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

        # Build series (use last value per step).
        by_step: dict[int, dict[str, Any]] = {}
        for it in items:
            try:
                s = int(it.get("step"))
            except Exception:
                continue
            by_step[s] = it
        steps_sorted = sorted(by_step.keys())
        def _f(v) -> float:
            try:
                x = float(v)
                return x if math.isfinite(x) else float("nan")
            except Exception:
                return float("nan")

        loss_total = [_f(by_step[s].get("loss_total")) for s in steps_sorted]
        loss_spec = [_f(by_step[s].get("loss_spec")) for s in steps_sorted]
        loss_reg = [_f(by_step[s].get("loss_reg")) for s in steps_sorted]
        loss_purity = [_f(by_step[s].get("loss_purity")) for s in steps_sorted]
        loss_fill = [_f(by_step[s].get("loss_fill")) for s in steps_sorted]
        latest = by_step[steps_sorted[-1]] if steps_sorted else {}

        return JSONResponse(
            _json_sanitize(
                {
                    "meta": meta,
                    "series": {
                        "steps": steps_sorted,
                    "loss_total": loss_total,
                    "loss_spec": loss_spec,
                    "loss_reg": loss_reg,
                    "loss_purity": loss_purity,
                    "loss_fill": loss_fill,
                },
                "latest": latest,
            }
        )
        )

    return app
