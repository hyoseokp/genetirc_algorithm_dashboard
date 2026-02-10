from __future__ import annotations

import io
import json
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from .config import load_config
from .losses import loss_definition_latex


@dataclass
class Tail:
    path: Path
    max_lines: int = 2000
    _lines: deque[str] = field(default_factory=deque)
    _pos: int = 0

    def read_new(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self._pos)
            data = f.read()
            self._pos = f.tell()
        if not data:
            return
        for line in data.splitlines():
            self._lines.append(line)
            if len(self._lines) > self.max_lines:
                self._lines.popleft()

    def lines(self) -> list[str]:
        self.read_new()
        return list(self._lines)


class RunManager:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.proc: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()

    def start(self, args: list[str]) -> None:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                raise RuntimeError("run already active")
            self.proc = subprocess.Popen(
                args,
                cwd=str(self.workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

    def stop(self) -> None:
        with self.lock:
            if not self.proc or self.proc.poll() is not None:
                return
            self.proc.terminate()

    def status(self) -> dict[str, Any]:
        with self.lock:
            if not self.proc:
                return {"running": False, "returncode": None}
            rc = self.proc.poll()
            return {"running": rc is None, "returncode": rc}


def _latest_npz(progress_dir: Path, mode: str) -> Path | None:
    pat = "topk_step-" if mode == "best" else "topk_cur_step-"
    files = sorted(progress_dir.glob(f"{pat}*.npz"))
    if not files:
        return None
    return files[-1]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _struct_png(struct01: np.ndarray, invert: bool) -> bytes:
    # struct01: (H,W) float 0..1
    img = struct01
    if invert:
        img = 1.0 - img
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    # simple PNG via pillow
    from PIL import Image

    im = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def create_app(progress_dir: str, cfg_path: str, paths_path: str) -> FastAPI:
    progress_dir_p = Path(progress_dir)
    progress_dir_p.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path, paths_path)

    app = FastAPI()
    runner = RunManager(workdir=Path.cwd())
    tail = Tail(progress_dir_p / "metrics.jsonl")

    static_dir = Path(__file__).parent / "dashboard_static"
    index_html = (static_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/")
    def root() -> HTMLResponse:
        return HTMLResponse(index_html)

    @app.get("/api/ping")
    def ping() -> dict[str, Any]:
        return {"ok": True, "ts": time.time()}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        st = runner.status()
        latest_best = _latest_npz(progress_dir_p, "best")
        latest_cur = _latest_npz(progress_dir_p, "cur")
        return {
            **st,
            "progress_dir": str(progress_dir_p),
            "latest_best": latest_best.name if latest_best else None,
            "latest_cur": latest_cur.name if latest_cur else None,
        }

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        return {
            "ga": {
                "generations": int(cfg.ga.generations),
                "population": int(cfg.ga.population),
                "elite": int(cfg.ga.elite),
            },
            "io": {"topk": int(cfg.io.topk), "print_every": int(cfg.io.print_every)},
        }

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        lines = tail.lines()
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return {"lines": out[-1500:]}

    @app.get("/api/topk/latest")
    def topk_latest(mode: str = "best") -> dict[str, Any]:
        mode2 = "best" if mode == "best" else "cur"
        p = _latest_npz(progress_dir_p, mode2)
        if not p:
            return {"ok": False, "detail": "no topk yet"}
        d = _load_npz(p)
        return {
            "ok": True,
            "file": p.name,
            "k": int(d["struct01"].shape[0]),
            "gen": int(p.stem.split("-")[-1]),
        }

    @app.get("/api/topk/image")
    def topk_image(mode: str = "best", k: int = 0, invert: int = 1) -> Response:
        mode2 = "best" if mode == "best" else "cur"
        p = _latest_npz(progress_dir_p, mode2)
        if not p:
            raise HTTPException(404, "no topk")
        d = _load_npz(p)
        struct = d["struct01"]
        if k < 0 or k >= struct.shape[0]:
            raise HTTPException(400, "k out of range")
        png = _struct_png(struct[k, 0], invert=bool(invert))
        return Response(content=png, media_type="image/png")

    @app.get("/api/topk/spectrum")
    def topk_spectrum(mode: str = "best", k: int = 0) -> dict[str, Any]:
        mode2 = "best" if mode == "best" else "cur"
        p = _latest_npz(progress_dir_p, mode2)
        if not p:
            raise HTTPException(404, "no topk")
        d = _load_npz(p)
        pred = d["pred"]
        if k < 0 or k >= pred.shape[0]:
            raise HTTPException(400, "k out of range")
        return {"pred_rgbc": pred[k].tolist()}

    @app.get("/api/loss_definition")
    def api_loss_definition() -> dict[str, Any]:
        return {"latex": loss_definition_latex(cfg.spectra, cfg.loss)}

    @app.post("/api/run/start")
    def run_start(device: str = "cuda") -> dict[str, Any]:
        if not Path(paths_path).exists():
            raise HTTPException(400, f"missing paths config: {paths_path}")
        # Spawn: python -m scripts.run_ga ...
        args = [
            os.environ.get("PYTHON", "python"),
            "-m",
            "scripts.run_ga",
            "--config",
            cfg_path,
            "--paths",
            paths_path,
            "--progress-dir",
            str(progress_dir_p),
            "--device",
            device,
        ]
        runner.start(args)
        return {"ok": True, "args": args}

    @app.post("/api/run/stop")
    def run_stop() -> dict[str, Any]:
        runner.stop()
        return {"ok": True}

    @app.post("/api/run/reset")
    def run_reset() -> dict[str, Any]:
        # delete progress artifacts
        for p in progress_dir_p.glob("topk*.npz"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            (progress_dir_p / "metrics.jsonl").unlink()
        except Exception:
            pass
        try:
            (progress_dir_p / "run_meta.json").unlink()
        except Exception:
            pass
        return {"ok": True}

    return app
