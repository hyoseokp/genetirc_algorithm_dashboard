from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import time


@dataclass
class MetricsLine:
    ts: str
    gen: int
    loss_total: float
    loss_spec: float
    loss_reg: float
    fill: float


class ProgressLogger:
    def __init__(self, progress_dir: str | Path):
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.progress_dir / "metrics.jsonl"

    def write_meta(self, meta: dict[str, Any]) -> None:
        p = self.progress_dir / "run_meta.json"
        p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_metrics(self, d: dict[str, Any]) -> None:
        # single-line JSONL, flush by write_text append
        line = json.dumps(d, ensure_ascii=False)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def now_iso() -> str:
        # UTC-ish monotonic timestamp for UI
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
