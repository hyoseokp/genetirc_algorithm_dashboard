"""Optimization logger for saving run summaries and results to organized log directories."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class OptimizationLogManager:
    """Manages automatic logging of optimization runs to D://optimization_log/."""

    def __init__(
        self,
        base_log_dir: Path | str,
        optimizer_type: str,
        seed: int | None = None,
        start_time: float | None = None,
    ):
        """Initialize the optimization logger.

        Args:
            base_log_dir: Base directory for logs (e.g., D://optimization_log)
            optimizer_type: Type of optimizer (GA, CMA-ES, etc.)
            seed: Random seed (for multi-seed runs)
            start_time: Start timestamp (from time.time())
        """
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp-based folder name
        if start_time is None:
            import time
            start_time = time.time()
        self.start_time = start_time

        # Format: YYYYMMDD_HHMMSS_GA_seed0
        dt = datetime.fromtimestamp(start_time)
        timestamp = dt.strftime("%Y%m%d_%H%M%S")
        opt_name = str(optimizer_type).upper()
        seed_label = f"seed{seed}" if seed and seed > 0 else "seed0"
        folder_name = f"{timestamp}_{opt_name}_{seed_label}"

        self.log_dir = self.base_log_dir / folder_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer_type = optimizer_type
        self.seed = seed or 0

        print(f"[LOG] OptimizationLogManager initialized: {self.log_dir}", flush=True)

    def finalize(
        self,
        progress_dir: Path | str,
        engine: str,
        success: bool,
        error_msg: str | None = None,
    ) -> None:
        """Finalize logging by copying and processing results from progress_dir.

        Args:
            progress_dir: The progress directory (data/progress/ or data/progress/seed_N/)
            engine: Engine type (ga or cmaes)
            success: Whether optimization completed successfully
            error_msg: Error message if unsuccessful
        """
        progress_dir = Path(progress_dir)

        print(f"[LOG] Starting finalization: progress_dir={progress_dir}, engine={engine}, success={success}", flush=True)

        try:
            # 1. Copy run config
            config_path = progress_dir / "run_config.yaml"
            if config_path.exists():
                shutil.copy2(config_path, self.log_dir / "run_config.yaml")
                print(f"[LOG] Copied run config", flush=True)
            else:
                print(f"[WARN] run_config.yaml not found at {config_path}", flush=True)

            # 2. Process metrics: metrics_{engine}.jsonl → loss_curve.npy
            self._process_metrics(progress_dir, engine)

            # 3. Extract best structure and losses
            self._extract_best_results(progress_dir, engine)

            # 4. Copy FDTD results if available
            self._copy_fdtd_results(progress_dir)

            # 5. Read meta and write summary
            self._write_summary(progress_dir, engine, success, error_msg)

            print(f"[LOG] Finalization complete: {self.log_dir}", flush=True)

        except Exception as e:
            import traceback
            print(f"[ERROR] Finalization error: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

    def _process_metrics(self, progress_dir: Path, engine: str) -> None:
        """Convert metrics JSONL to loss_curve NPY."""
        metrics_path = progress_dir / f"metrics_{engine}.jsonl"
        if not metrics_path.exists():
            print(f"[WARN] Metrics file not found: {metrics_path}", flush=True)
            return

        # Read all metrics lines
        metrics_list: list[dict[str, Any]] = []
        with metrics_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        metrics_list.append(obj)
                except Exception:
                    continue

        if not metrics_list:
            print(f"[WARN] No metrics found in {metrics_path}", flush=True)
            return

        # Extract columns
        steps = []
        loss_totals = []
        loss_specs = []
        loss_regs = []
        loss_purities = []
        loss_fills = []
        fills = []

        for m in metrics_list:
            steps.append(m.get("step", 0))
            loss_totals.append(m.get("loss_total", 0.0))
            loss_specs.append(m.get("loss_spec", 0.0))
            loss_regs.append(m.get("loss_reg", 0.0))
            loss_purities.append(m.get("loss_purity", 0.0))
            loss_fills.append(m.get("loss_fill", 0.0))
            fills.append(m.get("fill", 0.0))

        # Stack into single array: (N, 7)
        loss_curve = np.column_stack(
            [steps, loss_totals, loss_specs, loss_regs, loss_purities, loss_fills, fills]
        ).astype(np.float32)

        # Save
        output_path = self.log_dir / "loss_curve.npy"
        np.save(output_path, loss_curve)
        print(f"[LOG] Loss curve saved: {output_path} (shape={loss_curve.shape})", flush=True)

    def _extract_best_results(self, progress_dir: Path, engine: str) -> None:
        """Extract best structure and losses from latest topk_step-*.npz."""
        progress_dir = Path(progress_dir)

        # Find latest topk_step-*.npz
        topk_files = sorted(progress_dir.glob("topk_step-*.npz"))
        if not topk_files:
            print(f"[WARN] No topk_step-*.npz found in {progress_dir}", flush=True)
            return

        latest_topk = topk_files[-1]

        try:
            data = np.load(latest_topk, allow_pickle=False)

            # Extract arrays
            struct128 = data.get("struct128_topk")
            seed16 = data.get("seed16_topk")
            best_loss = data.get("metric_best_loss")

            if struct128 is not None:
                output_struct = self.log_dir / "best_structure.npy"
                np.save(output_struct, struct128.astype(np.uint8))
                print(f"[LOG] Best structure saved: {output_struct} (shape={struct128.shape})", flush=True)

            if seed16 is not None:
                output_seed = self.log_dir / "best_structure_seeds.npy"
                np.save(output_seed, seed16.astype(np.float32))
                print(f"[LOG] Best seeds saved: {output_seed} (shape={seed16.shape})", flush=True)

            if best_loss is not None:
                output_loss = self.log_dir / "best_loss_values.npy"
                np.save(output_loss, best_loss.astype(np.float32))
                print(f"[LOG] Best losses saved: {output_loss} (shape={best_loss.shape})", flush=True)

        except Exception as e:
            print(f"[WARN] Error extracting best results: {e}", flush=True)

    def _copy_fdtd_results(self, progress_dir: Path) -> None:
        """Copy FDTD spectrum results if available."""
        fdtd_files = sorted(progress_dir.glob("fdtd_rggb_step-*.npy"))
        if not fdtd_files:
            return

        latest_fdtd = fdtd_files[-1]

        try:
            data = np.load(latest_fdtd, allow_pickle=False)
            output_path = self.log_dir / "fdtd_spectrum.npy"
            np.save(output_path, data.astype(np.float32))
            print(f"[LOG] FDTD spectrum saved: {output_path} (shape={data.shape})", flush=True)

        except Exception as e:
            print(f"[WARN] Error copying FDTD results: {e}", flush=True)

    def _write_summary(
        self,
        progress_dir: Path,
        engine: str,
        success: bool,
        error_msg: str | None = None,
    ) -> None:
        """Write summary.md with metadata and results."""
        import time

        end_time = time.time()
        elapsed_seconds = end_time - self.start_time
        elapsed_str = self._format_duration(elapsed_seconds)

        # Read run_meta
        meta_path = progress_dir / f"run_meta_{engine}.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Extract key values
        n_start = meta.get("n_start", "?")
        n_steps = meta.get("n_steps", "?")
        topk = meta.get("topk", "?")
        device = meta.get("device", "?")
        ts_start = meta.get("ts_start", "?")

        # Read final best loss from saved loss_curve
        best_loss_value = None
        loss_curve_path = self.log_dir / "loss_curve.npy"
        if loss_curve_path.exists():
            try:
                loss_curve = np.load(loss_curve_path)
                if loss_curve.size > 0:
                    # Column 1 is loss_total
                    best_loss_value = float(loss_curve[-1, 1])
            except Exception:
                pass

        # Status
        status = "[OK] Completed" if success else "[ERROR] Failed / Interrupted"
        if error_msg:
            status += f" ({error_msg})"

        # Generate summary markdown
        summary_lines = [
            "# Optimization Run Summary\n",
            f"## Status\n**{status}**\n",
            f"## Timing\n",
            f"- **Start**: {ts_start}\n",
            f"- **End**: {datetime.fromtimestamp(end_time).isoformat()}Z\n",
            f"- **Duration**: {elapsed_str}\n",
            f"\n## Configuration\n",
            f"- **Optimizer**: {self.optimizer_type} ({engine})\n",
            f"- **Seed**: {self.seed}\n",
            f"- **Population**: {n_start}\n",
            f"- **Generations**: {n_steps}\n",
            f"- **TopK**: {topk}\n",
            f"- **Device**: {device}\n",
            f"\n## Results\n",
        ]

        if best_loss_value is not None:
            summary_lines.append(f"- **Best Loss**: {best_loss_value:.6f}\n")

        # List output files
        summary_lines.append(f"\n## Output Files\n")
        for f in sorted(self.log_dir.iterdir()):
            if f.is_file() and f.name != "summary.md":
                size_mb = f.stat().st_size / (1024 * 1024)
                summary_lines.append(f"- `{f.name}` ({size_mb:.2f} MB)\n")

        # Write to file
        summary_path = self.log_dir / "summary.md"
        try:
            with summary_path.open("w", encoding="utf-8") as f:
                f.write("".join(summary_lines))
            print(f"[LOG] Summary written: {summary_path}", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to write summary: {e}", flush=True)
            # Try with simple ASCII
            try:
                with summary_path.open("w", encoding="ascii", errors="replace") as f:
                    f.write("".join(summary_lines))
            except Exception as e2:
                print(f"[WARN] Failed to write summary (ASCII): {e2}", flush=True)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in seconds to readable string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")

        return " ".join(parts)
