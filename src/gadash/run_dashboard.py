from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
import yaml

from gadash.dashboard_app import create_app
from gadash.surrogate_interface import CRReconSurrogate
import torch
from gadash.config import load_config


def main() -> int:
    p = argparse.ArgumentParser(prog="gadash dashboard")
    p.add_argument("--progress-dir", default="data/progress")
    p.add_argument("--paths", default="configs/paths.yaml")
    p.add_argument("--device", default="cpu")
    p.add_argument("--surrogate", choices=["auto", "crrecon", "none"], default="auto")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8501)
    args = p.parse_args()

    surrogate = None
    if args.surrogate != "none":
        try:
            # Use the same env-var expansion logic as the GA runner.
            cfg = load_config("configs/ga.yaml", args.paths)
            root = str(cfg.paths.forward_model_root or "")
            ckpt = str(cfg.paths.forward_checkpoint or "")
            cfg_yaml = str(cfg.paths.forward_config_yaml or "")
            print(f"[DASHBOARD] Loading surrogate...")
            print(f"  forward_model_root: {root}")
            print(f"  checkpoint_path: {ckpt}")
            print(f"  config_yaml: {cfg_yaml}")
            print(f"  device: {args.device}")
            if args.surrogate in ("auto", "crrecon"):
                surrogate = CRReconSurrogate(
                    forward_model_root=Path(root),
                    checkpoint_path=Path(ckpt),
                    config_yaml=Path(cfg_yaml),
                    device=torch.device(str(args.device)),
                )
                print(f"[DASHBOARD] ✓ Surrogate loaded successfully")
        except Exception as e:
            print(f"[DASHBOARD] ✗ Surrogate load failed: {e}")
            import traceback
            traceback.print_exc()
            surrogate = None

    app = create_app(progress_dir=Path(args.progress_dir), surrogate=surrogate)
    print(f"[DASHBOARD] http://{args.host}:{int(args.port)}/")
    # Keep terminal quiet; dashboard polls endpoints periodically.
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
