from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
import yaml

from gadash.dashboard_app import create_app
from gadash.surrogate_interface import CRReconSurrogate
import torch


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
            paths = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8")) or {}
            root = str(paths.get("forward_model_root", ""))
            ckpt = str(paths.get("forward_checkpoint", ""))
            cfg_yaml = str(paths.get("forward_config_yaml", ""))
            if args.surrogate in ("auto", "crrecon"):
                surrogate = CRReconSurrogate(
                    forward_model_root=Path(root),
                    checkpoint_path=Path(ckpt),
                    config_yaml=Path(cfg_yaml),
                    device=torch.device(str(args.device)),
                )
        except Exception as e:
            print(f"[DASHBOARD] surrogate disabled: {e}")
            surrogate = None

    app = create_app(progress_dir=Path(args.progress_dir), surrogate=surrogate)
    print(f"[DASHBOARD] http://{args.host}:{int(args.port)}/")
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
