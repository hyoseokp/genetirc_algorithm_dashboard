from __future__ import annotations

import argparse

import uvicorn

from gadash.dashboard_app import create_app


def main() -> int:
    p = argparse.ArgumentParser(prog="gadash dashboard")
    p.add_argument("--progress-dir", default="data/progress")
    p.add_argument("--config", default="configs/ga.yaml")
    p.add_argument("--paths", default="configs/paths.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8501)
    args = p.parse_args()

    app = create_app(progress_dir=args.progress_dir, cfg_path=args.config, paths_path=args.paths)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
