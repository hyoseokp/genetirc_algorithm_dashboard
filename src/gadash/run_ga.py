from __future__ import annotations

import argparse
from pathlib import Path

from gadash.config import load_config
from gadash.ga_opt import run_ga


def main() -> int:
    p = argparse.ArgumentParser(prog="gadash ga")
    p.add_argument("--config", default="configs/ga.yaml")
    p.add_argument("--paths", default="configs/paths.yaml")
    p.add_argument("--progress-dir", default="data/progress")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fdtd-verify", choices=["on", "off"], default="off")
    p.add_argument("--fdtd-every", type=int, default=0, help="If >0, verify topk every N generations (non-blocking).")
    p.add_argument("--fdtd-k", type=int, default=None, help="How many topk items to verify (default: all in snapshot).")
    p.add_argument("--fdtd-config", default="configs/fdtd.yaml")
    args = p.parse_args()

    cfg = load_config(args.config, args.paths)
    Path(args.progress_dir).mkdir(parents=True, exist_ok=True)

    out = run_ga(
        cfg,
        progress_dir=args.progress_dir,
        device=args.device,
        dry_run=args.dry_run,
        fdtd_verify=(str(args.fdtd_verify) == "on"),
        fdtd_every=int(args.fdtd_every),
        fdtd_k=(int(args.fdtd_k) if args.fdtd_k is not None else None),
        fdtd_config=args.fdtd_config,
        paths_yaml=args.paths,
    )
    print(f"[OK] ga done: progress_dir={out['progress_dir']} gens={out['gens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
