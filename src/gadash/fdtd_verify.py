from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .fdtd_runner import FDTDRunPaths, FDTDRuntimeOptions, run_fdtd_batch
from .gds_export import export_struct128_to_gds
from .lumapi_bridge import LumapiBridge


def _load_yaml_dict(path: str | Path) -> dict[str, Any]:
    import yaml

    p = Path(path)
    if not p.exists():
        return {}
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(obj, dict):
        raise TypeError("YAML root must be a mapping")
    return obj


def _run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"fdtd_{ts}"


@dataclass(frozen=True)
class FDTDRuntimeCfg:
    lumerical_root: str
    template_fsp: str
    hide: bool = True
    chunk_size: int = 8
    max_retries: int = 2
    timeout_s: dict[str, int] | None = None


def resolve_fdtd_cfg(*, fdtd_yaml: str | Path, paths_yaml: str | Path) -> FDTDRuntimeCfg:
    """Load FDTD config and patch-in machine-local configs/paths.yaml if needed."""
    obj = _load_yaml_dict(fdtd_yaml)
    fd = obj.get("fdtd") if isinstance(obj.get("fdtd"), dict) else {}

    lumerical_root = str(fd.get("lumerical_root", "") or "")
    template_fsp = str(fd.get("template_fsp", "") or "")
    hide = bool(fd.get("hide", True))
    chunk_size = int(fd.get("chunk_size", 8) or 8)
    max_retries = int(fd.get("max_retries", 2) or 2)
    timeout_s = fd.get("timeout_s") if isinstance(fd.get("timeout_s"), dict) else None

    # Patch from paths.yaml when empty.
    paths = _load_yaml_dict(paths_yaml)
    if not lumerical_root.strip():
        lumerical_root = str(paths.get("lumerical_root", "") or "")
    if not template_fsp.strip():
        template_fsp = str(paths.get("fdtd_template_fsp", "") or "")

    if not lumerical_root.strip():
        raise ValueError("FDTD lumerical_root not set (configs/fdtd.yaml or configs/paths.yaml)")
    if not template_fsp.strip():
        raise ValueError("FDTD template_fsp not set (configs/fdtd.yaml or configs/paths.yaml)")

    return FDTDRuntimeCfg(
        lumerical_root=lumerical_root,
        template_fsp=template_fsp,
        hide=hide,
        chunk_size=chunk_size,
        max_retries=max_retries,
        timeout_s=timeout_s,
    )


def _load_npz_struct_topk(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    if "struct128_topk" not in z.files:
        raise KeyError(f"struct128_topk missing in {path}")
    arr = np.asarray(z["struct128_topk"])
    if arr.ndim != 3:
        raise ValueError(f"expected struct128_topk (K,128,128), got {arr.shape}")
    return arr


@dataclass(frozen=True)
class FDTDVerifyResult:
    out_dir: Path
    fdtd_rggb_path: Path
    done_ids: list[int]


def verify_topk_with_fdtd(
    *,
    topk_npz: str | Path,
    fdtd_cfg: FDTDRuntimeCfg,
    out_dir: str | Path = r"C:\gadash_fdtd_results",
    k: int | None = None,
    layer_map: str = "1:0",
    cell_prefix: str = "structure",
) -> FDTDVerifyResult:
    """Run Lumerical FDTD on Top-K structures and stack RGGB spectra."""
    topk_npz = Path(topk_npz)
    struct_topk = _load_npz_struct_topk(topk_npz)
    K = int(struct_topk.shape[0])
    if k is None:
        k_use = K
    else:
        k_use = max(1, min(K, int(k)))
        struct_topk = struct_topk[:k_use]

    out_root = Path(out_dir)
    run_dir = out_root / _run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    gds_dir = run_dir / "gds"
    gds_dir.mkdir(parents=True, exist_ok=True)

    spectra_dir = run_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[int, Path, str]] = []
    for i in range(k_use):
        sid = int(i)
        cell_name = f"{cell_prefix}_{sid:05d}"
        gds_path = gds_dir / f"{cell_name}.gds"
        export_struct128_to_gds(struct_topk[i], out_path=gds_path, structure_id=sid)
        items.append((sid, gds_path, cell_name))

    bridge = LumapiBridge(
        lumerical_root=Path(fdtd_cfg.lumerical_root),
        template_fsp=Path(fdtd_cfg.template_fsp),
        hide=bool(fdtd_cfg.hide),
        layer_map=str(layer_map),
    )
    paths = FDTDRunPaths(out_dir=spectra_dir)
    options = FDTDRuntimeOptions(chunk_size=int(fdtd_cfg.chunk_size), max_retries=int(fdtd_cfg.max_retries))
    done = run_fdtd_batch(bridge=bridge, items=items, paths=paths, options=options)

    rggb_list: list[np.ndarray] = []
    for sid in range(k_use):
        p = paths.spectra_path(sid)
        if not p.exists():
            raise FileNotFoundError(f"missing spectra for id={sid}: {p}")
        arr = np.load(p)
        if arr.ndim != 3 or arr.shape[0:2] != (2, 2):
            raise ValueError(f"expected spectra (2,2,C) for id={sid}, got {arr.shape}")
        rggb_list.append(arr.astype(np.float32))
    fdtd_rggb = np.stack(rggb_list, axis=0)  # (K,2,2,C)

    fdtd_rggb_path = run_dir / "fdtd_rggb.npy"
    np.save(fdtd_rggb_path, fdtd_rggb)
    return FDTDVerifyResult(out_dir=run_dir, fdtd_rggb_path=fdtd_rggb_path, done_ids=done)
