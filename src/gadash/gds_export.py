from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

try:
    import gdstk  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("gdstk is required for GDS export") from e


@dataclass(frozen=True)
class GDSExportOptions:
    layer: int = 1
    datatype: int = 0
    pixel_size: float = 1.0  # arbitrary unit


def _rectangles_from_binary_grid(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Convert binary grid to axis-aligned rectangles deterministically."""
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape={grid.shape}")
    h, w = grid.shape
    g = grid.astype(np.uint8) != 0

    segments_per_row: list[list[tuple[int, int]]] = []
    for y in range(h):
        row = g[y]
        segs: list[tuple[int, int]] = []
        x = 0
        while x < w:
            if not row[x]:
                x += 1
                continue
            x0 = x
            while x < w and row[x]:
                x += 1
            x1 = x
            segs.append((x0, x1))
        segments_per_row.append(segs)

    rects: list[tuple[int, int, int, int]] = []
    active: dict[tuple[int, int], tuple[int, int]] = {}

    for y in range(h + 1):
        segs = segments_per_row[y] if y < h else []
        seg_set = set(segs)

        for key in list(active.keys()):
            if key not in seg_set:
                x0, x1 = key
                y0, y1 = active.pop(key)
                rects.append((x0, y0, x1, y1))

        for seg in segs:
            if seg in active:
                y0, _y1 = active[seg]
                active[seg] = (y0, y + 1)
            else:
                active[seg] = (y, y + 1)

    return rects


def export_struct128_to_gds(
    struct128: np.ndarray,
    *,
    out_path: str | Path,
    structure_id: int,
    options: GDSExportOptions | None = None,
) -> Path:
    """Export a (H,W) binary structure to a GDS file."""
    options = options or GDSExportOptions()
    struct = np.asarray(struct128)
    if struct.ndim != 2:
        raise ValueError(f"struct128 must be 2D, got shape={struct.shape}")

    sid = int(structure_id)
    cell_name = f"structure_{sid:05d}"
    lib = gdstk.Library()
    cell = lib.new_cell(cell_name)

    rects = _rectangles_from_binary_grid(struct)
    ps = float(options.pixel_size)
    for x0, y0, x1, y1 in rects:
        r = gdstk.rectangle(
            (x0 * ps, y0 * ps),
            (x1 * ps, y1 * ps),
            layer=int(options.layer),
            datatype=int(options.datatype),
        )
        cell.add(r)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # gdstk (native) may fail to open paths containing non-ASCII characters on Windows.
    tmp_root = Path(os.environ.get("CRINV_GDS_TMP", r"C:\gdstk_tmp"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="crinv_", suffix=".gds", dir=str(tmp_root), delete=False) as tf:
        tmp_path = Path(tf.name)
    try:
        lib.write_gds(str(tmp_path))
        shutil.copyfile(tmp_path, out_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    return out_path

