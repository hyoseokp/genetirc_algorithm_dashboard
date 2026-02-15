from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .commit_protocol import CommitMarker, VALID_COMMITTED


class FDTDBridge(Protocol):
    """Backend interface to decouple runner logic from lumapi for testing."""

    def open(self) -> None: ...
    def close(self) -> None: ...
    def import_gds(self, *, gds_path: Path, cell_name: str) -> None: ...
    def run(self) -> None: ...
    def extract_spectra(self) -> np.ndarray: ...


@dataclass(frozen=True)
class FDTDRuntimeOptions:
    chunk_size: int = 8
    max_retries: int = 2


@dataclass(frozen=True)
class FDTDRunPaths:
    out_dir: Path

    def structure_dir(self, structure_id: int) -> Path:
        return self.out_dir / f"structure_{int(structure_id):05d}"

    def marker_path(self, structure_id: int) -> Path:
        return self.structure_dir(structure_id) / "valid.txt"

    def spectra_path(self, structure_id: int) -> Path:
        return self.structure_dir(structure_id) / "spectra.npy"


def run_fdtd_batch(
    *,
    bridge: FDTDBridge,
    items: list[tuple[int, Path, str]],  # (structure_id, gds_path, cell_name)
    paths: FDTDRunPaths,
    options: FDTDRuntimeOptions | None = None,
) -> list[int]:
    """Run FDTD in chunks with 2-phase commit markers."""
    options = options or FDTDRuntimeOptions()
    done: list[int] = []

    chunk = int(options.chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be > 0")

    for i in range(0, len(items), chunk):
        sub = items[i : i + chunk]
        bridge.open()
        try:
            for structure_id, gds_path, cell_name in sub:
                sdir = paths.structure_dir(structure_id)
                sdir.mkdir(parents=True, exist_ok=True)
                marker = CommitMarker(paths.marker_path(structure_id))
                if marker.read() == VALID_COMMITTED and paths.spectra_path(structure_id).exists():
                    done.append(structure_id)
                    continue

                retries = 0
                while True:
                    try:
                        marker.stage()
                        bridge.import_gds(gds_path=gds_path, cell_name=cell_name)
                        print(f"[FDTD-RUNNER] Running simulation for id={structure_id}...", flush=True)
                        bridge.run()
                        print(f"[FDTD-RUNNER] Extracting spectra for id={structure_id}...", flush=True)
                        spectra = bridge.extract_spectra()
                        np.save(paths.spectra_path(structure_id), spectra)
                        marker.commit()
                        done.append(structure_id)
                        print(f"[FDTD-RUNNER] id={structure_id} completed successfully", flush=True)
                        break
                    except Exception as e:
                        retries += 1
                        print(f"[FDTD-RUNNER] id={structure_id} attempt {retries} failed: {e}", flush=True)
                        if retries > int(options.max_retries):
                            raise RuntimeError(
                                f"fdtd item failed: id={int(structure_id)} cell={cell_name} "
                                f"gds={gds_path} retries={int(options.max_retries)} err={e}"
                            ) from e
                        # Close and reopen bridge for a clean session on retry
                        print(f"[FDTD-RUNNER] Reopening FDTD session for retry...", flush=True)
                        try:
                            bridge.close()
                        except Exception:
                            pass
                        bridge.open()
        finally:
            bridge.close()

    return done
