from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fdtd_scripts import extract_spectra_script, gds_import_script


def load_lumapi(*, lumerical_root: Path):
    """Load Lumerical's lumapi from the installation (Windows-friendly).

    Avoid importing a stray pip-installed lumapi that can't find interopapi.dll.
    """
    lumerical_root = Path(lumerical_root)
    api_py = lumerical_root / "api" / "python"
    lumapi_py = api_py / "lumapi.py"
    if not lumapi_py.exists():
        raise FileNotFoundError(f"lumapi.py not found at: {lumapi_py}")

    # Ensure DLL search path includes the directories containing interopapi.dll.
    dll_dirs = [
        api_py,
        lumerical_root / "bin",
        lumerical_root / "api" / "c",
    ]
    for d in dll_dirs:
        if d.exists():
            try:
                os.add_dll_directory(str(d))
            except Exception:
                pass

    # Ensure python can import modules relative to lumapi.py.
    if str(api_py) not in sys.path:
        sys.path.insert(0, str(api_py))

    spec = importlib.util.spec_from_file_location("lumapi", str(lumapi_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load spec for lumapi.py at {lumapi_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class LumapiBridge:
    """Concrete FDTDBridge using lumapi.FDTD session."""

    lumerical_root: Path
    template_fsp: Path
    hide: bool = True

    # Lumerical scripts are template-dependent; these are overridable hooks.
    layer_map: str = "1:0"

    def __post_init__(self) -> None:
        self.lumerical_root = Path(self.lumerical_root)
        self.template_fsp = Path(self.template_fsp)
        if not self.lumerical_root.exists():
            raise FileNotFoundError(f"lumerical_root not found: {self.lumerical_root}")
        if not self.template_fsp.exists():
            raise FileNotFoundError(f"template_fsp not found: {self.template_fsp}")
        # Lumerical (and/or Windows DLL loading) can be fragile with non-ASCII paths.
        # If needed, copy the template to an ASCII-only temp path and use that.
        try:
            str(self.template_fsp).encode("ascii")
        except Exception:
            tmp_root = Path(os.environ.get("GADASH_FDTD_TMP", r"C:\lumerical_tmp"))
            tmp_root.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_root / "template.fsp"
            try:
                tmp_path.write_bytes(self.template_fsp.read_bytes())
                self.template_fsp = tmp_path
            except Exception:
                # Best effort; fall back to original.
                pass
        self._lumapi = None
        self._fdtd = None

    def open(self) -> None:
        if self._fdtd is not None:
            return
        self._lumapi = load_lumapi(lumerical_root=self.lumerical_root)

        # Try common constructor patterns.
        fdtd = None
        try:
            fdtd = self._lumapi.FDTD(hide=bool(self.hide))
            try:
                fdtd.load(str(self.template_fsp))
            except Exception:
                # Some versions accept filename in constructor only.
                fdtd.close()
                fdtd = None
        except Exception:
            fdtd = None

        if fdtd is None:
            fdtd = self._lumapi.FDTD(filename=str(self.template_fsp), hide=bool(self.hide))

        self._fdtd = fdtd

    def close(self) -> None:
        if self._fdtd is None:
            return
        try:
            self._fdtd.close()
        finally:
            self._fdtd = None
            self._lumapi = None

    def import_gds(self, *, gds_path: Path, cell_name: str) -> None:
        if self._fdtd is None:
            raise RuntimeError("FDTD session not open")
        script = gds_import_script(gds_path=str(gds_path), cell_name=cell_name, layer_map=self.layer_map)
        self._fdtd.switchtolayout()
        self._fdtd.eval(script)

    def run(self) -> None:
        if self._fdtd is None:
            raise RuntimeError("FDTD session not open")
        self._fdtd.run()

    def extract_spectra(self) -> np.ndarray:
        """Extract template-dependent spectra as RGGB.

        Returns: (2,2,N) float32.
        """
        if self._fdtd is None:
            raise RuntimeError("FDTD session not open")
        self._fdtd.eval(extract_spectra_script())
        _f = np.asarray(self._fdtd.getv("f_vec")).astype(np.float32).reshape(-1)
        T = np.asarray(self._fdtd.getv("T")).astype(np.float32)

        if T.ndim == 3 and T.shape[0:2] == (2, 2):
            return T
        if T.ndim != 2:
            raise ValueError(f"expected T with ndim 2 (4,N) or (N,4), got shape={T.shape}")
        if 4 in T.shape:
            if T.shape[0] == 4:
                v = T
            else:
                v = T.T
            v = v.reshape(4, -1)
            return v.reshape(2, 2, -1)
        raise ValueError(f"expected T to have a 4-dim axis, got shape={T.shape}")
