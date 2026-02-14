from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    warnings.filterwarnings("ignore", category=SyntaxWarning)
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
    target_material: str = "Si3N4 (Silicon Nitride) - Phillip"
    z_min: float = 0.0
    z_max: float = 600e-9
    trans_1: str = "Trans_1"
    trans_2: str = "Trans_2"
    trans_3: str = "Trans_3"
    preclean_names: list[str] | None = None

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
            # Allow Lumerical to finish cleanup before closing
            # (prevents "QProcess: Destroyed while process is still running")
            import time
            time.sleep(1.0)
            self._fdtd.close()
        finally:
            self._fdtd = None
            self._lumapi = None

    def import_gds(self, *, gds_path: Path, cell_name: str) -> None:
        if self._fdtd is None:
            raise RuntimeError("FDTD session not open")
        fallback = cell_name
        if "_" in cell_name:
            # structure_00012 -> structure_12 fallback used in original workflow
            stem, _, tail = cell_name.rpartition("_")
            try:
                fallback = f"{stem}_{int(tail)}"
            except Exception:
                fallback = cell_name
        script = gds_import_script(
            gds_path=str(gds_path),
            cell_name_primary=cell_name,
            cell_name_fallback=fallback,
            layer_map=self.layer_map,
            target_material=self.target_material,
            z_min=self.z_min,
            z_max=self.z_max,
            preclean_names=(self.preclean_names or []),
        )
        self._fdtd.switchtolayout()
        try:
            self._fdtd.eval(script)
        except Exception as e:
            raise RuntimeError(
                f"fdtd.eval(gdsimport) failed: gds={gds_path} cellA={cell_name} "
                f"cellB={fallback} layer_map={self.layer_map} material={self.target_material} err={e}"
            ) from e
        import_ok = int(np.asarray(self._fdtd.getv("import_ok")).reshape(-1)[0])
        try:
            self._fdtd.eval('import_count=0; try{ import_count=getnamednumber("IMPORTED_GDS"); } catch(errCnt) { import_count=0; }')
            import_count = int(np.asarray(self._fdtd.getv("import_count")).reshape(-1)[0])
        except Exception:
            import_count = -1
        if import_ok != 1:
            import_err = str(self._fdtd.getv("import_err"))
            used_cell = str(self._fdtd.getv("used_cell"))
            raise RuntimeError(
                f"gdsimport failed (used_cell={used_cell}, import_count={import_count}): {import_err}"
            )
        if import_count == 0:
            used_cell = str(self._fdtd.getv("used_cell"))
            raise RuntimeError(f"gdsimport returned zero geometry (used_cell={used_cell}, import_count=0)")

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
        triples: list[tuple[str, str, str]] = []
        triples.append((self.trans_1, self.trans_2, self.trans_3))
        for tri in [
            ("Trans_1", "Trans_2", "Trans_3"),
            ("trans_1", "trans_2", "trans_3"),
            ("T1", "T2", "T3"),
            ("R", "G", "B"),
        ]:
            if tri not in triples:
                triples.append(tri)

        last_probe = ""
        t1 = t2 = t3 = None
        for m1, m2, m3 in triples:
            self._fdtd.eval(extract_spectra_script(trans_1=m1, trans_2=m2, trans_3=m3))
            vv1 = np.asarray(self._fdtd.getv("T1"), dtype=np.float32).reshape(-1)
            vv2 = np.asarray(self._fdtd.getv("T2"), dtype=np.float32).reshape(-1)
            vv3 = np.asarray(self._fdtd.getv("T3"), dtype=np.float32).reshape(-1)
            if (vv1.size > 1) or (vv2.size > 1) or (vv3.size > 1):
                t1, t2, t3 = vv1, vv2, vv3
                break
            last_probe = f"{m1},{m2},{m3}"

        if t1 is None or t2 is None or t3 is None:
            # Fallback: single monitor with T matrix (4,N)/(N,4)
            try:
                self._fdtd.eval(
                    "m='monitor'; f_vec=getdata(m,'f'); T=getdata(m,'T');"
                )
                T = np.asarray(self._fdtd.getv("T"), dtype=np.float32)
                if T.ndim == 2 and 4 in T.shape:
                    v = T if T.shape[0] == 4 else T.T
                    v = v.reshape(4, -1)
                    return v.reshape(2, 2, -1).astype(np.float32, copy=False)
            except Exception:
                pass
            raise RuntimeError(
                "no transmission monitor results found; "
                f"tried triples={triples}, last_probe={last_probe}, "
                "and fallback monitor='monitor'/T"
            )

        n = int(max(t1.size, t2.size, t3.size))

        def _fit(v: np.ndarray) -> np.ndarray:
            if v.size == n:
                return v
            if v.size <= 1:
                return np.zeros((n,), dtype=np.float32)
            x = np.linspace(0.0, 1.0, num=v.size, dtype=np.float32)
            xi = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
            return np.interp(xi, x, v).astype(np.float32)

        r = _fit(t1)
        g = _fit(t2)
        b = _fit(t3)
        # Build RGGB tensor from RGB monitors: [[R,G],[G,B]]
        out = np.stack([r, g, g, b], axis=0).reshape(2, 2, n)
        return out.astype(np.float32, copy=False)
