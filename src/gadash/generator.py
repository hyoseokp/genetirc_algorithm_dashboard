from __future__ import annotations

import torch

from .config import DesignConfig, GeneratorConfig
from .ops import enforce_diag_symmetry, gaussian_blur2d, resize_bicubic, soft_morph_close


class RuleMFSGenerator:
    def __init__(self, design: DesignConfig, gen: GeneratorConfig):
        self.design = design
        self.gen = gen

    def __call__(self, seed01: torch.Tensor) -> torch.Tensor:
        """seed01: (B,1,16,16) expected in [0,1] -> struct01: (B,1,128,128) in [0,1]."""
        x = seed01
        if self.design.enforce_symmetry:
            x = enforce_diag_symmetry(x)

        x = resize_bicubic(x, self.design.struct_size)
        x = gaussian_blur2d(x, sigma=self.gen.blur_sigma)

        # threshold -> soft-ish binary (still continuous)
        x = torch.sigmoid((x - self.gen.threshold) * 12.0)

        # enforce min feature size with circular padding (tile-friendly)
        x = soft_morph_close(x, radius=int(self.gen.mfs_radius_px), iters=int(self.gen.mfs_iters))
        # Match the original rule's final inversion: binary = ~enforce_mfs_final(...)
        x = x.clamp(0.0, 1.0)
        return (1.0 - x).clamp(0.0, 1.0)


class RuleMFSSciPyGenerator:
    """Exact (non-differentiable) rule matching the SciPy EDT-based dataset generator.

    CPU-only and slower, but useful when you want to match the original rule exactly.
    """

    def __init__(self, design: DesignConfig, gen: GeneratorConfig):
        self.design = design
        self.gen = gen

        try:
            from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "generator.backend=rule_mfs_scipy requires SciPy (scipy.ndimage). "
                "Install scipy or use generator.backend=rule_mfs."
            ) from e

        self._distance_transform_edt = distance_transform_edt
        self._gaussian_filter = gaussian_filter
        self._zoom = zoom

    def _apply_buffer_logic(self, binary, min_size: float, buffer: float = 1.0):
        import numpy as np

        radius = float(min_size) * 0.5
        dt = self._distance_transform_edt(binary)
        core = dt >= radius
        if not core.any():
            return np.zeros_like(binary, dtype=bool)

        dt_core = self._distance_transform_edt(~core)
        over = dt_core <= (radius + buffer)
        dt_over = self._distance_transform_edt(over)
        return dt_over >= buffer

    def _enforce_mfs_final(self, binary, min_size: float = 8.0, max_iter: int = 4):
        import numpy as np

        pad = int(float(min_size) * 2.0)
        buffer_val = 1.0

        img = binary.astype(bool)
        img = np.pad(img, pad, mode="wrap")

        for _ in range(int(max_iter)):
            prev = img
            img = self._apply_buffer_logic(img, min_size, buffer_val)  # solid
            img = ~self._apply_buffer_logic(~img, min_size, buffer_val)  # void
            if not np.any(img ^ prev):
                break

        img = img[pad:-pad, pad:-pad]
        img = np.triu(img) | np.triu(img, 1).T
        return img

    def __call__(self, seed01: torch.Tensor) -> torch.Tensor:
        """seed: (B,1,16,16) arbitrary -> internally normalized to [0,1] -> struct01: (B,1,128,128) in {0,1} (float32)."""
        import numpy as np

        x = seed01
        # Normalize seed into [0,1] for robustness (match dataset-style raw inputs).
        # - If the caller passes logits / unconstrained values, squash with sigmoid.
        # - If already in-range, just clamp (avoid distorting uniform[0,1] seeds).
        try:
            x_min = float(x.min().detach().cpu().item())
            x_max = float(x.max().detach().cpu().item())
        except Exception:
            x_min, x_max = -1.0, 2.0
        if x_min < 0.0 or x_max > 1.0:
            x = torch.sigmoid(x)
        else:
            x = x.clamp(0.0, 1.0)
        if self.design.enforce_symmetry:
            x = enforce_diag_symmetry(x)

        x_np = x.detach().float().cpu().numpy()  # (B,1,16,16)
        B = int(x_np.shape[0])
        H = int(self.design.struct_size)
        out = np.zeros((B, 1, H, H), dtype=np.float32)
        zf = float(self.design.struct_size) / float(self.design.seed_size)

        # Note: we interpret gen.mfs_radius_px as "radius in px", while the reference code uses
        # MIN_FEATURE_SIZE as a diameter-like size. Convert radius->min_size ~= 2*radius.
        min_size = float(self.gen.mfs_radius_px) * 2.0

        def _one(i: int):
            sym = x_np[i, 0]  # (16,16)
            up = self._zoom(sym, zf, order=3)
            blur = self._gaussian_filter(up, sigma=float(self.gen.blur_sigma))
            binary = blur > float(self.gen.threshold)
            binary = ~self._enforce_mfs_final(binary, min_size=min_size, max_iter=int(self.gen.mfs_iters))
            return i, binary.astype(np.float32)

        # The EDT kernels are in C and typically release the GIL; threads can speed up on multi-core CPUs.
        try:
            import os
            from concurrent.futures import ThreadPoolExecutor, as_completed

            max_workers = min(8, int(os.cpu_count() or 1))
            if B >= 4 and max_workers >= 2:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = [ex.submit(_one, i) for i in range(B)]
                    for f in as_completed(futs):
                        i, bi = f.result()
                        out[i, 0] = bi
            else:
                for i in range(B):
                    _, bi = _one(i)
                    out[i, 0] = bi
        except Exception:
            for i in range(B):
                _, bi = _one(i)
                out[i, 0] = bi

        return torch.from_numpy(out).to(device=seed01.device, dtype=torch.float32)


def build_generator(design: DesignConfig, gen: GeneratorConfig):
    if gen.backend == "rule_mfs":
        return RuleMFSGenerator(design, gen)
    if gen.backend == "rule_mfs_scipy":
        return RuleMFSSciPyGenerator(design, gen)
    raise ValueError(f"unsupported generator backend: {gen.backend}")
