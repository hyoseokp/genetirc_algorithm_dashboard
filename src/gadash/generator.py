from __future__ import annotations

import torch

from .config import DesignConfig, GeneratorConfig
from .ops import enforce_diag_symmetry, gaussian_blur2d, resize_bicubic, soft_morph_close


class RuleMFSGenerator:
    def __init__(self, design: DesignConfig, gen: GeneratorConfig):
        self.design = design
        self.gen = gen

    def __call__(self, seed01: torch.Tensor) -> torch.Tensor:
        """seed01: (B,1,16,16) in [0,1] -> struct01: (B,1,128,128) in [0,1]."""
        x = seed01
        if self.design.enforce_symmetry:
            x = enforce_diag_symmetry(x)

        x = resize_bicubic(x, self.design.struct_size)
        x = gaussian_blur2d(x, sigma=self.gen.blur_sigma)

        # threshold -> soft-ish binary (still continuous)
        x = torch.sigmoid((x - self.gen.threshold) * 12.0)

        # enforce min feature size with circular padding (tile-friendly)
        x = soft_morph_close(x, radius=int(self.gen.mfs_radius_px), iters=int(self.gen.mfs_iters))
        return x.clamp(0.0, 1.0)


def build_generator(design: DesignConfig, gen: GeneratorConfig):
    if gen.backend != "rule_mfs":
        raise ValueError(f"unsupported generator backend: {gen.backend}")
    return RuleMFSGenerator(design, gen)
