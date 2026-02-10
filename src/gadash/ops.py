from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def enforce_diag_symmetry(x: torch.Tensor) -> torch.Tensor:
    """Enforce symmetry over the main diagonal for last-2 dims."""
    return torch.triu(x) + torch.triu(x, diagonal=1).transpose(-1, -2)


def gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise gaussian blur. x: (B,1,H,W)"""
    if sigma <= 0:
        return x
    radius = int(max(1, math.ceil(3.0 * sigma)))
    k = 2 * radius + 1
    device = x.device
    dtype = x.dtype

    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()

    g1 = g.view(1, 1, 1, k)
    g2 = g.view(1, 1, k, 1)

    x = F.pad(x, (radius, radius, 0, 0), mode="circular")
    x = F.conv2d(x, g1)
    x = F.pad(x, (0, 0, radius, radius), mode="circular")
    x = F.conv2d(x, g2)
    return x


def disk_kernel(radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r = int(radius)
    if r <= 0:
        return torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
    ys, xs = torch.meshgrid(
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    m = (xs * xs + ys * ys) <= (float(r) * float(r))
    k = m.to(dtype)
    k = k / k.sum().clamp_min(1.0)
    return k.view(1, 1, 2 * r + 1, 2 * r + 1)


def soft_morph_close(x: torch.Tensor, radius: int, iters: int = 1) -> torch.Tensor:
    """Approximate close (dilate then erode) with disk kernel using conv + smoothstep.

    x is in [0,1]. returns in [0,1].
    """
    if radius <= 0 or iters <= 0:
        return x
    k = disk_kernel(radius, x.device, x.dtype)

    def _dilate(y: torch.Tensor) -> torch.Tensor:
        # max-pool with disk: use conv as weighted average then sharpen
        z = F.conv2d(F.pad(y, (radius, radius, radius, radius), mode="circular"), k)
        return torch.sigmoid((z - 0.5) * 12.0)

    def _erode(y: torch.Tensor) -> torch.Tensor:
        z = F.conv2d(F.pad(y, (radius, radius, radius, radius), mode="circular"), k)
        return torch.sigmoid((z - 0.5) * 12.0)

    out = x
    for _ in range(iters):
        out = _dilate(out)
        out = 1.0 - _dilate(1.0 - out)  # dual for erosion
    return out


def resize_bicubic(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
