from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import LossConfig, SpectraConfig
from .spectral import purity_matrix


def total_variation_2d(x: torch.Tensor) -> torch.Tensor:
    # x: (B,1,H,W)
    dx = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(1, 2, 3))
    dy = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(1, 2, 3))
    return dx + dy


def gray_penalty(x: torch.Tensor) -> torch.Tensor:
    # Encourage binarization (0/1)
    return (x * (1.0 - x)).mean(dim=(1, 2, 3))


def loss_from_pred(
    pred_rgbc: torch.Tensor,
    struct01: torch.Tensor,
    spectra: SpectraConfig,
    loss_cfg: LossConfig,
) -> dict[str, torch.Tensor]:
    """Dashboard-compatible loss (same knob names as Inverse_design_CR).

    pred_rgbc: (B,3,C) in [0,1]
    struct01: (B,1,H,W) in [0,1]
    """
    rgb_weights = spectra.rgb_weights or {"R": 1.0, "G": 2.0, "B": 1.0}
    A = purity_matrix(pred_rgbc, rgb_weights=rgb_weights)  # (B,3,3)

    I = torch.eye(3, device=A.device, dtype=A.dtype).unsqueeze(0)
    loss_purity = ((A - I) ** 2).sum(dim=(1, 2))

    diagA = torch.diagonal(A, dim1=1, dim2=2)  # (B,3)
    loss_abs = ((1.0 - diagA) ** 2).sum(dim=1)

    loss_gray = gray_penalty(struct01)
    loss_tv = total_variation_2d(struct01)

    fill = struct01.mean(dim=(1, 2, 3))
    lo = F.softplus(float(loss_cfg.fill_min) - fill)
    hi = F.softplus(fill - float(loss_cfg.fill_max))
    loss_fill = (lo * lo) + (hi * hi)

    loss_spec = float(loss_cfg.w_purity) * loss_purity + float(loss_cfg.w_abs) * loss_abs
    loss_reg = float(loss_cfg.w_gray) * loss_gray + float(loss_cfg.w_tv) * loss_tv
    loss_total = loss_spec + loss_reg + float(loss_cfg.w_fill) * loss_fill

    return {
        "A": A,
        "loss_total": loss_total,
        "loss_spec": loss_spec,
        "loss_reg": loss_reg,
        "loss_purity": loss_purity,
        "loss_abs": loss_abs,
        "loss_gray": loss_gray,
        "loss_tv": loss_tv,
        "loss_fill": loss_fill,
        "fill": fill,
    }

