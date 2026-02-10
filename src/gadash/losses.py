from __future__ import annotations

import torch

from .config import LossConfig, SpectraConfig
from .spectral import purity_matrix


def loss_from_pred(pred_rgbc: torch.Tensor, struct01: torch.Tensor, spectra: SpectraConfig, loss_cfg: LossConfig):
    """Compute losses.

    pred_rgbc: (B,3,C) in [0,1]
    struct01: (B,1,H,W) in [0,1]

    Loss uses purity matrix A (3x3): encourage diagonal high, off-diagonal low.
    """
    rgb_weights = spectra.rgb_weights or {"R": 1.0, "G": 2.0, "B": 1.0}
    A = purity_matrix(pred_rgbc, rgb_weights=rgb_weights)

    diag = torch.diagonal(A, dim1=1, dim2=2)  # (B,3)
    off = A - torch.diag_embed(diag)

    # purity spec loss: maximize diagonal, minimize off-diagonal (simple, stable)
    loss_spec = (1.0 - diag).mean(dim=1) + off.abs().mean(dim=(1, 2))

    # reg: total variation-like smoothness on structure
    dx = (struct01[..., 1:, :] - struct01[..., :-1, :]).abs().mean(dim=(1, 2, 3))
    dy = (struct01[..., :, 1:] - struct01[..., :, :-1]).abs().mean(dim=(1, 2, 3))
    loss_reg = dx + dy

    fill = struct01.mean(dim=(1, 2, 3))
    fill_pen = (loss_cfg.fill_min - fill).clamp_min(0.0)

    loss_total = (
        loss_cfg.spec_weight * loss_spec
        + loss_cfg.reg_weight * loss_reg
        + loss_cfg.fill_weight * fill_pen
    )

    return {
        "A": A,
        "loss_total": loss_total,
        "loss_spec": loss_spec,
        "loss_reg": loss_reg,
        "fill": fill,
    }


def loss_definition_latex(spectra: SpectraConfig, loss_cfg: LossConfig) -> str:
    wR = (spectra.rgb_weights or {}).get("R", 1.0)
    wG = (spectra.rgb_weights or {}).get("G", 2.0)
    wB = (spectra.rgb_weights or {}).get("B", 1.0)

    return r"""
\[
\begin{aligned}
&\textbf{Purity matrix (}A\in\mathbb{R}^{3\times 3}\textbf{):}\\
&A_{i,j}= w_i\;\frac{1}{|\mathcal{B}_j|}\sum_{c\in\mathcal{B}_j} \hat{s}_{i,c},\quad
i\in\{R,G,B\},\; j\in\{R,G,B\}\\[6pt]
&\textbf{Bands (}C=30\textbf{): }\mathcal{B}_B=\{0..9\},\;\mathcal{B}_G=\{10..19\},\;\mathcal{B}_R=\{20..29\}\\[6pt]
&\textbf{Weights: }(w_R,w_G,w_B)=(""" + str(wR) + r"," + str(wG) + r"," + str(wB) + r")\\[6pt]
&\textbf{Spec loss: }\mathcal{L}_{\text{spec}}=\frac{1}{3}\sum_{i}(1-A_{i,i}) + \frac{1}{9}\sum_{i\neq j}|A_{i,j}|\\[6pt]
&\textbf{Reg loss: }\mathcal{L}_{\text{reg}}=\|\nabla_x x\|_1+\|\nabla_y x\|_1\\[6pt]
&\textbf{Fill penalty: }\mathcal{L}_{\text{fill}}=\max(0,\,f_{\min}-\mathrm{mean}(x))\\[10pt]
&\textbf{Total: }\mathcal{L}=\lambda_{\text{spec}}\mathcal{L}_{\text{spec}}+\lambda_{\text{reg}}\mathcal{L}_{\text{reg}}+\lambda_{\text{fill}}\mathcal{L}_{\text{fill}}\\
&\lambda_{\text{spec}}=""" + str(loss_cfg.spec_weight) + r",\;\lambda_{\text{reg}}=""" + str(loss_cfg.reg_weight) + r",\;\lambda_{\text{fill}}=""" + str(loss_cfg.fill_weight) + r",\; f_{\min}=""" + str(loss_cfg.fill_min) + r".
\end{aligned}
\]
"""
