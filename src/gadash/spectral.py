from __future__ import annotations

import numpy as np
import torch


def wavelengths_nm(channels: int, wmin: float, wmax: float) -> np.ndarray:
    return np.linspace(float(wmin), float(wmax), int(channels), endpoint=True)


def bands_for_rgb(channels: int) -> dict[str, slice]:
    # 30 channels: B 0-9, G 10-19, R 20-29
    if channels != 30:
        # generic split into 3 equal-ish blocks
        b = channels // 3
        return {"B": slice(0, b), "G": slice(b, 2 * b), "R": slice(2 * b, channels)}
    return {"B": slice(0, 10), "G": slice(10, 20), "R": slice(20, 30)}


def purity_matrix(pred_rgbc: torch.Tensor, rgb_weights: dict[str, float]) -> torch.Tensor:
    """Build 3x3 purity matrix A for each sample.

    pred_rgbc: (B,3,C) values in [0,1].
    rows: detector channel (R,G,B). cols: wavelength band (R,G,B).

    A[i,j] = mean over channels in band j of spectrum at detector i.

    Additionally applies per-detector weights (e.g. G=2) directly to rows.
    """
    B, three, C = pred_rgbc.shape
    assert three == 3

    bands = bands_for_rgb(C)
    # order rows as R,G,B
    idx = {"R": 0, "G": 1, "B": 2}

    A = pred_rgbc.new_zeros((B, 3, 3))
    for det_name, det_i in idx.items():
        w_det = float(rgb_weights.get(det_name, 1.0))
        for band_name, sl in bands.items():
            band_j = idx[band_name]
            A[:, det_i, band_j] = pred_rgbc[:, det_i, sl].mean(dim=-1) * w_det
    return A
