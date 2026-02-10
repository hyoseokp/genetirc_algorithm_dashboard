from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import yaml


class ForwardSurrogate(Protocol):
    """Forward surrogate interface.

    Input: binary/continuous structure [B,128,128] float in [0,1]
    Output: RGGB spectra [B,2,2,C]
    """

    n_channels: int

    def predict(self, x_binary: torch.Tensor) -> torch.Tensor: ...


class MockSurrogate:
    """Cheap deterministic surrogate for dry-run/integration tests."""

    def __init__(self, n_channels: int = 30):
        self.n_channels = int(n_channels)

    def predict(self, x_binary: torch.Tensor) -> torch.Tensor:
        if x_binary.ndim != 3:
            raise ValueError(f"x_binary must be [B,H,W], got {tuple(x_binary.shape)}")
        B, H, W = x_binary.shape
        h2, w2 = H // 2, W // 2
        q00 = x_binary[:, :h2, :w2].mean(dim=(-1, -2))
        q01 = x_binary[:, :h2, w2:].mean(dim=(-1, -2))
        q10 = x_binary[:, h2:, :w2].mean(dim=(-1, -2))
        q11 = x_binary[:, h2:, w2:].mean(dim=(-1, -2))
        base = torch.stack([q00, q01, q10, q11], dim=-1).view(B, 2, 2)
        C = self.n_channels
        out = base[..., None].repeat(1, 1, 1, C)
        slope = torch.linspace(0.0, 0.01, steps=C, device=x_binary.device, dtype=x_binary.dtype)
        return out + slope.view(1, 1, 1, C)


@dataclass
class CRReconSurrogate:
    """Load and run CR_recon forward model checkpoints (copied from Inverse_design_CR pattern)."""

    forward_model_root: Path
    checkpoint_path: Path
    config_yaml: Path
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        self.forward_model_root = Path(self.forward_model_root)
        self.checkpoint_path = Path(self.checkpoint_path)
        self.config_yaml = Path(self.config_yaml)
        if not self.forward_model_root.exists():
            raise FileNotFoundError(f"forward_model_root not found: {self.forward_model_root}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint_path not found: {self.checkpoint_path}")
        if not self.config_yaml.exists():
            raise FileNotFoundError(f"config_yaml not found: {self.config_yaml}")

        # Git-LFS pointer detection
        try:
            head = self.checkpoint_path.read_bytes()[:200]
            if b"version https://git-lfs.github.com/spec/v1" in head:
                raise ValueError(
                    "checkpoint_path points to a Git-LFS pointer file, not real weights: "
                    f"{self.checkpoint_path}"
                )
        except ValueError:
            raise
        except Exception:
            pass

        cfg = _load_yaml_dict(self.config_yaml)
        model_name = cfg["model"]["name"]
        model_params = cfg["model"]["params"]
        out_len = int(model_params.get("out_len", cfg.get("data", {}).get("out_len", 30)))
        self.n_channels = out_len

        import importlib
        import sys

        sys.path.insert(0, str(self.forward_model_root))
        models = importlib.import_module("models")
        get_model = getattr(models, "get_model")

        self._model = get_model(model_name, **model_params)
        self._model.to(self.device)
        self._model.eval()

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self._model.load_state_dict(state, strict=True)

        self._map_to_pm1 = bool(cfg.get("data", {}).get("map_to_pm1", True))

    @torch.no_grad()
    def predict(self, x_binary: torch.Tensor) -> torch.Tensor:
        if x_binary.ndim != 3:
            raise ValueError(f"x_binary must be [B,H,W], got {tuple(x_binary.shape)}")
        x = x_binary.to(device=self.device, dtype=torch.float32)
        if self._map_to_pm1:
            x = (x * 2.0) - 1.0
        x = x.unsqueeze(1)  # (B,1,128,128)
        return self._model(x)


def _load_yaml_dict(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be a mapping, got: {type(obj).__name__}")
    return obj

