from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


class SurrogateError(RuntimeError):
    pass


@dataclass
class SurrogateOutput:
    # (B,3,C)
    pred_rgbc: torch.Tensor


class CRReconSurrogate:
    """Thin wrapper around external CR_recon code + checkpoint.

    Expects:
    - paths.cr_recon_root: directory containing CR_recon python package
    - paths.checkpoint_path: .pt checkpoint file

    This is intentionally minimal and may require adjusting if CR_recon API differs.
    """

    def __init__(self, cr_recon_root: str, checkpoint_path: str, device: str = "cpu"):
        self.cr_recon_root = str(cr_recon_root)
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device)

        if not Path(self.checkpoint_path).exists():
            raise SurrogateError(f"checkpoint not found: {self.checkpoint_path}")
        if not Path(self.cr_recon_root).exists():
            raise SurrogateError(f"cr_recon_root not found: {self.cr_recon_root}")

        # Put CR_recon on sys.path (root is expected to contain the package code)
        if self.cr_recon_root not in sys.path:
            sys.path.insert(0, self.cr_recon_root)

        # Lazy import: these module paths may need to be edited to match actual CR_recon code.
        try:
            self._cr = importlib.import_module("CR_recon")
        except Exception as e:
            raise SurrogateError(f"failed to import CR_recon from {self.cr_recon_root}: {e}")

        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self):
        """Try a few common builder APIs.

        If this fails, adjust this function for your CR_recon package.
        """
        # Common patterns:
        # - CR_recon.models.build_model(...)
        # - CR_recon.model.build(...)
        # - torch.load(state_dict) + model class
        candidates = [
            ("CR_recon.models", "build_model"),
            ("CR_recon.models", "create_model"),
            ("CR_recon.model", "build_model"),
        ]
        last_err: Exception | None = None
        for mod_name, fn_name in candidates:
            try:
                mod = importlib.import_module(mod_name)
                fn = getattr(mod, fn_name)
                model = fn()
                sd = torch.load(self.checkpoint_path, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                model.load_state_dict(sd, strict=False)
                return model
            except Exception as e:
                last_err = e

        # Fallback: try loading full model object
        try:
            obj = torch.load(self.checkpoint_path, map_location="cpu")
            if hasattr(obj, "to"):
                return obj
        except Exception as e:
            last_err = e

        raise SurrogateError(f"could not build CR_recon model from checkpoint: {last_err}")

    @torch.inference_mode()
    def predict(self, struct01: torch.Tensor) -> SurrogateOutput:
        """struct01: (B,1,H,W) -> pred_rgbc (B,3,C)"""
        x = struct01.to(self.device)
        out = self.model(x)

        # Heuristics: allow model to return dict/tuple
        if isinstance(out, dict):
            # pick first tensor-like
            for v in out.values():
                if torch.is_tensor(v):
                    out = v
                    break
        if isinstance(out, (tuple, list)):
            out = out[0]

        if not torch.is_tensor(out):
            raise SurrogateError(f"unexpected model output type: {type(out)}")

        # Try to coerce shapes to (B,3,C)
        if out.dim() == 2:
            # (B, 3*C)
            B, D = out.shape
            if D % 3 != 0:
                raise SurrogateError(f"cannot reshape output of shape {out.shape} to (B,3,C)")
            C = D // 3
            out = out.view(B, 3, C)
        elif out.dim() == 3:
            # assume already (B,3,C)
            pass
        elif out.dim() == 4:
            # (B,3,?,C) -> merge
            B = out.shape[0]
            out = out.view(B, 3, -1).mean(dim=2)
        else:
            raise SurrogateError(f"unexpected output dims: {out.shape}")

        return SurrogateOutput(pred_rgbc=out.clamp(0.0, 1.0))
