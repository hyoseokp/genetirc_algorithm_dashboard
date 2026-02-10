from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    checkpoint_path: str | None = None
    cr_recon_root: str | None = None


@dataclass(frozen=True)
class GAConfig:
    population: int = 128
    generations: int = 200
    elite: int = 8
    tournament_k: int = 5
    crossover_alpha: float = 0.5
    mutation_sigma: float = 0.15
    mutation_p: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class DesignConfig:
    seed_size: int = 16
    struct_size: int = 128
    enforce_symmetry: bool = True


@dataclass(frozen=True)
class GeneratorConfig:
    backend: str = "rule_mfs"  # currently only rule_mfs
    blur_sigma: float = 2.0
    threshold: float = 0.45
    mfs_radius_px: int = 8
    mfs_iters: int = 2


@dataclass(frozen=True)
class RobustnessConfig:
    samples: int = 2


@dataclass(frozen=True)
class SpectraConfig:
    channels: int = 30
    wavelength_min_nm: float = 400.0
    wavelength_max_nm: float = 700.0
    rgb_weights: dict[str, float] | None = None  # {'R':1,'G':2,'B':1}


@dataclass(frozen=True)
class LossConfig:
    spec_weight: float = 1.0
    reg_weight: float = 0.02
    fill_min: float = 0.2
    fill_weight: float = 0.2


@dataclass(frozen=True)
class IOConfig:
    topk: int = 8
    print_every: int = 10


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig = PathsConfig()
    ga: GAConfig = GAConfig()
    design: DesignConfig = DesignConfig()
    generator: GeneratorConfig = GeneratorConfig()
    robustness: RobustnessConfig = RobustnessConfig()
    spectra: SpectraConfig = SpectraConfig(rgb_weights={"R": 1.0, "G": 2.0, "B": 1.0})
    loss: LossConfig = LossConfig()
    io: IOConfig = IOConfig()


def _deep_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _expand_path(s: str | None) -> str | None:
    if s is None:
        return None
    # Allow ASCII-only configs using ${USERPROFILE} / %USERPROFILE%.
    s2 = os.path.expandvars(str(s))
    s2 = os.path.expanduser(s2)
    return s2


def load_config(main_cfg_path: str | Path, paths_cfg_path: str | Path | None = None) -> AppConfig:
    main = load_yaml(main_cfg_path)
    paths = load_yaml(paths_cfg_path) if paths_cfg_path else {}

    rgb_w = _deep_get(main, "spectra", "rgb_weights", default=None)
    if rgb_w is None:
        rgb_w = {"R": 1.0, "G": 2.0, "B": 1.0}

    return AppConfig(
        paths=PathsConfig(
            checkpoint_path=_expand_path(_deep_get(paths, "checkpoint_path", default=None)),
            cr_recon_root=_expand_path(_deep_get(paths, "cr_recon_root", default=None)),
        ),
        ga=GAConfig(
            population=int(_deep_get(main, "ga", "population", default=128)),
            generations=int(_deep_get(main, "ga", "generations", default=200)),
            elite=int(_deep_get(main, "ga", "elite", default=8)),
            tournament_k=int(_deep_get(main, "ga", "tournament_k", default=5)),
            crossover_alpha=float(_deep_get(main, "ga", "crossover_alpha", default=0.5)),
            mutation_sigma=float(_deep_get(main, "ga", "mutation_sigma", default=0.15)),
            mutation_p=float(_deep_get(main, "ga", "mutation_p", default=0.2)),
            seed=int(_deep_get(main, "ga", "seed", default=0)),
        ),
        design=DesignConfig(
            seed_size=int(_deep_get(main, "design", "seed_size", default=16)),
            struct_size=int(_deep_get(main, "design", "struct_size", default=128)),
            enforce_symmetry=bool(_deep_get(main, "design", "enforce_symmetry", default=True)),
        ),
        generator=GeneratorConfig(
            backend=str(_deep_get(main, "generator", "backend", default="rule_mfs")),
            blur_sigma=float(_deep_get(main, "generator", "blur_sigma", default=2.0)),
            threshold=float(_deep_get(main, "generator", "threshold", default=0.45)),
            mfs_radius_px=int(_deep_get(main, "generator", "mfs_radius_px", default=8)),
            mfs_iters=int(_deep_get(main, "generator", "mfs_iters", default=2)),
        ),
        robustness=RobustnessConfig(samples=int(_deep_get(main, "robustness", "samples", default=2))),
        spectra=SpectraConfig(
            channels=int(_deep_get(main, "spectra", "channels", default=30)),
            wavelength_min_nm=float(_deep_get(main, "spectra", "wavelength_min_nm", default=400.0)),
            wavelength_max_nm=float(_deep_get(main, "spectra", "wavelength_max_nm", default=700.0)),
            rgb_weights={k: float(v) for k, v in rgb_w.items()},
        ),
        loss=LossConfig(
            spec_weight=float(_deep_get(main, "loss", "spec_weight", default=1.0)),
            reg_weight=float(_deep_get(main, "loss", "reg_weight", default=0.02)),
            fill_min=float(_deep_get(main, "loss", "fill_min", default=0.2)),
            fill_weight=float(_deep_get(main, "loss", "fill_weight", default=0.2)),
        ),
        io=IOConfig(
            topk=int(_deep_get(main, "io", "topk", default=8)),
            print_every=int(_deep_get(main, "io", "print_every", default=10)),
        ),
    )
