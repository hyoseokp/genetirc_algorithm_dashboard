# Data Storage Structure: Best Seeds, Structures, Spectra, and FDTD Results

## Overview

The genetic algorithm optimization produces multiple types of data at each step. This document explains where each type is stored and how to locate it.

## Single Run (seed=0, default)

All files go to `data/progress/` directory:

```
data/progress/
├── run_meta.json                    # Optimization metadata (timestamps, config)
├── run_meta_ga.json                 # GA-specific metadata
├── metrics_ga.jsonl                 # GA metrics per step (loss, fill, etc)
├── run_config.yaml                  # Configuration used for this run
│
├── topk_step-0.npz                  # Best structures found so far (step 0)
├── topk_step-10.npz                 # Best structures found so far (step 10)
├── topk_step-100.npz                # ... continues
│
├── topk_cur_step-0.npz              # Current generation structures (step 0)
├── topk_cur_step-10.npz             # Current generation structures (step 10)
│
├── fdtd_rggb_step-0.npy             # FDTD-verified spectrum (step 0, if FDTD verify=ON)
├── fdtd_rggb_step-10.npy            # FDTD-verified spectrum (step 10)
│
├── fdtd_meta.json                   # FDTD verification metadata
└── purity_step-0.json               # Spectral purity analysis (optional)
```

## Multi-Start Runs (seed=0, 1000, 2000, ...)

Each seed runs independently with isolated files:

```
data/progress/                       # seed=0 (default location)
├── run_config.yaml
├── metrics_ga.jsonl
├── topk_step-*.npz
├── topk_cur_step-*.npz
├── fdtd_rggb_step-*.npy
│
├── seed_1000/                       # seed=1000 (isolated)
│   ├── run_config.yaml
│   ├── metrics_ga.jsonl
│   ├── topk_step-*.npz
│   ├── topk_cur_step-*.npz
│   └── fdtd_rggb_step-*.npy
│
└── seed_2000/                       # seed=2000 (isolated)
    ├── run_config.yaml
    ├── metrics_ga.jsonl
    ├── topk_step-*.npz
    ├── topk_cur_step-*.npz
    └── fdtd_rggb_step-*.npy

finetune_dataset.npz                 # Cumulative dataset from ALL seeds
```

---

## File Contents and Data Types

### 1. **Best Structures**
**Files:** `topk_step-{N}.npz` and `topk_cur_step-{N}.npz`

**Location:**
- Single run: `data/progress/topk_step-100.npz`
- Multi-seed: `data/progress/seed_1000/topk_step-100.npz`

**Content:**
```python
import numpy as np
z = np.load('topk_step-100.npz')

# Best-so-far top-K structures (K=10 by default, configurable with 'topk' parameter)
struct128_topk = z['struct128_topk']        # Shape: (K, 128, 128)
                                             # dtype: uint8
                                             # values: 0 (air) or 1 (silicon nitride)

# Current generation top-K (generation 100 only, not cumulative)
# Same format as above, but only from current generation
# File: topk_cur_step-100.npz

# Loss values for each structure
metric_best_loss = z['metric_best_loss']    # Shape: (K,)
                                             # dtype: float32
                                             # Lower is better
```

**Visualization:** Displayed in Dashboard as grayscale thumbnails

---

### 2. **Surrogate-Generated Spectra** (Forward Model)
**Type:** Not directly saved to disk (computed on demand)

**Access Points:**
1. **API Endpoint:** `/api/topk/{step}/{idx}/spectrum?mode=best&seed={seed}`
   ```
   GET /api/topk/100/5/spectrum?mode=best&seed=1000
   Response: {
     "rgb": [[R_data], [G_data], [B_data]],  # 3x(wavelength_points,)
     "seed": 1000,
     "n_channels": 30
   }
   ```

2. **Dashboard:** Click on any structure thumbnail → Spectrum displayed in graph

**Computation:**
- Surrogate model loads from `checkpoints/cnn_xattn_epoch_0499.pt`
- Input: 128x128 binary structure
- Output: 3x30 RGB spectrum (400-700 nm)
- Cached to avoid recomputation

**File References:**
- Config: `configs/paths.yaml` (forward_model_root, forward_checkpoint, forward_config_yaml)
- Model class: `src/gadash/surrogate_interface.py` (CRReconSurrogate)

---

### 3. **FDTD-Verified Spectra**
**Files:** `fdtd_rggb_step-{N}.npy`

**Location:**
- Single run: `data/progress/fdtd_rggb_step-10.npy`
- Multi-seed: `data/progress/seed_1000/fdtd_rggb_step-10.npy`

**Content:**
```python
import numpy as np
fdtd_data = np.load('fdtd_rggb_step-10.npy')
# Shape: (K, 2, 2, C)
# dtype: float32
# Format: RGGB (Red, Green, Green, Blue) Bayer pattern
# Values: [0, 1] transmission spectrum
# C: number of frequency points (typically 30)

# Access as:
# fdtd_data[0, 0, 0, :] = Red channel of 1st structure
# fdtd_data[0, 0, 1, :] = Green channel of 1st structure
# fdtd_data[0, 1, 1, :] = Blue channel of 1st structure
```

**When Generated:**
- Only if `fdtd_verify = on` in dashboard
- Only every N steps (configurable with `fdtd_every` parameter)
- Each simulation takes ~5-10 minutes per structure

**Metadata:**
```json
// fdtd_meta.json
{
  "step": 10,
  "k": 10,
  "out_dir": "D:\\gadash_fdtd_results\\fdtd_20260211_145800"
}
```

**Associated Temporary Files:**
- GDS files: `{out_dir}/gds/structure_*.gds` (can be deleted after verification)
- Lumerical results: `{out_dir}/structure_*/` (raw simulation data)

---

### 4. **Best Loss Values**
**Location:** Stored inside topk NPZ files

**File:** `data/progress/topk_step-100.npz`

**Content:**
```python
import numpy as np
z = np.load('topk_step-100.npz')

metric_best_loss = z['metric_best_loss']    # Shape: (K,)
                                             # dtype: float32
                                             # Example: [0.0234, 0.0251, 0.0289, ...]

# Total loss components (also available):
metric_cur_loss = z.get('metric_cur_loss')  # Current generation loss
```

**Visualization:**
- Dashboard: Each thumbnail shows loss below it
- Loss graph (bottom-left): minimum loss vs. generation

---

### 5. **Metrics Log**
**File:** `metrics_ga.jsonl` (JSON Lines format)

**Location:**
- Single run: `data/progress/metrics_ga.jsonl`
- Multi-seed: `data/progress/seed_1000/metrics_ga.jsonl`

**Content (1 line per generation):**
```json
{"ts": "2026-02-11T14:58:00+00:00", "step": 0, "loss_total": 0.1234, "loss_spec": 0.0234, "loss_fill": 0.05, "fill": 0.45}
{"ts": "2026-02-11T14:59:00+00:00", "step": 1, "loss_total": 0.1100, "loss_spec": 0.0210, "loss_fill": 0.05, "fill": 0.47}
```

**Visualization:**
- Dashboard: Loss curve, Fill ratio curve

---

## Multi-Seed Merged Data (Dashboard)

When running multiple seeds, dashboard shows **merged** results:

### 1. `/api/topk/latest` (no seed parameter)
```json
{
  "step": 100,
  "k": 10,
  "merged": true,
  "seed_steps": {"0": 100, "1000": 98, "2000": 100},
  "seed_origins": [1000, 0, 2000, 1000, 0, 1000, 2000, 0, 1000, 2000],
  "images": ["/api/topk/merged/0.png", "/api/topk/merged/1.png", ...],
  "metrics": {
    "metric_best_loss": [0.0234, 0.0251, 0.0289, ...]
  }
}
```

**seed_origins:** Which seed each top-10 structure came from

### 2. Spectrum Loading for Merged Results
When you click a thumbnail in merged view:
- UI extracts seed from `seed_origins` array
- Calls `/api/topk/{step}/{idx}/spectrum?seed={seed}`
- Displays spectrum from that specific seed's surrogate model

---

## Fine-Tuning Dataset (Optional)

**File:** `data/finetune_dataset.npz`

**Content:** Cumulative dataset of FDTD-verified structures
```python
import numpy as np
z = np.load('finetune_dataset.npz')

struct_list = z['struct_list']          # Shape: (N, 128, 128), dtype: uint8
spectrum_fdtd = z['spectrum_fdtd']      # Shape: (N, 2, 2, C), dtype: float32
```

**Usage:** Can be used to fine-tune surrogate model on FDTD-verified data

**Accumulation:** Grows across all seeds and all FDTD verification runs

---

## Archive and Reset

When you click "Reset" in dashboard:

**Before Reset:**
```
data/progress/
├── topk_step-*.npz
├── metrics_ga.jsonl
├── seed_1000/
│   ├── topk_step-*.npz
│   └── metrics_ga.jsonl
└── seed_2000/
```

**After Reset (files archived):**
```
data/progress/                    # cleared
data/progress_archive/
├── seed_0_20260211_145800/       # archived
│   ├── topk_step-*.npz
│   ├── metrics_ga.jsonl
│   └── fdtd_rggb_step-*.npy
├── seed_1000_20260211_145800/
└── seed_2000_20260211_145800/

finetune_dataset.npz              # NOT deleted (cumulative)
```

---

## Summary Table

| Data Type | File Pattern | Location | Format | Size |
|-----------|-------------|----------|--------|------|
| Best Structures | `topk_step-{N}.npz` | progress_dir | NPZ | ~2 MB per step |
| Current Gen Structures | `topk_cur_step-{N}.npz` | progress_dir | NPZ | ~2 MB per step |
| FDTD Spectra | `fdtd_rggb_step-{N}.npy` | progress_dir | NPY | ~1 MB per step |
| Metrics Log | `metrics_ga.jsonl` | progress_dir | JSONL | ~1 KB per step |
| Surrogate Spectra | (computed on demand) | memory cache | - | - |
| Fine-tuning Dataset | `finetune_dataset.npz` | data/ | NPZ | grows ~1 MB per FDTD run |
| Metadata | `run_meta.json` | progress_dir | JSON | <10 KB |

---

## Accessing Your Data Programmatically

```python
import numpy as np
import json
from pathlib import Path

progress_dir = Path("data/progress/seed_1000")

# Load best structures from step 100
z = np.load(progress_dir / "topk_step-100.npz")
struct = z['struct128_topk']           # (K, 128, 128)
loss = z['metric_best_loss']           # (K,)

# Load FDTD verified spectra
fdtd = np.load(progress_dir / "fdtd_rggb_step-100.npy")  # (K, 2, 2, 30)

# Load metrics
with open(progress_dir / "metrics_ga.jsonl") as f:
    for line in f:
        data = json.loads(line)
        print(f"Step {data['step']}: loss={data['loss_total']:.4f}")

# Load fine-tuning dataset
z_ft = np.load("data/finetune_dataset.npz")
all_structs = z_ft['struct_list']      # (N, 128, 128)
all_spectra = z_ft['spectrum_fdtd']    # (N, 2, 2, 30)
```

---

## Key Takeaways

✓ **Single Run:** Everything in `data/progress/`
✓ **Multi-Start:** Each seed in `data/progress/seed_{N}/`
✓ **Structures:** `topk_step-{N}.npz` (best cumulative), `topk_cur_step-{N}.npz` (current only)
✓ **Surrogate Spectra:** Computed on demand via API, cached in memory
✓ **FDTD Spectra:** Saved to `fdtd_rggb_step-{N}.npy` (only if FDTD verify=ON)
✓ **Loss Values:** Stored inside NPZ files or metrics log
✓ **Multi-Seed Dashboard:** Merges results from all seeds, tracks `seed_origins` for per-seed spectrum access
