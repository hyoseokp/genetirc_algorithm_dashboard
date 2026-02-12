# genetirc_algorithm_dashboard

GA/CMA-ES optimization dashboard for inverse design.

## 1. Setup
### Install
```powershell
cd C:\Users\연구실\genetirc_algorithm_dashboard
python -m pip install -e .
```

### Configure paths
`configs/paths.yaml` is required and gitignored.

```powershell
Copy-Item configs\paths.example.yaml configs\paths.yaml
Get-Content configs\paths.yaml
```

Expected keys:
```yaml
forward_model_root: ${USERPROFILE}/data_CR_dashboard_toggles_nosmudge/code/CR_recon
forward_config_yaml: ${USERPROFILE}/data_CR_dashboard_toggles_nosmudge/code/CR_recon/configs/default.yaml
forward_checkpoint: ${USERPROFILE}/checkpoints/cnn_xattn_epoch_0499.pt
```

Quick check:
```powershell
Test-Path $env:USERPROFILE\data_CR_dashboard_toggles_nosmudge\code\CR_recon
Test-Path $env:USERPROFILE\data_CR_dashboard_toggles_nosmudge\code\CR_recon\configs\default.yaml
Test-Path $env:USERPROFILE\checkpoints\cnn_xattn_epoch_0499.pt
```

## 2. Process Overview
1. Dashboard starts FastAPI app (`gadash.run_dashboard` -> `create_app`).
2. User clicks `Run` on UI.
3. Dashboard calls `/api/run/start`, writes run-local config, starts optimizer subprocess (`python -m gadash.run_ga`).
4. Subprocess writes progress artifacts into `data/progress` or `data/progress/seed_{seed}`.
5. UI polls `/api/status`, `/api/topk/latest`, `/api/run/status` and renders charts/images.
6. User clicks `Stop`.
7. `/api/run/stop` terminates process, saves logs and final result package:
   - logs: `D:\optimization_log\seed_{seed}_{timestamp}.log`
   - results: `D:\optimization_results\<timestamp>_<optimizer>_<backend>_<seed-info>\*`

## 3. Terminal Commands
### Run dashboard
```powershell
python -m gadash.run_dashboard --progress-dir data/progress --paths configs/paths.yaml --device cuda --host 127.0.0.1 --port 8501
```

Open:
`http://127.0.0.1:8501/`

### Run optimizer directly (CLI)
```powershell
python -m gadash.run_ga --config configs/ga.yaml --paths configs/paths.yaml --progress-dir data/progress --device cuda
```

### Useful API calls (optional)
```powershell
curl http://127.0.0.1:8501/api/run/status
curl -X POST "http://127.0.0.1:8501/api/run/start?n_start=200&n_steps=2000&topk=50&seed=0"
curl -X POST "http://127.0.0.1:8501/api/run/stop"
curl -X POST "http://127.0.0.1:8501/api/run/reset"
```

## 4. Dashboard Usage
### Run
1. Set parameters (`n_start`, `n_steps`, `topk`, device, optimizer, optional multi-run count).
2. Click `Run optimization`.
3. Check `Run Status` and charts.

### Stop
1. Click `Stop`.
2. Process is terminated.
3. Log file is saved to `D:\optimization_log`.
4. Best structure/spectrum package is saved to `D:\optimization_results`.

### Reset
1. Click `Reset`.
2. Current progress is archived into `data/progress_archive`.
3. Working `data/progress` is cleared.

## 5. Important Outputs
### Progress artifacts
`data/progress/` and `data/progress/seed_{seed}/`
- `run_config.yaml`
- `run_meta_ga.json` or `run_meta_cmaes.json`
- `metrics_ga.jsonl` or `metrics_cmaes.jsonl`
- `topk_step-<step>.npz` (best-so-far)
- `topk_cur_step-<step>.npz` (current generation)
- `fdtd_rggb_step-<step>.npy` (when FDTD verification is available)

TopK NPZ keys:
- `seed16_topk` (K,16,16)
- `struct128_topk` (K,128,128)
- `metric_best_loss` / `metric_cur_loss` (K,)

### Stop-time result package
`D:\optimization_results\<folder>/`
- `best_structure_topk.npy`
- `surrogate_spectrum.npy` (if surrogate enabled)
- `fdtd_spectrum.npy` (if FDTD exists)
- `structure_visualization.png`
- `spectrum_comparison.png`
- `run_meta.json`

## 6. File/Module Roles
### Entry points
- `src/gadash/run_dashboard.py`: dashboard launcher (FastAPI + optional surrogate load)
- `src/gadash/run_ga.py`: optimizer launcher (GA/CMA-ES selection)

### Dashboard/API
- `src/gadash/dashboard_app.py`: all dashboard APIs (`/api/run/*`, `/api/status`, `/api/topk/*`) and stop-time result collection
- `src/gadash/dashboard_static/index.html`: frontend UI, charts, Run/Stop/Reset interactions

### Optimizers
- `src/gadash/ga_opt.py`: GA loop and progress writing
- `src/gadash/cma_opt.py`: CMA-ES loop and progress writing

### Logging/progress
- `src/gadash/progress_logger.py`: per-step metrics/meta persistence
- `src/gadash/optimization_logger.py`: run-final logging/export helper

### Models/loss/generation
- `src/gadash/surrogate_interface.py`: surrogate model interface and inference
- `src/gadash/generator.py`: structure generator logic
- `src/gadash/losses.py`: optimization loss composition
- `src/gadash/spectral.py`: spectral utilities

### FDTD/integration
- `src/gadash/fdtd_verify.py`: FDTD verification pipeline
- `src/gadash/fdtd_runner.py`: FDTD execution wrapper
- `src/gadash/lumapi_bridge.py`: Lumerical API bridge
- `src/gadash/gds_export.py`, `src/gadash/fdtd_scripts.py`: export/script helpers

## 7. Notes
- Multi-seed runs use isolated directories (`seed_0`, `seed_1000`, `seed_2000`, ...).
- Dashboard can merge multi-seed metrics/top-k for visualization.
- `configs/paths.yaml` is local-only (not committed).
