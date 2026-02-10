# genetirc_algorithm_dashboard

Genetic Algorithm (GA) optimization loop driven by a surrogate model, with a local dashboard.

## Install

```powershell
cd C:\Users\연구실\genetirc_algorithm_dashboard
python -m pip install -e .
```

## Configure Paths (Required)
Create `configs/paths.yaml` (copy from `configs/paths.example.yaml`, this file is gitignored):

```yaml
checkpoint_path: C:/Users/연구실/checkpoints/cnn_xattn_epoch_0499.pt
cr_recon_root: C:/Users/연구실/data_CR/code/CR_recon
```

`cr_recon_root` must contain the CR_recon Python package/code used to build the surrogate model.

## Run GA (CLI)

```powershell
python -m scripts.run_ga --config configs/ga.yaml --progress-dir data/progress --device cuda
```

## Run Dashboard

```powershell
python -m scripts.run_dashboard --progress-dir data/progress --host 127.0.0.1 --port 8501
```

Open: `http://127.0.0.1:8501/`

## Notes
- Dashboard can start/stop/reset GA runs.
- Progress files written to `data/progress/`:
  - `metrics.jsonl`
  - `topk_step-<gen>.npz` (best-so-far)
  - `topk_cur_step-<gen>.npz` (current generation)
  - `run_meta.json`
