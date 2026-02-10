# genetirc_algorithm_dashboard

GA (Genetic Algorithm) optimization loop + dashboard.

Dashboard UI/UX is copied from `C:\Users\연구실\Inverse_design_CR` and adapted minimally for GA.

## Install (once)

```powershell
cd C:\Users\연구실\genetirc_algorithm_dashboard
python -m pip install -e .
```

## Configure Paths (required)

Create `configs/paths.yaml` (gitignored):

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

Quick path check:
```powershell
Test-Path $env:USERPROFILE\data_CR_dashboard_toggles_nosmudge\code\CR_recon
Test-Path $env:USERPROFILE\data_CR_dashboard_toggles_nosmudge\code\CR_recon\configs\default.yaml
Test-Path $env:USERPROFILE\checkpoints\cnn_xattn_epoch_0499.pt
```

## Run Dashboard

```powershell
python -m gadash.run_dashboard --progress-dir data/progress --paths configs/paths.yaml --device cuda --host 127.0.0.1 --port 8501
```

Open: `http://127.0.0.1:8501/`

Dashboard has Run/Stop/Reset buttons. Run triggers GA in a subprocess and writes progress artifacts.

## Run GA (CLI only)

```powershell
python -m gadash.run_ga --config configs/ga.yaml --paths configs/paths.yaml --progress-dir data/progress --device cuda
```

## Outputs

`data/progress/`
- `run_meta.json`
- `metrics.jsonl` (JSONL, key `step` = generation)
- `topk_step-<step>.npz` (best-so-far)
- `topk_cur_step-<step>.npz` (current generation)

TopK npz keys (dashboard-compatible):
- `seed16_topk` (K,16,16) float32
- `struct128_topk` (K,128,128) uint8 (0/1)
- `metric_best_loss` or `metric_cur_loss` (K,)