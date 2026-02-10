# genetirc_algorithm_dashboard

Surrogate 湲곕컲 Genetic Algorithm (GA) 理쒖쟻??+ 濡쒖뺄 ??쒕낫??

## 0) ?꾩튂

```powershell
cd C:\Users\?곌뎄??genetirc_algorithm_dashboard
```

## 1) ?ㅼ튂 (泥섏쓬 1??

```powershell
python -m pip install -e .
```

## 2) Path ?ㅼ젙 (?꾩닔)

`configs/paths.yaml`??留뚮뱾?댁꽌 surrogate 泥댄겕?ъ씤?몄? `CR_recon` 肄붾뱶 寃쎈줈瑜?吏?뺥빐????
(???뚯씪? `.gitignore` ?섏뼱 ?덉뼱 濡쒖뺄?먮쭔 議댁옱)

### 2-1) ?앹꽦

```powershell
Copy-Item configs\paths.example.yaml configs\paths.yaml
```

### 2-2) ?댁슜 ?뺤씤/?섏젙

```powershell
Get-Content configs\paths.yaml
```

?덉떆:
```yaml
checkpoint_path: C:/Users/?곌뎄??checkpoints/cnn_xattn_epoch_0499.pt
cr_recon_root: C:/Users/?곌뎄??data_CR_dashboard_toggles_nosmudge/code/CR_recon
```

### 2-3) 寃쎈줈 議댁옱 ?뺤씤

```powershell
Test-Path C:\Users\?곌뎄??checkpoints\cnn_xattn_epoch_0499.pt
Test-Path C:\Users\?곌뎄??data_CR_dashboard_toggles_nosmudge\code\CR_recon
```

????`True`?ъ빞 ?뺤긽.

## 3) GA ?ㅽ뻾 (CLI)

```powershell
python -m scripts.run_ga --config configs/ga.yaml --paths configs/paths.yaml --progress-dir data/progress --device cuda
```

## 4) ??쒕낫???ㅽ뻾

```powershell
python -m scripts.run_dashboard --progress-dir data/progress --config configs/ga.yaml --paths configs/paths.yaml --host 127.0.0.1 --port 8501
```

釉뚮씪?곗??먯꽌 ?닿린: `http://127.0.0.1:8501/`

??쒕낫?쒖뿉??`Run/Stop/Reset`?쇰줈 GA瑜??쒖뼱?????덉쓬.

## 異쒕젰臾?
`data/progress/`
- `metrics.jsonl`
- `topk_step-<gen>.npz` (best-so-far)
- `topk_cur_step-<gen>.npz` (current generation)
- `run_meta.json`
