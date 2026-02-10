# genetirc_algorithm_dashboard

Surrogate 기반 Genetic Algorithm (GA) 최적화 + 로컬 대시보드.

주의: 기존 프로젝트(`Inverse_design_CR`)에도 `scripts.*` 모듈이 있어서, 이 프로젝트는 `python -m gadash.run_dashboard` / `python -m gadash.run_ga`로 실행해야 충돌이 안 남.

## 0) 위치

```powershell
cd C:\Users\연구실\genetirc_algorithm_dashboard
```

## 1) 설치 (처음 1회)

```powershell
python -m pip install -e .
```

## 2) Path 설정 (필수)

`configs/paths.yaml`을 만들어서 surrogate 체크포인트와 `CR_recon` 코드 경로를 지정해야 함.
(이 파일은 `.gitignore` 되어 있어 로컬에만 존재)

한글 경로가 터미널에서 깨지는 경우가 있어서, `${USERPROFILE}` 환경변수를 쓰는 방식을 기본으로 둠.

### 2-1) 생성

```powershell
Copy-Item configs\paths.example.yaml configs\paths.yaml
```

### 2-2) 내용 확인/수정

```powershell
Get-Content configs\paths.yaml
```

예시:
```yaml
checkpoint_path: ${USERPROFILE}/checkpoints/cnn_xattn_epoch_0499.pt
cr_recon_root: ${USERPROFILE}/data_CR_dashboard_toggles_nosmudge/code/CR_recon
```

### 2-3) 경로 존재 확인

```powershell
Test-Path $env:USERPROFILE\checkpoints\cnn_xattn_epoch_0499.pt
Test-Path $env:USERPROFILE\data_CR_dashboard_toggles_nosmudge\code\CR_recon
```

둘 다 `True`여야 정상.

## 3) GA 실행 (CLI)

```powershell
python -m gadash.run_ga --config configs/ga.yaml --paths configs/paths.yaml --progress-dir data/progress --device cuda
```

## 4) 대시보드 실행

```powershell
python -m gadash.run_dashboard --progress-dir data/progress --config configs/ga.yaml --paths configs/paths.yaml --host 127.0.0.1 --port 8501
```

브라우저에서 열기: `http://127.0.0.1:8501/`

대시보드에서 `Run/Stop/Reset`으로 GA를 제어할 수 있음.

## 출력물

`data/progress/`
- `metrics.jsonl`
- `topk_step-<gen>.npz` (best-so-far)
- `topk_cur_step-<gen>.npz` (current generation)
- `run_meta.json`