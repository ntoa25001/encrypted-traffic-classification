# Full experiment suite on the XS dataset. Run from the project root:  powershell -File run_all.ps1
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "."
$T = @("--test-periods","W-2022-13","W-2022-16","W-2022-22","W-2022-34","W-2022-46")
python scripts/01_explore.py $T
python scripts/02_baseline_ml.py --model rf --features stats $T
python scripts/02_baseline_ml.py --model rf --features stats+ppi $T
python scripts/03_train_dl.py --model cnn $T
python scripts/03_train_dl.py --model cnn --no-stats $T
python scripts/03_train_dl.py --model lstm $T
python scripts/03_train_dl.py --model transformer $T
# pre-training and low-label
python scripts/04_pretrain_transformer.py
foreach ($f in 0.01, 0.1) {
  python scripts/03_train_dl.py --model cnn --label-fraction $f --test-periods W-2022-13
  python scripts/03_train_dl.py --model transformer --label-fraction $f --test-periods W-2022-13
  python scripts/03_train_dl.py --model transformer --label-fraction $f --test-periods W-2022-13 --init-from results/pretrain_transformer.pt
}
# open-set
python scripts/05_open_set.py --model cnn --test-periods W-2022-13 W-2022-34
# exporter-change effect (train before week 10, test after)
python scripts/03_train_dl.py --model cnn --train-period W-2022-6 --test-periods W-2022-12 --tag _preW10
python scripts/99_collect.py > results/summary.md
