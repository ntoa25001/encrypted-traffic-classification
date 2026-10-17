# Encrypted traffic classification on CESNET-TLS-Year22

Course project skeleton: classical ML vs deep learning vs self-supervised pre-training on
**CESNET-TLS-Year22** (TLS flows from a 100 Gbps ISP backbone, 180 services, all of 2022).
Input is the first 30 packets of each flow (inter-packet time, direction, size) plus flow statistics.
There is **no payload** in this dataset, so everything below works on side-channel features only.

## 1. Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
(For GPU, install the CUDA build of PyTorch first from pytorch.org.)

## 2. Smoke test (no download, ~1 minute)

```
python scripts/03_train_dl.py --synthetic --model cnn --epochs 3 --test-periods W-2022-13 W-2022-16
```
Uses random synthetic data with the same shapes as CESNET. Use it to check your environment.
Scores on synthetic data mean nothing.

## 3. Download the dataset

See **[DATA.md](DATA.md)** for all links (automatic and manual download).

```
python scripts/00_download.py --data-root data --size XS
python scripts/01_explore.py  --data-root data --size XS
```
`XS` = 10 M flows sampled over the year (start here), `S`=25 M, `M`=50 M, `L`=100 M.
The download size is printed first; re-running resumes an interrupted download.
Data hosted by CESNET (via the `cesnet-datazoo` package); also on Zenodo: https://zenodo.org/records/10608607

## 4. Experiments (each maps to a section of the report)

| # | Question | Command |
|---|---|---|
| 1 | Classical ML baseline | `02_baseline_ml.py --model rf --features stats` and `--features stats+ppi` |
| 2 | Deep learning | `03_train_dl.py --model cnn` / `lstm` / `transformer` |
| 3 | What is the model relying on? | `03_train_dl.py --model cnn --no-stats` (packet sequence only) |
| 4 | Concept drift | `--test-periods W-2022-13 W-2022-16 W-2022-22 W-2022-34 W-2022-46` (train once, test on later weeks) |
| 5 | Few labels | `--label-fraction 0.01 / 0.1`, from scratch vs `--init-from results/pretrain_transformer.pt` |
| 6 | Pre-training | `04_pretrain_transformer.py` (masked-packet reconstruction, no labels) |
| 7 | Unknown services | `05_open_set.py` (top-K known, the rest unknown; max-softmax AUROC / FPR@95TPR) |

Run everything: `powershell -File run_all.ps1` (Windows) or `bash run_all.sh`. Collect a table: `python scripts/99_collect.py`.
Results are JSON files in `results/` (per-class report, confusion-relevant metrics, latency, #params).

## 5. Things to know (also good answers for Q&A)

* **Week-10 exporter change.** In week 10 (7-13 Mar 2022) CESNET updated the flow exporter, which changed the
  distribution of packet-sequence data (see the dataset paper, https://doi.org/10.1038/s41597-024-03927-4).
  Default train period is therefore `W-2022-12` (after the change). To show the effect, train on `W-2022-6` and test on `W-2022-12`.
* **Shortcut features are excluded on purpose.** The dataset also has SNI, JA3, IPs and destination ASN.
  They identify services almost directly, so using them would make the task trivial and is not "classifying encrypted traffic".
  They are never loaded (`return_other_fields=False`). A good extra ablation for the report: add SNI/JA3 and show the score jump.
* **Split is by time, not random**, which is what makes the drift experiment honest. Random splits leak near-duplicate flows.
* **Top-K classes.** `--topk 50` keeps the 50 most frequent services; rare services have too few samples.
  Changing `--topk` or the period changes the class list, so always report the exact settings (they are saved in each JSON).
* **No ET-BERT here.** ET-BERT/YaTC need payload bytes, which CESNET does not contain. `04_pretrain_transformer.py` is the flow-level
  analogue (BERT-style masking over packets). Explain this difference in the report.
* Macro-F1 is the main metric (classes are imbalanced); accuracy is reported too.

## 6. Layout

```
etc_lib/data.py        only module touching cesnet-datazoo; scaling; label-fraction sampling
etc_lib/models.py      PpiCNN, PpiLSTM, PpiTransformer
etc_lib/train.py       training loop, metrics, open-set metrics, latency
scripts/               00 download, 01 explore, 02 ML, 03 DL, 04 pretrain, 05 open-set, 99 collect
```
