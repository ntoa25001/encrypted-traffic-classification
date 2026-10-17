"""Experiments 1b / 2 / 3: train CNN, LSTM or Transformer (optionally from a pre-trained checkpoint).

    # closed-set baseline + concept drift (several test periods in one run)
    python scripts/03_train_dl.py --model cnn --test-periods W-2022-13 W-2022-16 W-2022-24 W-2022-40
    # low-label (10 % of train labels), from scratch vs pre-trained
    python scripts/03_train_dl.py --model transformer --label-fraction 0.1
    python scripts/03_train_dl.py --model transformer --label-fraction 0.1 --init-from results/pretrain_transformer.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from etc_lib.data import Preprocessor, add_data_args, load_experiment, stratified_fraction
from etc_lib.models import build_model, count_params
from etc_lib.train import (closed_set_metrics, fit_classifier, get_device, inference_latency, predict_logits,
                           save_json, seed_everything)


def main():
    ap = argparse.ArgumentParser(); add_data_args(ap)
    ap.add_argument("--model", default="cnn", choices=["cnn", "lstm", "transformer"])
    ap.add_argument("--no-stats", action="store_true", help="Ablation: packet sequence only, no flow statistics")
    ap.add_argument("--label-fraction", type=float, default=1.0)
    ap.add_argument("--init-from", default=None, help="Pre-trained checkpoint (from 04_pretrain_transformer.py)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seed_everything(a.seed); dev = get_device(); print("device:", dev)

    exp = load_experiment(a, a.test_periods)
    pre = Preprocessor().fit(exp.train)
    train = pre(stratified_fraction(exp.train, a.label_fraction, a.seed)); val = pre(exp.val)
    tests = {k: pre(v) for k, v in exp.tests.items()}

    model = build_model(a.model, exp.n_classes, train.ppi.shape[1], train.stats.shape[1], use_stats=not a.no_stats)
    if a.init_from:
        ck = torch.load(a.init_from, map_location="cpu")
        missing = model.load_state_dict(ck["state_dict"], strict=False)
        print("loaded pre-trained weights; classifier head re-initialised. missing:", len(missing.missing_keys))
    print(f"{a.model}: {count_params(model):,} params, {len(train):,} train flows, {exp.n_classes} classes")

    fit = fit_classifier(model, train, val, dev, epochs=a.epochs, lr=a.lr, bs=a.batch_size)
    res = {"model": a.model, "use_stats": not a.no_stats, "label_fraction": a.label_fraction,
           "pretrained": bool(a.init_from), "params": count_params(model), "train_flows": len(train),
           "train_period": a.train_period, **fit, "tests": {}}
    for p, t in tests.items():
        m = closed_set_metrics(t.y, predict_logits(model, t, dev).argmax(1), exp.class_names)
        res["tests"][p] = m
        print(f"{p}: acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}")
    res.update(inference_latency(model, next(iter(tests.values())), dev))
    name = f"dl_{a.model}{'_nostats' if a.no_stats else ''}{'_pre' if a.init_from else ''}_f{a.label_fraction}{a.tag}"
    out = a.out or f"results/{name}.json"
    save_json(res, out); print("saved", out)


if __name__ == "__main__":
    main()
