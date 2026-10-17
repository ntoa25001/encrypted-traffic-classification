"""Experiment 4: open-set. Top-K services are known; all other services appear only at test time.

Measures how well a max-softmax score separates unknown from known flows (AUROC, FPR@95%TPR).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etc_lib.data import Preprocessor, add_data_args, load_experiment
from etc_lib.models import build_model
from etc_lib.train import fit_classifier, get_device, open_set_metrics, predict_logits, save_json, seed_everything


def main():
    ap = argparse.ArgumentParser(); add_data_args(ap)
    ap.add_argument("--model", default="cnn", choices=["cnn", "lstm", "transformer"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--unknown-size", type=int, default=20000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seed_everything(a.seed); dev = get_device()

    exp = load_experiment(a, a.test_periods, open_world=True, unknown_size=a.unknown_size)
    pre = Preprocessor().fit(exp.train)
    train, val = pre(exp.train), pre(exp.val)
    model = build_model(a.model, exp.n_classes, train.ppi.shape[1], train.stats.shape[1])
    fit_classifier(model, train, val, dev, epochs=a.epochs)
    res = {"model": a.model, "topk_known": exp.n_classes, "tests": {}}
    for p, t in exp.tests.items():
        m = open_set_metrics(predict_logits(model, pre(t), dev), t.y, exp.unknown_label)
        res["tests"][p] = m
        print(f"{p}: AUROC {m['auroc_unknown_detection']:.3f}  FPR@95TPR {m['fpr_at_95tpr']:.3f}  "
              f"known macro-F1 {m['closed_set_macro_f1_on_known']:.3f}")
    out = a.out or f"results/openset_{a.model}.json"
    save_json(res, out); print("saved", out)


if __name__ == "__main__":
    main()
