"""Experiment 1a: classical ML baseline on flow statistics + flattened packet sequence."""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from etc_lib.data import add_data_args, load_experiment, stratified_fraction
from etc_lib.train import closed_set_metrics, save_json


def flat(s, use_ppi=True):
    parts = [s.stats]
    if use_ppi:
        parts.append(s.ppi.reshape(len(s), -1))
    return np.concatenate(parts, 1)


def main():
    ap = argparse.ArgumentParser(); add_data_args(ap)
    ap.add_argument("--model", default="rf", choices=["rf", "hgb"])
    ap.add_argument("--features", default="stats+ppi", choices=["stats", "stats+ppi"],
                    help="'stats' = only flow statistics (classic ML); 'stats+ppi' adds the 30-packet sequence")
    ap.add_argument("--label-fraction", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    exp = load_experiment(a, a.test_periods)
    train = stratified_fraction(exp.train, a.label_fraction, a.seed)
    use_ppi = a.features == "stats+ppi"
    clf = (RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=a.seed, class_weight="balanced_subsample")
           if a.model == "rf" else HistGradientBoostingClassifier(max_iter=200, random_state=a.seed))
    t0 = time.time(); clf.fit(flat(train, use_ppi), train.y); fit_s = time.time() - t0
    res = {"model": f"{a.model}[{a.features}]", "label_fraction": a.label_fraction, "train_flows": len(train),
           "train_period": a.train_period, "fit_seconds": fit_s, "tests": {}}
    for p, t in exp.tests.items():
        t1 = time.time(); pred = clf.predict(flat(t, use_ppi)); dt = time.time() - t1
        m = closed_set_metrics(t.y, pred, exp.class_names)
        m["predict_us_per_flow"] = dt / len(t) * 1e6
        res["tests"][p] = m
        print(f"{p}: acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}")
    out = a.out or f"results/ml_{a.model}_{a.features}_f{a.label_fraction}.json"
    save_json(res, out); print("saved", out)


if __name__ == "__main__":
    main()
