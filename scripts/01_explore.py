"""Print dataset facts needed for the report: classes per period, class imbalance, PPI shape."""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etc_lib.data import add_data_args, load_experiment
from etc_lib.train import save_json


def main():
    ap = argparse.ArgumentParser(); add_data_args(ap)
    ap.add_argument("--out", default="results/explore.json")
    a = ap.parse_args()
    exp = load_experiment(a, a.test_periods)
    c = Counter(exp.train.y.tolist())
    counts = sorted(c.values(), reverse=True)
    info = {
        "train_flows": len(exp.train), "val_flows": len(exp.val),
        "test_flows": {k: len(v) for k, v in exp.tests.items()},
        "n_classes": exp.n_classes,
        "ppi_shape": list(exp.train.ppi.shape), "stats_dim": int(exp.train.stats.shape[1]),
        "class_count_max": counts[0], "class_count_min": counts[-1],
        "imbalance_ratio": counts[0] / max(1, counts[-1]),
        "top10_classes": [(exp.class_names[k], v) for k, v in c.most_common(10)],
    }
    for k, v in info.items():
        print(f"{k:18s}: {v}")
    save_json(info, a.out)


if __name__ == "__main__":
    main()
