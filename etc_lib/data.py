"""Data access layer. The only module that touches cesnet-datazoo (or the synthetic fallback)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

# Channel order of the packet sequence (PPI) in CESNET datasets
IPT, DIR, SIZE = 0, 1, 2


@dataclass
class Split:
    ppi: np.ndarray    # (N, C, 30) float32: inter-packet time, direction, size
    stats: np.ndarray  # (N, F) float32 flow statistics (+ booleans)
    y: np.ndarray      # (N,) int64; unknown class (open-set) = n_classes

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        return Split(self.ppi[idx], self.stats[idx], self.y[idx])


@dataclass
class Experiment:
    train: Split
    val: Split
    tests: dict[str, Split]
    class_names: list[str]
    unknown_label: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_classes(self):
        return len(self.class_names)


def add_data_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("data")
    g.add_argument("--data-root", default="data", help="Folder where cesnet-datazoo stores the dataset")
    g.add_argument("--size", default="XS", choices=["XS", "S", "M", "L"], help="Subsampled dataset size")
    g.add_argument("--train-period", default="W-2022-12",
                   help="Train period. NOTE: use weeks >= 10 (exporter was updated in week 10, see README)")
    g.add_argument("--test-periods", nargs="+", default=["W-2022-13"],
                   help="One or more test periods, e.g. W-2022-13 W-2022-16 W-2022-24")
    g.add_argument("--topk", type=int, default=50, help="Keep the K most frequent services as known classes")
    g.add_argument("--min-train-samples", type=int, default=100)
    g.add_argument("--max-train", type=int, default=0, help="Optional cap on #train flows (0 = all)")
    g.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    g.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke test, no download)")
    g.add_argument("--seed", type=int, default=42)


def _load_synthetic(args, test_periods, open_world):
    from etc_lib import synthetic as syn
    k = min(args.topk, 20)
    tr = syn.make_split(6000, k, drift=0.0, seed=args.seed)
    va = syn.make_split(1500, k, drift=0.0, seed=args.seed + 1)
    tests = {}
    for i, p in enumerate(test_periods):
        ppi, st, y = syn.make_split(2000, k, drift=0.1 * (i + 1), seed=args.seed + 10 + i)
        if open_world:
            uppi, ust, _ = syn.make_split(500, 5, drift=0.1 * (i + 1), seed=args.seed + 50 + i, class_offset=k)
            ppi = np.concatenate([ppi, uppi]); st = np.concatenate([st, ust])
            y = np.concatenate([y, np.full(500, k, dtype="int64")])
        tests[p] = Split(ppi, st, y)
    return Experiment(Split(*tr), Split(*va), tests, [f"class_{i}" for i in range(k)],
                      unknown_label=k if open_world else None, meta={"synthetic": True})


def _arrays_from_loader(loader):
    from cesnet_datazoo.datasets.loaders import load_from_dataloader
    _, ppi, stats, y = load_from_dataloader(loader, silent=False)
    return Split(ppi.astype("float32"), stats.astype("float32"), y.astype("int64"))


def load_experiment(args, test_periods=None, open_world=False, unknown_size=20000) -> Experiment:
    """Load train/val/test arrays.

    open_world=True: only the top-K services are 'known'; the remaining services are returned as
    extra test samples labelled `unknown_label` (= n_classes). Used for the open-set experiment.
    """
    test_periods = test_periods or args.test_periods
    if args.synthetic:
        return _load_synthetic(args, test_periods, open_world)

    from cesnet_datazoo.config import AppSelection, DatasetConfig
    from cesnet_datazoo.datasets import CESNET_TLS_Year22

    ds = CESNET_TLS_Year22(args.data_root, size=args.size)  # downloads if missing
    tests, train, val, class_names, unk = {}, None, None, None, None
    for i, tp in enumerate(test_periods):
        cfg = DatasetConfig(
            dataset=ds,
            apps_selection=AppSelection.TOPX_KNOWN,
            apps_selection_topx=args.topk,
            min_train_samples_per_app=args.min_train_samples,
            train_period_name=args.train_period,
            test_period_name=tp,
            train_size=args.max_train if args.max_train > 0 else "all",
            test_unknown_size=unknown_size if open_world else 0,
            val_unknown_size=0,
            train_workers=args.workers, val_workers=args.workers, test_workers=args.workers,
            random_state=args.seed,
            use_packet_histograms=False,
            use_tcp_features=False,
            return_other_fields=False,   # SNI / JA3 / IPs / ASN are deliberately NOT used (see README)
        )
        ds.set_dataset_config_and_initialize(cfg, silent_warning=True)
        if i == 0:
            train = _arrays_from_loader(ds.get_train_dataloader())
            val = _arrays_from_loader(ds.get_val_dataloader())
            class_names = list(ds.get_known_apps())
            unk = ds.class_info.unknown_class_label if open_world else None
        else:
            assert list(ds.get_known_apps()) == class_names, "Known classes changed between test periods"
        tests[tp] = _arrays_from_loader(ds.get_test_dataloader())
    return Experiment(train, val, tests, class_names, unknown_label=unk,
                      meta={"size": args.size, "train_period": args.train_period, "topk": args.topk})


class Preprocessor:
    """Scales PPI and flow statistics. Fit on train only."""

    def fit(self, train: Split):
        s = self._stats_t(train.stats)
        self.mu, self.sd = s.mean(0), s.std(0) + 1e-6
        return self

    @staticmethod
    def _stats_t(x):
        return np.sign(x) * np.log1p(np.abs(x))

    def ppi(self, x):
        x = x.copy()
        x[:, IPT] = np.log1p(np.abs(x[:, IPT]))
        x[:, SIZE] = np.abs(x[:, SIZE]) / 1500.0
        return x.astype("float32")

    def stats(self, x):
        return ((self._stats_t(x) - self.mu) / self.sd).astype("float32")

    def __call__(self, s: Split) -> Split:
        return Split(self.ppi(s.ppi), self.stats(s.stats), s.y)


def stratified_fraction(split: Split, fraction: float, seed: int) -> Split:
    """Keep `fraction` of the train flows in every class (for the low-label experiment)."""
    if fraction >= 1.0:
        return split
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(split.y):
        idx = np.where(split.y == c)[0]
        n = max(2, int(round(len(idx) * fraction)))
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return split.subset(np.sort(np.concatenate(keep)))
