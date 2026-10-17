"""Synthetic stand-in for CESNET data, only for smoke-testing the pipeline without downloading anything.

It mimics the shapes returned by cesnet-datazoo: PPI = (N, 3, 30) with channels
[IPT, DIR, SIZE] and flow statistics = (N, F). Each class has its own packet-size
template; later "periods" add increasing noise to emulate concept drift.
"""
import numpy as np

PPI_LEN = 30
N_STATS = 8


def _templates(n_classes, rng):
    return {
        "size": rng.uniform(60, 1400, size=(n_classes, PPI_LEN)),
        "ipt": rng.uniform(0, 80, size=(n_classes, PPI_LEN)),
        "dir": rng.choice([-1.0, 1.0], size=(n_classes, PPI_LEN)),
    }


def make_split(n, n_classes, drift=0.0, seed=0, class_offset=0, template_seed=123):
    rng = np.random.default_rng(seed)
    t = _templates(class_offset + n_classes, np.random.default_rng(template_seed))
    y = rng.integers(0, n_classes, size=n)
    c = y + class_offset
    noise = 0.15 + drift
    size = t["size"][c] * (1 + noise * rng.standard_normal((n, PPI_LEN)))
    ipt = np.abs(t["ipt"][c] * (1 + noise * rng.standard_normal((n, PPI_LEN))))
    d = t["dir"][c]
    flip = rng.random((n, PPI_LEN)) < 0.05 + drift / 2
    d = np.where(flip, -d, d)
    ppi = np.stack([ipt, d, np.abs(size)], axis=1).astype("float32")
    stats = np.column_stack([
        np.abs(size).sum(1), np.abs(size).sum(1) * 0.4, np.full(n, PPI_LEN), np.full(n, PPI_LEN) * 0.5,
        ipt.sum(1), np.full(n, PPI_LEN), rng.integers(1, 5, n), ipt.sum(1) * 1.1,
    ]).astype("float32")
    return ppi, stats, y.astype("int64")
