"""Self-supervised pre-training of the packet-sequence Transformer (masked packet reconstruction).

CESNET data contains no payload, so ET-BERT-style byte pre-training is not possible here. This is the
analogue for flow data: hide ~30 % of the packets (IPT, direction, size) and predict them from context.
Labels are NOT used, so the whole training period (or more) can be used as unlabeled data.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from etc_lib.data import Preprocessor, add_data_args, load_experiment
from etc_lib.models import build_model
from etc_lib.train import get_device, save_json, seed_everything


def main():
    ap = argparse.ArgumentParser(); add_data_args(ap)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/pretrain_transformer.pt")
    a = ap.parse_args()
    seed_everything(a.seed); dev = get_device()

    exp = load_experiment(a, a.test_periods[:1])
    pre = Preprocessor().fit(exp.train)
    tr = pre(exp.train)
    model = build_model("transformer", exp.n_classes, tr.ppi.shape[1], tr.stats.shape[1], use_stats=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(torch.from_numpy(tr.ppi)), batch_size=a.batch_size, shuffle=True)
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train(); tot = n = 0
        for (x,) in dl:
            x = x.to(dev)
            mask = torch.rand(x.size(0), x.size(2), device=dev) < a.mask_ratio
            rec = model.reconstruct(x, mask)                                  # (B, C, L)
            m = mask.unsqueeze(1).expand_as(x)
            loss = F.mse_loss(rec[m], x[m])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        hist.append(tot / n); print(f"epoch {ep:02d} masked-packet MSE {tot / n:.4f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(a)}, a.out)
    save_json({"loss_history": hist}, a.out.replace(".pt", "_log.json")); print("saved", a.out)


if __name__ == "__main__":
    main()
