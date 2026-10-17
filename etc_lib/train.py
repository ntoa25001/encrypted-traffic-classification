import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(split, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(split.ppi), torch.from_numpy(split.stats), torch.from_numpy(split.y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict_logits(model, split, device, bs=2048):
    model.eval(); outs = []
    for ppi, st, _ in _loader(split, bs, False):
        outs.append(model(ppi.to(device), st.to(device)).float().cpu())
    return torch.cat(outs).numpy()


def fit_classifier(model, train, val, device, epochs=20, lr=2e-3, bs=256, patience=4, wd=1e-4, log=print):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    best, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0; n = 0
        for ppi, st, y in _loader(train, bs, True):
            ppi, st, y = ppi.to(device), st.to(device), y.to(device)
            opt.zero_grad(); loss = lossf(model(ppi, st), y); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        pred = predict_logits(model, val, device).argmax(1)
        f1 = f1_score(val.y, pred, average="macro")
        log(f"epoch {ep:02d} loss {tot / n:.4f} val macro-F1 {f1:.4f}")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log("early stop"); break
    model.load_state_dict(best_state)
    return {"best_val_macro_f1": float(best), "train_seconds": time.time() - t0}


def closed_set_metrics(y_true, y_pred, class_names=None):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": classification_report(y_true, y_pred, labels=list(range(len(class_names))) if class_names else None,
                                           target_names=class_names, output_dict=True, zero_division=0) if class_names else None,
    }


def inference_latency(model, split, device, n=2000, bs=1):
    """Mean per-flow latency (ms) at batch size `bs`, and throughput (flows/s) at batch size 1024."""
    model.eval()
    idx = np.arange(min(n, len(split)))
    x = split.subset(idx)
    ppi, st = torch.from_numpy(x.ppi).to(device), torch.from_numpy(x.stats).to(device)
    with torch.no_grad():
        t = time.perf_counter()
        for i in range(0, min(200, len(idx)), bs):
            model(ppi[i:i + bs], st[i:i + bs])
        if device.type == "cuda": torch.cuda.synchronize()
        lat = (time.perf_counter() - t) / min(200, len(idx)) * 1000
        t = time.perf_counter(); big = 1024
        for i in range(0, len(idx), big):
            model(ppi[i:i + big], st[i:i + big])
        if device.type == "cuda": torch.cuda.synchronize()
        thr = len(idx) / (time.perf_counter() - t)
    return {"latency_ms_per_flow_bs1": lat, "throughput_flows_per_s_bs1024": thr}


def open_set_metrics(logits, y, unknown_label):
    """Max-softmax baseline. Positive = unknown. Returns AUROC and FPR@95%TPR for detecting unknown flows."""
    p = torch.softmax(torch.from_numpy(logits), 1).numpy()
    score_unknown = 1.0 - p.max(1)
    is_unk = (y == unknown_label).astype(int)
    auroc = roc_auc_score(is_unk, score_unknown)
    thr = np.quantile(score_unknown[is_unk == 1], 0.05)       # keeps 95% of unknown flows
    fpr = float((score_unknown[is_unk == 0] >= thr).mean())
    known = is_unk == 0
    pred = logits.argmax(1)
    return {"auroc_unknown_detection": float(auroc), "fpr_at_95tpr": fpr,
            "closed_set_macro_f1_on_known": float(f1_score(y[known], pred[known], average="macro")),
            "n_known": int(known.sum()), "n_unknown": int(is_unk.sum())}


def save_json(obj, path):
    import json, os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
