"""Collect results/*.json into one Markdown table (paste into the report)."""
import glob
import json
import os

rows = []
for f in sorted(glob.glob(os.path.join("results", "*.json"))):
    try:
        r = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    if "tests" not in r:
        continue
    for period, m in r["tests"].items():
        if "macro_f1" in m:
            rows.append((r.get("model"), r.get("label_fraction", ""), "pre" if r.get("pretrained") else "",
                         period, f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", os.path.basename(f)))
        elif "auroc_unknown_detection" in m:
            rows.append((r.get("model") + " (open-set)", "", "", period, f"AUROC {m['auroc_unknown_detection']:.3f}",
                         f"FPR95 {m['fpr_at_95tpr']:.3f}", os.path.basename(f)))
print("| model | label frac | pretrained | test period | acc | macro-F1 | file |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(map(str, r)) + " |")
