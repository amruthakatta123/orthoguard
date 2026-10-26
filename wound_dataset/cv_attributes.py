"""
cv_attributes.py
5-fold cross-validation for the finding detector. Every labeled image is used as a test image
exactly once, so rare findings (swelling, pus, edges...) get ~10-17 test cases instead of 0-3.

Run (from the wound_dataset folder, after apply_ai_labels.py):
    python cv_attributes.py
Options: --folds 5  --head-epochs 5  --finetune-epochs 8  --batch 16  --min-positives 8

Writes cv_results.txt (numbers for the writeup). It does not change attribute_model.pt.
Per finding it reports: positives, AUC (threshold-free), and caught / false alarms at 0.5.
AUC of 0.5 = coin flip, 1.0 = perfect. Under ~0.7 means the finding is not reliable.
"""
import argparse
import csv
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from train_attributes import MEAN, STD, AttrDataset, prf


def read_all(path):
    rows, names = [], None
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        names = [c for c in r.fieldnames if c not in ("file", "done")]
        for row in r:
            if os.path.exists(row["file"]):
                rows.append((row["file"], [int(row[k] or 0) for k in names]))
    return rows, names


def make_folds(rows, k, seed=42):
    """Spread rare findings evenly: images with the rarest positives are dealt out first."""
    rng = random.Random(seed)
    n = len(rows[0][1])
    freq = [sum(r[1][j] for r in rows) for j in range(n)]
    def rarity(r):
        pos = [freq[j] for j in range(n) if r[1][j]]
        return min(pos) if pos else 10 ** 9
    order = list(range(len(rows)))
    rng.shuffle(order)
    order.sort(key=lambda i: rarity(rows[i]))
    fold = [0] * len(rows)
    for pos, i in enumerate(order):
        fold[i] = pos % k
    return fold


def auc(y, s):
    """Probability a random positive scores higher than a random negative (ties count half)."""
    pos = [x for x, t in zip(s, y) if t == 1]
    neg = [x for x, t in zip(s, y) if t == 0]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def fit_predict(train_rows, test_rows, n_out, args, device):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                  transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tl = DataLoader(AttrDataset(train_rows, train_tf), batch_size=args.batch, shuffle=True)
    el = DataLoader(AttrDataset(test_rows, eval_tf), batch_size=args.batch)
    y = torch.tensor([r[1] for r in train_rows], dtype=torch.float32)
    pos = y.sum(0)
    crit = nn.BCEWithLogitsLoss(pos_weight=((len(y) - pos) / pos.clamp(min=1)).to(device))
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, n_out)
    model.to(device)

    def run(params, lr, epochs):
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(epochs):
            model.train()
            for x, t in tl:
                opt.zero_grad()
                crit(model(x.to(device)), t.to(device)).backward()
                opt.step()
    for p in model.features.parameters():
        p.requires_grad = False
    run(model.classifier.parameters(), 1e-3, args.head_epochs)
    for p in model.parameters():
        p.requires_grad = True
    run(model.parameters(), 1e-4, args.finetune_epochs)
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in el:
            out.append(torch.sigmoid(model(x.to(device))).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--min-positives", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows, all_names = read_all(args.labels)
    totals = [sum(r[1][j] for r in rows) for j in range(len(all_names))]
    keep = [j for j, t in enumerate(totals) if t >= args.min_positives]
    skipped = [(all_names[j], totals[j]) for j in range(len(all_names)) if j not in keep]
    names = [all_names[j] for j in keep]
    rows = [(p, [y[j] for j in keep]) for p, y in rows]
    print(f"{len(rows)} images, findings: {names}; skipped: {skipped}")

    fold = make_folds(rows, args.folds)
    probs = torch.zeros(len(rows), len(names))
    for k in range(args.folds):
        tr = [r for r, f in zip(rows, fold) if f != k]
        idx = [i for i, f in enumerate(fold) if f == k]
        print(f"fold {k + 1}/{args.folds}: train {len(tr)}, test {len(idx)}")
        probs[idx] = fit_predict(tr, [rows[i] for i in idx], len(names), args, device)

    y = torch.tensor([r[1] for r in rows], dtype=torch.float32)
    lines = [f"{args.folds}-fold cross-validation, {len(rows)} images (each image tested once)", "",
             f"{'finding':<11}{'positives':>10}{'AUC':>7}{'caught':>8}{'missed':>8}{'false alarms':>14}"]
    for j, n in enumerate(names):
        tp, fp, fn, tn = prf(y[:, j], (probs[:, j] >= 0.5).float())
        a = auc(y[:, j].tolist(), probs[:, j].tolist())
        lines.append(f"{n:<11}{tp + fn:>10}{(f'{a:.2f}' if a is not None else 'n/a'):>7}{tp:>8}{fn:>8}{fp:>14}")
    if skipped:
        lines += ["", "Not tested (too few examples): " + ", ".join(f"{n} ({c})" for n, c in skipped)]
    report = "\n".join(lines)
    print("\n" + report)
    with open("cv_results.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
