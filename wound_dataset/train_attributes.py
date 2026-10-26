"""
train_attributes.py  (v2: any number of findings)
Trains a multi-label MobileNetV2 that predicts several independent yes/no findings for one
photo (redness, swelling, clear drainage, open edges, pus, spreading redness, dark skin),
using labels.csv made with label_images.py / apply_ai_labels.py.

Run:      python train_attributes.py
Options:  --head-epochs 5  --finetune-epochs 10  --batch 16  --out attribute_model.pt
          --min-positives 5   (findings with fewer training examples are not trained)

Uses the same train/val/test split as the dataset folder (no leakage between splits).
Outputs:  attribute_model.pt and attribute_results.txt (numbers for your writeup)

Findings with too few positive examples are skipped and listed, and the app leaves those
as normal checkboxes. Findings with fewer than 15 examples are trained but flagged as weak.
Results are reported as raw counts (caught X of Y) because test sets are small.
"""
import argparse
import copy
import csv
import os

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class AttrDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, y = self.rows[i]
        with Image.open(path) as im:
            x = self.tf(im.convert("RGB"))
        return x, torch.tensor(y, dtype=torch.float32)


def read_rows(labels_file):
    """Returns (splits, label_names). Every row of labels.csv is used; 'done' only tracks
    your review progress in the labeling tool."""
    splits = {"train": [], "val": [], "test": []}
    with open(labels_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        names = [c for c in reader.fieldnames if c not in ("file", "done")]
        for r in reader:
            path = r["file"]
            parts = path.replace("\\", "/").split("/")
            split = next((p for p in parts if p in splits), None)
            if split is None or not os.path.exists(path):
                continue
            splits[split].append((path, [int(r[k] or 0) for k in names]))
    return splits, names


def prf(y, p):
    """Counts from 0/1 tensors: (tp, fp, fn, tn)."""
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    return tp, fp, fn, tn


def balanced(tp, fp, fn, tn):
    """Average of recall and specificity; None when a side has no examples."""
    if tp + fn == 0 or tn + fp == 0:
        return None
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


@torch.no_grad()
def predict(model, loader, device, thr=0.5):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        pr = torch.sigmoid(model(x.to(device))).cpu()
        ps.append((pr >= thr).float())
        ys.append(y)
    return torch.cat(ys), torch.cat(ps)


def val_score(y, p):
    scores = [balanced(*prf(y[:, j], p[:, j])) for j in range(y.shape[1])]
    scores = [s for s in scores if s is not None]
    return sum(scores) / len(scores) if scores else 0.0


def train_phase(name, model, params, lr, epochs, loaders, criterion, device, best):
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        total, loss_sum = 0, 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            total += len(y)
        yv, pv = predict(model, loaders["val"], device)
        score = val_score(yv, pv)
        print(f"[{name}] epoch {ep}/{epochs}  train loss {loss_sum / total:.3f}  val balanced acc {score:.3f}")
        if score > best["score"]:
            best["score"] = score
            best["state"] = copy.deepcopy(model.state_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="attribute_model.pt")
    ap.add_argument("--min-positives", type=int, default=5)
    ap.add_argument("--no-pretrained", action="store_true", help="testing only")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, all_names = read_rows(args.labels)
    for s, rows in splits.items():
        print(f"{s}: {len(rows)} labeled images")
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise SystemExit("Need labeled images in train, val and test. Check labels.csv.")

    tr_all = torch.tensor([y for _, y in splits["train"]], dtype=torch.float32)
    keep, skipped = [], []
    for j, name in enumerate(all_names):
        npos = int(tr_all[:, j].sum())
        total_pos = sum(r[1][j] for rows in splits.values() for r in rows)
        print(f"  {name:<10} {npos:>3} positive of {len(tr_all)} training images ({total_pos} in all splits)")
        if npos < args.min_positives:
            skipped.append((name, npos))
        else:
            keep.append(j)
            if npos < 15:
                print(f"     WARNING: only {npos} training examples, so '{name}' will be weak.")
    if not keep:
        raise SystemExit("No finding has enough positive examples to train.")
    names = [all_names[j] for j in keep]
    for name, npos in skipped:
        print(f"  Skipping '{name}': {npos} training examples is too few (need {args.min_positives}).")

    for s in splits:
        splits[s] = [(p, [y[j] for j in keep]) for p, y in splits[s]]
    tr = torch.tensor([y for _, y in splits["train"]], dtype=torch.float32)
    pos = tr.sum(0)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),   # no hue/saturation shifts: color is the signal
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    loaders = {
        "train": DataLoader(AttrDataset(splits["train"], train_tf), batch_size=args.batch, shuffle=True),
        "val": DataLoader(AttrDataset(splits["val"], eval_tf), batch_size=args.batch),
        "test": DataLoader(AttrDataset(splits["test"], eval_tf), batch_size=args.batch),
    }

    # rarer positives get a larger weight
    pos_weight = ((len(tr) - pos) / pos.clamp(min=1)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    w = None if args.no_pretrained else models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=w)
    model.classifier[1] = nn.Linear(model.last_channel, len(names))
    model.to(device)

    best = {"score": -1.0, "state": None}
    for p in model.features.parameters():
        p.requires_grad = False
    train_phase("head", model, model.classifier.parameters(), 1e-3, args.head_epochs,
                loaders, criterion, device, best)
    for p in model.parameters():
        p.requires_grad = True
    train_phase("finetune", model, model.parameters(), 1e-4, args.finetune_epochs,
                loaders, criterion, device, best)

    model.load_state_dict(best["state"])
    torch.save({"state_dict": best["state"], "labels": names, "threshold": 0.5}, args.out)

    yt, pt = predict(model, loaders["test"], device)
    lines = [f"Best validation balanced accuracy: {best['score']:.3f}",
             f"Test images: {len(yt)}", "",
             f"{'finding':<11}{'train pos':>10}{'test pos':>9}{'caught':>8}{'missed':>8}{'false alarms':>14}{'precision':>11}{'recall':>8}"]
    for j, name in enumerate(names):
        tp, fp, fn, tn = prf(yt[:, j], pt[:, j])
        prec = f"{tp / (tp + fp):.2f}" if tp + fp else "n/a"
        rec = f"{tp / (tp + fn):.2f}" if tp + fn else "n/a"
        lines.append(f"{name:<11}{int(pos[j]):>10}{tp + fn:>9}{tp:>8}{fn:>8}{fp:>14}{prec:>11}{rec:>8}")
    if skipped:
        lines += ["", "Not trained (too few examples): " + ", ".join(f"{n} ({c})" for n, c in skipped)]
    lines += ["", "Small test sets make these numbers noisy: report them as 'caught X of Y', not only as percentages."]
    report = "\n".join(lines)
    print("\n" + report)
    with open("attribute_results.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved {args.out} and attribute_results.txt")


if __name__ == "__main__":
    main()
