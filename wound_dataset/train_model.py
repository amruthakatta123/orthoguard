"""
train_model.py
Fine-tunes a pretrained MobileNetV2 on dataset/{train,val,test}/<class>/ images
(made by prepare_dataset.py) and reports honest per-class metrics.

Install:   pip install torch torchvision
           (On Windows with a very new Python, PyTorch may not install. If so, use
            Google Colab: upload the `dataset` folder and this file, then run it there.)
Run:       python train_model.py
Options:   --head-epochs 5  --finetune-epochs 10  --batch 16  --out model.pt

Outputs:   model.pt (weights + class names) and results.txt (metrics for your writeup)

The test set is evaluated ONCE at the end. The best epoch is chosen on the
validation set by balanced accuracy (average recall across classes), so a model
that ignores the rare "urgent" class does not win.
"""
import argparse
import copy
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
ALERT_CLASSES = ("urgent", "needs_attention")  # classes whose recall matters most


def make_data(root, batch):
    # Mild augmentation. No hue shift, because redness is the signal.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    sets = {
        "train": datasets.ImageFolder(os.path.join(root, "train"), train_tf),
        "val": datasets.ImageFolder(os.path.join(root, "val"), eval_tf),
        "test": datasets.ImageFolder(os.path.join(root, "test"), eval_tf),
    }
    if not (sets["train"].classes == sets["val"].classes == sets["test"].classes):
        raise SystemExit("Class folders differ between train/val/test. Re-run prepare_dataset.py.")
    loaders = {
        k: DataLoader(v, batch_size=batch, shuffle=(k == "train"), num_workers=0)
        for k, v in sets.items()
    }
    return sets, loaders


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys), torch.cat(ps)


def confusion(y, p, n):
    cm = torch.zeros(n, n, dtype=torch.long)
    for t, q in zip(y.tolist(), p.tolist()):
        cm[t, q] += 1
    return cm


def recalls(cm):
    return [cm[i, i].item() / cm[i].sum().item() if cm[i].sum() > 0 else None for i in range(len(cm))]


def balanced_acc(cm):
    r = [x for x in recalls(cm) if x is not None]
    return sum(r) / len(r) if r else 0.0


def train_phase(name, model, params, lr, epochs, loaders, criterion, device, n_classes, best):
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
        cm = confusion(yv, pv, n_classes)
        score = balanced_acc(cm)
        print(f"[{name}] epoch {ep}/{epochs}  train loss {loss_sum / total:.3f}  val balanced acc {score:.3f}")
        if score > best["score"]:
            best["score"] = score
            best["state"] = copy.deepcopy(model.state_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--no-pretrained", action="store_true", help="skip downloading ImageNet weights (testing only)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets, loaders = make_data(args.data, args.batch)
    classes = sets["train"].classes
    n = len(classes)
    counts = torch.bincount(torch.tensor(sets["train"].targets), minlength=n).float()
    print("Classes:", classes)
    print("Train counts:", counts.int().tolist(), "| val:", len(sets["val"]), "| test:", len(sets["test"]))

    # rarer classes get larger loss weights
    weights = (counts.sum() / (n * counts.clamp(min=1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    w = None if args.no_pretrained else models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=w)
    model.classifier[1] = nn.Linear(model.last_channel, n)
    model.to(device)

    best = {"score": -1.0, "state": None}

    # Phase 1: train only the new classifier head
    for p in model.features.parameters():
        p.requires_grad = False
    train_phase("head", model, model.classifier.parameters(), 1e-3, args.head_epochs,
                loaders, criterion, device, n, best)

    # Phase 2: unfreeze everything, small learning rate
    for p in model.parameters():
        p.requires_grad = True
    train_phase("finetune", model, model.parameters(), 1e-4, args.finetune_epochs,
                loaders, criterion, device, n, best)

    model.load_state_dict(best["state"])
    torch.save({"state_dict": best["state"], "classes": classes}, args.out)

    # Final, one-time test evaluation
    yt, pt = predict(model, loaders["test"], device)
    cm = confusion(yt, pt, n)
    lines = [f"Best validation balanced accuracy: {best['score']:.3f}",
             f"Test images: {len(yt)}",
             f"Test accuracy: {(yt == pt).float().mean().item():.3f}",
             f"Test balanced accuracy: {balanced_acc(cm):.3f}", "",
             f"{'class':<18}{'support':>8}{'precision':>11}{'recall':>9}"]
    rec = recalls(cm)
    for i, c in enumerate(classes):
        support = cm[i].sum().item()
        predicted = cm[:, i].sum().item()
        prec = cm[i, i].item() / predicted if predicted else float("nan")
        r = rec[i] if rec[i] is not None else float("nan")
        lines.append(f"{c:<18}{support:>8}{prec:>11.2f}{r:>9.2f}")
    lines += ["", "Confusion matrix (rows = true, columns = predicted):",
              "".ljust(18) + "".join(f"{c[:10]:>12}" for c in classes)]
    for i, c in enumerate(classes):
        lines.append(f"{c:<18}" + "".join(f"{cm[i, j].item():>12}" for j in range(n)))
    for i, c in enumerate(classes):
        if c in ALERT_CLASSES and rec[i] is not None:
            lines += ["", f"Recall on '{c}': {rec[i]:.2f} (the share of truly serious cases the model caught)"]
    report = "\n".join(lines)
    print("\n" + report)
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved {args.out} and results.txt")


if __name__ == "__main__":
    main()
