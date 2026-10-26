"""
prepare_dataset.py
Run this AFTER you have manually reviewed/sorted images in raw/<class>/...

Steps:
  1. Drop corrupt, tiny, or non-RGB-convertible images
  2. Remove exact and near-duplicate images ACROSS all classes (perceptual hash)
  3. Stratified split into train/val/test (70/15/15) BEFORE any augmentation
  4. Write dataset/<split>/<class>/ plus a manifest.csv

Install:   pip install pillow imagehash
Run:       python prepare_dataset.py
"""
import csv
import os
import random
import shutil

import imagehash
from PIL import Image

RAW_DIR = "raw"
OUT_DIR = "dataset"
MIN_SIDE = 128           # discard images smaller than this
HASH_DISTANCE = 6        # <= this Hamming distance counts as a duplicate
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def iter_images():
    for cls in sorted(os.listdir(RAW_DIR)):
        cls_dir = os.path.join(RAW_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for root, _, files in os.walk(cls_dir):
            for name in files:
                yield cls, os.path.join(root, name)


def main():
    random.seed(SEED)
    kept = []            # (class, path, hash)
    seen_hashes = []     # list of ImageHash
    stats = {"corrupt": 0, "too_small": 0, "duplicate": 0}

    for cls, path in iter_images():
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                if min(im.size) < MIN_SIDE:
                    stats["too_small"] += 1
                    continue
                h = imagehash.phash(im)
        except Exception:
            stats["corrupt"] += 1
            continue

        if any(h - s <= HASH_DISTANCE for s in seen_hashes):
            stats["duplicate"] += 1
            continue
        seen_hashes.append(h)
        kept.append((cls, path, str(h)))

    # stratified split
    by_class = {}
    for item in kept:
        by_class.setdefault(item[0], []).append(item)

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    rows = []
    for cls, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        bounds = {
            "train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:],
        }
        for split, group in bounds.items():
            dest_dir = os.path.join(OUT_DIR, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for i, (_, src, h) in enumerate(group):
                ext = os.path.splitext(src)[1].lower() or ".jpg"
                dest = os.path.join(dest_dir, f"{cls}_{split}_{i:04d}{ext}")
                shutil.copy2(src, dest)
                rows.append([dest, cls, split, src, h])

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "split", "original_path", "phash"])
        w.writerows(rows)

    print("Removed:", stats)
    print("Kept per class:")
    for cls, items in by_class.items():
        print(f"  {cls}: {len(items)}")
    print(f"Dataset written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
