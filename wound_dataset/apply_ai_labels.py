"""
apply_ai_labels.py
Merges AI-assisted labels (ai_labels.txt) into labels.csv.

ai_labels.txt has one line per image, "sheet.cell: findings", for example:
    1.1: redness, pus
    1.2: none
    1.3: swelling, clear, edges
Findings: redness, swelling, clear, edges, pus, spreading, dark   (or "none").
"junk" marks an image that is not a usable wound photo (slide, diagram, arthroscope view...):
it is left out of labels.csv and listed in junk_images.txt. If an image appears on two lines,
the LAST line wins, so corrections can simply be added at the end of the file.
Lines starting with # are ignored. Append new batches to the same file.

Rules
  * The AI labels replace your earlier redness / pus labels (KEEP_HUMAN is empty). To keep your
    own labels instead, set KEEP_HUMAN = {"redness", "pus"}.
  * If a row was already reviewed with label_images.py (done = 1), it is left untouched.
  * Your original labels.csv is copied to labels_backup_before_ai.csv the first time.

Run (from the wound_dataset folder):
    python apply_ai_labels.py
"""
import csv
import json
import os
import re
import shutil
import sys

from label_images import FIELDS, LABELS

AI_FILE = "ai_labels.txt"
INDEX_FILE = os.path.join("contact_sheets", "index.json")
LABELS_FILE = "labels.csv"
KEEP_HUMAN = set()   # AI labels replace your earlier hand labels for every finding

ALIASES = {"spread": "spreading", "open": "edges", "drainage": "clear", "dark_skin": "dark", "swell": "swelling"}
LINE = re.compile(r"^\s*(\d+)\s*[.:/-]\s*(\d+)\s*[:=-]\s*(.*?)\s*$")


def parse_ai(path, index):
    ai, errors = {}, []
    with open(path, encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE.match(line)
            if not m:
                errors.append(f"line {ln}: cannot read '{line}'")
                continue
            sheet, cell, rest = m.group(1), int(m.group(2)), m.group(3)
            files = index.get(sheet)
            if files is None or not (1 <= cell <= len(files)):
                errors.append(f"line {ln}: no image for sheet {sheet} cell {cell}")
                continue
            toks = [ALIASES.get(t, t) for t in re.split(r"[,\s;]+", rest.lower()) if t]
            bad = [t for t in toks if t not in LABELS and t not in ("none", "junk")]
            if bad:
                errors.append(f"line {ln}: unknown finding(s) {bad}")
                continue
            ai[files[cell - 1].replace("\\", "/")] = {t for t in toks if t in LABELS or t == "junk"}
    return ai, errors


def read_existing(path):
    rows = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["file"].replace("\\", "/")] = r
    return rows


def main():
    if not os.path.exists(INDEX_FILE):
        sys.exit("contact_sheets/index.json not found. Run make_contact_sheets.py first.")
    if not os.path.exists(AI_FILE):
        sys.exit(f"{AI_FILE} not found. Paste the labels from the chat into a file with that name.")
    with open(INDEX_FILE, encoding="utf-8") as f:
        index = json.load(f)
    all_files = [p.replace("\\", "/") for s in sorted(index, key=int) for p in index[s]]

    ai, errors = parse_ai(AI_FILE, index)
    if errors:
        print("Problems in ai_labels.txt (fix these lines and run again):")
        for e in errors[:30]:
            print("  ", e)
        sys.exit(1)

    existing = read_existing(LABELS_FILE)
    if existing and not os.path.exists("labels_backup_before_ai.csv"):
        shutil.copy(LABELS_FILE, "labels_backup_before_ai.csv")

    out, disagreements, untouched, junk = [], [], 0, []
    for path in all_files:
        old = existing.get(path)
        if old and int(old.get("done") or 0) == 1:
            out.append([path] + [int(old.get(n) or 0) for n in LABELS] + [1])
            untouched += 1
            continue
        if path not in ai and not old:
            continue                       # nothing known about this image yet
        found = ai.get(path)
        if found is not None and "junk" in found:
            junk.append(path)              # not a usable wound photo: leave it out of labels.csv
            continue
        vals = {}
        for n in LABELS:
            if n in KEEP_HUMAN and old is not None and old.get(n) not in (None, ""):
                vals[n] = int(old[n])
                if found is not None and (n in found) != bool(vals[n]):
                    disagreements.append((path, n, "you: yes, AI: no" if vals[n] else "you: no, AI: yes"))
            elif found is not None:
                vals[n] = int(n in found)
            else:
                vals[n] = int(old.get(n) or 0) if old else 0
        out.append([path] + [vals[n] for n in LABELS] + [0])

    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(out)

    with open("ai_disagreements.txt", "w", encoding="utf-8") as f:
        f.write("Images where the AI disagreed with your redness / pus label (check these):\n")
        for p, n, d in disagreements:
            f.write(f"{p}\t{n}\t{d}\n")

    with open("junk_images.txt", "w", encoding="utf-8") as f:
        f.write("Images marked as not usable (diagrams, slides, no wound visible). Left out of labels.csv.\n")
        f.write("You can delete these files from the dataset folder.\n")
        for p in junk:
            f.write(p + "\n")

    print(f"AI labels read for {len(ai)} of {len(all_files)} images.")
    print(f"{len(junk)} images marked junk and left out (listed in junk_images.txt).")
    print(f"labels.csv now has {len(out)} rows ({untouched} already reviewed by you and left alone).")
    print("Positives per finding:")
    for i, n in enumerate(LABELS):
        print(f"  {n:<10}{sum(r[1 + i] for r in out):>4}")
    print(f"{len(disagreements)} disagreements on redness/pus written to ai_disagreements.txt")


if __name__ == "__main__":
    main()
