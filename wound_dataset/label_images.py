"""
label_images.py  (v2: seven findings)
Fast keyboard labeling tool. Shows each image from dataset/ (train, val, test) and
records yes/no labels for seven visible findings into labels.csv.

Run (from the wound_dataset folder, after prepare_dataset.py):
    python label_images.py

Keys (press a key to toggle it ON/OFF for the current image)
    R  redness      clear redness of the skin beyond the thin pink line at the incision edge
    S  swelling     visible puffiness / swelling around the incision
    C  clear        clear or pink-tinged drainage (wet, weeping wound), no pus
    E  edges        incision edges visibly open or pulled apart
    P  pus          cloudy yellow / white / green discharge or pustules
    A  spreading    redness extending well away from the incision (red streaks, large red area)
    D  dark         dark, black or blue-gray skin or tissue at the wound

    Space / Enter / Right   save this image as reviewed and go to the next one
    Left                    go back one image
    Q / Esc                 save and quit

Your earlier redness / pus labels are kept and shown pre-set. Every image shows up once
more so you can review it for the new findings (press Space if nothing changes).
Progress saves automatically and the tool resumes where you stopped.
The class folder name is hidden on purpose so it does not bias your labels.
"""
import csv
import os
import sys

DATASET_DIR = "dataset"
LABELS_FILE = "labels.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# key -> (label name, short description). Label names match the app's symptom keys.
LABEL_KEYS = {
    "r": ("redness", "redness"),
    "s": ("swelling", "swelling"),
    "c": ("clear", "clear drainage"),
    "e": ("edges", "edges open"),
    "p": ("pus", "pus"),
    "a": ("spreading", "spreading redness"),
    "d": ("dark", "dark skin/tissue"),
}
LABELS = [name for name, _ in LABEL_KEYS.values()]
FIELDS = ["file"] + LABELS + ["done"]


class LabelStore:
    """Image list + labels, with load/save/resume. No GUI code, so it is easy to test."""

    def __init__(self, dataset_dir=DATASET_DIR, labels_file=LABELS_FILE):
        self.labels_file = labels_file
        self.files = []
        for split in ("train", "val", "test"):
            split_dir = os.path.join(dataset_dir, split)
            for root, _, names in os.walk(split_dir):
                for n in sorted(names):
                    if os.path.splitext(n)[1].lower() in IMAGE_EXTS:
                        self.files.append(os.path.join(root, n).replace("\\", "/"))
        self.labels = {}   # file -> {label: 0/1, ..., "done": 0/1}
        if os.path.exists(labels_file):
            with open(labels_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    entry = {name: int(row.get(name) or 0) for name in LABELS}
                    entry["done"] = int(row.get("done") or 0)   # old files have no "done": review again
                    self.labels[row["file"].replace("\\", "/")] = entry

    def first_unreviewed(self):
        for i, f in enumerate(self.files):
            if not self.labels.get(f, {}).get("done"):
                return i
        return max(0, len(self.files) - 1)

    def get(self, f):
        base = {name: 0 for name in LABELS}
        base.update({k: v for k, v in self.labels.get(f, {}).items() if k in LABELS})
        return base

    def is_done(self, f):
        return bool(self.labels.get(f, {}).get("done"))

    def set(self, f, values, done):
        entry = {name: int(values.get(name, 0)) for name in LABELS}
        entry["done"] = int(bool(done) or self.is_done(f))
        self.labels[f] = entry

    def save(self):
        tmp = self.labels_file + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(FIELDS)
            for f in self.files:
                if f in self.labels:
                    l = self.labels[f]
                    w.writerow([f] + [l[name] for name in LABELS] + [l["done"]])
        os.replace(tmp, self.labels_file)

    def counts(self):
        """(images reviewed, {label: positives among reviewed images})"""
        done = [l for l in self.labels.values() if l["done"]]
        return len(done), {name: sum(l[name] for l in done) for name in LABELS}


def run_gui():
    import tkinter as tk
    from PIL import Image, ImageTk

    store = LabelStore()
    if not store.files:
        sys.exit("No images found in dataset/. Run prepare_dataset.py first.")

    root = tk.Tk()
    root.title("Label images: toggle findings with the keys shown, Space = next, Left = back, Q = quit")
    root.geometry("980x860")

    info = tk.Label(root, font=("Segoe UI", 11))
    info.pack(pady=4)
    canvas = tk.Label(root)
    canvas.pack(expand=True)

    row = tk.Frame(root)
    row.pack(pady=6)
    chips = {}
    for key, (name, desc) in LABEL_KEYS.items():
        chip = tk.Label(row, text=f"[{key.upper()}] {desc}", font=("Segoe UI", 11, "bold"),
                        padx=8, pady=6, relief="groove", bd=2)
        chip.pack(side="left", padx=3)
        chips[name] = chip

    tk.Label(root, font=("Segoe UI", 10), fg="#555",
             text="Space / Enter = save + next      Left = back      Q = save + quit").pack(pady=4)

    state = {"i": 0, "vals": {}, "photo": None}

    def refresh_chips():
        for name, chip in chips.items():
            on = state["vals"].get(name, 0)
            chip.config(bg="#c62828" if on else "#e8e8e8", fg="white" if on else "#333")

    def load(i):
        i = max(0, min(i, len(store.files) - 1))
        state["i"] = i
        f = store.files[i]
        state["vals"] = store.get(f)
        try:
            im = Image.open(f).convert("RGB")
            im.thumbnail((900, 560))
            state["photo"] = ImageTk.PhotoImage(im)
            canvas.config(image=state["photo"], text="")
        except Exception as e:
            canvas.config(image="", text=f"Could not open image: {e}")
        n, pos = store.counts()
        summary = ", ".join(f"{name} {c}" for name, c in pos.items())
        info.config(text=f"Image {i + 1} of {len(store.files)}   |   reviewed {n}   |   positives so far: {summary}")
        refresh_chips()

    def commit(done):
        store.set(store.files[state["i"]], state["vals"], done)
        store.save()

    def make_toggle(name):
        def toggle(_=None):
            state["vals"][name] = 0 if state["vals"].get(name, 0) else 1
            refresh_chips()
        return toggle

    def nxt(_=None):
        commit(True)
        if state["i"] >= len(store.files) - 1:
            quit_(None, already=True)
        else:
            load(state["i"] + 1)

    def back(_=None):
        commit(False)
        load(state["i"] - 1)

    def quit_(_=None, already=False):
        if not already:
            commit(False)
        root.destroy()

    for key, (name, _) in LABEL_KEYS.items():
        t = make_toggle(name)
        root.bind(key, t)
        root.bind(key.upper(), t)
    for key in ("<space>", "<Return>", "<Right>"):
        root.bind(key, nxt)
    root.bind("<Left>", back)
    for key in ("q", "Q", "<Escape>"):
        root.bind(key, quit_)
    root.protocol("WM_DELETE_WINDOW", lambda: quit_(None))

    load(store.first_unreviewed())
    root.mainloop()
    n, pos = store.counts()
    print(f"Saved labels for {n} reviewed images to {LABELS_FILE}")
    for name, c in pos.items():
        print(f"  {name}: {c} positive")


if __name__ == "__main__":
    run_gui()
