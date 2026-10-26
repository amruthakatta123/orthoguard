"""
make_contact_sheets.py
Builds numbered contact sheets (12 images each) from the dataset/ folder so they can be
uploaded to Claude in batches for labeling.

Run (from the wound_dataset folder):
    python make_contact_sheets.py

Output:
    contact_sheets/sheet_001.jpg, sheet_002.jpg, ...   (upload these to the chat)
    contact_sheets/index.json                         (maps sheet + cell number -> image file)

Cell numbers (1 to 12) are drawn in the top-left corner of each picture.
Sheets show no class names, so labeling is not biased by which folder an image came from.
"""
import json
import os

from PIL import Image, ImageDraw, ImageFont

from label_images import LabelStore   # same image order as the labeling tool

COLS, ROWS = 4, 3
CELL_W, CELL_H = 400, 330
OUT = "contact_sheets"


def get_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def main():
    store = LabelStore()
    files = store.files
    if not files:
        raise SystemExit("No images in dataset/. Run prepare_dataset.py first.")
    os.makedirs(OUT, exist_ok=True)
    per = COLS * ROWS
    font = get_font(34)
    index = {}
    n_sheets = (len(files) + per - 1) // per

    for s in range(n_sheets):
        chunk = files[s * per:(s + 1) * per]
        sheet = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H), (28, 28, 28))
        draw = ImageDraw.Draw(sheet)
        for k, path in enumerate(chunk):
            x0, y0 = (k % COLS) * CELL_W, (k // COLS) * CELL_H
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((CELL_W - 8, CELL_H - 8))
                    sheet.paste(im, (x0 + (CELL_W - im.width) // 2, y0 + (CELL_H - im.height) // 2))
            except Exception:
                draw.text((x0 + 20, y0 + 20), "unreadable", fill=(255, 80, 80), font=font)
            draw.rectangle([x0 + 4, y0 + 4, x0 + 70, y0 + 52], fill=(0, 0, 0))
            draw.text((x0 + 12, y0 + 8), str(k + 1), fill=(255, 235, 59), font=font)
            draw.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1], outline=(60, 60, 60))
        sheet.save(os.path.join(OUT, f"sheet_{s + 1:03d}.jpg"), quality=88)
        index[str(s + 1)] = chunk

    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    print(f"Wrote {n_sheets} sheets ({len(files)} images) to ./{OUT}/")
    print("Upload them to the chat in batches (about 5 sheets at a time works well).")


if __name__ == "__main__":
    main()
