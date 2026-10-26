# Wound Image Dataset: Collection Guide

## Run order
1. `pip install icrawler pillow imagehash`
2. `python collect_images.py`  (downloads into `raw/<class>/<surgery>/`, can take a while)
3. **Manually review every image** (see below)
4. `python prepare_dataset.py`  (cleans, dedups, splits into `dataset/train|val|test`)
5. Train a MobileNetV2 / ResNet18 on `dataset/train`, tune on `val`, report once on `test`

## Classes
Use the folder name as the label. Move files between folders when the guess is wrong.

| Class | What it looks like |
|---|---|
| `normal` | Incision edges closed and flat, mild pink at the edge, small amount of bruising, dry scab or clean sutures |
| `concerning` | Redness extending noticeably beyond the incision, new swelling, warmth-looking tight shiny skin, irritation around sutures, small clear or cloudy drainage |
| `urgent` | Spreading redness, pus, wound edges separated (dehiscence), foul-looking discharge, dark or black tissue, large blisters |

Base these on a clinical source and cite it in your writeup (CDC surgical site infection guidelines, plus the signs listed in the cesarean paper's references are good starting points). If you are unsure about an image, **delete it**. A clean small dataset beats a large mislabeled one.

## Review checklist (do this for every image)
- [ ] Shows a real surgical incision or wound (delete diagrams, X-rays, MRI, stock art, cartoons, text screenshots)
- [ ] Shows the right body area (knee, shoulder, or Achilles/ankle)
- [ ] Label matches the criteria above
- [ ] Not a watermarked ad or a screenshot with large text overlays
- [ ] Skip anything showing a face or identifiable tattoos

## Targets
- Aim for **150+ images per class** after cleaning (450+ total). Expect to throw away 50-70% of what downloads.
- `urgent` will be your scarcest class. If you end up with fewer than about 60, add more queries in `collect_images.py` (for example "cellulitis after knee surgery", "postoperative wound infection leg").
- Try for varied skin tones. Add queries like "surgical site infection dark skin" and "incision healing brown skin" and check your final mix.

## Notes for your writeup
- Keep `queries_log.txt` and `dataset/manifest.csv`. They document how you sourced and cleaned the data.
- Use the images for model training only. Do not display or redistribute them in the app.
- Call the model a proof of concept trained on web-sourced images, with physician-labeled validation as the next step.
