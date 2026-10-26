# OrthoGuard

**Post-surgery recovery monitoring for athletes after sports-injury surgery.**

OrthoGuard helps athletes recovering from orthopedic surgery (ACL reconstruction, meniscus repair, labrum repair, rotator cuff repair, and Achilles tendon repair) track their incision and symptoms each day, and helps their doctor spot problems early.

Built for the 2026 Congressional App Challenge.

> ⚠️ **Medical disclaimer:** OrthoGuard is a student prototype for educational purposes only. It is **not a medical device** and does **not** provide medical advice, diagnosis, or treatment. Always follow your surgeon's instructions. If you think you have an infection or emergency, contact your doctor or call 911. Do not enter real patient information.

## Features

**For patients**
- Daily check-in: temperature, pain level, and a photo of the incision
- A green / yellow / red status for each check-in
- Recovery progress and a recovery plan for their surgery

**For doctors**
- A dashboard of their own patients, each with a profile
- Patients join a doctor's care team with a 6-character code

**How the status works**
- A rule-based engine assigns the green / yellow / red tier from the check-in answers.
- Two image models review the incision photo:
  - `model.pt`: classifies the wound as *normal* or *needs attention*
  - `attribute_model.pt`: detects visible findings such as redness
- The models can only **raise** a tier to yellow, never lower it, so the rules stay the safety net. Their findings pre-fill the check-in for the patient to confirm.
- An optional AI assistant answers recovery questions (Gemini or Claude).

## Model results

The wound classifier (MobileNet/ResNet-style CNN, PyTorch) on a held-out test set of 35 images:

| Metric | Value |
|---|---|
| Test accuracy | 94.3% |
| Balanced accuracy | 92.5% |
| Recall on "needs attention" | 89% |

This is a proof of concept trained on a small set of web-sourced images. Validation on physician-labeled clinical images is the next step.

## Getting started

Requires Python 3.10+.

```bash
git clone https://github.com/amruthakatta123/orthoguard.git
cd orthoguard
pip install -r requirements.txt
cd app
python -m streamlit run app.py
```

Then open the link Streamlit prints (usually http://localhost:8501).

**Demo accounts** are included in `app/data.json`: a demo doctor (`demo.doctor@example.com`, care-team code `DEMO42`) and a demo patient (`demo.patient@example.com`). You can also register a new account.

**AI assistant (optional):** copy `.env.example` to `.env` and add a `GEMINI_API_KEY` (free tier) or an `ANTHROPIC_API_KEY`. The rest of the app works without a key.

## Project structure

```
app/
  app.py                Streamlit app (patient + doctor views)
  model.pt              Wound classifier (normal vs needs_attention)
  attribute_model.pt    Visible-findings model (e.g. redness)
  data.json             Demo accounts and patients
  config.toml           Theme settings
wound_dataset/
  collect_images.py     Downloads candidate images by search query
  label_images.py       Labeling tool
  prepare_dataset.py    Cleans, de-duplicates and splits train/val/test
  train_model.py        Trains the wound classifier
  train_attributes.py   Trains the attribute model
  labels.csv            Image labels
  queries_log.txt       Search queries used to source images
  dataset/manifest.csv  Record of every image in each split
```

### About the training images

The training images were collected from the public web and belong to their original owners, so **they are not included in this repository**. The scripts in `wound_dataset/`, `queries_log.txt`, and `dataset/manifest.csv` document how the dataset was built so it can be reproduced.

## Acknowledgments

Thank you to the two medical students who helped shape the app's core concepts and gave clinical feedback throughout development: **[Name 1]** and **[Name 2]**.

## License

[MIT](LICENSE) © 2026 Amrutha
