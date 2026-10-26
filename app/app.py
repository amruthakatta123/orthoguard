"""
OrthoGuard - prototype for post-surgery sports-injury recovery monitoring.

Run:  python -m streamlit run app.py

Two kinds of accounts:
  * Patient - daily check-in (temperature, pain, incision photo), progress, recovery plan.
  * Doctor  - dashboard with a clickable list of THEIR patients; each opens its own profile.
Patients join a doctor's care team with the doctor's 6-character code when they register.

A rule-based engine assigns a green / yellow / red tier. Two photo models can only RAISE a tier
to yellow, never lower it: model.pt (normal vs needs_attention) and attribute_model.pt (visible
findings such as redness). Their findings pre-fill the check-in for the patient to confirm.
If a model file is missing, the app runs normally without it.

Prototype only: not a medical device, not medical advice. Do not enter real patient information.
Passwords are stored as salted PBKDF2 hashes, but this is a demo login, not production security.
"""
import hashlib
import hmac
import html
import json
import math
import os
import random
import string
import uuid
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

APP_NAME = "OrthoGuard"  # change this one line to rename the app

# Logo: a flexed arm (bicep) with a bandage, drawn as inline SVG so it works everywhere
LOGO_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="16" fill="#0B1412"/><g fill="#fff" stroke="#fff" stroke-linecap="round" stroke-linejoin="round"><path d="M10 50 L48 50" stroke-width="11" fill="none"/><path d="M48 49 L47 24" stroke-width="11" fill="none"/><rect x="38" y="8" width="19" height="17" rx="7" stroke="none"/><path d="M12 46 C10 30 22 20 33 24 C40 27 42 35 40 46 Z" stroke="none"/></g><path d="M41 17 H55 M41 21 H55" stroke="#0B1412" stroke-width="1.2" stroke-linecap="round" opacity=".5"/><g transform="rotate(-35 26 37)"><rect x="13" y="31.5" width="26" height="11" rx="5.5" fill="#0D9488" stroke="#0B1412" stroke-width="1.6"/><rect x="21.5" y="31.5" width="9" height="11" fill="#7FD6CC"/><g fill="#0B1412" opacity=".55"><circle cx="25" cy="35" r=".9"/><circle cx="27.6" cy="35" r=".9"/><circle cx="25" cy="37.2" r=".9"/><circle cx="27.6" cy="37.2" r=".9"/><circle cx="25" cy="39.4" r=".9"/><circle cx="27.6" cy="39.4" r=".9"/></g></g></svg>'''
DATA_FILE = "data.json"
UPLOAD_DIR = "uploads"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pt")
PHOTO_THRESHOLD = 0.40  # flag when P(needs_attention) >= this. Low on purpose: a false alarm beats a miss.
ATTR_MODEL_PATH = os.path.join(HERE, "attribute_model.pt")
ATTR_THRESHOLD = 0.50   # a finding is "detected" when its probability >= this
# Which findings may pre-fill the check-in. None = every finding the model was trained on.
# After cross-validation, restrict this to the reliable ones, e.g. {"redness", "pus"}.
PREFILL_FINDINGS = None

DEMO_DOCTOR_EMAIL = "demo.doctor@example.com"
DEMO_PATIENT_EMAIL = "demo.patient@example.com"
DEMO_CODE = "DEMO42"
PBKDF2_ROUNDS = 200_000
MAX_FAILS, LOCK_MINUTES = 5, 5

st.set_page_config(page_title=APP_NAME, layout="centered",
                   initial_sidebar_state="collapsed")

TIER_RANK = {"green": 0, "yellow": 1, "red": 2}
TIER_NAME = {"green": "On track", "yellow": "Watch closely", "red": "Contact surgeon"}

# key -> (label, level). Example criteria; thresholds should be set with a clinician.
SYMPTOMS = {
    "redness": ("Mild redness right at the incision edge", "yellow"),
    "swelling": ("Swelling is increasing", "yellow"),
    "warm": ("Skin around the incision feels warm", "yellow"),
    "clear": ("Small amount of clear or pink drainage", "yellow"),
    "edges": ("Incision edges are opening or separating", "red"),
    "pus": ("Pus or thick yellow/green or foul-smelling drainage", "red"),
    "spreading": ("Redness is spreading away from the incision", "red"),
    "dark": ("Dark, black, or blue-gray skin near the incision", "red"),
    "calf": ("New calf pain, tightness, or swelling", "red"),
    "breath": ("Chest pain or trouble breathing", "red"),
}
PATIENT_KEYS = ["warm", "calf", "breath"]  # only the patient can tell us these
VISUAL_KEYS = ["redness", "swelling", "clear", "edges", "pus", "spreading", "dark"]
FINDING_NAMES = {
    "redness": "redness", "swelling": "swelling", "clear": "clear drainage",
    "edges": "open incision edges", "pus": "pus", "spreading": "spreading redness",
    "dark": "dark skin",
}

# General example milestones. Protocols vary by surgeon.
SURGERIES = {
    "ACL reconstruction": [
        ("Days 0-14", "Control swelling, protect the incision, start early knee motion and quad activation as your therapist directs."),
        ("Weeks 2-6", "Work toward full knee extension and walking without crutches as cleared."),
        ("Months 3-6", "Build strength and neuromuscular control."),
        ("Months 6-9+", "Sport-specific testing. Return to play only with surgeon clearance."),
    ],
    "Rotator cuff repair": [
        ("Weeks 0-6", "Sling as directed. Passive motion only, per your protocol."),
        ("Weeks 6-12", "Begin active motion if cleared."),
        ("Months 3-6", "Start strengthening."),
        ("Months 6+", "Gradual return to overhead or throwing with clearance."),
    ],
    "Meniscus repair": [
        ("Weeks 0-6", "Protected weight bearing, brace and motion limits as directed."),
        ("Weeks 6-12", "Progress weight bearing and strengthening."),
        ("Months 4-6", "Running and sport progression with clearance."),
    ],
    "Shoulder labrum / stabilization": [
        ("Weeks 0-6", "Sling and passive motion per protocol."),
        ("Weeks 6-12", "Begin active range of motion."),
        ("Months 3-4", "Strengthening phase."),
        ("Months 4-6+", "Sport-specific progression with surgeon clearance."),
    ],
    "Achilles tendon repair": [
        ("Weeks 0-2", "Splint or boot, limited weight bearing as directed. Wound check around 2 weeks."),
        ("Weeks 2-6", "Boot with gradual weight bearing per protocol."),
        ("Weeks 6-12", "Transition out of the boot, begin strengthening."),
        ("Months 4-6+", "Sport-specific work. Return only with clearance."),
    ],
}


# ---------- look and feel ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{--teal:#0D9488;--teal-d:#0F766E;--teal-l:#E3F4F1;--ink:#0B1412;--muted:#6B7A78;--bg:#F3F6F6;--line:#E2E9E8;}
html, body, .stApp, .stApp *{font-family:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',sans-serif;}
.stApp{background:var(--bg);color:var(--ink);}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none!important;}
.block-container{max-width:640px;padding:1.1rem 1rem 4rem;}
h1,h2,h3{letter-spacing:-.02em;color:var(--ink);}
h2{font-size:1.5rem!important;} h3{font-size:1.15rem!important;}

[data-testid="stButton"], .stButton, [data-testid="stFormSubmitButton"], .stFormSubmitButton{width:100%;}
[data-testid="stIconMaterial"], [data-testid="stExpanderIcon"], .material-symbols-rounded{font-family:'Material Symbols Rounded'!important;}
.st-key-topbar [data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;align-items:center;}
.st-key-topbar [data-testid="stColumn"]{min-width:0!important;}
.st-key-topbar button{padding:.4rem .9rem;font-size:.85rem;}
/* buttons: black pill = main action, outlined pill = secondary */
.stButton button, .stFormSubmitButton button, .stDownloadButton button{
  width:100%;border-radius:999px;padding:.7rem 1.2rem;font-weight:700;
  border:1.5px solid var(--ink);background:#fff;color:var(--ink);transition:transform .05s;}
.stButton button:hover, .stFormSubmitButton button:hover{border-color:var(--teal);color:var(--teal-d);background:#fff;}
.stButton button:active{transform:scale(.99);}
button[kind="primary"], button[kind="primaryFormSubmit"],
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"]{
  background:var(--ink)!important;color:#fff!important;border-color:var(--ink)!important;}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover,
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover{
  background:var(--teal-d)!important;border-color:var(--teal-d)!important;color:#fff!important;}
button[kind="primary"] *, button[kind="primaryFormSubmit"] *,
button[data-testid="stBaseButton-primary"] *, button[data-testid="stBaseButton-primaryFormSubmit"] *{color:#fff!important;}

[data-testid="stElementContainer"]:has(button){width:100%!important;}
/* inputs */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"]>div{
  background:#F1F5F5!important;border:1px solid #D3DFDD!important;}
[data-baseweb="base-input"], [data-baseweb="input"] input, textarea{background:transparent!important;}
[data-testid$="RootElement"], [data-testid="stNumberInputContainer"], [data-testid="stSelectbox"] [role="combobox"],
[data-testid="stDateInput"] [role="group"]{background:#F1F5F5!important;border:1px solid #D3DFDD!important;border-radius:14px!important;}
[data-testid="stTextInputField"], [data-testid="stNumberInputField"]{background:transparent!important;}
[data-baseweb="input"], [data-baseweb="select"]>div, [data-baseweb="textarea"]{border-radius:14px!important;}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within{border-color:var(--teal)!important;}
.stTextInput label, .stNumberInput label, .stSelectbox label, .stTextArea label, .stDateInput label, .stSlider label,
.stRadio label, .stFileUploader label{font-weight:600;}
[data-testid="stFileUploaderDropzone"]{border-radius:18px;background:#fff;border:1.5px dashed #B7C7C5;}

/* cards: bordered containers and forms */
div[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stForm"]{
  background:#fff;border:1px solid var(--line)!important;border-radius:22px!important;
  box-shadow:0 6px 22px rgba(11,20,18,.05);}
[data-testid="stForm"]{padding:1.1rem;}

/* tabs as pills */
[role="tablist"], .stTabs [data-baseweb="tab-list"]{gap:4px!important;background:#E4ECEB!important;padding:5px!important;border-radius:999px!important;border-bottom:none!important;}
[role="tab"], .stTabs [data-baseweb="tab"]{border-radius:999px!important;height:40px!important;padding:0 10px!important;font-weight:700;color:var(--muted);flex:1 1 0!important;justify-content:center;}
[role="tab"][aria-selected="true"]{background:var(--ink)!important;}
[role="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] *{color:#fff!important;}
[data-testid="stTabHighlight"], [data-testid="stTabBorder"], [role="tab"]::after, [role="tablist"]::after, [role="tablist"]::before{display:none!important;}
[role="tab"], [role="tablist"]{border-bottom:none!important;box-shadow:none!important;outline:none!important;}
[role="tab"] *{border-bottom:none!important;box-shadow:none!important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none;}
[data-testid="stExpander"]{border-radius:18px;background:#fff;border:1px solid var(--line);}
[data-testid="stAlert"]{border-radius:16px;}

/* custom pieces */
.brand{display:flex;align-items:center;gap:8px;font-weight:800;font-size:1.05rem;letter-spacing:-.01em;}
.brand .dot{width:44px;height:44px;display:block;flex:none;}
.brand .dot svg{width:100%;height:100%;display:block;}
.hero{background:linear-gradient(180deg,#DDF2EE 0%,#F6FBFA 70%);border-radius:30px;padding:1.4rem 1.2rem 1.5rem;
  text-align:center;box-shadow:0 10px 30px rgba(13,148,136,.12);margin:.4rem 0 1rem;}
.hero h1{font-size:2.05rem;line-height:1.1;font-weight:800;margin:.4rem 0 .5rem;}
.hero h1 span{color:var(--teal);}
.hero p{color:var(--muted);font-size:.98rem;margin:0 auto;max-width:420px;}
.feat{display:flex;gap:10px;margin:.2rem 0 1rem;}
.feat div{flex:1;background:#fff;border:1px solid var(--line);border-radius:20px;padding:.8rem .6rem;text-align:center;font-size:.78rem;font-weight:600;line-height:1.25;}
.feat b{display:block;margin-bottom:.3rem;}
.feat b svg{display:block;margin:0 auto;}
.fine{color:var(--muted);font-size:.78rem;text-align:center;margin-top:.8rem;}
.dark-card{background:var(--ink);color:#fff;border-radius:26px;padding:1.1rem 1.2rem;margin:.3rem 0 1rem;display:flex;align-items:center;justify-content:space-between;gap:12px;}
.dark-card .big{font-size:1.9rem;font-weight:800;line-height:1;}
.dark-card .sm{color:#9FB3B0;font-size:.8rem;font-weight:600;}
.dark-card .hi{font-size:1.1rem;font-weight:700;margin-bottom:.15rem;}
.tag{display:inline-block;border-radius:999px;padding:.28rem .7rem;font-size:.78rem;font-weight:700;}
.tag-teal{background:var(--teal);color:#fff;} .tag-soft{background:#1F2F2D;color:#C9DAD7;}
.chip{display:inline-block;border-radius:999px;padding:.25rem .7rem;font-size:.76rem;font-weight:700;white-space:nowrap;}
.chip-green{background:#DDF3E6;color:#116A3A;} .chip-yellow{background:#FFF1C9;color:#8A5A00;}
.chip-red{background:#FDDCDC;color:#A31616;} .chip-none{background:#E8EEED;color:#5B6B69;}
.stats{display:flex;gap:10px;margin:.2rem 0 1rem;}
.stat{flex:1;background:#fff;border:1px solid var(--line);border-radius:20px;padding:.8rem .5rem;text-align:center;}
.stat b{display:block;font-size:1.6rem;line-height:1.1;font-weight:800;}
.stat span{font-size:.72rem;color:var(--muted);font-weight:600;}
.stat.hot b{color:#C62828;}
.prow{display:flex;align-items:center;gap:12px;}
.avatar{width:46px;height:46px;border-radius:50%;background:var(--teal-l);color:var(--teal-d);display:grid;place-items:center;font-weight:800;flex:none;}
.avatar.lg{width:54px;height:54px;background:var(--teal);color:#fff;font-size:1.1rem;}
.pinfo{flex:1;min-width:0;} .pinfo b{display:block;font-size:1rem;}
.pinfo span{color:var(--muted);font-size:.82rem;}
.flag{color:#C62828;font-weight:700;font-size:.78rem;}
.code-box{background:var(--teal-l);border-radius:18px;padding:.8rem 1rem;display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;}
.code-box b{font-size:1.3rem;letter-spacing:.15em;color:var(--teal-d);}
.msg{display:flex;flex-direction:column;margin:.4rem 0;}
.msg.mine{align-items:flex-end;} .msg.theirs{align-items:flex-start;}
.bubble{max-width:82%;padding:.55rem .9rem;border-radius:18px;font-size:.92rem;word-break:break-word;}
.mine .bubble{background:var(--ink);color:#fff;border-bottom-right-radius:6px;}
.theirs .bubble{background:#fff;border:1px solid var(--line);border-bottom-left-radius:6px;}
.meta{font-size:.7rem;color:var(--muted);margin-top:.15rem;padding:0 .3rem;}
.newmsg{color:var(--teal-d);font-weight:700;font-size:.78rem;}
.st-key-demobar [data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;gap:10px;}
.st-key-demobar [data-testid="stColumn"]{min-width:0!important;}
.st-key-resetbar{display:flex;justify-content:center;}
.st-key-resetbar button{width:auto!important;border:none!important;background:transparent!important;color:#C3CFCD!important;
  font-size:.68rem!important;font-weight:500!important;padding:.2rem .6rem!important;min-height:0!important;}
.st-key-resetbar button:hover{color:#8A9997!important;background:transparent!important;}
.reason{background:#fff;border:1px solid var(--line);border-radius:14px;padding:.55rem .8rem;margin:.35rem 0;font-size:.9rem;}
@media (max-width:640px){.block-container{padding:.8rem .8rem 4rem;}.hero h1{font-size:1.75rem;}}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def arc_svg(filled, total=7, size=300):
    """Segmented half-ring (a nod to a progress gauge) for the landing page."""
    cx, cy, r = size / 2, size / 2 + 10, size / 2 - 28
    step = 180 / total
    parts = []
    for i in range(total):
        a1, a2 = 180 - i * step - 7, 180 - (i + 1) * step + 7
        x1, y1 = cx + r * math.cos(math.radians(a1)), cy - r * math.sin(math.radians(a1))
        x2, y2 = cx + r * math.cos(math.radians(a2)), cy - r * math.sin(math.radians(a2))
        col = "#0D9488" if i < filled else "#D5E1DF"
        parts.append(f'<path d="M{x1:.1f} {y1:.1f} A{r} {r} 0 0 1 {x2:.1f} {y2:.1f}" stroke="{col}" '
                     f'stroke-width="20" stroke-linecap="round" fill="none"/>')
    return (f'<svg viewBox="0 0 {size} {size / 2 + 40}" width="100%" style="max-width:300px">{"".join(parts)}'
            f'<text x="{cx}" y="{cy - 22}" text-anchor="middle" font-size="13" font-weight="700" fill="#6B7A78">RECOVERY DAY</text>'
            f'<text x="{cx}" y="{cy + 22}" text-anchor="middle" font-size="46" font-weight="800" fill="#0B1412">12</text></svg>')


def chip(tier):
    if tier is None:
        return '<span class="chip chip-none">No check-ins</span>'
    return f'<span class="chip chip-{tier}">{TIER_NAME[tier]}</span>'


def initials(name):
    parts = [w for w in name.replace(".", " ").split() if w and w.lower() != "dr"]
    return ("".join(w[0] for w in parts[:2]) or "?").upper()


def first_name(name):
    parts = [w for w in name.split() if w.lower().strip(".") != "dr"]
    return parts[0] if parts else name


def nice_list(items):
    names = [FINDING_NAMES.get(i, i) for i in items]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


# ---------- storage ----------
def load_data():
    data = {}
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.setdefault("users", {})
    data.setdefault("patients", {})
    return data


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------- accounts ----------
def hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return salt.hex(), digest.hex()


def norm_email(e):
    return e.strip().lower()


def valid_email(e):
    return "@" in e and "." in e.split("@")[-1] and " " not in e and len(e) <= 120


def new_care_code(data):
    existing = {u.get("code") for u in data["users"].values()}
    alphabet = "".join(c for c in string.ascii_uppercase + string.digits if c not in "O0I1L")
    while True:
        code = "".join(random.choice(alphabet) for _ in range(6))
        if code not in existing:
            return code


def doctor_by_code(data, code):
    code = code.strip().upper()
    for email, u in data["users"].items():
        if u["role"] == "doctor" and u.get("code") == code:
            return email
    return None


def create_user(data, role, name, email, password, **extra):
    salt, digest = hash_password(password)
    data["users"][email] = {"role": role, "name": name, "salt": salt, "hash": digest,
                            "fails": 0, "locked_until": None, **extra}


def register(data, role, name, email, password, confirm, agreed, surgery=None, sdate=None, code=None, clinic=""):
    """Returns an error message, or None after the account has been created."""
    email = norm_email(email)
    if not name.strip():
        return "Please enter your name."
    if not valid_email(email):
        return "Enter a valid email address."
    if email in data["users"]:
        return "An account with that email already exists. Try logging in."
    if len(password) < 8:
        return "Use a password with at least 8 characters."
    if password != confirm:
        return "The two passwords don't match."
    if not agreed:
        return "Please confirm that you won't enter real medical information."
    if role == "doctor":
        create_user(data, "doctor", name.strip(), email, password,
                    code=new_care_code(data), clinic=clinic.strip())
    else:
        doctor = doctor_by_code(data, code or "")
        if doctor is None:
            return "That care-team code wasn't found. Ask your doctor for their 6-character code."
        pid = uuid.uuid4().hex[:8]
        data["patients"][pid] = {"name": name.strip(), "surgery": surgery, "surgery_date": sdate.isoformat(),
                                 "doctor": doctor, "owner": email, "checkins": []}
        create_user(data, "patient", name.strip(), email, password, pid=pid)
    save_data(data)
    return None


def authenticate(data, role, email, password):
    """Returns (email, None) on success or (None, error message)."""
    email = norm_email(email)
    u = data["users"].get(email)
    salt = bytes.fromhex(u["salt"]) if u else b"\x00" * 16
    _, digest = hash_password(password, salt)            # same work whether or not the account exists
    if u is None:
        return None, "Email or password is incorrect."
    if u.get("locked_until") and datetime.fromisoformat(u["locked_until"]) > datetime.now():
        return None, f"Too many attempts. Try again in {LOCK_MINUTES} minutes."
    if not hmac.compare_digest(digest, u["hash"]):
        u["fails"] = u.get("fails", 0) + 1
        if u["fails"] >= MAX_FAILS:
            u["locked_until"] = (datetime.now() + timedelta(minutes=LOCK_MINUTES)).isoformat()
            u["fails"] = 0
        save_data(data)
        return None, "Email or password is incorrect."
    if u["role"] != role:
        return None, f"That account is registered as a {u['role']}. Switch the option above."
    u["fails"], u["locked_until"] = 0, None
    save_data(data)
    return email, None


def log_in(email):
    st.session_state["user"] = email
    st.session_state.pop("open_pid", None)
    st.session_state["page"] = "home"


def log_out():
    for k in ("user", "open_pid", "last_result", "photo_cache"):
        st.session_state.pop(k, None)
    st.session_state["page"] = "landing"


# ---------- photo models and alert engine ----------
@st.cache_resource(show_spinner=False)
def load_photo_model():
    """Load model.pt (normal vs needs_attention, made by train_model.py).
    Returns None if it can't be loaded, so the app keeps working without it."""
    return _load_mobilenet(MODEL_PATH, "classes")


@st.cache_resource(show_spinner=False)
def load_attribute_model():
    """Load attribute_model.pt (visible findings, made by train_attributes.py)."""
    return _load_mobilenet(ATTR_MODEL_PATH, "labels")


def _load_mobilenet(path, names_key):
    if not os.path.exists(path):
        return None
    try:
        import torch
        from torch import nn
        from torchvision import models, transforms

        ckpt = torch.load(path, map_location="cpu")
        names = ckpt[names_key]
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, len(names))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return {"model": model, "names": names, "tf": tf, "torch": torch}
    except Exception as e:  # missing torch, blocked DLL, bad file...
        print(f"Model unavailable ({path}):", e)
        return None


def _run(m, photo_path, activation):
    from PIL import Image
    torch = m["torch"]
    with Image.open(photo_path) as im:
        x = m["tf"](im.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out = m["model"](x)[0]
    return activation(torch, out)


def score_photo(photo_path):
    """P(needs_attention) between 0 and 1, or None if the model isn't available."""
    m = load_photo_model()
    if m is None:
        return None
    try:
        probs = _run(m, photo_path, lambda t, o: t.softmax(o, dim=0))
        return float(probs[m["names"].index("needs_attention")])
    except Exception as e:
        print("Photo scoring failed:", e)
        return None


def score_attributes(photo_path):
    """{"redness": p, "pus": p, ...} for every trained finding, or None if unavailable."""
    m = load_attribute_model()
    if m is None:
        return None
    try:
        probs = _run(m, photo_path, lambda t, o: t.sigmoid(o))
        return {name: float(probs[i]) for i, name in enumerate(m["names"])}
    except Exception as e:
        print("Attribute scoring failed:", e)
        return None


def analyze_photo(photo_path):
    """Run the photo models. Returns None if none are available, otherwise
    {"flag_label": "concerning"/"normal"/None, "flag_score": float/None,
     "findings": [detected findings], "scores": {...}}.
    Photo results can only RAISE a tier (to yellow), never lower it or make it red."""
    score = score_photo(photo_path)
    attrs = score_attributes(photo_path)
    if score is None and attrs is None:
        return None
    findings = [k for k in VISUAL_KEYS
                if attrs and attrs.get(k, 0.0) >= ATTR_THRESHOLD
                and (PREFILL_FINDINGS is None or k in PREFILL_FINDINGS)]
    label = None if score is None else ("concerning" if score >= PHOTO_THRESHOLD else "normal")
    return {"flag_label": label, "flag_score": score, "findings": findings, "scores": attrs or {}}


def evaluate(entry, prev=None, photo_label=None, photo_score=None, findings=None):
    """Return (tier, reasons, emergency)."""
    red, yellow = [], []
    for key, (label, level) in SYMPTOMS.items():
        if entry["symptoms"].get(key):
            (red if level == "red" else yellow).append(label)

    temp = entry["temp"]
    if temp >= 101.0:
        red.append(f"Fever of {temp:.1f} F")
    elif temp >= 100.0:
        yellow.append(f"Low-grade temperature of {temp:.1f} F")

    pain = entry["pain"]
    if pain >= 8:
        red.append(f"Severe pain ({pain}/10)")
    elif pain >= 6:
        yellow.append(f"Elevated pain ({pain}/10)")
    if prev is not None and pain - prev["pain"] >= 3:
        yellow.append("Pain jumped 3+ points since the last check-in")

    tier = "red" if red else "yellow" if yellow else "green"

    # Photo checks can only raise a green check-in to yellow. Red comes from symptoms.
    findings = findings or []
    if photo_label == "concerning" or findings:
        if tier == "green":
            tier = "yellow"
        if findings:
            yellow.append("Photo check: " + nice_list(findings) + " detected in the incision photo")
        else:
            yellow.append("Photo check: incision photo flagged for physician review")

    emergency = bool(entry["symptoms"].get("breath"))
    return tier, red + yellow, emergency


# ---------- shared UI ----------
def show_result(tier, reasons, emergency):
    if emergency:
        st.error("Chest pain or trouble breathing can be an emergency. Call 911 now.")
    if tier == "red":
        st.error("**Contact your surgeon's office today.** Your check-in has been flagged for your care team's review. If you can't reach them and symptoms are getting worse, go to urgent care or the ER.")
    elif tier == "yellow":
        st.warning("**Watch closely.** Your check-in has been flagged for your care team. Check in again tomorrow, and call your surgeon's office if anything gets worse.")
    else:
        st.success("**On track.** Nothing concerning in today's check-in. Keep going and check in again tomorrow.")
    for r in reasons:
        st.markdown(f'<div class="reason">{r}</div>', unsafe_allow_html=True)


def history_section(p):
    cks = p["checkins"]
    if not cks:
        st.info("No check-ins yet.")
        return
    df = pd.DataFrame([{"Day": c["day"], "Pain": c["pain"], "Temp": c["temp"]} for c in cks])
    st.caption("Pain (0-10) by recovery day")
    st.line_chart(df, x="Day", y="Pain", color="#0D9488", height=190)
    st.caption("Temperature (F) by recovery day")
    st.line_chart(df, x="Day", y="Temp", color="#0B1412", height=190)
    photos = [c for c in cks if c.get("photo") and os.path.exists(c["photo"])][-4:]
    if photos:
        st.markdown("### Incision photos over time")
        cols = st.columns(min(len(photos), 2))
        for i, c in enumerate(photos):
            cols[i % len(cols)].image(c["photo"], caption=f"Day {c['day']}")


def top_bar(user):
    with st.container(key="topbar"):
        c1, c2 = st.columns([3, 1.3])
        c1.markdown(f'<div class="brand"><div class="dot">{LOGO_SVG}</div>{APP_NAME}</div>', unsafe_allow_html=True)
        if c2.button("Log out", key="logout"):
            log_out()
            st.rerun()


# ---------- landing, login, register ----------
def landing_page():
    st.markdown(f'<div class="brand"><div class="dot">{LOGO_SVG}</div>{APP_NAME}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero">{arc_svg(5)}<h1>Your recovery,<br><span>watched over.</span></h1>'
        '<p>Daily check-ins and AI photo screening that help athletes and surgeons catch wound problems early.</p></div>'
        '<div class="feat"><div><b><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="#0D9488" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/></svg></b>60-second daily check-in</div>'
        '<div><b><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="#0D9488" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8h3l1.5-2h7L17 8h3v11H4z"/><circle cx="12" cy="13.5" r="3.5"/></svg></b>AI incision photo check</div><div><b><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="#0D9488" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M6 16v-5a6 6 0 0 1 12 0v5l2 2H4z"/><path d="M10 20a2 2 0 0 0 4 0"/></svg></b>Alerts for your surgeon</div></div>',
        unsafe_allow_html=True)
    if st.button("Get started", type="primary", key="go_register"):
        st.session_state["page"] = "register"
        st.rerun()
    if st.button("I already have an account", key="go_login"):
        st.session_state["page"] = "login"
        st.rerun()
    demo_buttons(load_data())
    st.markdown('<div class="fine">Prototype for demonstration. Not a medical device and not medical advice. '
                'In an emergency, call 911.</div>', unsafe_allow_html=True)
    reset_footer()


def demo_buttons(data):
    """Two one-tap demo logins shown under the register / log in buttons."""
    ensure_demo(data)
    st.markdown('<div class="fine" style="margin:.9rem 0 .4rem">or explore with demo data</div>', unsafe_allow_html=True)
    with st.container(key="demobar"):
        c1, c2 = st.columns(2)
        if c1.button("Demo patient", key="demo_pat"):
            log_in(DEMO_PATIENT_EMAIL)
            st.rerun()
        if c2.button("Demo doctor", key="demo_doc"):
            log_in(DEMO_DOCTOR_EMAIL)
            st.rerun()


def reset_footer():
    """A tiny, faint reset button at the very bottom of the logged-out pages."""
    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
    with st.container(key="resetbar"):
        if st.button("Reset all data", key="reset_all"):
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            log_out()
            st.rerun()


def auth_header(title, sub):
    if st.button("← Back", key="back_landing"):
        st.session_state["page"] = "landing"
        st.rerun()
    st.markdown(f"## {title}")
    st.caption(sub)


def login_page(data):
    auth_header("Welcome back", "Log in to your account.")
    role = st.radio("I am a", ["Patient", "Doctor"], horizontal=True, key="auth_role").lower()
    with st.form("login_form"):
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        go = st.form_submit_button("Log in", type="primary")
    if go:
        who, err = authenticate(data, role, email, pw)
        if err:
            st.error(err)
        else:
            log_in(who)
            st.rerun()
    if st.button("New here? Create an account", key="to_register"):
        st.session_state["page"] = "register"
        st.rerun()
    reset_footer()


def register_page(data):
    auth_header("Create your account", "Choose how you'll use the app.")
    role = st.radio("I am a", ["Patient", "Doctor"], horizontal=True, key="auth_role").lower()
    with st.form("register_form"):
        if role == "doctor":
            name = st.text_input("Your name", placeholder="Dr. Rivera")
            clinic = st.text_input("Practice or clinic (optional)")
        else:
            name = st.text_input("First name and last initial", placeholder="Alex K.",
                                 help="Use a nickname or initials. Don't enter real patient information in this prototype.")
            clinic = ""
        email = st.text_input("Email")
        pw = st.text_input("Password (8+ characters)", type="password")
        pw2 = st.text_input("Confirm password", type="password")
        surgery = sdate = code = None
        if role == "patient":
            surgery = st.selectbox("Surgery", list(SURGERIES))
            sdate = st.date_input("Surgery date", value=date.today(), max_value=date.today())
            code = st.text_input("Your doctor's care-team code", placeholder="6 characters",
                                 help=f"Your doctor sees this on their dashboard. Demo code: {DEMO_CODE}")
        agreed = st.checkbox("I understand this is a prototype and I won't enter real medical information.")
        go = st.form_submit_button("Create account", type="primary")
    if go:
        err = register(data, role, name, email, pw, pw2, agreed, surgery, sdate, code, clinic)
        if err:
            st.error(err)
        else:
            log_in(norm_email(email))
            st.rerun()
    if st.button("Already have an account? Log in", key="to_login"):
        st.session_state["page"] = "login"
        st.rerun()
    reset_footer()


# ---------- messages ----------
def esc(x):
    return html.escape(str(x))


def fmt_time(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %I:%M %p").replace(" 0", " ")
    except Exception:
        return ""


def unread_count(p, viewer):
    return sum(1 for m in p.get("messages", []) if m["from"] != viewer and not m.get("read"))


def doctor_name(data, p):
    return data["users"].get(p.get("doctor"), {}).get("name", "your care team")


def messages_panel(data, p, viewer, other_name):
    """Short (100-character) messages between a patient and their doctor."""
    msgs = p.setdefault("messages", [])
    st.caption(f"Short messages with {other_name}, 100 characters max. Messages aren't watched around the clock: "
               "for anything urgent call the surgeon's office, or 911 in an emergency.")
    if not msgs:
        st.info("No messages yet.")
    for m in msgs[-30:]:
        mine = m["from"] == viewer
        new = (not mine) and not m.get("read")
        who = "You" if mine else esc(other_name)
        st.markdown(
            f'<div class="msg {"mine" if mine else "theirs"}"><div class="bubble">{esc(m["text"])}</div>'
            f'<div class="meta">{who} · {fmt_time(m["time"])}{" · <span class=newmsg>New</span>" if new else ""}</div></div>',
            unsafe_allow_html=True)
    with st.form(f"msg_form_{viewer}", clear_on_submit=True):
        text = st.text_input("Message", max_chars=100, placeholder="Type a short message (100 characters max)",
                             label_visibility="collapsed")
        sent = st.form_submit_button("Send", type="primary")
    if sent:
        text = text.strip()[:100]
        if not text:
            st.warning("Type a message first.")
        else:
            msgs.append({"from": viewer, "text": text, "time": datetime.now().isoformat(timespec="seconds"), "read": False})
            save_data(data)
            st.rerun()
    changed = False                      # shown now; cleared on the next refresh
    for m in msgs:
        if m["from"] != viewer and not m.get("read"):
            m["read"], changed = True, True
    if changed:
        save_data(data)


# ---------- patient experience ----------
def photo_uploader(key):
    return st.file_uploader(
        "Incision photo (same angle and lighting each day works best)",
        type=["jpg", "jpeg", "png"], key=key)


def show_photo_findings(analysis):
    if analysis is None:
        st.caption("Photo saved for progress tracking and physician review.")
        return
    found = analysis["findings"]
    if found:
        st.warning("**Photo check:** " + nice_list(found) + " detected. "
                   "Please confirm below. Your care team will see this photo.")
    elif analysis["flag_label"] == "concerning":
        st.warning("**Photo check:** this photo looks different from typical healing, "
                   "so it will be flagged for your care team.")
    else:
        st.info("**Photo check:** nothing concerning detected. This tool can miss things, "
                "so still report anything below that worries you.")


# ---------------------------------------------------------------- AI recovery assistant
ASSISTANT_MODEL = "claude-haiku-5-5"
ASSISTANT_MAX_CHARS = 300
RED_FLAG_WORDS = (
    "chest pain", "can't breathe", "cant breathe", "short of breath", "fainting", "passed out",
    "pus", "wound opened", "wound open", "bleeding through", "fever", "calf pain", "numb",
    "suicid", "kill myself",
)


def _secret(name):
    val = os.environ.get(name, "")
    if not val:
        try:
            val = st.secrets.get(name, "")
        except Exception:
            val = ""
    return val


def _provider():
    """Gemini (free tier) if GEMINI_API_KEY is set, otherwise Claude if ANTHROPIC_API_KEY is set."""
    if _secret("GEMINI_API_KEY"):
        return "gemini"
    if _secret("ANTHROPIC_API_KEY"):
        return "claude"
    return None


def _api_key():
    return _provider()


def assistant_system_prompt(p, day, tier_label):
    return (
        "You are the recovery assistant inside OrthoGuard, an app for athletes recovering from sports surgery. "
        f"The patient had: {p['surgery']}. They are on day {day} of recovery. "
        f"Their latest check-in status: {tier_label}.\n"
        "Rules:\n"
        "- Answer in plain, calm language, in 4 sentences or fewer unless asked for more.\n"
        "- Give general education only. You are not their doctor. Never diagnose, never say a wound is or is not infected, "
        "and never give medication names or doses.\n"
        "- For activity, weight-bearing, driving or return-to-sport questions, say the surgeon's protocol decides and "
        "suggest they message their care team in the Chat tab.\n"
        "- If the patient describes fever, spreading redness, pus or drainage, an open wound, heavy bleeding, calf pain or "
        "swelling, chest pain, trouble breathing, or feeling very unwell, tell them to contact their surgeon today, and to "
        "call 911 for chest pain or trouble breathing.\n"
        "- If asked about anything unrelated to recovery, politely steer back.\n"
        "- If they mention wanting to hurt themselves, respond with care and tell them to call or text 988 (US)."
    )


def ask_assistant(history, system):
    history = history[-10:]
    if _provider() == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=_secret("GEMINI_API_KEY"))
        contents = [types.Content(role="user" if m["role"] == "user" else "model",
                                  parts=[types.Part(text=m["content"])]) for m in history]
        resp = client.models.generate_content(
            model=_secret("GEMINI_MODEL") or "gemini-3.1-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=500))
        return (resp.text or "").strip()
    import anthropic
    client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"), timeout=20.0)
    resp = client.messages.create(
        model=ASSISTANT_MODEL, max_tokens=400, system=system, messages=history)
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def assistant_tab(data, p, pid, day):
    st.caption("Ask general questions about your recovery. This is not medical advice and does not replace your care team.")
    key = f"assist_{pid}"
    hist = st.session_state.setdefault(key, [])
    last = p["checkins"][-1] if p["checkins"] else None
    tier_label = last.get("tier", "no check-in yet") if last else "no check-in yet"

    if not _api_key():
        st.info("The assistant is not set up on this device. Add GEMINI_API_KEY to turn it on.")
        return

    for m in hist:
        who = "You" if m["role"] == "user" else "Assistant"
        cls = "reason" if m["role"] == "assistant" else "reason"
        st.markdown(f'<div class="{cls}"><b>{who}</b><br>{esc(m["content"])}</div>', unsafe_allow_html=True)

    with st.form(f"assist_form_{pid}", clear_on_submit=True):
        q = st.text_input("Your question", max_chars=ASSISTANT_MAX_CHARS,
                          placeholder="For example: Is some swelling normal at this stage?")
        sent = st.form_submit_button("Ask", use_container_width=True)
    if sent and q.strip():
        q = q.strip()[:ASSISTANT_MAX_CHARS]
        hist.append({"role": "user", "content": q})
        try:
            with st.spinner("Thinking..."):
                ans = ask_assistant(hist, assistant_system_prompt(p, day, tier_label))
            if not ans:
                ans = "I could not answer that. Please message your care team in the Chat tab."
        except Exception as e:
            print("Assistant error:", repr(e))
            ans = "The assistant is unavailable right now. Please message your care team in the Chat tab."
        if any(w in q.lower() for w in RED_FLAG_WORDS):
            ans += "\n\nIf this is happening now, contact your surgeon today. Call 911 for chest pain or trouble breathing."
        hist.append({"role": "assistant", "content": ans})
        st.rerun()
    if hist and st.button("Clear conversation", key=f"assist_clear_{pid}"):
        st.session_state[key] = []
        st.rerun()


def patient_view(data, user):
    pid = user["pid"]
    p = data["patients"][pid]
    day = (date.today() - date.fromisoformat(p["surgery_date"])).days
    last = p["checkins"][-1] if p["checkins"] else None
    done_today = bool(last) and last["date"] == date.today().isoformat()
    doc = esc(doctor_name(data, p))
    state = '<span class="tag tag-teal">Today\'s check-in done</span>' if done_today \
        else '<span class="tag tag-soft">Check-in due today</span>'
    st.markdown(
        f'<div class="dark-card"><div><div class="hi">Hi, {esc(first_name(p["name"]))}</div>'
        f'<div class="sm">{p["surgery"]}</div><div class="sm">Care team: <b style="color:#fff">{doc}</b></div>'
        f'<div style="margin-top:.55rem">{state}</div></div>'
        f'<div style="text-align:right"><div class="big">Day {day}</div><div class="sm">of recovery</div></div></div>',
        unsafe_allow_html=True)

    unread = unread_count(p, "patient")
    t1, t2, t3, t4, t5 = st.tabs(["Check-in", "Progress", f"Chat ({unread})" if unread else "Chat", "Ask", "Plan"])
    with t1:
        checkin_tab(data, p, pid, day)
    with t2:
        history_section(p)
    with t3:
        messages_panel(data, p, "patient", doc)
    with t4:
        assistant_tab(data, p, pid, day)
    with t5:
        st.caption("General examples. Follow your surgeon's protocol.")
        for span, text in SURGERIES[p["surgery"]]:
            st.markdown(f'<div class="reason"><b>{span}</b><br>{text}</div>', unsafe_allow_html=True)


def checkin_tab(data, p, pid, day):
    last_res = st.session_state.pop("last_result", None)
    if last_res and last_res["pid"] == pid:
        show_result(last_res["tier"], last_res["reasons"], last_res["emergency"])

    st.markdown("### Today's check-in")
    n = st.session_state.get("upl_n", 0)
    photo = photo_uploader(f"photo_{pid}_{n}")

    photo_path, analysis, ck = None, None, "none"
    if photo is not None:
        ck = f"{pid}_{n}_{photo.name}_{photo.size}"
        cache = st.session_state.setdefault("photo_cache", {})
        if ck not in cache:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            ext = os.path.splitext(photo.name)[1].lower() or ".jpg"
            path = os.path.join(UPLOAD_DIR, f"{pid}_{uuid.uuid4().hex[:8]}{ext}")
            with open(path, "wb") as f:
                f.write(photo.getbuffer())
            with st.spinner("Checking your photo..."):
                cache[ck] = {"path": path, "analysis": analyze_photo(path)}
        photo_path = cache[ck]["path"]
        analysis = cache[ck]["analysis"]
        show_photo_findings(analysis)

    found = set(analysis["findings"]) if analysis else set()
    flagged = bool(analysis) and (bool(found) or analysis["flag_label"] == "concerning")

    with st.form("checkin", clear_on_submit=True):
        c1, c2 = st.columns(2)
        temp = c1.number_input("Temperature (F)", 95.0, 106.0, 98.6, 0.1)
        pain = c2.slider("Pain (0-10)", 0, 10, 3)

        st.markdown("**Does any of this apply today?**")
        sym = {}
        for key in PATIENT_KEYS:
            sym[key] = st.checkbox(SYMPTOMS[key][0], key=f"sym_{key}_{ck}")

        # Open when we have no photo (we need the patient's input) or the photo raised a flag.
        with st.expander("What does the incision look like?", expanded=(analysis is None or flagged)):
            if analysis is not None:
                st.caption("Pre-filled from your photo. Please correct anything that's wrong.")
            for key in VISUAL_KEYS:
                sym[key] = st.checkbox(SYMPTOMS[key][0], value=(key in found), key=f"sym_{key}_{ck}")

        notes = st.text_area("Notes for your care team (optional)")
        submitted = st.form_submit_button("Submit check-in", type="primary")

    if submitted:
        prev = p["checkins"][-1] if p["checkins"] else None
        a = analysis or {}
        entry = {
            "date": date.today().isoformat(), "day": day, "temp": float(temp), "pain": int(pain),
            "symptoms": sym, "notes": notes.strip(), "photo": photo_path, "reviewed": False,
        }
        # Detected findings only pre-fill the checkboxes above. The tier uses what the patient CONFIRMED
        # (entry["symptoms"]) plus the whole-photo flag, so an unconfirmed false alarm can't raise the tier.
        tier, reasons, emergency = evaluate(
            entry, prev, a.get("flag_label"), a.get("flag_score"), None)
        entry.update(
            tier=tier, reasons=reasons, emergency=emergency,
            photo_score=a.get("flag_score"), photo_findings=a.get("findings", []),
            photo_flag=bool(a.get("findings")) or a.get("flag_label") == "concerning",
        )
        p["checkins"].append(entry)
        save_data(data)
        st.session_state["last_result"] = {"pid": pid, "tier": tier, "reasons": reasons, "emergency": emergency}
        st.session_state["upl_n"] = n + 1   # fresh, empty photo box next time
        st.rerun()


# ---------- doctor experience ----------
def summarize(pid, p):
    last = p["checkins"][-1] if p["checkins"] else None
    tier = last["tier"] if last else None
    flagged = bool(last) and last["tier"] != "green" and not last.get("reviewed")
    day = last["day"] if last else (date.today() - date.fromisoformat(p["surgery_date"])).days
    return {"pid": pid, "p": p, "last": last, "tier": tier, "flagged": flagged, "day": day,
            "rank": TIER_RANK[tier] if tier else -1}


def doctor_view(data, user, email):
    mine = {pid: p for pid, p in data["patients"].items() if p.get("doctor") == email}
    open_pid = st.session_state.get("open_pid")
    if open_pid in mine:
        return patient_profile(data, open_pid, mine[open_pid])

    st.markdown(f"## Hello, {user['name']}")
    st.markdown(f'<div class="code-box"><div><div style="font-size:.78rem;color:#4B6360;font-weight:600">'
                f'Your care-team code</div><b>{user["code"]}</b></div>'
                f'<div style="font-size:.78rem;color:#4B6360;max-width:170px;text-align:right">'
                f'Patients enter this when they sign up.</div></div>', unsafe_allow_html=True)

    rows = sorted((summarize(pid, p) for pid, p in mine.items()),
                  key=lambda r: (-int(r["flagged"]), -r["rank"], r["p"]["name"]))
    st.markdown(
        f'<div class="stats"><div class="stat"><b>{len(rows)}</b><span>Patients</span></div>'
        f'<div class="stat {"hot" if any(r["flagged"] for r in rows) else ""}"><b>{sum(r["flagged"] for r in rows)}</b>'
        f'<span>Need review</span></div>'
        f'<div class="stat"><b>{sum(r["rank"] == 2 for r in rows)}</b><span>Red status</span></div></div>',
        unsafe_allow_html=True)

    if not rows:
        st.info("No patients yet. When a patient registers with your code, they'll appear here.")
        return

    st.markdown("### Your patients")
    q = st.text_input("Search", placeholder="Search by name or surgery", label_visibility="collapsed")
    only = st.radio("Show", ["All", "Needs review"], horizontal=True, label_visibility="collapsed")
    shown = [r for r in rows if (not q or q.lower() in (r["p"]["name"] + r["p"]["surgery"]).lower())
             and (only == "All" or r["flagged"])]
    if not shown:
        st.caption("No patients match.")
    for r in shown:
        p = r["p"]
        with st.container(border=True, key=f"card_{r['pid']}"):
            flag = ' <span class="flag">Needs review</span>' if r["flagged"] else ""
            if unread_count(p, "doctor"):
                flag += ' <span class="newmsg">New message</span>'
            st.markdown(
                f'<div class="prow"><div class="avatar">{esc(initials(p["name"]))}</div>'
                f'<div class="pinfo"><b>{esc(p["name"])}</b><span>{p["surgery"]} · Day {r["day"]}</span><br>'
                f'{chip(r["tier"])}{flag}</div></div>', unsafe_allow_html=True)
            if st.button("Open profile  →", key=f"open_{r['pid']}"):
                st.session_state["open_pid"] = r["pid"]
                st.rerun()


def patient_profile(data, pid, p):
    s = summarize(pid, p)
    last = s["last"]
    if st.button("← All patients", key="back_list"):
        st.session_state.pop("open_pid", None)
        st.rerun()
    st.markdown(
        f'<div class="dark-card"><div class="prow"><div class="avatar lg">{esc(initials(p["name"]))}</div>'
        f'<div><div class="hi">{esc(p["name"])}</div><div class="sm">{p["surgery"]}</div>'
        f'<div class="sm">Surgery {p["surgery_date"]}</div></div></div>'
        f'<div style="text-align:right"><div class="big">Day {s["day"]}</div><div style="margin-top:.4rem">{chip(s["tier"])}</div></div></div>',
        unsafe_allow_html=True)

    unread = unread_count(p, "doctor")
    t1, t2, t3, t4 = st.tabs(["Overview", "Trends", "Check-ins", f"Chat ({unread})" if unread else "Chat"])
    with t1:
        if not last:
            st.info("This patient hasn't checked in yet.")
        else:
            st.markdown(f"**Latest check-in:** {last['date']} (day {last['day']})")
            if last["reasons"]:
                for r in last["reasons"]:
                    st.markdown(f'<div class="reason">{r}</div>', unsafe_allow_html=True)
            else:
                st.success("Nothing concerning reported.")
            c1, c2 = st.columns(2)
            c1.metric("Temperature", f"{last['temp']:.1f} F")
            c2.metric("Pain", f"{last['pain']}/10")
            if last.get("notes"):
                st.markdown(f"**Patient note:** {last['notes']}")
            if last.get("photo") and os.path.exists(last["photo"]):
                st.image(last["photo"], caption="Latest incision photo")
                found = last.get("photo_findings") or []
                if found:
                    st.warning("AI photo screening detected: " + nice_list(found))
                elif last.get("photo_flag"):
                    st.warning("AI photo screening flagged this photo for review.")
            if last["tier"] != "green":
                if last.get("reviewed"):
                    st.success("Reviewed.")
                elif st.button("Mark latest check-in as reviewed", type="primary", key="mark_reviewed"):
                    last["reviewed"] = True
                    save_data(data)
                    st.rerun()
    with t2:
        history_section(p)
    with t3:
        if not p["checkins"]:
            st.info("No check-ins yet.")
        else:
            log = pd.DataFrame([{
                "Date": c["date"], "Day": c["day"], "Status": c["tier"].capitalize(),
                "Temp (F)": c["temp"], "Pain": c["pain"], "Flags": "; ".join(c["reasons"]),
            } for c in reversed(p["checkins"])])
            st.dataframe(log, hide_index=True)
    with t4:
        messages_panel(data, p, "doctor", p["name"])


# ---------- demo data ----------
def ensure_demo(data):
    """Creates the demo doctor, three demo patients and the demo patient login (once)."""
    if DEMO_DOCTOR_EMAIL in data["users"]:
        return
    today = date.today()

    def mk(day, temp, pain, **syms):
        return {"day": day, "temp": temp, "pain": pain, "symptoms": syms}

    create_user(data, "doctor", "Dr. Rivera", DEMO_DOCTOR_EMAIL, uuid.uuid4().hex, code=DEMO_CODE, clinic="Demo Sports Medicine")
    demos = [
        ("Jordan M.", "ACL reconstruction", 12,
         [mk(d, 98.4 + (d % 3) * 0.2, max(1, 6 - d // 2)) for d in range(2, 13)]),
        ("Maya R.", "Achilles tendon repair", 10,
         [mk(d, 98.6, 4) for d in range(2, 7)]
         + [mk(7, 99.0, 4, redness=True), mk(8, 99.4, 5, redness=True, warm=True),
            mk(9, 99.8, 6, redness=True, swelling=True, warm=True),
            mk(10, 100.2, 6, redness=True, swelling=True, warm=True, clear=True)]),
        ("Chris T.", "Rotator cuff repair", 9,
         [mk(d, 98.6, 3) for d in range(2, 7)]
         + [mk(7, 99.5, 4, redness=True), mk(8, 100.4, 5, redness=True, swelling=True),
            mk(9, 101.6, 8, spreading=True, pus=True, warm=True, swelling=True)]),
    ]
    for name, surgery, days_ago, checkins in demos:
        sdate = today - timedelta(days=days_ago)
        pid = uuid.uuid4().hex[:8]
        p = {"name": name, "surgery": surgery, "surgery_date": sdate.isoformat(),
             "doctor": DEMO_DOCTOR_EMAIL, "owner": None, "checkins": [], "demo": True}
        prev = None
        for c in checkins:
            entry = {**c, "date": (sdate + timedelta(days=c["day"])).isoformat(),
                     "notes": "", "photo": None, "reviewed": False}
            tier, reasons, emergency = evaluate(entry, prev)
            entry.update(tier=tier, reasons=reasons, emergency=emergency)
            p["checkins"].append(entry)
            prev = entry
        if name == "Maya R.":
            p["messages"] = [{"from": "doctor", "text": "Hi Maya, how does the incision look today? Send a photo in your check-in.",
                              "time": datetime.now().isoformat(timespec="seconds"), "read": False}]
        if name == "Maya R.":   # the demo patient logs in as Maya
            p["owner"] = DEMO_PATIENT_EMAIL
            create_user(data, "patient", name, DEMO_PATIENT_EMAIL, uuid.uuid4().hex, pid=pid)
        data["patients"][pid] = p
    save_data(data)


# ---------- main ----------
def main():
    inject_css()
    data = load_data()
    page = st.session_state.get("page", "landing")
    email = st.session_state.get("user")
    user = data["users"].get(email) if email else None
    if email and user is None:       # account vanished (data reset)
        log_out()
        page, email = "landing", None

    if user is None:
        if page == "login":
            login_page(data)
        elif page == "register":
            register_page(data)
        else:
            landing_page()
        return

    top_bar(user)
    if user["role"] == "doctor":
        doctor_view(data, user, email)
    else:
        patient_view(data, user)
    st.markdown('<div class="fine">Prototype for demonstration. Not a medical device and not medical advice. '
                'Do not enter real patient information. In an emergency, call 911.</div>', unsafe_allow_html=True)


main()
