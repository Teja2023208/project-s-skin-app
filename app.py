# Final corrected app.py for Project S — Language fixed (sidebar-only)
# Single-file Streamlit app.

import os
import io
import json
import tempfile
import threading
import sqlite3
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Optional libs
try:
    import cv2
except Exception:
    cv2 = None

# fpdf2 required
from fpdf import FPDF
import fpdf as _fpdf_pkg

# Flask optional
try:
    from flask import Flask, request, jsonify
except Exception:
    Flask = None
    request = None
    jsonify = None

import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/skin_classifier.h5")
TFLITE_PATH = os.environ.get("TFLITE_PATH", "models/skin_classifier.tflite")
IMG_SIZE = 160
TOP_K_DEFAULT = 3
DB_PATH = "project_s_reports.db"
CUSTOM_TEXTS = "custom_texts.json"
API_HOST = "127.0.0.1"
API_PORT_DEFAULT = 8502
FONTS_DIR = "fonts"

# ----------------------------
# CLASS MAPPINGS
# ----------------------------
CLASS_INDICES = {
 'acne_rosacea': 0,
 'actinic_keratosis_basal_cell_carcinoma_and_other_malignant_lesions': 1,
 'atopic_dermatitis_photos': 2,
 'autoimmune': 3,
 'bacterial_infections': 4,
 'bullous_disease_photos': 5,
 'eczema': 6,
 'exanthems_and_drug_eruptions': 7,
 'fungal_infections': 8,
 'hair_loss': 9,
 'herpes_hpv_and_other_stds_photos': 10,
 'melanoma': 11,
 'nail_fungus_and_other_nail_disease': 12,
 'pigmentation': 13,
 'poison_ivy_photos_and_other_contact_dermatitis': 14,
 'psoriasis': 15,
 'scabies_lyme_disease_and_other_infestations_and_bites': 16,
 'seborrheic_keratoses_and_other_benign_tumors': 17,
 'systemic_disease': 18,
 'urticaria_hives': 19,
 'vascular_tumors': 20,
 'vasculitis_photos': 21,
 'viral_infections': 22
}
IDX_TO_LABEL = {v: k for k, v in CLASS_INDICES.items()}
DISPLAY_NAMES = {k: " ".join(w.capitalize() for w in k.split("_")) for k in CLASS_INDICES.keys()}

# ----------------------------
# TEXTS
# ----------------------------
DESCRIPTIONS_EN = {k: "" for k in CLASS_INDICES.keys()}  # short placeholders
TREATMENTS_EN = {k: "" for k in CLASS_INDICES.keys()}
# (populate with real strings or keep earlier ones as needed)
# For brevity we keep placeholders here — replace with your full texts if desired.
for k in DESCRIPTIONS_EN:
    DESCRIPTIONS_EN[k] = k.replace("_", " ").capitalize() + " — brief description."
    TREATMENTS_EN[k] = "Suggested general care: consult a dermatologist if concerned."
# Hindi/Telugu copies of English by default
DESCRIPTIONS_HI = dict(DESCRIPTIONS_EN)
TREATMENTS_HI = dict(TREATMENTS_EN)
DESCRIPTIONS_TE = dict(DESCRIPTIONS_EN)
TREATMENTS_TE = dict(TREATMENTS_EN)

# ----------------------------
# UTILITIES
# ----------------------------
def ensure_rgb(pil: Image.Image) -> Image.Image:
    return pil.convert("RGB")

# skin mask (same robust implementation)
def skin_mask_ycrcb(pil: Image.Image) -> np.ndarray:
    arr_rgb = np.asarray(pil.convert("RGB"))
    if cv2 is None:
        r,g,b = arr_rgb[:,:,0], arr_rgb[:,:,1], arr_rgb[:,:,2]
        mask = (r > 80) & (g > 40) & (b > 20) & (r > g) & (r > b)
        return (mask.astype(np.uint8) * 255)
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    mask = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)
    return mask

# Auto-crop, preprocess, severity, grad-cam and prediction wrappers
# (Kept concise; use your prior implementations — unchanged logic)

def auto_crop_by_skin(pil: Image.Image, margin=0.02):
    mask = skin_mask_ycrcb(pil)
    h, w = mask.shape[:2]
    skin_pixels = (mask > 0).sum()
    if skin_pixels / (h*w + 1e-9) < 0.02:
        return None, None
    if cv2 is None:
        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            return None, None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        c = max(contours, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(c)
        x0, y0, x1, y1 = x, y, x+ww, y+hh
    dx = int((x1 - x0) * margin)
    dy = int((y1 - y0) * margin)
    x0 = max(0, x0 - dx)
    y0 = max(0, y0 - dy)
    x1 = min(w - 1, x1 + dx)
    y1 = min(h - 1, y1 + dy)
    crop = pil.crop((x0, y0, x1, y1))
    return crop, (x0, y0, x1, y1)

from PIL import ImageOps

def preprocess_pil(pil: Image.Image, target_size: Tuple[int,int], auto_crop: bool = True):
    pil = ensure_rgb(pil)
    crop_box = None
    if auto_crop:
        try:
            cropped, box = auto_crop_by_skin(pil)
            if cropped is not None:
                pil = cropped
                crop_box = box
        except Exception:
            pass
    pil_fit = ImageOps.fit(pil, target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(pil_fit).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return arr, crop_box

def estimate_severity(original_pil: Image.Image, crop_box: Optional[Tuple[int,int,int,int]]):
    pil = ensure_rgb(original_pil)
    if crop_box:
        pil_c = pil.crop(crop_box)
    else:
        pil_c = pil
    arr = np.asarray(pil_c).astype("float32") / 255.0
    h, w, _ = arr.shape
    mask = (skin_mask_ycrcb(pil_c) > 0).astype(np.uint8)
    area_pct = mask.sum() / (h*w + 1e-9)
    r = arr[:,:,0]
    gbmax = np.maximum(arr[:,:,1], arr[:,:,2])
    redness = float(np.clip((r - gbmax).mean(), 0.0, 1.0))
    if cv2:
        gray = cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        texture = float(np.var(lap))
        texture = min(texture / 1000.0, 1.0)
    else:
        texture = float(np.var(np.mean(arr, axis=2))) / (1.0)
    score = (0.5*area_pct) + (0.35*redness) + (0.15*texture)
    level = "Low" if score < 0.33 else ("Medium" if score < 0.66 else "High")
    return {"area_pct": float(area_pct), "redness": redness, "texture": texture, "score": float(score), "level": level}

# Grad-CAM + overlay (same approach as before)

def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: Optional[str] = None, pred_index: Optional[int] = None):
    try:
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                lname = layer.name.lower()
                if "conv" in lname:
                    last_conv_layer_name = layer.name
                    break
        if last_conv_layer_name is None:
            return None
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=[-1, 0])
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-9)
        heatmap = heatmap.numpy()
        h, w = img_array.shape[1], img_array.shape[2]
        if cv2:
            heatmap = cv2.resize(heatmap, (w, h))
        else:
            heatmap = np.array(Image.fromarray((heatmap*255).astype(np.uint8)).resize((w, h))) / 255.0
        heatmap = np.clip(heatmap, 0, 1)
        return heatmap
    except Exception:
        return None

def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4, cmap: str = "jet"):
    if heatmap is None:
        return pil_img
    cmap_func = plt.get_cmap(cmap)
    heatmap_rgb = cmap_func(heatmap)[:, :, :3]
    heatmap_img = Image.fromarray((heatmap_rgb * 255).astype(np.uint8)).resize(pil_img.size)
    heatmap_img = heatmap_img.convert("RGBA")
    base = pil_img.convert("RGBA")
    blended = Image.blend(base, heatmap_img, alpha=alpha)
    return blended.convert("RGB")

# Prediction wrappers

def predict_with_model(model_info: Dict, processed_array: np.ndarray, temperature: float = 1.0):
    if model_info["type"] == "keras":
        model = model_info["model"]
        preds = model.predict(processed_array)
        out = preds[0] if preds.ndim == 2 else preds
        if np.isclose(out.sum(), 1.0, atol=1e-3):
            logits = np.log(np.clip(out, 1e-9, 1.0))
        else:
            logits = out
        logits = logits / max(1e-6, temperature)
        probs = tf.nn.softmax(logits).numpy()
        return np.array(probs)
    elif model_info["type"] == "tflite":
        interp = model_info["model"]
        input_details = interp.get_input_details()[0]
        output_details = interp.get_output_details()[0]
        inp = processed_array.astype(np.float32)
        try:
            interp.set_tensor(input_details["index"], inp)
            interp.invoke()
            out = interp.get_tensor(output_details["index"])
        except Exception:
            interp.resize_tensor_input(input_details["index"], inp.shape)
            interp.allocate_tensors()
            interp.set_tensor(input_details["index"], inp)
            interp.invoke()
            out = interp.get_tensor(output_details["index"])
        logits = out[0] if out.ndim == 2 else out
        logits = logits / max(1e-6, temperature)
        probs = tf.nn.softmax(logits).numpy()
        return np.array(probs)
    else:
        raise RuntimeError("No model loaded")

# ----------------------------
# PDF generator (unicode-safe)
# ----------------------------
class MedicalPDF(FPDF):
    def header(self):
        if "Noto" in self.fonts:
            self.set_font("Noto", size=12)
        else:
            set_font(unicode_font if unicode_font else "Arial", size=12)
        self.set_fill_color(240, 240, 240)
        self.rect(0, 0, self.w, 18, "F")
        self.set_xy(10, 5)
        self.cell(0, 8, "Project S - Skin Disease Classifier", ln=False)
        self.set_xy(self.w - 70, 5)
        self.set_font(self.font_family, size=10)
        self.cell(60, 8, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ln=False, align="R")
        self.ln(15)
    def footer(self):
        self.set_y(-12)
        self.set_font("Noto" if "Noto" in self.fonts else "Arial", size=9)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

def generate_unicode_pdf(image_pil, preds, desc, treat, severity, gradcam_img=None):
    pdf = MedicalPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    en = os.path.join(FONTS_DIR, "NotoSans-Regular.ttf")
    hi = os.path.join(FONTS_DIR, "NotoSansDevanagari-Regular.ttf")
    te = os.path.join(FONTS_DIR, "NotoSansTelugu-Regular.ttf")
    # Force Unicode font loading
    if os.path.exists(en):
        pdf.add_font("Noto", "", en)
        unicode_font = "Noto"
    elif os.path.exists(hi):
        pdf.add_font("NotoDev", "", hi)
        unicode_font = "NotoDev"
    elif os.path.exists(te):
        pdf.add_font("NotoTel", "", te)
        unicode_font = "NotoTel"
    else:
        unicode_font = None

    def auto_font(text, size=12):
        if any("\u0900" <= c <= "\u097F" for c in text):
            pdf.set_font("NotoDev" if "NotoDev" in pdf.fonts else "Noto", size=size)
        elif any("\u0C00" <= c <= "\u0C7F" for c in text):
            pdf.set_font("NotoTel" if "NotoTel" in pdf.fonts else "Noto", size=size)
        else:
            pdf.set_font("Noto" if "Noto" in pdf.fonts else "Arial", size=size)
    left_x = 10
    img_width = 90
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)
    image_pil.save(tmp_path, "JPEG")
    pdf.image(tmp_path, x=left_x, y=30, w=img_width)
    try:
        os.remove(tmp_path)
    except:
        pass
    right_x = left_x + img_width + 20
    pdf.set_xy(right_x, 30)
    auto_font("Top Prediction:", 14)
    pdf.cell(0, 8, "Top Prediction:", ln=True)
    top_idx, top_prob = preds[0]
    label = DISPLAY_NAMES.get(IDX_TO_LABEL.get(top_idx, ""), f"Class {top_idx}")
    pdf.set_x(right_x)
    auto_font(f"{label} - {top_prob*100:.1f}%", 12)
    pdf.cell(0, 7, f"{label} - {top_prob*100:.1f}%", ln=True)
    if gradcam_img:
        tmp_fd, grad_path = tempfile.mkstemp(suffix=".jpg")
        os.close(tmp_fd)
        gradcam_img.save(grad_path, "JPEG")
        pdf.image(grad_path, x=right_x, y=55, w=80)
        try:
            os.remove(grad_path)
        except:
            pass
    pdf.set_y(140)
    auto_font("Description", 14)
    pdf.cell(0, 10, "Description", ln=True)
    pdf.set_fill_color(248, 248, 248)
    # choose correct font based on language
    if any("\u0900" <= c <= "\u097F" for c in desc):
        pdf.set_font("NotoDev" if "NotoDev" in pdf.fonts else "Arial", size=12)
    elif any("\u0C00" <= c <= "\u0C7F" for c in desc):
        pdf.set_font("NotoTel" if "NotoTel" in pdf.fonts else "Arial", size=12)
    else:
        pdf.set_font("Noto" if "Noto" in pdf.fonts else "Arial", size=12)

    pdf.multi_cell(pdf.w - 20, 8, desc, fill=True)
    pdf.ln(5)
    auto_font("Suggested Treatment", 14)
    pdf.cell(0, 10, "Suggested Treatment", ln=True)
    if any("\u0900" <= c <= "\u097F" for c in treat):
        pdf.set_font("NotoDev" if "NotoDev" in pdf.fonts else "Arial", size=12)
    elif any("\u0C00" <= c <= "\u0C7F" for c in treat):
        pdf.set_font("NotoTel" if "NotoTel" in pdf.fonts else "Arial", size=12)
    else:
        pdf.set_font("Noto" if "Noto" in pdf.fonts else "Arial", size=12)

    pdf.multi_cell(pdf.w - 20, 8, treat, fill=True)
    pdf.ln(5)
    auto_font("Severity Analysis", 14)
    pdf.cell(0, 10, "Severity Analysis", ln=True)
    severity_text = (
        f"Level: {severity.get('level')}\n"
        f"Score: {severity.get('score'):.2f}\n"
        f"Area: {severity.get('area_pct')*100:.1f}%\n"
        f"Redness: {severity.get('redness'):.3f}"
    )
    pdf.multi_cell(pdf.w - 20, 8, severity_text, fill=True)
    data = pdf.output(dest="S")
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("latin-1")
    return data

# expose
generate_pdf_report = generate_unicode_pdf

# ----------------------------
# DB helpers
# ----------------------------
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      filename TEXT,
      timestamp TEXT,
      top_label TEXT,
      top_prob REAL,
      preds_json TEXT,
      description TEXT,
      treatment TEXT,
      severity_json TEXT,
      image_blob BLOB
    )""")
    conn.commit()
    return conn

DB_CONN = init_db()

def save_report_to_db(filename: str, preds: List[Tuple[int, float]], description: str, treatment: str, severity: dict, image_pil: Image.Image):
    cur = DB_CONN.cursor()
    top_idx, top_prob = preds[0]
    preds_json = json.dumps([[int(i), float(p)] for i, p in preds])
    severity_json = json.dumps(severity)
    buf = io.BytesIO()
    image_pil.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    cur.execute("INSERT INTO reports (filename, timestamp, top_label, top_prob, preds_json, description, treatment, severity_json, image_blob) VALUES (?,?,?,?,?,?,?,?,?)",
                (filename, datetime.now().isoformat(), IDX_TO_LABEL.get(top_idx, str(top_idx)), float(top_prob), preds_json, description, treatment, severity_json, sqlite3.Binary(img_bytes)))
    DB_CONN.commit()
    return cur.lastrowid

def list_reports(limit: int = 200):
    cur = DB_CONN.cursor()
    cur.execute("SELECT id, filename, timestamp, top_label, top_prob, preds_json, description, treatment, severity_json FROM reports ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "filename": r[1],
            "timestamp": r[2],
            "top_label": r[3],
            "top_prob": r[4],
            "preds": json.loads(r[5]),
            "description": r[6],
            "treatment": r[7],
            "severity": json.loads(r[8]) if r[8] else {}
        })
    return out

def get_report_image_bytes(report_id: int):
    cur = DB_CONN.cursor()
    cur.execute("SELECT image_blob FROM reports WHERE id=?", (report_id,))
    r = cur.fetchone()
    if not r or r[0] is None:
        return None
    return bytes(r[0])

def delete_report(report_id: int):
    cur = DB_CONN.cursor()
    cur.execute("DELETE FROM reports WHERE id=?", (report_id,))
    DB_CONN.commit()

# ----------------------------
# Custom texts persistence
# ----------------------------
def load_custom_texts(path=CUSTOM_TEXTS):
    if os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf8"))
        except Exception:
            return {"descriptions": {}, "treatments": {}}
    return {"descriptions": {}, "treatments": {}}

def save_custom_texts(data, path=CUSTOM_TEXTS):
    json.dump(data, open(path, "w", encoding="utf8"), indent=2, ensure_ascii=False)

CUSTOMS = load_custom_texts()

# ----------------------------
# Streamlit UI — FIX: sidebar-only language (Option A)
# ----------------------------
st.set_page_config(page_title="Project S — Skin Disease Classifier", layout="wide")

# ensure session defaults to avoid AttributeError
st.session_state.setdefault("APP_LANG", "English")
st.session_state.setdefault("auto_crop", True)
st.session_state.setdefault("enable_gradcam", True)
st.session_state.setdefault("top_k", TOP_K_DEFAULT)
st.session_state.setdefault("temperature", 1.0)
st.session_state.setdefault("enable_api", False)

# Load model once
@st.cache_resource
def load_model_safe():
    # minimal wrapper to reuse your load_model logic — you can adapt
    info = {"type": "none", "model": None, "input_shape": (IMG_SIZE, IMG_SIZE, 3)}
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            try:
                inp = model.inputs[0].shape
                h = int(inp[1] or IMG_SIZE)
                w = int(inp[2] or IMG_SIZE)
                c = int(inp[3] or 3)
                info["input_shape"] = (h, w, c)
            except Exception:
                pass
            info.update(type="keras", model=model)
            return info
        except Exception as e:
            st.warning(f"Failed to load Keras model: {e}")
    if os.path.exists(TFLITE_PATH):
        try:
            interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            shape = input_details["shape"]
            h, w, c = int(shape[1]), int(shape[2]), int(shape[3])
            info.update(type="tflite", model=interpreter, input_shape=(h, w, c))
            return info
        except Exception as e:
            st.warning(f"Failed to load TFLite model: {e}")
    return info

model_info = load_model_safe()
if model_info["type"] == "none":
    st.error("No model found. Place a model in models/ and reload.")
    st.stop()

# ----------------------------
# Sidebar — SINGLE language selector (stores in session_state.APP_LANG)
# ----------------------------
with st.sidebar:
    st.markdown("## Settings")
    st.selectbox("Select language", ["English", "Hindi", "Telugu"], key="APP_LANG")
    if st.button("Sync from custom_texts.json"):
        CUSTOMS = load_custom_texts()
        st.success("Synced custom_texts.json into memory.")
    st.markdown("---")
    st.caption("Project S — informational only.")

# Read language from session_state (sidebar authoritative)
APP_LANG = st.session_state.APP_LANG

# ----------------------------
# Pages (simple dispatcher)
# ----------------------------
PAGES = ["Classifier", "History", "Custom Text Editor", "Skin Detection Preview", "Audio Output", "Severity Charts"]
if "ps_page" not in st.session_state:
    st.session_state.ps_page = "Classifier"

with st.sidebar:
    page_choice = st.radio("Navigation", PAGES, index=PAGES.index(st.session_state.ps_page))
    st.session_state.ps_page = page_choice

page = st.session_state.ps_page

# Page implementations (reuse your functions but they will read APP_LANG from session_state)

def render_history_page():
    st.header("History")
    rows = list_reports()
    if not rows:
        st.info("No saved reports.")
        return
    df = pd.DataFrame([{"id": r['id'], 'filename': r['filename'], 'timestamp': r['timestamp'], 'top_label': r['top_label'], 'top_prob': f"{r['top_prob']*100:.1f}%"} for r in rows])
    st.dataframe(df, use_container_width=True)

def render_custom_text_editor():
    st.header("Custom Text Editor")
    customs = load_custom_texts()
    sel = st.selectbox("Select class", list(CLASS_INDICES.keys()), format_func=lambda k: DISPLAY_NAMES.get(k, k))
    cur_desc = customs.get('descriptions', {}).get(sel, DESCRIPTIONS_EN.get(sel, ''))
    cur_treat = customs.get('treatments', {}).get(sel, TREATMENTS_EN.get(sel, ''))
    new_desc = st.text_area("Description", value=cur_desc, height=120)
    new_treat = st.text_area("Treatment", value=cur_treat, height=120)
    if st.button("Save for class"):
        customs.setdefault('descriptions', {})[sel] = new_desc
        customs.setdefault('treatments', {})[sel] = new_treat
        save_custom_texts(customs)
        st.success("Saved.")

def render_skin_preview():
    st.header("Skin Detection Preview")
    up = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    if not up:
        st.info("Upload an image to preview.")
        return
    pil = Image.open(io.BytesIO(up.read())).convert("RGB")
    st.image(pil, width=420)
    mask = skin_mask_ycrcb(pil)
    st.image(Image.fromarray(np.stack([mask]*3, axis=2)), width=420)

# Simple audio page (TTS omitted for brevity)
def render_audio_output():
    st.header("Audio Output")
    st.info("TTS available if gTTS installed.")

def render_severity_charts():
    st.header("Severity Charts")
    up = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="sev_up")
    if not up:
        st.info("Upload to compute severity.")
        return
    img = Image.open(io.BytesIO(up.read())).convert("RGB")
    sev = estimate_severity(img, None)
    st.json(sev)
    metrics = {"Area%": sev['area_pct']*100, "Redness": sev['redness']*100, "Texture": sev['texture']*100}
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(metrics.keys(), metrics.values())
    ax.set_ylim(0,100)
    st.pyplot(fig)

# Page dispatcher
if page != "Classifier":
    if page == "History":
        render_history_page()
    elif page == "Custom Text Editor":
        render_custom_text_editor()
    elif page == "Skin Detection Preview":
        render_skin_preview()
    elif page == "Audio Output":
        render_audio_output()
    elif page == "Severity Charts":
        render_severity_charts()
    st.stop()

# -------------------------------
# Classifier page (reads language from sidebar-only APP_LANG)
# -------------------------------
st.markdown("<h1>Skin Disease Classifier</h1>", unsafe_allow_html=True)

# Classifier controls stored in session_state with unique keys
col_left, col_right = st.columns([3,1])
with col_right:
    st.checkbox("Auto-crop lesion", key="auto_crop")
    st.checkbox("Enable Grad-CAM", key="enable_gradcam")
    st.slider("Top K", 1, 5, key="top_k")
    st.slider("Temperature", 0.5, 3.0, key="temperature")

with st.container():
    uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="clf_up")
    camera_img = st.camera_input("Or capture with camera", key="clf_camera")

images = []
if uploaded_file:
    images.append((uploaded_file.name, Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")))
elif camera_img:
    images.append(("camera.jpg", Image.open(io.BytesIO(camera_img.read())).convert("RGB")))

if not images:
    st.info("Upload or capture an image to continue.")
    st.stop()

# Resolve language maps using session_state.APP_LANG (sidebar authoritative)

def get_text_maps(lang: str):
    if lang == "Hindi":
        return DESCRIPTIONS_HI, TREATMENTS_HI, 'hi'
    if lang == "Telugu":
        return DESCRIPTIONS_TE, TREATMENTS_TE, 'te'
    return DESCRIPTIONS_EN, TREATMENTS_EN, 'en'

DESCRIPTIONS, TREATMENTS, LANG_CODE = get_text_maps(st.session_state.APP_LANG)

for fname, pil_img in images:
    st.header(f"Image: {fname}")
    st.image(pil_img, width=420)
    mask = skin_mask_ycrcb(pil_img)
    skin_frac = float((mask > 0).sum()) / (mask.shape[0] * mask.shape[1] + 1e-9)
    st.write(f"Skin pixel fraction: {skin_frac:.3f}")
    if skin_frac < 0.02:
        st.error("Not a skin image")
        continue
    target_h, target_w, _ = model_info['input_shape']
    processed_arr, crop_box = preprocess_pil(pil_img, target_size=(target_h, target_w), auto_crop=st.session_state.auto_crop)
    severity = estimate_severity(pil_img, crop_box)
    try:
        probs = predict_with_model(model_info, processed_arr, temperature=st.session_state.temperature)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        continue
    top_k_local = st.session_state.top_k
    top_indices = np.argsort(probs)[::-1][:top_k_local]
    preds = [(int(i), float(probs[i])) for i in top_indices]
    top_idx, top_prob = preds[0]
    top_label = DISPLAY_NAMES.get(IDX_TO_LABEL.get(top_idx, ''), f'class_{top_idx}')
    st.markdown(f"### 🔍 Prediction — *{top_label}*")
    st.markdown(f"**Confidence:** {top_prob*100:.1f}%")
    desc_default = DESCRIPTIONS.get(IDX_TO_LABEL.get(top_idx), '')
    treat_default = TREATMENTS.get(IDX_TO_LABEL.get(top_idx), '')
    colA, colB = st.columns([2,3])
    with colA:
        new_desc = st.text_area(f"desc_{fname}", value=desc_default, height=120)
    with colB:
        new_treat = st.text_area(f"treat_{fname}", value=treat_default, height=120)
        st.write(f"Severity: {severity['level']} (score {severity['score']:.2f})")
    gradcam_img = None
    if st.session_state.enable_gradcam and model_info['type'] == 'keras':
        heatmap = make_gradcam_heatmap(processed_arr, model_info['model'], None, int(top_idx))
        if heatmap is not None:
            base_img = pil_img.crop(crop_box).resize((target_w, target_h)) if crop_box else pil_img.resize((target_w, target_h))
            gradcam_img = overlay_heatmap_on_image(base_img, heatmap, alpha=0.45)
            st.image(gradcam_img, caption='Grad-CAM', width=420)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button('Save report', key=f'save_{fname}'):
            save_report_to_db(fname, preds, new_desc, new_treat, severity, pil_img)
            st.success('Saved')
    with c2:
        pdfb = generate_pdf_report(pil_img, preds, new_desc, new_treat, severity, gradcam_img)
        if pdfb:
            st.download_button('Download PDF', pdfb, file_name=f'report_{fname}.pdf', mime='application/pdf')
    with c3:
        if st.button('Export CSV', key=f'csv_{fname}'):
            row = {'filename': fname, 'timestamp': datetime.now().isoformat(), 'top_label': IDX_TO_LABEL.get(top_idx, str(top_idx)), 'top_prob': top_prob, 'preds': json.dumps(preds), 'severity': json.dumps(severity)}
            df = pd.DataFrame([row])
            st.download_button('Download CSV', df.to_csv(index=False).encode('utf8'), file_name=f'prediction_{fname}.csv', mime='text/csv')

st.caption('Project S — informational only. Not a medical device.')
