# final_app.py  (drop-in replacement for your app.py)
# Project S — Streamlit app (PDF Unicode-safe, language fix, safer PDF fallback)

import os
import io
import json
import tempfile
import threading
import sqlite3
import re
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

try:
    import cv2
except Exception:
    cv2 = None

# PDF
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
FONTS_DIR = "fonts"  # place Noto fonts here

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
# DEFAULT TEXTS (short placeholders - you can replace with your full texts)
# ----------------------------
DESCRIPTIONS_EN = {k: (k.replace("_"," ").capitalize() + " — brief description.") for k in CLASS_INDICES.keys()}
TREATMENTS_EN = {k: "Suggested care: consult dermatologist if concerned." for k in CLASS_INDICES.keys()}
DESCRIPTIONS_HI = dict(DESCRIPTIONS_EN)
TREATMENTS_HI = dict(TREATMENTS_EN)
DESCRIPTIONS_TE = dict(DESCRIPTIONS_EN)
TREATMENTS_TE = dict(TREATMENTS_EN)

# ----------------------------
# UTILITIES
# ----------------------------
def ensure_rgb(pil: Image.Image) -> Image.Image:
    return pil.convert("RGB")

def sanitize_for_pdf(text: str) -> str:
    """
    Replace problematic unicode characters with ascii equivalents.
    Used when a Unicode-capable font is not available.
    """
    if not isinstance(text, str):
        return text
    replace_map = {
        "\u2014": "-",  # em-dash
        "\u2013": "-",  # en-dash
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
        "\u2026": "...",
    }
    for k, v in replace_map.items():
        text = text.replace(k, v)
    # remove any remaining characters outside latin-1 if desired:
    text = re.sub(r"[^\x00-\xFF]", "?", text)
    return text

# ----------------------------
# SKIN MASK & PREPROCESS (same logic as previously)
# ----------------------------
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

# Grad-CAM & predict wrappers (kept compact)
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
# PDF: Unicode-safe using fpdf2 and Noto fonts (robust fallback)
# ----------------------------
class MedicalPDF(FPDF):
    def header(self):
        # Use self.set_font (not a bare set_font call)
        if "Noto" in self.fonts:
            self.set_font("Noto", size=12)
        else:
            # fallback to core font (may not support Unicode)
            self.set_font("Helvetica", size=12)
        self.set_fill_color(240, 240, 240)
        self.rect(0, 0, self.w, 18, "F")
        self.set_xy(10, 5)
        self.cell(0, 8, "Project S - Skin Disease Classifier")
        self.set_xy(self.w - 70, 5)
        self.set_font(self.font_family, size=10)
        self.cell(60, 8, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), align="R")
        self.ln(15)

    def footer(self):
        self.set_y(-12)
        self.set_font("Noto" if "Noto" in self.fonts else "Helvetica", size=9)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

def generate_unicode_pdf(image_pil, preds, desc, treat, severity, gradcam_img=None):
    """
    Creates a PDF, uses Noto fonts if present in fonts/; otherwise falls back.
    When falling back to non-Unicode fonts, non-latin text is sanitized to avoid crashes.
    """
    pdf = MedicalPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    # Add page (header will be called)
    pdf.add_page()

    # Try to register Noto fonts if available
    en_path = os.path.join(FONTS_DIR, "NotoSans-Regular.ttf")
    hi_path = os.path.join(FONTS_DIR, "NotoSansDevanagari-Regular.ttf")
    te_path = os.path.join(FONTS_DIR, "NotoSansTelugu-Regular.ttf")

    # Add fonts if files exist (uni=True)
    try:
        if os.path.exists(en_path):
            pdf.add_font("Noto", "", en_path, uni=True)
        if os.path.exists(hi_path):
            pdf.add_font("NotoDev", "", hi_path, uni=True)
        if os.path.exists(te_path):
            pdf.add_font("NotoTel", "", te_path, uni=True)
    except Exception:
        # ignore font registration errors, we'll fallback
        pass

    # helper to write text with fallback
    def write_block(title: str, content: str):
        pdf.set_font("Noto" if "Noto" in pdf.fonts else "Helvetica", size=12)
        pdf.cell(0, 8, title)
        pdf.ln(6)
        # If chosen font doesn't support Unicode (core font), sanitize content
        use_sanitize = False
        if not ("Noto" in pdf.fonts or "NotoDev" in pdf.fonts or "NotoTel" in pdf.fonts):
            use_sanitize = True
        # If language-specific font available, try to set it
        # Decide based on presence of Devanagari / Telugu chunks:
        if any("\u0900" <= c <= "\u097F" for c in content) and "NotoDev" in pdf.fonts:
            pdf.set_font("NotoDev", size=12)
            use_sanitize = False
        elif any("\u0C00" <= c <= "\u0C7F" for c in content) and "NotoTel" in pdf.fonts:
            pdf.set_font("NotoTel", size=12)
            use_sanitize = False
        elif "Noto" in pdf.fonts:
            pdf.set_font("Noto", size=12)
            use_sanitize = False
        else:
            pdf.set_font("Helvetica", size=12)
            use_sanitize = True

        text_to_write = sanitize_for_pdf(content) if use_sanitize else content
        try:
            pdf.multi_cell(pdf.w - 20, 8, text_to_write, fill=False)
        except Exception:
            # As a last resort, force sanitize and write
            pdf.multi_cell(pdf.w - 20, 8, sanitize_for_pdf(content), fill=False)
        pdf.ln(3)

    # Image (left)
    left_x = 10
    img_width = 90
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(tmp_fd)
        image_pil.save(tmp_path, "JPEG")
        pdf.image(tmp_path, x=left_x, y=30, w=img_width)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    except Exception:
        # ignore image errors
        pass

    right_x = left_x + img_width + 20
    pdf.set_xy(right_x, 30)
    pdf.set_font("Noto" if "Noto" in pdf.fonts else "Helvetica", size=14)
    pdf.cell(0, 8, "Top Prediction:")
    pdf.ln(8)
    if preds:
        top_idx, top_prob = preds[0]
        label = DISPLAY_NAMES.get(IDX_TO_LABEL.get(top_idx, ""), f"Class {top_idx}")
        pdf.set_font("Noto" if "Noto" in pdf.fonts else "Helvetica", size=12)
        pdf.cell(0, 7, f"{label} - {top_prob*100:.1f}%")
        pdf.ln(10)

    # Description, Treatment, Severity
    write_block("Description", desc or "")
    write_block("Suggested Treatment", treat or "")
    sev_text = f"Level: {severity.get('level')}\nScore: {severity.get('score'):.2f}\nArea: {severity.get('area_pct')*100:.1f}%\nRedness: {severity.get('redness'):.3f}"
    write_block("Severity Analysis", sev_text)

    out = pdf.output(dest="S")
    if isinstance(out, bytearray):
        return bytes(out)
    if isinstance(out, str):
        return out.encode("latin-1", errors="replace")
    return out

generate_pdf_report = generate_unicode_pdf

# ----------------------------
# DB helpers (unchanged)
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
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Project S — Skin Disease Classifier", layout="wide")

# session defaults (avoid AttributeError)
st.session_state.setdefault("APP_LANG", "English")
st.session_state.setdefault("auto_crop", True)
st.session_state.setdefault("enable_gradcam", True)
st.session_state.setdefault("top_k", TOP_K_DEFAULT)
st.session_state.setdefault("temperature", 1.0)
st.session_state.setdefault("ps_page", "Classifier")

# Load model (cached)
@st.cache_resource(show_spinner=False)
def load_model_safe():
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
    st.warning("No model found. Place a Keras model at models/skin_classifier.h5 or a TFLite model at models/skin_classifier.tflite")
    # Do not exit immediately to let user interact with other pages
    # st.stop()

# Sidebar (single language selector, unique keys)
with st.sidebar:
    st.markdown("## Settings")
    st.selectbox("Select language", ["English", "Hindi", "Telugu"], key="APP_LANG")
    if st.button("Sync from custom_texts.json", key="sync_btn"):
        CUSTOMS = load_custom_texts()
        st.success("Synced custom_texts.json into memory.")
    st.markdown("---")
    st.caption("Project S — informational only.")

# Navigation radio
PAGES = ["Classifier", "History", "Custom Text Editor", "Skin Detection Preview", "Audio Output", "Severity Charts"]
with st.sidebar:
    page_choice = st.radio("Navigation", PAGES, index=PAGES.index(st.session_state.ps_page))
    st.session_state.ps_page = page_choice

page = st.session_state.ps_page
APP_LANG = st.session_state.APP_LANG

# Page functions (concise)
def render_history_page():
    st.header("History — Saved Reports")
    rows = list_reports(limit=500)
    if not rows:
        st.info("No saved reports.")
        return
    df_rows = [{"id": r["id"], "filename": r["filename"], "timestamp": r["timestamp"], "top_label": r["top_label"], "top_prob": f"{r['top_prob']*100:.1f}%"} for r in rows]
    st.dataframe(pd.DataFrame(df_rows), use_container_width=True)

def render_custom_text_editor():
    st.header("Custom Text Editor")
    customs = load_custom_texts()
    sel = st.selectbox("Select class", list(CLASS_INDICES.keys()), format_func=lambda k: DISPLAY_NAMES.get(k, k))
    cur_desc = customs.get("descriptions", {}).get(sel, DESCRIPTIONS_EN.get(sel, ""))
    cur_treat = customs.get("treatments", {}).get(sel, TREATMENTS_EN.get(sel, ""))
    new_desc = st.text_area("Description", value=cur_desc, height=120)
    new_treat = st.text_area("Treatment", value=cur_treat, height=120)
    if st.button("Save changes for selected class", key="save_custom"):
        customs.setdefault("descriptions", {})[sel] = new_desc
        customs.setdefault("treatments", {})[sel] = new_treat
        save_custom_texts(customs)
        st.success("Saved custom text.")

def render_skin_preview():
    st.header("Skin Detection Preview")
    up = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="preview_up")
    if not up:
        st.info("Upload an image to preview.")
        return
    pil = Image.open(io.BytesIO(up.read())).convert("RGB")
    st.image(pil, caption="Original", width=420)
    mask = skin_mask_ycrcb(pil)
    mask_img = Image.fromarray(np.stack([mask]*3, axis=2))
    st.image(mask_img, caption="Skin mask (white = skin)", width=420)
    crop, box = auto_crop_by_skin(pil)
    if crop is not None:
        st.image(crop, caption=f"Auto-crop preview {box}", width=360)
    else:
        st.info("Auto-crop heuristic couldn't find a dominant skin region.")

def render_audio_output():
    st.header("Audio Output")
    st.info("TTS is available if gTTS is installed. (Not included in this file.)")

def render_severity_charts():
    st.header("Severity Charts")
    up = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="sev_up")
    if not up:
        st.info("Upload an image to compute severity.")
        return
    img = Image.open(io.BytesIO(up.read())).convert("RGB")
    sev = estimate_severity(img, None)
    st.json(sev)
    metrics = {"Area%": sev['area_pct']*100, "Redness": sev['redness']*100, "Texture": sev['texture']*100}
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(metrics.keys(), metrics.values())
    ax.set_ylim(0,100)
    st.pyplot(fig)

# Dispatcher
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
# Classifier page
# -------------------------------
st.markdown("<h1>Skin Disease Classifier</h1>", unsafe_allow_html=True)

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

# Resolve language maps
def get_text_maps(lang: str):
    if lang == "Hindi":
        base_desc, base_treat = DESCRIPTIONS_HI, TREATMENTS_HI
    elif lang == "Telugu":
        base_desc, base_treat = DESCRIPTIONS_TE, TREATMENTS_TE
    else:
        base_desc, base_treat = DESCRIPTIONS_EN, TREATMENTS_EN
    customs = load_custom_texts()
    descs = dict(base_desc)
    treats = dict(base_treat)
    for k, v in customs.get("descriptions", {}).items():
        descs[k] = v
    for k, v in customs.get("treatments", {}).items():
        treats[k] = v
    # choose language code for potential TTS later
    lang_code = "hi" if lang == "Hindi" else ("te" if lang == "Telugu" else "en")
    return descs, treats, lang_code

DESCRIPTIONS, TREATMENTS, LANG_CODE = get_text_maps(APP_LANG)

for fname, pil_img in images:
    st.header(f"Image: {fname}")
    st.image(pil_img, caption="Original", width=420)

    mask = skin_mask_ycrcb(pil_img)
    skin_frac = float((mask > 0).sum()) / (mask.shape[0] * mask.shape[1] + 1e-9)
    st.write(f"Skin pixel fraction: {skin_frac:.3f}")
    if skin_frac < 0.02:
        st.error("Not a skin image (low skin pixel fraction). Try another photo.")
        continue

    target_h, target_w, _ = model_info.get("input_shape", (IMG_SIZE, IMG_SIZE, 3))
    processed_arr, crop_box = preprocess_pil(pil_img, target_size=(target_h, target_w), auto_crop=st.session_state.auto_crop)
    severity = estimate_severity(pil_img, crop_box)

    try:
        if model_info["type"] == "none":
            # If no model, skip prediction but allow PDF export & description display
            probs = np.zeros(len(IDX_TO_LABEL))
        else:
            probs = predict_with_model(model_info, processed_arr, temperature=st.session_state.temperature)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        continue

    top_k_local = st.session_state.top_k
    top_indices = np.argsort(probs)[::-1][:top_k_local]
    preds = [(int(i), float(probs[i])) for i in top_indices]

    top_idx, top_prob = preds[0] if preds else (0, 0.0)
    top_label = DISPLAY_NAMES.get(IDX_TO_LABEL.get(top_idx, ""), f"class_{top_idx}")
    st.markdown(f"### 🔍 Prediction — *{top_label}*")
    st.markdown(f"**Confidence:** {top_prob*100:.1f}%")

    desc_default = DESCRIPTIONS.get(IDX_TO_LABEL.get(top_idx), "")
    treat_default = TREATMENTS.get(IDX_TO_LABEL.get(top_idx), "")

    colA, colB = st.columns([2,3])
    with colA:
        new_desc = st.text_area(f"desc_{fname}", value=desc_default, height=120)
    with colB:
        new_treat = st.text_area(f"treat_{fname}", value=treat_default, height=120)
        st.write(f"Severity: {severity['level']} (score {severity['score']:.2f})")

    gradcam_img = None
    if st.session_state.enable_gradcam and model_info.get('type') == 'keras':
        heatmap = make_gradcam_heatmap(processed_arr, model_info['model'], None, int(top_idx))
        if heatmap is not None:
            base_img = pil_img.crop(crop_box).resize((target_w, target_h)) if crop_box else pil_img.resize((target_w, target_h))
            gradcam_img = overlay_heatmap_on_image(base_img, heatmap, alpha=0.45)
            st.image(gradcam_img, caption="Grad-CAM", width=420)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Save report", key=f"save_{fname}"):
            save_report_to_db(fname, preds, new_desc, new_treat, severity, pil_img)
            st.success("Saved report to DB.")
    with c2:
        try:
            pdfb = generate_pdf_report(pil_img, preds, new_desc, new_treat, severity, gradcam_img)
            if pdfb:
                st.download_button("Download PDF", pdfb, file_name=f"report_{fname}.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
    with c3:
        if st.button("Export CSV", key=f"csv_{fname}"):
            row = {
                "filename": fname,
                "timestamp": datetime.now().isoformat(),
                "top_label": IDX_TO_LABEL.get(top_idx, str(top_idx)),
                "top_prob": top_prob,
                "preds": json.dumps(preds),
                "severity": json.dumps(severity)
            }
            df = pd.DataFrame([row])
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf8"), file_name=f"prediction_{fname}.csv", mime="text/csv")

st.caption("Project S — informational only. Not a medical device. Always consult a dermatologist.")

# End of file
