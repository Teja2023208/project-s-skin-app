"""
app.py - Project S (Option B)
Full-featured stable single-file Streamlit app:
 - Keras (.h5) and optional TFLite support
 - Skin detection, auto-crop, severity heuristics
 - Grad-CAM overlay (Keras)
 - Multilanguage UI (English / Hindi / Telugu)
 - AI auto-description (OpenAI optional)
 - gTTS voice explanations
 - JSON sync for descriptions/treatments
 - SQLite history (save/list/delete)
 - ASCII-safe PDF export using fpdf2 (no unicode in PDF)
 - Robust handling for fonts / fpdf2 / byte conversions
Disclaimer: informational only. Not medical advice.
"""

import os
import io
import sys
import json
import math
import base64
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

# PDF (fpdf2)
try:
    from fpdf import FPDF
    import fpdf as _fpdf_pkg
except Exception:
    FPDF = None
    _fpdf_pkg = None

# TTS
try:
    from gtts import gTTS
except Exception:
    gTTS = None

# OpenAI (optional)
try:
    import openai
except Exception:
    openai = None

# Matplotlib for colormap if needed
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------
# CONFIG
# ---------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/skin_classifier.h5")
TFLITE_PATH = os.environ.get("TFLITE_PATH", "models/skin_classifier.tflite")
IMG_SIZE_DEFAULT = 160
TOP_K_DEFAULT = 3
DB_PATH = "project_s_reports.db"
CUSTOM_TEXTS = "custom_texts.json"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
if openai is not None and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# ---------------------
# CLASS MAPS (example)
# ---------------------
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

# ---------------------
# DEFAULT TEXTS (EN/HI/TE)
# Keep these brief; custom_texts.json can override
# ---------------------
DESCRIPTIONS_EN = {k: f"Informational description for {DISPLAY_NAMES[k]}." for k in CLASS_INDICES.keys()}
TREATMENTS_EN = {k: f"Informational treatment suggestions for {DISPLAY_NAMES[k]}." for k in CLASS_INDICES.keys()}

DESCRIPTIONS_HI = {k: DESCRIPTIONS_EN[k] for k in CLASS_INDICES.keys()}  # you can replace with real Hindi
TREATMENTS_HI = {k: TREATMENTS_EN[k] for k in CLASS_INDICES.keys()}

DESCRIPTIONS_TE = {k: DESCRIPTIONS_EN[k] for k in CLASS_INDICES.keys()}  # placeholder Telugu
TREATMENTS_TE = {k: TREATMENTS_EN[k] for k in CLASS_INDICES.keys()}

# ---------------------
# UTILITIES
# ---------------------
def pretty_pct(x: float) -> str:
    return f"{x*100:5.1f}%"

def confidence_color(percent: float) -> str:
    if percent >= 0.7:
        return "#198754"
    if percent >= 0.4:
        return "#fd7e14"
    return "#dc3545"

def ensure_rgb(pil: Image.Image) -> Image.Image:
    return pil.convert("RGB")

def sanitize_for_pdf(text: str) -> str:
    """
    Make text ASCII-safe for the PDF:
    - Replace common Unicode punctuation with ASCII equivalents
    - Finally encode to ascii with replacement for any remaining non-ascii
    """
    if not isinstance(text, str):
        text = str(text)
    rep = (text.replace("—", "-")
               .replace("–", "-")
               .replace("“", '"')
               .replace("”", '"')
               .replace("…", "...")
               .replace("’", "'"))
    # convert to ascii (non-ascii become '?') to ensure PDF won't crash
    return rep.encode("ascii", "replace").decode("ascii")

# ---------------------
# SKIN MASK / AUTO-CROP / PREPROCESS
# ---------------------
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
    arr = np.expand_dims(arr, 0)  # 1,H,W,3
    return arr, crop_box

# ---------------------
# SEVERITY
# ---------------------
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

# ---------------------
# Grad-CAM (Keras)
# ---------------------
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

# ---------------------
# MODEL LOADING & PREDICT
# ---------------------
@st.cache_resource(show_spinner=False)
def load_model(path_keras: str = MODEL_PATH, path_tflite: str = TFLITE_PATH):
    info = {"type": "none", "model": None, "input_shape": (IMG_SIZE_DEFAULT, IMG_SIZE_DEFAULT, 3)}
    if os.path.exists(path_keras):
        try:
            model = tf.keras.models.load_model(path_keras, compile=False)
            try:
                inp = model.inputs[0].shape
                h = int(inp[1] or IMG_SIZE_DEFAULT)
                w = int(inp[2] or IMG_SIZE_DEFAULT)
                c = int(inp[3] or 3)
                info["input_shape"] = (h, w, c)
            except Exception:
                pass
            info.update(type="keras", model=model)
            return info
        except Exception as e:
            st.warning(f"Failed to load Keras model: {e}")
    # tflite
    if os.path.exists(path_tflite):
        try:
            interpreter = tf.lite.Interpreter(model_path=path_tflite)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            shape = input_details["shape"]
            h, w, c = int(shape[1]), int(shape[2]), int(shape[3])
            info.update(type="tflite", model=interpreter, input_shape=(h, w, c))
            return info
        except Exception as e:
            st.warning(f"Failed to load TFLite model: {e}")
    return info

def predict_with_model(model_info: Dict, processed_array: np.ndarray, temperature: float = 1.0):
    if model_info["type"] == "keras":
        model = model_info["model"]
        preds = model.predict(processed_array)
        if preds.ndim == 2:
            out = preds[0]
        else:
            out = preds
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
            try:
                interp.resize_tensor_input(input_details["index"], inp.shape)
                interp.allocate_tensors()
                interp.set_tensor(input_details["index"], inp)
                interp.invoke()
                out = interp.get_tensor(output_details["index"])
            except Exception as e:
                raise RuntimeError("TFLite inference failed: " + str(e))
        if out.ndim == 2:
            logits = out[0]
        else:
            logits = out
        logits = logits / max(1e-6, temperature)
        probs = tf.nn.softmax(logits).numpy()
        return np.array(probs)
    else:
        raise RuntimeError("No model loaded")

# ---------------------
# DB (SQLite)
# ---------------------
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
                (filename, datetime.now().isoformat(), DISPLAY_NAMES.get(IDX_TO_LABEL.get(top_idx, ""), str(top_idx)), float(top_prob), preds_json, description, treatment, severity_json, sqlite3.Binary(img_bytes)))
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

# ---------------------
# CUSTOM TEXTS JSON
# ---------------------
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

# ---------------------
# AI auto-description (OpenAI optional)
# ---------------------
def ai_generate_texts(label_key: str, lang: str = "English"):
    label_display = DISPLAY_NAMES.get(label_key, label_key)
    if OPENAI_API_KEY and openai is not None:
        try:
            prompt_desc = f"Write a short, plain-language medical description (1-2 sentences) for a skin condition named '{label_display}'. Keep it informational and non-diagnostic. Language: {lang}."
            prompt_treat = f"Write concise informational treatment suggestions (1-2 sentences) for '{label_display}'. Language: {lang}."
            resp1 = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt_desc}], max_tokens=150, temperature=0.2)
            resp2 = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt_treat}], max_tokens=150, temperature=0.2)
            desc = resp1.choices[0].message.content.strip()
            treat = resp2.choices[0].message.content.strip()
            return desc, treat
        except Exception:
            pass
    # fallback templates
    if lang == "Hindi":
        desc_template = f"{label_display} एक त्वचा संबंधी समस्या है जो आमतौर पर त्वचा पर परिवर्तन दिखाती है।"
        treat_template = "क्षेत्र को साफ रखें, ट्रिगर से बचें और आवश्यक होने पर त्वचा विशेषज्ञ से सलाह लें।"
    elif lang == "Telugu":
        desc_template = f"{label_display} ఒక చర్మ సమస్య. అవసరమైతే డాక్టర్ ని సంప్రదించండి."
        treat_template = "ప్రాంతాన్ని శుభ్రంగా ఉంచండి, ట్రిగ్గర్లను నివారించండి."
    else:
        desc_template = f"{label_display} is a skin condition that typically causes localized changes to the skin."
        treat_template = "Keep the area clean and consult a dermatologist for persistent or concerning lesions."
    return desc_template, treat_template

# ---------------------
# TTS using gTTS
# ---------------------
def text_to_audio_bytes(text: str, lang_code: str = "en"):
    if gTTS is None:
        return None
    try:
        tts = gTTS(text=text, lang=lang_code)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception:
        return None

# ---------------------
# ASCII-safe PDF generator (fpdf2)
# - This version intentionally makes PDF ASCII-only to avoid unicode crashes.
# ---------------------
class SimpleReportPDF(FPDF):
    def header(self):
        # Use ASCII-safe header (avoid unicode chars)
        self.set_fill_color(240, 240, 240)
        self.rect(0, 0, self.w, 18, "F")
        self.set_xy(10, 5)
        # default font (Helvetica/Arial) is fine for ASCII
        self.set_font("Arial", size=12)
        self.cell(0, 8, "Project S - Skin Disease Classifier", ln=False)
        self.set_xy(self.w - 80, 5)
        self.set_font("Arial", size=10)
        self.cell(70, 8, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ln=False, align="R")
        self.ln(15)

    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", size=9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

def generate_ascii_pdf(image_pil: Image.Image, preds: List[Tuple[int, float]], desc: str, treat: str, severity: dict, gradcam_img: Optional[Image.Image] = None) -> Optional[bytes]:
    """
    Generates a PDF that is ASCII-only (non-ascii replaced). Returns bytes or None.
    """
    if FPDF is None:
        return None
    pdf = SimpleReportPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # Save main image
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)
    image_pil.save(tmp_path, "JPEG")

    left_x = 10
    img_width = 90
    pdf.image(tmp_path, x=left_x, y=30, w=img_width)
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    right_x = left_x + img_width + 20
    pdf.set_xy(right_x, 30)
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 8, "Top Prediction:", ln=True)

    # top label -> display name
    top_idx, top_prob = preds[0]
    label = DISPLAY_NAMES.get(IDX_TO_LABEL.get(top_idx, ""), f"Class {top_idx}")
    label_line = sanitize_for_pdf(f"{label} - {top_prob*100:.1f}%")
    pdf.set_x(right_x)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 7, label_line, ln=True)

    # gradcam thumbnail (if provided)
    if gradcam_img is not None:
        g_fd, g_path = tempfile.mkstemp(suffix=".jpg")
        os.close(g_fd)
        gradcam_img.save(g_path, "JPEG")
        pdf.image(g_path, x=right_x, y=55, w=80)
        try:
            os.remove(g_path)
        except:
            pass

    # Move below image
    pdf.set_y(140)

    # Description box
    pdf.set_font("Arial", size=13)
    pdf.cell(0, 10, "Description", ln=True)
    pdf.set_fill_color(248, 248, 248)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(pdf.w - 20, 7, sanitize_for_pdf(desc), fill=True)
    pdf.ln(5)

    # Treatment box
    pdf.set_font("Arial", size=13)
    pdf.cell(0, 10, "Suggested Treatment", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(pdf.w - 20, 7, sanitize_for_pdf(treat), fill=True)
    pdf.ln(5)

    # Severity
    pdf.set_font("Arial", size=13)
    pdf.cell(0, 10, "Severity Analysis", ln=True)
    sev_text = (
        f"Level: {severity.get('level')}\n"
        f"Score: {severity.get('score'):.2f}\n"
        f"Area: {severity.get('area_pct')*100:.1f}%\n"
        f"Redness: {severity.get('redness'):.3f}"
    )
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(pdf.w - 20, 7, sanitize_for_pdf(sev_text))

    # Convert to bytes (fpdf2 returns bytearray)
    return bytes(pdf.output(dest="S"))

# ---------------------
# Flask API starter (background)
# ---------------------
def start_api_server(host="127.0.0.1", port=8502):
    try:
        from flask import Flask, request, jsonify
    except Exception:
        st.warning("Flask not installed - API disabled.")
        return None
    app = Flask("project_s_api")
    model_info_local = load_model()
    @app.route("/predict", methods=["POST"])
    def api_predict():
        try:
            img = None
            if request.files and 'file' in request.files:
                f = request.files['file']; img = Image.open(io.BytesIO(f.read())).convert("RGB")
            else:
                data = request.get_json(force=True, silent=True) or {}
                b64 = data.get("image_b64") or data.get("image")
                if b64:
                    img = Image.open(io.BytesIO(base64.b64decode(b64.split(",")[-1]))).convert("RGB")
            if img is None:
                return jsonify({"error":"no image provided"}), 400
            arr, crop_box = preprocess_pil(img, target_size=model_info_local["input_shape"][:2], auto_crop=True)
            probs = predict_with_model(model_info_local, arr, temperature=float(request.args.get("temperature",1.0)))
            top_k = int(request.args.get("top_k", TOP_K_DEFAULT))
            top_indices = np.argsort(probs)[::-1][:top_k]
            preds = [[int(i), float(probs[i])] for i in top_indices]
            severity = estimate_severity(img, crop_box)
            return jsonify({"preds": preds, "severity": severity})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    thread = threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True)
    thread.start()
    return app

# ---------------------
# STREAMLIT UI
# ---------------------
st.set_page_config(page_title="Project S — Skin Disease Classifier (Option B)", layout="wide")
with st.sidebar:
    st.title("Project S — Option B (Full)")
    st.markdown("Informational only — not medical advice.")
    st.markdown("---")
    st.write("Model (Keras):"); st.code(MODEL_PATH)
    st.write("Model (TFLite optional):"); st.code(TFLITE_PATH)
    auto_crop = st.checkbox("Auto-crop lesion (heuristic)", value=True)
    enable_gradcam = st.checkbox("Enable Grad-CAM (Keras only)", value=True)
    top_k = st.slider("Top K predictions", 1, 5, TOP_K_DEFAULT)
    temperature = st.slider("Temperature (calibration)", 0.5, 3.0, 1.0, step=0.1)
    st.markdown("---")
    st.subheader("Persistence & API")
    enable_api = st.checkbox("Enable REST API (Flask background)", value=False)
    api_port = st.number_input("API port", min_value=1025, max_value=65535, value=8502)
    st.markdown("---")
    st.subheader("Language")
    LANGUAGES = ["English", "Hindi", "Telugu"]
    APP_LANG = st.selectbox("App language", LANGUAGES, index=0)
    st.button("Sync descriptions from JSON", key="sync_json")
    st.caption("Not medical advice — consult dermatologist.")

# Load model
with st.spinner("Loading model..."):
    model_info = load_model()
if model_info["type"] == "none":
    st.error("No model found. Place a Keras .h5 at models/ or a TFLite .tflite")
    st.stop()

if enable_api:
    start_api_server(port=int(api_port))
    st.success(f"API started on port {api_port}")

# INPUTS
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload images (multiple allowed)", type=["jpg","jpeg","png"], accept_multiple_files=True)
    cam = st.camera_input("Or capture with camera")
with col2:
    st.write("Tips:")
    st.write("- Take a focused close-up of the lesion")
    st.write("- Natural lighting, avoid heavy filters")
    st.write("- For urgent growth/bleeding, seek clinician")

images: List[Tuple[str, Image.Image]] = []
if uploaded:
    for f in uploaded:
        try:
            images.append((f.name, Image.open(io.BytesIO(f.read())).convert("RGB")))
        except Exception as e:
            st.warning(f"Could not open {f.name}: {e}")
if cam:
    try:
        images.insert(0, ("camera.jpg", Image.open(io.BytesIO(cam.read())).convert("RGB")))
    except Exception:
        pass

if not images:
    st.info("Upload or capture an image to run the classifier.")
    st.stop()

# language maps
def get_text_maps(lang: str):
    if lang == "Hindi":
        base_desc = DESCRIPTIONS_HI
        base_treat = TREATMENTS_HI
        lang_code = "hi"
    elif lang == "Telugu":
        base_desc = DESCRIPTIONS_TE
        base_treat = TREATMENTS_TE
        lang_code = "te"
    else:
        base_desc = DESCRIPTIONS_EN
        base_treat = TREATMENTS_EN
        lang_code = "en"
    customs = load_custom_texts()
    descs = dict(base_desc)
    treats = dict(base_treat)
    for k, v in customs.get("descriptions", {}).items():
        descs[k] = v
    for k, v in customs.get("treatments", {}).items():
        treats[k] = v
    return descs, treats, lang_code

TEXTS, TREAT_MAP, LANG_CODE = get_text_maps(APP_LANG)

# process images
for fname, pil_img in images:
    st.markdown("---")
    st.header(f"Image: {fname}")
    st.image(pil_img, caption="Original", use_container_width=False, width=420)

    # skin check
    mask = skin_mask_ycrcb(pil_img)
    skin_frac = float((mask > 0).sum()) / (mask.shape[0]*mask.shape[1] + 1e-9)
    st.write(f"Skin pixel fraction: {skin_frac:.3f}")
    if skin_frac < 0.02:
        st.error("Not a skin image (low skin pixel fraction). Try another photo.")
        continue

    # preprocess
    target_h, target_w, target_c = model_info["input_shape"]
    processed_arr, crop_box = preprocess_pil(pil_img, target_size=(target_h, target_w), auto_crop=auto_crop)
    st.write(f"Auto-crop applied: {'Yes' if crop_box else 'No'}")

    # severity
    severity = estimate_severity(pil_img, crop_box)

    # predict
    try:
        probs = predict_with_model(model_info, processed_arr, temperature=temperature)
    except Exception as e:
        st.exception(e)
        st.error("Prediction failed.")
        continue

    # top-k
    top_indices = np.argsort(probs)[::-1][:top_k]
    preds = [(int(i), float(probs[i])) for i in top_indices]

    # display
    top_idx, top_prob = preds[0]
    top_label = DISPLAY_NAMES.get(IDX_TO_LABEL.get(top_idx, ""), f"class_{top_idx}")
    color = confidence_color(top_prob)
    st.markdown(f"### 🔍 Prediction Result — *{top_label}*")
    st.markdown(f"<div style='padding:10px;border-left:6px solid {color};border-radius:6px;'>Confidence: <strong>{top_prob*100:5.1f}%</strong></div>", unsafe_allow_html=True)

    # description & treatment UI (editable)
    default_desc = TEXTS.get(IDX_TO_LABEL.get(top_idx, ""), f"No description available for {top_label}.")
    default_treat = TREAT_MAP.get(IDX_TO_LABEL.get(top_idx, ""), f"No treatment suggestions available for {top_label}.")
    colA, colB = st.columns([2, 3])
    with colA:
        st.markdown("#### Brief description")
        new_desc = st.text_area(f"desc_{fname}", value=default_desc, height=140)
        st.markdown("#### Suggested next steps (informational)")
        st.write("- Re-take image with close-up, good lighting.")
        st.write("- Avoid topical creams before imaging.")
        st.write("- Seek dermatologist for persistent or suspicious lesions.")
    with colB:
        st.markdown("#### Suggested treatment ideas (informational)")
        new_treat = st.text_area(f"treat_{fname}", value=default_treat, height=140)
        st.markdown("#### Severity")
        st.write(f"Level: *{severity['level']}* — Score: *{severity['score']:.2f}*")
        st.write(f"Estimated lesion area fraction: *{severity['area_pct']*100:.1f}%*")
        st.write(f"Redness metric: *{severity['redness']:.3f}*")

    # top-k bar
    st.markdown("### 📊 Top predictions")
    for idx, prob in preds:
        label = DISPLAY_NAMES.get(IDX_TO_LABEL.get(idx, ""), f"class_{idx}")
        pct = prob * 100.0
        bar_html = (
            f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:6px;'>"
            f"<div style='flex:1'><strong>{label}</strong></div>"
            f"<div style='width:60%;background:#efefef;border-radius:6px;padding:4px;'>"
            f"<div style='width:{pct}%;background:{confidence_color(prob)};height:12px;border-radius:6px;'></div>"
            f"</div>"
            f"<div style='width:80px;text-align:right'>{pct:5.1f}%</div>"
            f"</div>"
        )
        st.markdown(bar_html, unsafe_allow_html=True)

    # Grad-CAM
    gradcam_img = None
    if enable_gradcam and model_info["type"] == "keras":
        heatmap = make_gradcam_heatmap(processed_arr, model_info["model"], last_conv_layer_name=None, pred_index=int(top_idx))
        try:
            if crop_box:
                display_base = pil_img.crop(crop_box).resize((target_w, target_h))
            else:
                display_base = pil_img.resize((target_w, target_h))
            gradcam_img = overlay_heatmap_on_image(display_base, heatmap, alpha=0.45)
            st.image(gradcam_img, caption="Grad-CAM overlay", width=420)
        except Exception:
            gradcam_img = None

    # AI generate
    if st.button(f"AI-generate description/treatment for {top_label}"):
        desc_gen, treat_gen = ai_generate_texts(IDX_TO_LABEL.get(top_idx, ""), lang=APP_LANG)
        st.info("AI-generated texts (preview)")
        st.write("Description:", desc_gen)
        st.write("Treatment:", treat_gen)
        if st.button("Use AI-generated texts (save)"):
            CUSTOMS.setdefault("descriptions", {})[IDX_TO_LABEL.get(top_idx, "")] = desc_gen
            CUSTOMS.setdefault("treatments", {})[IDX_TO_LABEL.get(top_idx, "")] = treat_gen
            save_custom_texts(CUSTOMS)
            st.success("Saved AI-generated texts to custom_texts.json")

    # Voice TTS
    full_text = f"{new_desc}. Treatment: {new_treat}."
    if st.button("🔊 Voice Explanation"):
        audio_bytes = text_to_audio_bytes(full_text, lang_code=LANG_CODE)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.error("Text-to-speech unavailable (gTTS not installed or TTS failed).")

    # Auto-play if enabled from sidebar
    if st.session_state.get("auto_play_audio", False):
        audio_bytes = text_to_audio_bytes(full_text, lang_code=LANG_CODE)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")

    # Save to DB
    if st.button(f"Save report to History for {fname}"):
        img_for_save = gradcam_img if gradcam_img is not None else pil_img
        rid = save_report_to_db(fname, preds, new_desc, new_treat, severity, img_for_save)
        st.success(f"Saved report id {rid}")

    # PDF export (ASCII-safe)
    if st.button(f"Export PDF report for {fname}"):
        pdf_bytes = generate_ascii_pdf(gradcam_img if gradcam_img is not None else pil_img, preds, new_desc, new_treat, severity, gradcam_img)
        if pdf_bytes is None:
            st.error("FPDF2 not installed. Install fpdf2 to enable PDF reports.")
        else:
            # ensure bytes (fpdf2 returns bytearray sometimes)
            pdf_bytes = bytes(pdf_bytes) if isinstance(pdf_bytes, bytearray) else pdf_bytes
            st.download_button("Download report (PDF)", pdf_bytes, file_name=f"report_{fname}.pdf", mime="application/pdf")

    # CSV export
    if st.button(f"Export CSV row for {fname}"):
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

    # Debug expander
    with st.expander("Model debug info"):
        st.write("Raw probabilities (first 50):")
        st.write(probs[:min(50, len(probs))])
        mapping = {i: DISPLAY_NAMES.get(IDX_TO_LABEL.get(i, ""), f"class_{i}") for i in range(len(IDX_TO_LABEL))}
        st.json(mapping)

# History viewer
st.markdown("---")
st.header("Saved Reports (History)")
reports = list_reports(limit=200)
for r in reports:
    cols = st.columns([1, 3, 2, 2])
    with cols[0]:
        st.write(r["id"])
    with cols[1]:
        st.write(r["filename"])
        st.write(r["timestamp"])
    with cols[2]:
        st.write(r["top_label"])
        st.write(f"{r['top_prob']*100:.1f}%")
    with cols[3]:
        if st.button(f"Download image {r['id']}", key=f"dlimg{r['id']}"):
            img_bytes = get_report_image_bytes(r['id'])
            if img_bytes:
                st.download_button("Download image", img_bytes, file_name=f"report_img_{r['id']}.jpg", mime="image/jpeg")
        if st.button(f"Delete {r['id']}", key=f"del{r['id']}"):
            try:
                delete_report(r['id'])
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")

st.caption("Project S — informational only. Not a medical device. Always consult a dermatologist for diagnosis.")
