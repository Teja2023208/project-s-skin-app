# =========================
# app.py — Project S (Clean Final, Option A - Full)
# Single-file Streamlit app (final corrected)
# - Sidebar navigation
# - All pages (Classifier, History, Custom Text Editor, Skin Detection Preview, Audio Output, Severity Charts)
# - Classifier page: clean UI, language selector in sidebar & classifier, Grad-CAM enabled by default
# =========================

import os
import io
import sys
import json
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

# fpdf2 required
try:
    from fpdf import FPDF
    import fpdf as _fpdf_pkg
except Exception:
    FPDF = None
    _fpdf_pkg = None

# Flask for API (optional)
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
# DEFAULT TEXTS (English / Hindi / Telugu)
# ----------------------------
DESCRIPTIONS_EN = {
    "acne_rosacea": "A chronic skin condition causing redness, bumps, or pimple-like eruptions on the face. Often worsened by heat, sun, spicy foods, or stress.",
    "actinic_keratosis_basal_cell_carcinoma_and_other_malignant_lesions": "Sun-induced precancerous and cancerous lesions. Appear as rough, scaly patches or non-healing sores.",
    "atopic_dermatitis_photos": "A chronic itchy inflammatory skin condition known as eczema. Often dry and triggered by irritants or allergies.",
    "autoimmune": "Autoimmune skin diseases where the immune system attacks the skin, causing inflammation and rashes.",
    "bacterial_infections": "Infections caused by bacteria like Staphylococcus. Appear as red, swollen, painful lesions with possible pus.",
    "bullous_disease_photos": "Conditions with large fluid-filled blisters, often autoimmune.",
    "eczema": "Dry, itchy, inflamed skin with recurring flares triggered by irritants.",
    "exanthems_and_drug_eruptions": "Rashes caused by viral infection or medication reaction.",
    "fungal_infections": "Ringworm-like infections with circular, red, itchy patches.",
    "hair_loss": "Hair thinning due to autoimmune, hormonal, or infectious causes.",
    "herpes_hpv_and_other_stds_photos": "Viral STDs causing blisters, ulcers, or wart-like growths.",
    "melanoma": "A dangerous skin cancer arising from pigment cells. Appears as changing or irregular moles.",
    "nail_fungus_and_other_nail_disease": "Nail thickening, discoloration, brittleness caused by fungal infection.",
    "pigmentation": "Darkening or lightening of skin due to melanin imbalance (melasma, vitiligo).",
    "poison_ivy_photos_and_other_contact_dermatitis": "Allergic skin reaction causing redness, itching, and blisters.",
    "psoriasis": "Autoimmune condition with thick, red, scaly plaques.",
    "scabies_lyme_disease_and_other_infestations_and_bites": "Mite/tick/insect bites causing intense itching and irritation.",
    "seborrheic_keratoses_and_other_benign_tumors": "Harmless raised skin growths common in older adults.",
    "systemic_disease": "Skin changes caused by internal diseases (diabetes, thyroid, liver).",
    "urticaria_hives": "Raised itchy wheals often caused by allergies.",
    "vascular_tumors": "Red/purple growths formed by abnormal blood vessels.",
    "vasculitis_photos": "Inflamed blood vessels causing red or purple spots.",
    "viral_infections": "Rashes or lesions caused by common skin viruses.",
}
TREATMENTS_EN = {
    "acne_rosacea": "Avoid heat, spicy food. Use gentle cleansers. Topical antibiotics help.",
    "actinic_keratosis_basal_cell_carcinoma_and_other_malignant_lesions": "Needs dermatologist evaluation. Removal or freezing required.",
    "atopic_dermatitis_photos": "Moisturize often. Use steroid creams for flares.",
    "autoimmune": "Requires specialist treatment with steroids or immunosuppressants.",
    "bacterial_infections": "Topical/oral antibiotics. Seek care for spreading redness.",
    "bullous_disease_photos": "Do not burst blisters. Requires dermatology supervision.",
    "eczema": "Moisturizers, steroid creams, antihistamines during flares.",
    "exanthems_and_drug_eruptions": "Stop triggering drug (under doctor advice). Antihistamines help.",
    "fungal_infections": "Apply antifungal creams. Keep area dry.",
    "hair_loss": "Use minoxidil. Rule out nutritional deficiencies.",
    "herpes_hpv_and_other_stds_photos": "Avoid contact. Antivirals or cryotherapy needed.",
    "melanoma": "URGENT dermatologist evaluation. Surgical removal required.",
    "nail_fungus_and_other_nail_disease": "Antifungal creams or oral medicines.",
    "pigmentation": "Daily sunscreen. Vitamin C/niacinamide creams help.",
    "poison_ivy_photos_and_other_contact_dermatitis": "Cold compress, steroid creams, avoid irritant.",
    "psoriasis": "Steroid creams, vitamin D creams, phototherapy.",
    "scabies_lyme_disease_and_other_infestations_and_bites": "Permethrin cream. Treat itching.",
    "seborrheic_keratoses_and_other_benign_tumors": "Removal optional (laser/freezing).",
    "systemic_disease": "Treat underlying condition. Dermatologist guidance needed.",
    "urticaria_hives": "Antihistamines. Avoid triggers.",
    "vascular_tumors": "Laser or surgical removal.",
    "vasculitis_photos": "Requires urgent evaluation. Steroids may be needed.",
    "viral_infections": "Supportive care. Antivirals for specific infections.",
}

# Hindi & Telugu (short copies of English for now — can be customized)
DESCRIPTIONS_HI = {k: v for k, v in DESCRIPTIONS_EN.items()}
TREATMENTS_HI = {k: v for k, v in TREATMENTS_EN.items()}
DESCRIPTIONS_TE = {k: v for k, v in DESCRIPTIONS_EN.items()}
TREATMENTS_TE = {k: v for k, v in TREATMENTS_EN.items()}

# ----------------------------
# UTILS
# ----------------------------
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

# ----------------------------
# Skin mask / strict check
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

def strict_skin_check(pil):
    img = np.array(pil.convert("RGB"))
    if cv2 is None:
        return True, 0.5, "OK (cv2 missing)"
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    mask1 = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask2 = ((h < 25) | (h > 330)) & (s > 30) & (v > 50)
    mask2 = mask2.astype(np.uint8)
    mask = cv2.bitwise_and(mask1, mask2)
    skin_ratio = float(mask.sum()) / (mask.shape[0] * mask.shape[1] + 1e-9)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    std_gray = float(np.std(gray))
    if std_gray < 18:
        return False, skin_ratio, "Looks like an artwork / edited poster"
    try:
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_detector.detectMultiScale(gray, 1.2, 6)
    except:
        faces = []
    face_count = len(faces)
    if skin_ratio < 0.10 and face_count == 0:
        return False, skin_ratio, "Very low skin detected and no real face"
    if skin_ratio < 0.05:
        return False, skin_ratio, "Not enough skin pixels"
    if face_count == 0 and skin_ratio < 0.20:
        return False, skin_ratio, "Face not detected; looks like illustration"
    return True, skin_ratio, "OK"

# ----------------------------
# Auto-crop & preprocess
# ----------------------------
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

# ----------------------------
# SEVERITY ESTIMATION
# ----------------------------
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

# ----------------------------
# Grad-CAM utilities
# ----------------------------
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

# ----------------------------
# PREDICTION wrappers (Keras / TFLite)
# ----------------------------
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

# ----------------------------
# PDF: Unicode-safe using fpdf2 and Noto fonts
# ----------------------------
class MedicalPDF(FPDF):
    def header(self):
        if "Noto" in self.fonts:
            self.set_font("Noto", size=12)
        else:
            self.set_font("Arial", size=12)
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

def sanitize_ascii(text: str) -> str:
    if not isinstance(text, str):
        return text
    return (
        text.replace("—", "-")
            .replace("–", "-")
            .replace("“", '"')
            .replace("”", '"')
            .replace("…", "...")
            .replace("’", "'")
    )

def generate_unicode_pdf(image_pil, preds, desc, treat, severity, gradcam_img=None):
    if FPDF is None:
        return None
    pdf = MedicalPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    en = os.path.join(FONTS_DIR, "NotoSans-Regular.ttf")
    hi = os.path.join(FONTS_DIR, "NotoSansDevanagari-Regular.ttf")
    te = os.path.join(FONTS_DIR, "NotoSansTelugu-Regular.ttf")
    if os.path.exists(en): pdf.add_font("Noto", "", en, uni=True)
    if os.path.exists(hi): pdf.add_font("NotoDev", "", hi, uni=True)
    if os.path.exists(te): pdf.add_font("NotoTel", "", te, uni=True)
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
    label_line = sanitize_ascii(f"{label} - {top_prob*100:.1f}%")
    pdf.set_x(right_x)
    auto_font(label_line, 12)
    pdf.cell(0, 7, label_line, ln=True)

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
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Description", ln=True)
    pdf.set_fill_color(248, 248, 248)
    pdf.multi_cell(pdf.w - 20, 8, desc, fill=True)
    pdf.ln(5)
    auto_font("Suggested Treatment", 14)
    pdf.cell(0, 10, "Suggested Treatment", ln=True)
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

# public generate_pdf_report
generate_pdf_report = generate_unicode_pdf

# ----------------------------
# FLASK API (background)
# ----------------------------
flask_app = None
def start_api_server(host=API_HOST, port=API_PORT_DEFAULT):
    global flask_app
    if Flask is None:
        st.warning("Flask not installed - API disabled.")
        return
    if flask_app is not None:
        return
    flask_app = Flask("project_s_api")

    @flask_app.route("/predict", methods=["POST"])
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
                return jsonify({"error": "No image provided"}), 400
            temp = float(request.args.get("temperature", 1.0))
            model_info_local = load_model()
            target_h, target_w, _ = model_info_local["input_shape"]
            arr, crop_box = preprocess_pil(img, target_size=(target_h, target_w), auto_crop=True)
            probs = predict_with_model(model_info_local, arr, temperature=temp)
            top_k = int(request.args.get("top_k", TOP_K_DEFAULT))
            top_indices = np.argsort(probs)[::-1][:top_k]
            preds = [[int(i), float(probs[i])] for i in top_indices]
            severity = estimate_severity(img, crop_box)
            if request.args.get("save", "0") == "1":
                desc = CUSTOMS.get("descriptions", {}).get(IDX_TO_LABEL.get(int(preds[0][0]), ""), DESCRIPTIONS_EN.get(IDX_TO_LABEL.get(int(preds[0][0]), ""), ""))
                treat = CUSTOMS.get("treatments", {}).get(IDX_TO_LABEL.get(int(preds[0][0]), ""), TREATMENTS_EN.get(IDX_TO_LABEL.get(int(preds[0][0]), ""), ""))
                save_report_to_db("api_upload.jpg", preds, desc, treat, severity, img)
            return jsonify({"preds": preds, "severity": severity})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/reports", methods=["GET"])
    def api_reports():
        r = list_reports(limit=200)
        return jsonify({"reports": r})

    def run():
        flask_app.run(host=host, port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

# ----------------------------
# MODEL LOADING (Keras / TFLite)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(path_keras: str = MODEL_PATH, path_tflite: str = TFLITE_PATH):
    info = {"type": "none", "model": None, "input_shape": (IMG_SIZE, IMG_SIZE, 3)}
    if os.path.exists(path_keras):
        try:
            model = tf.keras.models.load_model(path_keras, compile=False)
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

# ----------------------------
# STORAGE: SQLite helpers
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
# CUSTOM TEXTS (JSON persistence)
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
# AI AUTO DESCRIPTION GENERATOR (fallback only)
# ----------------------------
def ai_generate_texts(label_key: str, lang: str = "English"):
    label_display = DISPLAY_NAMES.get(label_key, label_key)
    desc_template = f"{label_display} is a skin condition that typically shows localized changes to the skin (redness, bumps, or texture change). Seek clinical evaluation for persistent or concerning lesions."
    treat_template = "Keep the area clean, avoid triggers, use OTC topical measures when appropriate, and consult a dermatologist."
    if lang == "Hindi":
        desc_template = f"{label_display} एक त्वचा संबंधी समस्या है जो आमतौर पर त्वचा पर लालिमा, दाने या बनावट में बदलाव दिखाती है।"
        treat_template = "क्षेत्र को साफ रखें, ट्रिगर से बचें, और आवश्यक होने पर त्वचा विशेषज्ञ से सलाह लें।"
    if lang == "Telugu":
        desc_template = f"{label_display} ఒక చెర్మ్ సమస్య, సాధారణంగా చర్మంలో ఎరుపు, గడ్డలు లేదా టెక్స్చర్ మార్పులను చూపుతుంది."
        treat_template = "ప్రాంతాన్ని శుభ్రంగా ఉంచండి, ట్రిగ్గర్లను నివారించండి, అవసరమైతే డెర్మాటాలజిస్ట్ ని సంప్రదించండి."
    return desc_template, treat_template

# ----------------------------
# TEXT-TO-SPEECH helper (optional, not required by UI)
# ----------------------------
try:
    from gtts import gTTS
except Exception:
    gTTS = None

def text_to_audio_bytes(text: str, lang_code: str = "en") -> Optional[bytes]:
    if gTTS is None:
        return None
    try:
        tts = gTTS(text=text, lang=lang_code)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception:
        return None

# ----------------------------
# STREAMLIT UI START
# ----------------------------
st.set_page_config(page_title="Project S — Skin Disease Classifier (Final)", layout="wide")

# quick fpdf2 version check
if _fpdf_pkg is None:
    st.sidebar.error("fpdf2 not installed. Install with: pip install fpdf2 to enable Unicode PDF reports.")
else:
    try:
        ver = _fpdf_pkg.__version__
        if int(ver.split(".")[0]) < 2:
            st.sidebar.error("Old fpdf detected. Install fpdf2 (pip uninstall fpdf; pip install fpdf2).")
    except Exception:
        pass

# Load model
with st.spinner("Loading model..."):
    model_info = load_model()

if model_info["type"] == "none":
    st.error("No model found. Place a Keras model at models/skin_classifier.h5 or a TFLite model at models/skin_classifier.tflite")
    st.stop()

# Defaults
enable_api = False
auto_crop = True
enable_gradcam = True  # Grad-CAM enabled by default (user confirmed)
top_k = TOP_K_DEFAULT
temperature = 1.0

# ----------------------------
# SIDEBAR NAV + PAGES
# ----------------------------
PAGES = [
    "Classifier",
    "History",
    "Custom Text Editor",
    "Skin Detection Preview",
    "Audio Output",
    "Severity Charts",
]
if "ps_page" not in st.session_state:
    st.session_state.ps_page = "Classifier"

with st.sidebar:
    LANGUAGES = ["English", "Hindi", "Telugu"]

    if "APP_LANG" not in st.session_state:
        st.session_state.APP_LANG 

    st.session_state.APP_LANG = st.selectbox(
        "Select language",
        LANGUAGES,
        index=LANGUAGES.index(st.session_state.APP_LANG)
    )

    st.button("Sync from custom_texts.json", key="sync_json_sidebar")

    st.markdown("---")
    st.caption("Project S — informational only.")

    page_choice = st.radio("", PAGES, index=PAGES.index(st.session_state.ps_page))
    st.session_state.ps_page = page_choice

page = st.session_state.ps_page

# ---------- Helper small UI utilities ----------
def safe_get_report_image(report_id: int):
    b = get_report_image_bytes(report_id)
    if b:
        return Image.open(io.BytesIO(b)).convert("RGB")
    return None

# ---------- Page: History ----------
def render_history_page():
    st.header("📁 History — Saved Reports")
    rows = list_reports(limit=500)
    if not rows:
        st.info("No saved reports.")
        return
    df_rows = []
    for r in rows:
        df_rows.append({
            "id": r["id"],
            "filename": r["filename"],
            "timestamp": r["timestamp"],
            "top_label": r["top_label"],
            "top_prob": f"{r['top_prob']*100:.1f}%"
        })
    df = pd.DataFrame(df_rows)
    st.dataframe(df, use_container_width=True)
    st.markdown("### Actions")
    cols = st.columns(3)
    sel_id = st.number_input("Enter report id to view/delete", min_value=1, value=rows[0]["id"], step=1)
    if cols[0].button("View report"):
        rep = next((r for r in rows if r["id"] == sel_id), None)
        if not rep:
            st.error("Report not found.")
        else:
            st.subheader(f"Report #{rep['id']} — {rep['filename']}")
            img = safe_get_report_image(rep["id"])
            if img:
                st.image(img, width=420)
            st.markdown("**Prediction**: " + str(rep["top_label"]))
            st.markdown("**Confidence**: " + f"{rep['top_prob']*100:.1f}%")
            st.markdown("**Description**")
            st.write(rep.get("description", ""))
            st.markdown("**Treatment**")
            st.write(rep.get("treatment", ""))
            st.markdown("**Severity**")
            st.json(rep.get("severity", {}))
            if cols[1].button("Download PDF", key=f"dlpdf_{rep['id']}"):
                img = safe_get_report_image(rep["id"])
                pdfb = generate_pdf_report(img if img else Image.new("RGB",(100,100),(255,255,255)),
                                           rep.get("preds", []), rep.get("description",""), rep.get("treatment",""), rep.get("severity", {}))
                if pdfb:
                    st.download_button("Download report PDF", pdfb, file_name=f"report_{rep['id']}.pdf", mime="application/pdf")
                else:
                    st.error("PDF generation unavailable.")
            if cols[2].button("Delete", key=f"del_{rep['id']}"):
                delete_report(rep["id"])
                st.success("Report deleted. Refresh page to update list.")

# ---------- Page: Custom Text Editor ----------
def render_custom_text_editor():
    st.header("📝 Custom Text Editor (Descriptions & Treatments)")
    customs = load_custom_texts()
    descs = customs.get("descriptions", {})
    treats = customs.get("treatments", {})
    st.markdown("Edit descriptions and treatments for classes. Click Save to persist to `custom_texts.json`.")
    options = list(CLASS_INDICES.keys())
    sel = st.selectbox("Select class key", options, format_func=lambda k: DISPLAY_NAMES.get(k, k))
    cur_desc = descs.get(sel, DESCRIPTIONS_EN.get(sel, ""))
    cur_treat = treats.get(sel, TREATMENTS_EN.get(sel, ""))
    new_desc = st.text_area("Description", value=cur_desc, height=120)
    new_treat = st.text_area("Treatment", value=cur_treat, height=120)
    if st.button("Save changes for selected class"):
        customs.setdefault("descriptions", {})[sel] = new_desc
        customs.setdefault("treatments", {})[sel] = new_treat
        save_custom_texts(customs)
        st.success("Saved custom text. Use Sync button in sidebar or reload app to reload into memory.")
        global CUSTOMS
        CUSTOMS = load_custom_texts()
    if st.button("Reset all custom texts (delete custom_texts.json)"):
        if os.path.exists(CUSTOM_TEXTS):
            os.remove(CUSTOM_TEXTS)
        CUSTOMS = {"descriptions": {}, "treatments": {}}
        st.success("Custom texts removed; defaults restored.")

# ---------- Page: Skin Detection Preview ----------
def render_skin_preview():
    st.header("🖼️ Skin Detection Preview")
    st.markdown("Upload an image to preview skin mask, detected region bounding box and suggested auto-crop.")
    up = st.file_uploader("Upload image for skin preview", type=["jpg","jpeg","png"])
    if not up:
        st.info("Upload an image to preview.")
        return
    pil = Image.open(io.BytesIO(up.read())).convert("RGB")
    st.image(pil, caption="Original", width=420)
    mask = skin_mask_ycrcb(pil)
    skin_frac = float((mask > 0).sum()) / (mask.shape[0]*mask.shape[1] + 1e-9)
    st.write(f"Skin pixel fraction: {skin_frac:.3f}")
    mask_img = Image.fromarray(np.stack([mask]*3, axis=2))
    st.image(mask_img, caption="Skin mask (white = skin)", width=420)
    crop, box = auto_crop_by_skin(pil)
    if crop is not None:
        st.image(crop, caption=f"Auto-crop preview {box}", width=360)
    else:
        st.info("Auto-crop heuristic couldn't find a dominant skin region.")
    st.markdown("Tip: if mask misses area, try a closer/clearer photo or disable auto-crop in main UI.")

# ---------- Page: Audio Output (TTS) ----------
def render_audio_output():
    st.header("🔊 Text-to-Speech (Audio Output)")
    st.markdown("Generate audio from text or use description text from Classifier page.")
    txt = st.text_area("Text to speak (leave empty to test custom text)", height=160)
    lang_choice = st.selectbox("Language for speech", ["English", "Hindi", "Telugu"], index=0, key="audio_lang")
    lang_map = {"English":"en","Hindi":"hi","Telugu":"te"}
    lang_code_local = lang_map.get(lang_choice,"en")
    if st.button("Generate audio"):
        if not txt:
            st.warning("Please provide text to synthesize.")
        else:
            b = text_to_audio_bytes(txt, lang_code=lang_code_local)
            if b:
                st.audio(b, format="audio/mp3")
            else:
                st.error("TTS unavailable (gTTS not installed or failed).")

# ---------- Page: Severity Charts ----------
def render_severity_charts():
    st.header("📊 Severity Charts")
    st.markdown("Upload an image to compute severity metrics and show charts.")
    upload = st.file_uploader("Upload image for severity plot", type=["jpg","jpeg","png"])
    if not upload:
        st.info("Upload an image to generate severity charts.")
        return
    img = Image.open(io.BytesIO(upload.read())).convert("RGB")
    severity = estimate_severity(img, None)
    st.write("Severity values:")
    st.json(severity)
    metrics = {"Area%": severity["area_pct"]*100, "Redness": severity["redness"]*100, "Texture": severity["texture"]*100}
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(metrics.keys(), metrics.values())
    ax.set_ylabel("Scaled percent")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

# ---------- Page Dispatcher ----------
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
    else:
        st.info("Unknown page")
    st.stop()

# -------------------------------
# CLEAN CLASSIFIER UI
# -------------------------------
st.markdown(
    """
    <style>
    .title {font-size:30px; font-weight:700; margin-bottom:4px;}
    .subtitle {color:#6b7280; margin-top:0px; margin-bottom:18px;}
    .upload-box {border:2px dashed #e6e6e6;padding:18px;border-radius:10px;text-align:center}
    .small {font-size:13px;color:#6b7280}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Skin Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a close-up skin image. Diagnostic use is not intended.</div>', unsafe_allow_html=True)

# Classifier controls (minimal): language selector + auto-crop toggle + Grad-CAM toggle
ccol1, ccol2 = st.columns([3, 1])
with ccol2:
    auto_crop = st.checkbox("Auto-crop lesion (heuristic)", value=auto_crop, key="auto_crop_ui")
    enable_gradcam = st.checkbox("Enable Grad-CAM (Keras only)", value=enable_gradcam, key="gradcam_ui")
    st.markdown("---")
    enable_api = st.checkbox("Enable REST API (Flask, background)", value=enable_api, key="api_ui")
    api_port = st.number_input("API port", min_value=1025, max_value=65535, value=API_PORT_DEFAULT, key="api_port_ui")
    st.markdown("---")
    st.subheader("Language & Texts (also in sidebar)")
    LANGUAGES = ["English", "Hindi", "Telugu"]
    APP_LANG = st.selectbox("Select language", LANGUAGES, index=0, key="lang_ui")
    if st.button("Sync custom texts", key="sync_json_classifier"):
        CUSTOMS = load_custom_texts()
        st.success("Synced custom_texts.json into memory (reloaded).")

with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    st.markdown('</div>', unsafe_allow_html=True)

camera_img = st.camera_input("Or capture with camera", key="camera_ui")

images = []
if uploaded_file:
    pil = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    images.append((uploaded_file.name, pil))
elif camera_img:
    pil = Image.open(io.BytesIO(camera_img.read())).convert("RGB")
    images.append(("camera.jpg", pil))

if not images:
    st.info("Upload or capture an image to continue.")
    st.stop()

# Map language -> texts
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

DESCRIPTIONS, TREATMENTS, LANG_CODE = get_text_maps(APP_LANG)

# Start API if requested
if enable_api:
    try:
        start_api_server(host=API_HOST, port=int(api_port))
        st.success(f"REST API started at http://{API_HOST}:{int(api_port)} (predict & reports)")
    except Exception as e:
        st.warning(f"API failed to start: {e}")

for fname, pil_img in images:
    st.markdown("---")
    st.header(f"Image: {fname}")
    st.image(pil_img, caption="Original", width=420)

    # skin check
    mask = skin_mask_ycrcb(pil_img)
    skin_frac = float((mask > 0).sum()) / (mask.shape[0] * mask.shape[1] + 1e-9)
    st.write(f"Skin pixel fraction: {skin_frac:.3f}")
    if skin_frac < 0.02:
        st.error("Not a skin image (low skin pixel fraction). Try another photo.")
        continue

    # preprocess
    target_h, target_w, _ = model_info["input_shape"]
    processed_arr, crop_box = preprocess_pil(
        pil_img, target_size=(target_h, target_w), auto_crop=auto_crop
    )
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

    # display top result card
    top_idx, top_prob = preds[0]
    top_label = DISPLAY_NAMES.get(IDX_TO_LABEL.get(top_idx, ""), f"class_{top_idx}")
    color = confidence_color(top_prob)
    st.markdown(f"### 🔍 Prediction — *{top_label}*")
    st.markdown(
        f"<div style='padding:10px;border-left:6px solid {color};border-radius:6px;'>"
        f"Confidence: <strong>{top_prob*100:5.1f}%</strong></div>",
        unsafe_allow_html=True,
    )

    # description & treatment (language-aware)
    desc_default = DESCRIPTIONS.get(IDX_TO_LABEL.get(top_idx), "")
    treat_default = TREATMENTS.get(IDX_TO_LABEL.get(top_idx), "")

    colA, colB = st.columns([2, 3])
    with colA:
        st.markdown("#### Brief description")
        new_desc = st.text_area(f"desc_{fname}", value=desc_default, height=120)
    with colB:
        st.markdown("#### Suggested treatment")
        new_treat = st.text_area(f"treat_{fname}", value=treat_default, height=120)
        st.markdown("#### Severity")
        st.write(f"Level: *{severity['level']}* — Score: *{severity['score']:.2f}*")
        st.write(f"Area: *{severity['area_pct']*100:.1f}%*")
        st.write(f"Redness: *{severity['redness']:.3f}*")

    # Grad-CAM
    gradcam_img = None
    if enable_gradcam and model_info["type"] == "keras":
        heatmap = make_gradcam_heatmap(processed_arr, model_info["model"], None, int(top_idx))
        if heatmap is not None:
            base_img = pil_img.crop(crop_box).resize((target_w, target_h)) if crop_box else pil_img.resize((target_w, target_h))
            gradcam_img = overlay_heatmap_on_image(base_img, heatmap, alpha=0.45)
            st.image(gradcam_img, caption="Grad-CAM", width=420)

    # Save / Download options
    ops_col1, ops_col2, ops_col3 = st.columns([1,1,1])
    with ops_col1:
        if st.button("Save report to history", key=f"save_{fname}"):
            save_report_to_db(fname, preds, new_desc, new_treat, severity, pil_img)
            st.success("Saved report to local DB (ephemeral on cloud).")
    with ops_col2:
        if FPDF is not None:
            pdf_bytes = generate_pdf_report(pil_img, preds, new_desc, new_treat, severity, gradcam_img)
            if pdf_bytes:
                st.download_button("Download PDF", pdf_bytes, file_name=f"report_{fname}.pdf", mime="application/pdf")
            else:
                st.error("PDF generation failed.")
        else:
            st.error("FPDF/fpdf2 or fonts not installed. Install fpdf2 and add fonts to enable PDF reports.")
    with ops_col3:
        if st.button(f"Export CSV row for {fname}", key=f"csv_{fname}"):
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

# Footer
st.markdown("---")
st.caption("Project S — informational only. Not a medical device. Always consult a dermatologist for diagnosis.")
# End of file
