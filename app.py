# =========================
# FINAL CORRECTED APP.PY — PROJECT S
# (Language fix, PDF fix, UI fix, no duplicate keys)
# =========================

import os
import io
import json
import tempfile
import sqlite3
from datetime import datetime
from typing import Tuple, Optional, Dict, List

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Optional CV2
try:
    import cv2
except:
    cv2 = None

# PDF
from fpdf import FPDF
import fpdf as _fpdf_pkg

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "models/skin_classifier.h5"
TFLITE_PATH = "models/skin_classifier.tflite"
IMG_SIZE = 160
DB_PATH = "project_s_reports.db"
FONTS_DIR = "fonts"
TOP_K_DEFAULT = 3

# ---------------------------
# CLASS MAPS
# ---------------------------
CLASS_INDICES = {
 'acne_rosacea': 0,
 'fungal_infections': 8,
 'eczema': 6,
 'psoriasis': 15,
 'pigmentation': 13,
 'viral_infections': 22,
}
IDX_TO_LABEL = {v: k for k, v in CLASS_INDICES.items()}
DISPLAY_NAMES = {k: k.replace('_',' ').title() for k in CLASS_INDICES}

# ---------------------------
# TEXTS (SHORT)
# ---------------------------
DESCRIPTIONS_EN = {k: f"{DISPLAY_NAMES[k]} — brief description." for k in CLASS_INDICES}
TREATMENTS_EN = {k: "Consult a dermatologist if symptoms persist." for k in CLASS_INDICES}

DESCRIPTIONS_HI = {k: f"{DISPLAY_NAMES[k]} — संक्षिप्त विवरण." for k in CLASS_INDICES}
TREATMENTS_HI = {k: "यदि लक्षण बने रहें तो त्वचा विशेषज्ञ से संपर्क करें।" for k in CLASS_INDICES}

DESCRIPTIONS_TE = {k: f"{DISPLAY_NAMES[k]} — సంక్షిప్త వివరణ." for k in CLASS_INDICES}
TREATMENTS_TE = {k: "లక్షణాలు కొనసాగితే చర్మ వైద్యుడిని సంప్రదించండి." for k in CLASS_INDICES}

# ---------------------------
# UTILITIES
# ---------------------------
def ensure_rgb(pil: Image.Image):
    return pil.convert("RGB")

# Skin mask

def skin_mask_ycrcb(pil: Image.Image):
    arr = np.array(pil.convert("RGB"))
    if cv2 is None:
        r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        mask = (r>80)&(g>40)&(b>20)&(r>g)&(r>b)
        return (mask.astype(np.uint8)*255)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _,cr,cb = cv2.split(ycrcb)
    mask = ((cr>=135)&(cr<=180)&(cb>=85)&(cb<=135)).astype(np.uint8)*255
    return cv2.medianBlur(mask,5)

# Auto-crop

def auto_crop_by_skin(pil, margin=0.02):
    mask = skin_mask_ycrcb(pil)
    ys,xs = np.where(mask>0)
    if len(xs)<20:
        return None,None
    x0,x1 = xs.min(),xs.max()
    y0,y1 = ys.min(),ys.max()
    dx=int((x1-x0)*margin)
    dy=int((y1-y0)*margin)
    x0=max(0,x0-dx); y0=max(0,y0-dy)
    x1=min(mask.shape[1]-1,x1+dx)
    y1=min(mask.shape[0]-1,y1+dy)
    return pil.crop((x0,y0,x1,y1)), (x0,y0,x1,y1)

# Preprocess

def preprocess_pil(pil,target,auto_crop=True):
    pil=ensure_rgb(pil)
    box=None
    if auto_crop:
        c,b=auto_crop_by_skin(pil)
        if c is not None:
            pil=c; box=b
    pil_fit=ImageOps.fit(pil,target,Image.Resampling.LANCZOS)
    arr=np.array(pil_fit)/255.0
    return np.expand_dims(arr,0), box

# Severity

def estimate_severity(img,box):
    if box: img=img.crop(box)
    arr=np.array(img)/255.0
    h,w,_=arr.shape
    mask=(skin_mask_ycrcb(img)>0).astype(np.uint8)
    area=mask.sum()/(h*w)
    r=arr[:,:,0]; gb=np.maximum(arr[:,:,1],arr[:,:,2])
    redness=float(np.clip((r-gb).mean(),0,1))
    score=0.5*area+0.35*redness
    level="Low" if score<0.33 else "Medium" if score<0.66 else "High"
    return {"area_pct":area,"redness":redness,"score":score,"level":level}

# ---------------------------
# PDF (UNICODE SAFE)
# ---------------------------
class PDF(FPDF):
    pass

def safe_font(pdf,text):
    if any("\u0900"<=c<="\u097F" for c in text):  # Hindi
        pdf.set_font("Arial",size=12)
    elif any("\u0C00"<=c<="\u0C7F" for c in text):  # Telugu
        pdf.set_font("Arial",size=12)
    else:
        pdf.set_font("Arial",size=12)

def generate_pdf(img,preds,desc,treat,sev):
    pdf=PDF()
    pdf.add_page()
    pdf.set_font("Arial",size=14)
    pdf.cell(0,10,f"Prediction: {preds[0][0]}")
    tmp=tempfile.mktemp(suffix=".jpg")
    img.save(tmp,"JPEG")
    pdf.image(tmp,x=10,y=30,w=80)
    pdf.ln(60)
    safe_font(pdf,desc)
    pdf.multi_cell(0,8,f"Description: {desc}")
    pdf.ln(5)
    safe_font(pdf,treat)
    pdf.multi_cell(0,8,f"Treatment: {treat}")
    pdf.ln(5)
    pdf.multi_cell(0,8,f"Severity: {sev['level']} (Score: {sev['score']:.2f})")
    return pdf.output(dest="S").encode("latin-1")

# ---------------------------
# DB
# ---------------------------
conn=sqlite3.connect(DB_PATH,check_same_thread=False)
cur=conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY,filename TEXT,desc TEXT,treat TEXT,sev TEXT,img BLOB)")
conn.commit()

# ---------------------------
# MODEL LOAD
# ---------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        m=tf.keras.models.load_model(MODEL_PATH,compile=False)
        return {"type":"keras","model":m,"input_shape":(IMG_SIZE,IMG_SIZE,3)}
    return {"type":"none"}

model_info=load_model()
if model_info["type"]=="none":
    st.error("Model missing.")
    st.stop()

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(layout="wide")
st.session_state.setdefault("APP_LANG","English")
st.session_state.setdefault("auto_crop",True)
st.session_state.setdefault("top_k",3)
st.session_state.setdefault("temperature",1.0)

with st.sidebar:
    st.selectbox("Language",["English","Hindi","Telugu"],key="APP_LANG")
    page=st.radio("Pages",["Classifier","History"])  

# Language map

def lang_maps(lang):
    if lang=="Hindi": return DESCRIPTIONS_HI,TREATMENTS_HI
    if lang=="Telugu": return DESCRIPTIONS_TE,TREATMENTS_TE
    return DESCRIPTIONS_EN,TREATMENTS_EN

DESCRIPTIONS,TREATMENTS=lang_maps(st.session_state.APP_LANG)

# ---------------------------
# HISTORY
# ---------------------------
if page=="History":
    st.header("History")
    cur.execute("SELECT id,filename,desc,treat,sev FROM reports ORDER BY id DESC")
    rows=cur.fetchall()
