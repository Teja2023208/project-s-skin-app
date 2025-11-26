# ----------------------------
# IMAGE + TOP PREDICTION LAYOUT
# ----------------------------
left_x = 10
img_width = 90

# ---- Save main image to temp path ----
tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
os.close(tmp_fd)
image_pil.save(tmp_path, "JPEG")

# ---- Insert image ----
pdf.image(tmp_path, x=left_x, y=30, w=img_width)

# ---- Always delete AFTER using in PDF ----
try:
    os.remove(tmp_path)
except:
    pass

# ---- Right column start (fixed) ----
right_x = left_x + img_width + 20   # 10 + 90 + 20 = 120 px

pdf.set_xy(right_x, 30)
auto_font("Top Prediction:", 14)
pdf.cell(0, 8, "Top Prediction:", ln=True)

# ---- Prepare text ----
top_idx, top_prob = preds[0]
label = DISPLAY_NAMES.get(IDX_TO_LABEL.get(top_idx, ""), f"Class {top_idx}")
label_line = sanitize_ascii(f"{label} - {top_prob*100:.1f}%")

# ---- Print prediction ----
pdf.set_x(right_x)
auto_font(label_line, 12)
pdf.cell(0, 7, label_line, ln=True)

# ---- If Grad-CAM exists, show it under prediction ----
if gradcam_img:
    tmp_fd, grad_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)
    gradcam_img.save(grad_path, "JPEG")

    pdf.image(grad_path, x=right_x, y=55, w=80)

    try:
        os.remove(grad_path)
    except:
        pass

# ---- Move cursor below image area ----
pdf.set_y(140)
