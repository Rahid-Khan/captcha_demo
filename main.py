import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import re
import os

# ---------------------------------------
# Page Config
# ---------------------------------------
st.set_page_config(page_title="Smart CAPTCHA Solver", page_icon="🔍")

st.title("🔍 Smart Grid CAPTCHA Solver")
st.write("Automatically reads instruction → detects target number → finds matching tiles")

# =======================================
# EASY OCR (LOCAL MODELS ONLY)
# =======================================
@st.cache_resource
def load_reader():
    """
    Smart loader:
    • First run → downloads models automatically
    • Later runs → loads from local folder
    """

    model_path = "models"
    os.makedirs(model_path, exist_ok=True)

    model_file = os.path.join(model_path, "craft_mlt_25k.pth")

    if not os.path.exists(model_file):
        st.info("⬇️ First run: downloading EasyOCR models (one-time setup)...")

        # allow download
        reader = easyocr.Reader(
            ['en'],
            gpu=False,
            model_storage_directory=model_path,
            download_enabled=True
        )

        st.success("✅ Models downloaded and cached locally")
        return reader

    # models already exist → no download
    return easyocr.Reader(
        ['en'],
        gpu=False,
        model_storage_directory=model_path,
        download_enabled=False
    )


reader = load_reader()


# =======================================
# IMAGE ENHANCEMENT (Zoom + Sharpen + Denoise)
# =======================================
def enhance_image(img, scale=2):
    h, w = img.shape[:2]

    zoomed = cv2.resize(
        img,
        (w * scale, h * scale),
        interpolation=cv2.INTER_CUBIC
    )

    denoise = cv2.bilateralFilter(zoomed, 9, 75, 75)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(denoise, -1, kernel)

    return sharp


# =======================================
# DETECT INSTRUCTION + TARGET NUMBER
# =======================================
def extract_target_number(img):

    results = reader.readtext(
        img,
        detail=0,
        paragraph=True
    )

    text = " ".join(results).lower()

    match = re.search(r'number\s*(\d+)', text)

    if match:
        return match.group(1), text

    return None, text


# =======================================
# PREPROCESS TILE
# =======================================
def preprocess_tile(tile):

    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3,3), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return clean


# =======================================
# SPLIT GRID
# =======================================
def split_grid(img, rows, cols):

    h, w = img.shape[:2]
    th, tw = h // rows, w // cols

    tiles = []

    for r in range(rows):
        for c in range(cols):
            tiles.append(img[r*th:(r+1)*th, c*tw:(c+1)*tw])

    return tiles


# =======================================
# OCR TILE
# =======================================
def read_tile(tile):

    processed = preprocess_tile(tile)

    results = reader.readtext(
        processed,
        allowlist="0123456789",
        detail=1
    )

    if not results:
        return "", 0, processed

    text = "".join([r[1] for r in results])
    conf = round(np.mean([r[2] for r in results]), 2)

    text = re.sub(r'[^0-9]', '', text)

    return text, conf, processed


# =======================================
# UI
# =======================================
file = st.file_uploader("Upload CAPTCHA", type=["png", "jpg", "jpeg"])

if file:

    image = Image.open(file)
    img_np = np.array(image)

    enhanced = enhance_image(img_np)

    st.subheader("Enhanced Image (Zoomed)")
    st.image(enhanced, use_container_width=True)

    # -----------------------------------
    # STEP 1 → Extract instruction
    # -----------------------------------
    target, instruction_text = extract_target_number(enhanced)

    st.subheader("Detected Instruction Text")
    st.write(instruction_text)

    if target:
        st.success(f"🎯 Target Number Detected: {target}")
    else:
        st.warning("Target number not detected automatically")

    rows = st.slider("Rows", 2, 5, 3)
    cols = st.slider("Columns", 2, 5, 3)

    # -----------------------------------
    # STEP 2 → Detect tiles
    # -----------------------------------
    if st.button("🚀 Run Full Detection"):

        tiles = split_grid(enhanced, rows, cols)

        data = []
        previews = []

        for i, tile in enumerate(tiles):

            text, conf, processed = read_tile(tile)

            match_flag = (text == target)

            data.append({
                "tile_id": i + 1,
                "row": i // cols,
                "col": i % cols,
                "text": text,
                "confidence": conf,
                "match_target": match_flag
            })

            previews.append((processed, match_flag))

        df = pd.DataFrame(data)

        st.subheader("📊 Structured OCR Results")
        st.dataframe(df, use_container_width=True)

        st.subheader("🎯 Matching Tiles")
        st.dataframe(df[df["match_target"] == True], use_container_width=True)

        st.subheader("🧩 Tile Preview")
        grid_cols = st.columns(cols)

        for i, (p, match_flag) in enumerate(previews):
            caption = "MATCH ✅" if match_flag else ""
            grid_cols[i % cols].image(p, clamp=True, caption=caption)

else:
    st.info("Upload an image to start.")