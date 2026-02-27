import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Smart CAPTCHA Solver",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Smart Grid CAPTCHA Solver")
st.caption("Instruction → Target number → Tile detection → Structured output")


# ======================================================
# LOAD OCR (CACHED + AUTO DOWNLOAD)
# ======================================================
@st.cache_resource
def load_reader():
    return easyocr.Reader(
        ['en'],
        gpu=False,                       # Streamlit Cloud CPU only
        download_enabled=True,           # auto download first run
        model_storage_directory=".easyocr"
    )

reader = load_reader()


# ======================================================
# IMAGE ENHANCEMENT
# ======================================================
def enhance(img, scale=2):
    h, w = img.shape[:2]

    zoom = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)

    sharp_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(gray, -1, sharp_kernel)

    return sharp


# ======================================================
# EXTRACT INSTRUCTION NUMBER
# ======================================================
def get_target_number(img):
    text_list = reader.readtext(img, detail=0)
    full_text = " ".join(text_list).lower()

    match = re.search(r'number\s*(\d+)', full_text)

    if match:
        return match.group(1), full_text

    return None, full_text


# ======================================================
# TILE PREPROCESS
# ======================================================
def preprocess_tile(tile):
    blur = cv2.GaussianBlur(tile, (3,3), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    return th


# ======================================================
# SPLIT GRID
# ======================================================
def split_grid(img, rows, cols):
    h, w = img.shape
    th, tw = h // rows, w // cols

    tiles = []
    for r in range(rows):
        for c in range(cols):
            tile = img[r*th:(r+1)*th, c*tw:(c+1)*tw]
            tiles.append(tile)

    return tiles


# ======================================================
# READ TILE OCR
# ======================================================
def read_tile(tile):
    processed = preprocess_tile(tile)

    result = reader.readtext(
        processed,
        allowlist="0123456789",
        detail=1
    )

    if not result:
        return "", 0, processed

    text = "".join([r[1] for r in result])
    conf = round(np.mean([r[2] for r in result]), 2)

    text = re.sub(r'[^0-9]', '', text)

    return text, conf, processed


# ======================================================
# UI
# ======================================================
uploaded = st.file_uploader("Upload CAPTCHA image", type=["png","jpg","jpeg"])


if uploaded:

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    enhanced = enhance(img_np)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(img_np, use_container_width=True)

    with col2:
        st.subheader("Enhanced (Zoom + Sharpen)")
        st.image(enhanced, use_container_width=True)


    # -------------------------
    # STEP 1: Instruction OCR
    # -------------------------
    target, instruction = get_target_number(enhanced)

    st.subheader("Detected Instruction Text")
    st.write(instruction)

    if target:
        st.success(f"🎯 Target Number: {target}")
    else:
        st.warning("Target number not detected automatically")


    rows = st.slider("Rows", 2, 5, 3)
    cols = st.slider("Columns", 2, 5, 3)


    # -------------------------
    # STEP 2: Tiles OCR
    # -------------------------
    if st.button("🚀 Run Detection"):

        tiles = split_grid(enhanced, rows, cols)

        results_data = []
        previews = []

        for i, tile in enumerate(tiles):

            text, conf, processed = read_tile(tile)

            match_flag = (text == target)

            results_data.append({
                "tile_id": i+1,
                "row": i // cols,
                "col": i % cols,
                "text": text,
                "confidence": conf,
                "match_target": match_flag
            })

            previews.append((processed, match_flag))

        df = pd.DataFrame(results_data)

        st.subheader("📊 Structured Results")
        st.dataframe(df, use_container_width=True)

        st.subheader("🎯 Matching Tiles")
        st.write(df[df["match_target"] == True])

        # grid preview
        st.subheader("🧩 Tile Preview")
        grid_cols = st.columns(cols)

        for i, (img_tile, flag) in enumerate(previews):
            caption = "MATCH ✅" if flag else ""
            grid_cols[i % cols].image(img_tile, clamp=True, caption=caption)

else:
    st.info("Upload an image to start")