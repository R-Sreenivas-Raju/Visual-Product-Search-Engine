import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import faiss
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO
import os

# --- Configuration ---
MODEL_PATH = "clip_ft_s456"
EMBEDDINGS_PATH = "s456_gvC.npy"
GALLERY_CSV_PATH = "s456_gal.csv"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Color palette for bounding boxes (one per detection)
BOX_COLORS = [
    (255, 80,  80),   # red
    (80,  180, 255),  # blue
    (80,  220, 80),   # green
    (255, 180, 0),    # orange
    (200, 80,  255),  # purple
    (0,   220, 200),  # teal
]

st.set_page_config(page_title="Visual Product Search", layout="wide")

# --- Load Models & Data ---
@st.cache_resource
def load_models():
    yolo_model = YOLO('best_yolo.pt')
    try:
        clip_model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
        clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except:
        st.warning(f"Could not load fine-tuned CLIP from '{MODEL_PATH}'. Falling back to base CLIP.")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return yolo_model, clip_model, clip_proc

@st.cache_resource
def load_gallery():
    try:
        gallery_df  = pd.read_csv(GALLERY_CSV_PATH)
        gallery_embs = np.load(EMBEDDINGS_PATH).astype('float32')
        index = faiss.IndexHNSWFlat(gallery_embs.shape[1], 32)
        index.hnsw.efConstruction = 200
        index.add(gallery_embs)
        return gallery_df, index
    except Exception as e:
        st.error(f"Error loading gallery data: {e}")
        return None, None

yolo_model, clip_model, clip_proc = load_models()
gallery_df, index = load_gallery()

# --- Helper Functions ---

def detect_clothing_items(image: Image.Image):
    """
    Run YOLO on the image and return a list of detected items.
    Each item: {'label': str, 'confidence': float, 'box': [x1,y1,x2,y2], 'crop': PIL.Image}
    Items are sorted by area (largest first).
    """
    # ==============================================================
    # Calling YOLO for Cropping/Localization
    # Model: yolov8n.pt (Object Detection)
    # Purpose: Detect the person/clothing and crop the bounding box
    # ==============================================================
    results = yolo_model(image)
    boxes   = results[0].boxes
    names   = results[0].names  # {id: class_name}

    detections = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf   = float(box.conf[0].item())
        label  = names.get(cls_id, f'class_{cls_id}')
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        area = (x2 - x1) * (y2 - y1)
        crop = image.crop((x1, y1, x2, y2))
        detections.append({
            'label':      label,
            'confidence': conf,
            'box':        [x1, y1, x2, y2],
            'area':       area,
            'crop':       crop,
        })

    # Sort by area descending (biggest item first)
    detections.sort(key=lambda d: d['area'], reverse=True)
    return detections


def draw_detections(image: Image.Image, detections, highlight_idx=None):
    """
    Draw all bounding boxes on the image.
    highlight_idx: if set, that box is drawn thicker/brighter.
    Returns annotated PIL image.
    """
    annotated = image.copy().convert("RGBA")
    overlay   = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw      = ImageDraw.Draw(overlay)

    for i, det in enumerate(detections):
        color    = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = det['box']
        thickness = 5 if (highlight_idx is not None and i == highlight_idx) else 3
        alpha     = 220 if (highlight_idx is not None and i == highlight_idx) else 160

        # Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 40))
        # Border
        for t in range(thickness):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=(*color, alpha))

        # Label badge
        label_text = f"[{i+1}] {det['label']}  {det['confidence']*100:.0f}%"
        badge_x, badge_y = x1, max(0, y1 - 24)
        draw.rectangle([badge_x, badge_y, badge_x + len(label_text)*7 + 8, badge_y + 22],
                       fill=(*color, 210))
        draw.text((badge_x + 4, badge_y + 3), label_text, fill=(255, 255, 255, 255))

    result = Image.alpha_composite(annotated, overlay)
    return result.convert("RGB")


@torch.no_grad()
def get_image_embedding(image: Image.Image):
    # ==============================================================
    # Calling CLIP for Embedding Extraction
    # Model: Fine-tuned OpenAI CLIP (openai/clip-vit-base-patch32)
    # Purpose: Extract 512-dimensional visual feature vector
    # ==============================================================
    inputs = clip_proc(images=image, return_tensors="pt").to(DEVICE)
    vision_outputs = clip_model.vision_model(pixel_values=inputs.pixel_values)
    embedding = clip_model.visual_projection(vision_outputs.pooler_output)
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return embedding.cpu().numpy()


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🛍️ Visual Product Search Engine")
st.markdown("Upload a photo — we'll detect all clothing items and let you choose which one to search.")

k_results = st.sidebar.slider("Number of results to show (K)", min_value=5, max_value=20, value=10, step=5)

uploaded_file = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    orig_image = Image.open(uploaded_file).convert('RGB')

    # ── Step 1: Detect all clothing items ──────────────────────
    if 'detections' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
        with st.spinner("Running YOLO detection..."):
            detections = detect_clothing_items(orig_image)
        st.session_state.detections  = detections
        st.session_state.last_file   = uploaded_file.name
        st.session_state.selected_idx = None   # reset selection
        st.session_state.search_done  = False

    detections = st.session_state.detections

    # ── Step 2: Show annotated image + detection list ───────────
    st.markdown("---")

    if len(detections) == 0:
        st.warning("YOLO couldn't detect any items. Falling back to full image search.")
        detections = [{
            'label': 'full image', 'confidence': 1.0,
            'box': [0, 0, orig_image.width, orig_image.height],
            'area': orig_image.width * orig_image.height,
            'crop': orig_image,
        }]
        st.session_state.detections = detections

    col_img, col_sel = st.columns([3, 2])

    with col_img:
        st.subheader("Detected Items")
        highlight = st.session_state.get('selected_idx', None)
        annotated_img = draw_detections(orig_image, detections, highlight_idx=highlight)
        st.image(annotated_img, use_container_width=True)

    with col_sel:
        st.subheader("Choose an item to search")
        st.markdown("Select which clothing item you want to find similar products for:")
        st.markdown("")

        for i, det in enumerate(detections):
            color_hex = "#{:02x}{:02x}{:02x}".format(*BOX_COLORS[i % len(BOX_COLORS)])
            label     = det['label'].title()
            conf_pct  = det['confidence'] * 100

            # Coverage % relative to original image area
            img_area  = orig_image.width * orig_image.height
            coverage  = det['area'] / img_area * 100

            # Show thumbnail + button side by side
            tcol, bcol = st.columns([1, 3])
            with tcol:
                st.image(det['crop'], use_container_width=True)
            with bcol:
                st.markdown(
                    f"<span style='color:{color_hex}; font-weight:700; font-size:15px;'>▐ [{i+1}] {label}</span><br>"
                    f"<span style='font-size:12px; color:gray;'>Confidence: {conf_pct:.0f}% &nbsp;|&nbsp; Coverage: {coverage:.0f}%</span>",
                    unsafe_allow_html=True
                )
                if st.button(f"🔍 Search this item", key=f"select_{i}"):
                    st.session_state.selected_idx = i
                    st.session_state.search_done  = False
                    st.rerun()

            st.markdown("")

        # ── Manual Crop Option ──────────────────────────────────
        st.markdown("---")
        with st.expander("✂️ Or Manually Crop an Item"):
            st.markdown("Use the sliders to define your crop region, then preview and search.")
            W, H = orig_image.width, orig_image.height

            col_l, col_r = st.columns(2)
            with col_l:
                x1 = st.slider("Left (X1)",   0, W-1, 0,        key="cx1")
                x2 = st.slider("Right (X2)",  1, W,   W,        key="cx2")
            with col_r:
                y1 = st.slider("Top (Y1)",    0, H-1, 0,        key="cy1")
                y2 = st.slider("Bottom (Y2)", 1, H,   H,        key="cy2")

            # Live preview: draw crop box on image
            preview = orig_image.copy()
            draw = ImageDraw.Draw(preview)
            if x2 > x1 and y2 > y1:
                draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=4)
            st.image(preview, caption="Green box = your crop", use_container_width=True)

            if st.button("🔍 Search Manual Crop", type="primary", key="manual_search_btn"):
                if x2 <= x1 or y2 <= y1:
                    st.error("Invalid crop! Make sure X2 > X1 and Y2 > Y1.")
                else:
                    manual_crop = orig_image.crop((x1, y1, x2, y2))
                    st.session_state.detections.append({
                        'label': 'Manual Crop',
                        'confidence': 1.0,
                        'box': [x1, y1, x2, y2],
                        'area': (x2 - x1) * (y2 - y1),
                        'crop': manual_crop
                    })
                    st.session_state.selected_idx = len(st.session_state.detections) - 1
                    st.session_state.search_done  = False
                    st.rerun()

    # ── Step 3: Show selected crop confirmation ─────────────────
    selected_idx = st.session_state.get('selected_idx', None)

    if selected_idx is not None:
        st.markdown("---")
        selected_det = detections[selected_idx]
        color_hex    = "#{:02x}{:02x}{:02x}".format(*BOX_COLORS[selected_idx % len(BOX_COLORS)])

        st.markdown(
            f"### Searching for: "
            f"<span style='color:{color_hex};'>{selected_det['label'].title()}</span>",
            unsafe_allow_html=True
        )

        conf_col, crop_col = st.columns([1, 3])
        with conf_col:
            st.image(selected_det['crop'], caption="Selected crop", use_container_width=True)
        with crop_col:
            st.info(
                f"**Item:** {selected_det['label'].title()}  \n"
                f"**Confidence:** {selected_det['confidence']*100:.1f}%  \n"
                f"**Bounding box:** {selected_det['box']}"
            )
            c1, c2 = st.columns(2)
            with c1:
                confirm = st.button("✅ Confirm & Search", type="primary")
            with c2:
                if st.button("↩️ Choose a different item"):
                    st.session_state.selected_idx = None
                    st.session_state.search_done  = False
                    st.rerun()

        # ── Step 4: Search ──────────────────────────────────────
        if confirm or st.session_state.get('search_done', False):
            if index is None:
                st.error("Gallery not loaded. Check file paths.")
                st.stop()

            if not st.session_state.get('search_done', False):
                with st.spinner("Extracting features and searching catalog..."):
                    query_emb = get_image_embedding(selected_det['crop'])
                    # ==============================================================
                    # Calling FAISS for Vector Retrieval
                    # Model: FAISS HNSW (Hierarchical Navigable Small World) Index
                    # Purpose: Fast approximate nearest neighbor search in gallery
                    # ==============================================================
                    distances, indices = index.search(query_emb.astype('float32'), k_results)
                st.session_state.search_results   = (distances, indices)
                st.session_state.search_done      = True

            distances, indices = st.session_state.search_results

            st.markdown("---")
            st.subheader(f"Top {k_results} Matches for '{selected_det['label'].title()}'")

            cols = st.columns(5)
            for i, idx in enumerate(indices[0]):
                match_row = gallery_df.iloc[idx]
                score     = 1.0 / (1.0 + distances[0][i])

                img_path = match_row['full_path']
                img_path = img_path.replace(
                    "/kaggle/input/datasets/sasank93/cropped-images-vr-final-project/cropped_img/cropped_img/",
                    "cropped_img/")
                img_path = img_path.replace(
                    "/kaggle/input/datasets/sasank93/cropped-images-vr-final-project/cropped_img/",
                    "cropped_img/")

                with cols[i % 5]:
                    try:
                        res_img = Image.open(img_path)
                        st.image(res_img, use_container_width=True)
                    except:
                        st.info("Image not on disk")
                    st.caption(f"**Item ID:** {match_row['item_id']}")
                    st.caption(f"**Score:** {score:.3f}")
                    with st.expander("Caption"):
                        st.write(match_row['caption'])
