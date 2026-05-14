import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
import faiss
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO
import os

# --- Configuration ---
# Update these paths once you download the Kaggle outputs!
MODEL_PATH = "clip_ft_s456" # Path to downloaded fine-tuned model folder
EMBEDDINGS_PATH = "s456_gvC.npy" # Path to gallery visual embeddings
GALLERY_CSV_PATH = "s456_gal.csv" # Path to gallery metadata
TEXT_EMBS_PATH = "s456_qt.npy" # Optional: if you want to fuse text in queries

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ALPHA = 0.7 # Best alpha from our experiments

st.set_page_config(page_title="Visual Product Search", layout="wide")

# --- Load Models & Data ---
@st.cache_resource
def load_models():
    # 1. Load YOLO (will download yolov8n.pt automatically on first run)
    yolo_model = YOLO('yolov8n.pt') 
    
    # 2. Load fine-tuned CLIP
    try:
        clip_model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except:
        st.warning(f"Could not load fine-tuned CLIP from {MODEL_PATH}. Using base CLIP for now.")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    
    return yolo_model, clip_model, clip_proc

@st.cache_resource
def load_gallery():
    try:
        gallery_df = pd.read_csv(GALLERY_CSV_PATH)
        gallery_embs = np.load(EMBEDDINGS_PATH).astype('float32')
        
        # Build FAISS Index
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
def crop_with_yolo(image):
    # Run YOLO detection
    results = yolo_model(image)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        return image, False # No detection, return original
        
    # Get the largest bounding box (likely the main person/clothing)
    largest_box = None
    max_area = 0
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_box = [int(x1), int(y1), int(x2), int(y2)]
            
    if largest_box:
        # Crop image
        cropped_img = image.crop((largest_box[0], largest_box[1], largest_box[2], largest_box[3]))
        return cropped_img, True
    return image, False

@torch.no_grad()
def get_image_embedding(image):
    inputs = clip_proc(images=image, return_tensors="pt").to(DEVICE)
    vision_outputs = clip_model.vision_model(pixel_values=inputs.pixel_values)
    embedding = clip_model.visual_projection(vision_outputs.pooler_output)
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return embedding.cpu().numpy()

# --- UI ---
st.title("🛍️ Visual Product Search Engine")
st.markdown("Upload a photo of clothing, and we'll find similar items in our catalog.")

# Sidebar for K selection
k_results = st.sidebar.slider("Number of results to show (K)", min_value=5, max_value=20, value=10, step=5)

uploaded_file = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Show original
    orig_image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Upload")
        st.image(orig_image, use_container_width=True)
        
    # 2. YOLO Crop
    if 'cropped_img' not in st.session_state:
        cropped, success = crop_with_yolo(orig_image)
        st.session_state.cropped_img = cropped
        st.session_state.crop_success = success
        
    with col2:
        st.subheader("YOLO Localization")
        st.image(st.session_state.cropped_img, use_container_width=True, caption="Cropped to main item")
        if not st.session_state.crop_success:
            st.warning("YOLO didn't detect a clear bounding box. Using original image.")
            
    # 3. User Confirmation (Assignment Requirement!)
    st.markdown("---")
    st.subheader("Confirm Crop")
    
    col3, col4 = st.columns([1, 4])
    with col3:
        confirm = st.button("✅ Confirm & Search", type="primary")
    with col4:
        if st.button("❌ Re-crop (Use Original)"):
            st.session_state.cropped_img = orig_image
            st.rerun()

    # 4. Search & Display Results
    if confirm:
        if index is None:
            st.error("Gallery data not loaded. Please check file paths.")
            st.stop()
            
        with st.spinner('Extracting visual features and searching catalog...'):
            # Encode query
            query_emb = get_image_embedding(st.session_state.cropped_img)
            
            # FAISS Search
            distances, indices = index.search(query_emb.astype('float32'), k_results)
            
            # Display
            st.markdown("---")
            st.subheader(f"Top {k_results} Matches")
            
            # Create columns for results grid (e.g., 5 per row)
            cols = st.columns(5)
            
            for i, idx in enumerate(indices[0]):
                match_row = gallery_df.iloc[idx]
                score = 1.0 / (1.0 + distances[0][i]) # Convert L2 dist to similarity score roughly
                
                # Fix Kaggle paths to point to the local directory
                img_path = match_row['full_path']
                img_path = img_path.replace("/kaggle/input/datasets/sasank93/cropped-images-vr-final-project/cropped_img/cropped_img/", "cropped_img/")
                img_path = img_path.replace("/kaggle/input/datasets/sasank93/cropped-images-vr-final-project/cropped_img/", "cropped_img/") 
                
                with cols[i % 5]:
                    try:
                        # Attempt to load local image
                        res_img = Image.open(img_path)
                        st.image(res_img, use_container_width=True)
                    except:
                        # Fallback if image not on local disk
                        st.info("Image not found on disk")
                        
                    st.caption(f"**Item ID:** {match_row['item_id']}")
                    st.caption(f"**Score:** {score:.3f}")
                    with st.expander("Caption"):
                        st.write(match_row['caption'])

