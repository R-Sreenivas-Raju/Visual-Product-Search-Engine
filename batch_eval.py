import torch
import numpy as np
import pandas as pd
from PIL import Image
import faiss
from transformers import CLIPModel, CLIPProcessor
import random

# --- Configuration ---
MODEL_PATH = "clip_ft_s456"
GALLERY_CSV = "s456_gal.csv"
GALLERY_EMBS = "s456_gvC.npy"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K_LIST = [5, 10, 15]
NUM_TEST_QUERIES = 500 # Keep it to 500 so the demo runs fast in front of the prof!

def calc_metrics(q_ids, g_ids, I, ks):
    ret = g_ids[I]
    rel = (ret == q_ids[:, None]).astype(float)
    res = {}
    for k in ks:
        r = rel[:, :k]
        rk = np.arange(1, k+1)
        res[f'Recall@{k}'] = float(r.any(1).mean())
        
        dcg = (r / np.log2(rk+1)).sum(1)
        n_rel = rel.sum(1)
        ideal = np.zeros_like(dcg)
        for i in range(len(dcg)):
            nr = int(min(n_rel[i], k))
            if nr > 0:
                ideal[i] = sum(1/np.log2(j+2) for j in range(nr))
        ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal>0)
        res[f'NDCG@{k}'] = float(ndcg.mean())
        
        ch = r.cumsum(1)
        ap = (r * ch / rk).sum(1) / np.maximum(n_rel, 1).clip(max=k)
        res[f'mAP@{k}'] = float(ap.mean())
    return res

def main():
    print(f"Loading Fine-Tuned CLIP from {MODEL_PATH}...")
    try:
        model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading gallery data...")
    df = pd.read_csv(GALLERY_CSV)
    g_embs_all = np.load(GALLERY_EMBS).astype('float32')
    
    print("Auto-generating robust query/gallery split from local data...")
    random.seed(42)
    np.random.seed(42)
    
    # Find items with at least 2 images
    item_counts = df['item_id'].value_counts()
    valid_items = item_counts[item_counts >= 2].index.tolist()
    
    # Pick a random subset of items to act as queries (for fast demo execution)
    test_items = random.sample(valid_items, min(NUM_TEST_QUERIES, len(valid_items)))
    
    query_idx = []
    gallery_idx = []
    
    for item in test_items:
        idx_list = df[df['item_id'] == item].index.tolist()
        q_i = random.choice(idx_list)
        query_idx.append(q_i)
        
        # The rest of the images for this item go to the gallery
        g_i = [i for i in idx_list if i != q_i]
        gallery_idx.extend(g_i)
        
    # Add all other unused items to the gallery as distractors
    sampled_set = set(query_idx + gallery_idx)
    all_set = set(range(len(df)))
    gallery_idx.extend(list(all_set - sampled_set))
    
    query_df = df.iloc[query_idx].reset_index(drop=True)
    gallery_df = df.iloc[gallery_idx].reset_index(drop=True)
    true_gallery_embs = g_embs_all[gallery_idx]

    print("Building FAISS HNSW Index...")
    index = faiss.IndexHNSWFlat(true_gallery_embs.shape[1], 32)
    index.hnsw.efConstruction = 200
    index.add(true_gallery_embs)
    
    print(f"Running end-to-end inference on {len(query_df)} queries...")
    q_embs = []
    
    batch_size = 32
    for i in range(0, len(query_df), batch_size):
        batch = query_df.iloc[i:i+batch_size]
        imgs = []
        for p in batch['full_path']:
            # Fix Kaggle paths to point to local directory
            p = p.replace("/kaggle/input/datasets/sasank93/cropped-images-vr-final-project/cropped_img/cropped_img/", "cropped_img/")
            p = p.replace("/kaggle/input/datasets/sasank93/cropped-images-vr-final-project/cropped_img/", "cropped_img/")
            try: 
                imgs.append(Image.open(p).convert('RGB'))
            except: 
                imgs.append(Image.new('RGB', (224,224)))
        
        inputs = processor(images=imgs, return_tensors='pt', padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.vision_model(pixel_values=inputs.pixel_values)
            e = torch.nn.functional.normalize(model.visual_projection(outputs.pooler_output), dim=-1)
            q_embs.append(e.cpu().numpy())
            
        if (i // batch_size) % 5 == 0:
            print(f"  Processed {i}/{len(query_df)} images...")
            
    q_embs = np.vstack(q_embs)
    
    print("Searching index...")
    index.hnsw.efSearch = max(100, max(K_LIST)*4)
    _, I = index.search(q_embs.astype('float32'), k=max(K_LIST))
    
    q_arr = query_df['item_id'].values
    g_arr = gallery_df['item_id'].values
    metrics = calc_metrics(q_arr, g_arr, I, K_LIST)
    
    print("\n" + "="*50)
    print("BATCH EVALUATION RESULTS (End-to-End Pipeline)")
    print("="*50)
    print(f"Recall@5:  {metrics['Recall@5']:.4f}  |  NDCG@5:  {metrics['NDCG@5']:.4f}  |  mAP@5:  {metrics['mAP@5']:.4f}")
    print(f"Recall@10: {metrics['Recall@10']:.4f}  |  NDCG@10: {metrics['NDCG@10']:.4f}  |  mAP@10: {metrics['mAP@10']:.4f}")
    print(f"Recall@15: {metrics['Recall@15']:.4f}  |  NDCG@15: {metrics['NDCG@15']:.4f}  |  mAP@15: {metrics['mAP@15']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
