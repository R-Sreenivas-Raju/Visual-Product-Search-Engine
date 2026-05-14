# Visual Product Search Engine

A query-by-image product search system built using fine-tuned CLIP, BLIP-2 captions, and FAISS vector search on the **DeepFashion In-Shop Clothes Retrieval** dataset.

## Results

| Config | Recall@10 | NDCG@10 | mAP@10 |
|--------|-----------|---------|--------|
| A: Frozen CLIP (visual only) | 59.2% ± 7.9% | 43.6% ± 6.1% | 37.4% ± 5.4% |
| B: Frozen CLIP + Captions | 65.6% ± 7.6% | 44.9% ± 8.5% | 37.5% ± 8.1% |
| **C: Fine-tuned CLIP + Captions** | **96.8% ± 1.1%** | **81.9% ± 5.2%** | **74.6% ± 6.1%** |

*Results averaged over 3 random seeds. Evaluated on DeepFashion In-Shop dataset.*

## System Architecture

```
User Query Image
       |
  [YOLO Detection]  ← crops clothing item from background
       |
  [CLIP Visual Encoder]  ← fine-tuned on DeepFashion (last 4 blocks)
       |
  [FAISS HNSW Index]  ← searches 12,000+ gallery embeddings
       |
  Top-K Similar Products
```

**Offline Indexing Pipeline:**
1. **YOLO** — Localizes and crops the clothing item
2. **BLIP-2** — Generates semantic captions (color, style, material)
3. **CLIP** — Fuses visual + text embeddings: `v = α·visual + (1-α)·text`
4. **FAISS HNSW** — Stores and indexes the fused embeddings

**Online Query Pipeline:**
1. Upload image → YOLO crop → CLIP encode → FAISS search → Top-K results

## Repository Structure

```
├── vr-project-2.ipynb    # Main training & ablation study notebook (run on Kaggle)
├── app.py                # Streamlit interactive demo application
├── batch_eval.py         # End-to-end batch evaluation script
├── captions.json         # Pre-generated BLIP-2 captions for all 49k images
├── final_results.txt     # Final evaluation metrics
└── VR-Final-Project.pdf  # Assignment specification
```

## Setup & Running

### 1. Install Dependencies
```bash
pip install streamlit ultralytics faiss-cpu transformers torch torchvision pandas numpy pillow
```

### 2. Download Required Files (from Kaggle)
Download these from the Kaggle notebook output and place them in the project root:
- `clip_ft_s456/` — Fine-tuned CLIP model weights
- `s456_gvC.npy` — Pre-computed gallery embeddings
- `s456_gal.csv` — Gallery metadata

### 3. Run the Streamlit Demo
```bash
streamlit run app.py
```
- Upload a clothing image
- YOLO will detect and crop the main item
- Confirm the crop and search the catalog
- View top-K visually similar results

### 4. Run Batch Evaluation
```bash
python batch_eval.py
```
Outputs Recall@K, NDCG@K, mAP@K for K ∈ {5, 10, 15}.

## Fine-Tuning Details

- **Model:** `openai/clip-vit-base-patch32`
- **Strategy:** Fine-tune last 4 vision encoder blocks; text encoder frozen
- **Loss:** InfoNCE contrastive loss
- **Optimizer:** AdamW with cosine LR schedule
- **Seeds:** 3 random seeds for ablation study
- **Dataset:** DeepFashion In-Shop (~49k images, ~8k items)

## Key Findings

- Fine-tuning CLIP with contrastive loss boosted Recall@10 from **59% → 97%**
- Optimal fusion weight: **α = 0.7** (visual dominant, text adds context)
- BLIP-2 re-ranking was found to *reduce* performance due to non-discriminative captions
