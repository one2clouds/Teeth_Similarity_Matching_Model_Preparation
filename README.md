# Teeth Similarity Matching — Model Preparation
 
Data and model pipeline for learning tooth-shape embeddings from intraoral 3D scans, so that similar teeth end up close together in vector space. Built for crown-template retrieval: given a client's scanned tooth, find the most visually/geometrically similar teeth (and their crown designs) from a reference library.


## Pipeline overview
 
```
Raw IOS meshes (.obj, pervertex FDI labels in .json)
        │
        ▼
1. Extract individual teeth  ─────────────>  src/data_preprocess/extract_individual_teeth.py
   (splits a full jaw scan into one .obj per tooth, by FDI number)
        │
        ▼
2. Mesh → point cloud  ────────────────────>  src/data_preprocess/load_obj_save_pcd.py
   (sample 2048 points, center + scale-normalize, save as .pt tensor)
        │
        ▼
3. Train DGCNN classifier  ────────────────>  src/train.py + models/dgcnn.py
   (32-way FDI tooth-type classification, used as a pretext task)
        │
        ▼
4. Strip classification head → embedding model
        │
        ▼
5. Store feature embeddings  ──────────────>  src/store_feature_embeddings.py
   (256-d feature vector + thumbnail + label per tooth → feature_info.json)
        │
        ▼
6. Similarity search  ─────────────────────>  src/similarity_search.ipynb,
                                              src/similarity_search_multiple_teeth.ipynb
   (nearest-neighbor lookup in embedding space)
```

Thumbnails for each tooth mesh (used for visual inspection / UI previews) are rendered offline with `src/generate_thumbnail.py`.
 
A parallel `src_client_data/` pipeline mirrors steps 2 and 5 for new client scans (`.stl` files instead of `.obj`), so incoming client teeth can be embedded and matched against the reference library without retraining.
 
## Repository structure
 
```
models/
  dgcnn.py                         # DGCNN (Dynamic Graph CNN) point-cloud backbone
 
src/
  data_preprocess/
    extract_individual_teeth.py    # split a full-arch .obj mesh into per-tooth .obj files by FDI label
    load_obj_save_pcd.py           # mesh -> sampled, normalized point cloud tensor (.pt)
    point_cloud_dataset.py         # PyTorch Dataset over the .pt point-cloud tensors
  train.py                         # trains DGCNN for FDI tooth-type classification (pretext task)
  store_feature_embeddings.py      # runs the trained (head-removed) DGCNN and dumps feature_info.json
  generate_thumbnail.py            # renders a PNG thumbnail for each tooth .obj (via VTK)
  similarity_search.ipynb          # nearest-neighbor search over stored embeddings
  similarity_search_multiple_teeth.ipynb  # similarity search across several teeth at once
 
src_client_data/
  load_obj_save_pcd_client.py      # same as above, but for client .stl scans
  store_feature_embedding_client.py
 
feature_info_client_data.json      # stored embeddings/metadata for client teeth
feature_vectors_crown.json         # stored embeddings/metadata for crown templates
teeth_similarity_playbook.ipynb    # end-to-end walkthrough notebook
 
*convert_ly_to_obj.py                                   # scratch/experimental scripts (prefixed with `*`)
*tests_check_DGCNN_embeddings.ipynb                      # ad-hoc test/debug notebooks
*tests_client_data_extract_client_crown_template.ipynb
*tests_extract_client_crown_template.ipynb
*tests_get_feature_data_of_selected_labels_only.ipynb
*tests_json_chaniging.ipynb
```
 
> Files prefixed with `*` are exploratory/debugging notebooks and scripts, not part of the core pipeline.
 
## Model
 
**DGCNN** (`models/dgcnn.py`): a Dynamic Graph CNN operating directly on 3D point clouds:
- Builds a k-NN graph (k=20 by default) over the points at each layer and computes edge features (`get_graph_feature`), re-computing neighbors from the updated feature space at every layer (the "dynamic" part).
- 4 EdgeConv blocks (64 → 64 → 128 → 256 channels) followed by a shared Conv1d to a 1024-d embedding, max- and average-pooled and concatenated.
- 3 fully-connected layers (with dropout) reduce this to `output_channels` logits.
For training, `output_channels=32` (one class per FDI tooth number: 11–18, 21–28, 31–38, 41–48). For embedding extraction, the final dropout + linear layer are swapped for `nn.Identity()`, so the model outputs a 256-d feature vector instead of class logits — this is the embedding used for similarity search.
 
## Data preparation
 
1. **Extract individual teeth**: `extract_individual_teeth.py` reads a full-arch `.obj` mesh plus a per-vertex label `.json` (FDI numbering), and writes one `.obj` per tooth by keeping only the faces whose three vertices all share that tooth's label.
2. **Mesh → point cloud**: `load_obj_save_pcd.py` uses Open3D to uniformly sample 2048 points from each tooth mesh, centers the points on their centroid, and scales them to unit max-radius, then saves the result as a `.pt` tensor.
3. **Dataset**: `point_cloud_dataset.py` loads these tensors, transposes them to `(3, 2048)` for DGCNN's expected input layout, and infers the FDI label from the filename (`..._fid<NN>.pt`).
Expect ~24,000 individual teeth across ~900 patients (1,800 intraoral scans) once fully processed (see comments in `train.py`).
 
## Training
 
```bash
python src/train.py
```
 
- Uses [Weights & Biases](https://wandb.ai/) for experiment tracking (`wandb.init`), set up a W&B account/API key before running.
- 85/15 train/val split, `AdamW` optimizer, `OneCycleLR` schedule, cross entropy loss over the 32 FDI classes.
- Tracks accuracy/precision/recall/F1 (micro averaged) per epoch.
- Checkpoint saving / early stopping logic exists but is currently commented out in `train.py`, enable it before running long training jobs, or you'll lose the trained weights.
- Update `base_dir` in `train.py` to point at your local `pcd_tensors` directory before running.
## Generating embeddings
 
Once a classifier checkpoint exists:
 
```bash
python src/store_feature_embeddings.py
```
 
This loads the checkpoint, replaces the last dropout/linear layer with `Identity`, runs every point-cloud tensor through the network, and writes a `feature_info.json` containing, per tooth:
- `mesh_location` / `thumbnail_location`
- `label` (FDI tooth number, parsed from the filename)
- `feature_vector` (256-d embedding)
The client-data variant (`src_client_data/store_feature_embedding_client.py`) does the same for incoming client scans, writing into `feature_info_client_data.json`.
 
## Similarity search
 
`src/similarity_search.ipynb` and `src/similarity_search_multiple_teeth.ipynb` load the stored embeddings and perform nearest-neighbor lookups (e.g. by tooth label / FDI number) to find the closest matches for a query tooth — single-tooth and multi-tooth (full arch) respectively. `teeth_similarity_playbook.ipynb` walks through the full pipeline end-to-end.
 
## Requirements

You'll need:
```
torch
torchvision  
open3d
numpy
scikit-learn
tqdm
wandb
vtk              # for thumbnail rendering
```
 
