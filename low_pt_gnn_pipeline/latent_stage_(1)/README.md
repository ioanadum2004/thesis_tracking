# Stage 1: Latent/Metric Learning

Graph construction using metric learning to build candidate edges via learned embeddings.

## Scripts

- **`acts_custom_low_pt_reader.py`** - Custom ACTS data reader with time-based trajectory ordering for looping particles
- **`convert_csv_to_pyg_sets.py`** - Convert ACTS CSV files to PyTorch Geometric graph format
- **`train_latent_cluster_learning.py`** - Train metric learning model (MLP embedding) using contrastive hinge loss
- **`build_latent_graphs.py`** - Build KNN graphs in learned latent space using trained model
- **`test_my_latent_model.py`** - Evaluate metric learning model performance on test set
- **`run_convert_chunk.py`** - Run CSV-to-PyG conversion on specific chunk (for parallel processing)
- **`combine_chunks.py`** - Combine chunked data and split into train/val/test sets

## Configuration

Config files in `../acorn_configs/latent_stage_(1)/`:
- `convert_csv_to_pyg_sets.yaml` - CSV conversion parameters
- `latent_cluster_learning_train.yaml` - Metric learning training config
- `graph_construction_latent.yaml` - Graph construction via latent space

## Usage

All scripts should be run from this stage directory:

```bash
cd /data/alice/bkuipers/low_pt_gnn_pipeline/latent_stage_\(1\)

# Convert CSV to PyG format
python convert_csv_to_pyg_sets.py

# Train metric learning model
python train_latent_cluster_learning.py

# Build graphs using trained model
python build_latent_graphs.py last    # or specify checkpoint name

# Test model
python test_my_latent_model.py
```

## Data Flow

- **Input:** `../data/csv/` (from Stage 0)
- **Intermediate:** `../data/feature_store/` (PyG graphs without edges)
- **Output:** `../data/graph_constructed_latent/` (PyG graphs with KNN edges)
- **Checkpoints:** `../saved_models/` (trained metric learning models)
