# Stage 2: GNN Edge Classifier

Train and run Graph Neural Network to classify edges as true/false connections.

## Scripts

- **`train_myGNN.py`** - Train InteractionGNN edge classifier with message passing
- **`infer_gnn.py`** - Run trained GNN on graphs to generate edge scores
- **`test_my_gnn.py`** - Evaluate GNN accuracy on test or validation set
- **`generate_grid_search.py`** - Generate grid search configs and HTCondor batch submission files

## Configuration

Config files in `../acorn_configs/gnn_stage_(2)/`:
- `gnn_train.yaml` - GNN training parameters
- `gnn_infer.yaml` - GNN inference parameters
- `gnn_grid.yml` - Hyperparameter grid search specification

## Usage

All scripts should be run from this stage directory:

```bash
cd /data/alice/bkuipers/low_pt_gnn_pipeline/gnn_stage_\(2\)

# Train GNN
python train_myGNN.py

# Run inference
python infer_gnn.py

# Test model
python test_my_gnn.py <model_name>

# Generate grid search configs for hyperparameter tuning
python generate_grid_search.py
```

## Model Architecture

- **WeightedInteractionGNN** - Custom subclass of InteractionGNN with weighted loss and per-batch W&B logging
- **Message passing** - Node and edge embeddings updated through multiple graph iterations
- **Edge classification** - Binary classification (true/false edge)

## Data Flow

- **Input:** `../data/graph_constructed_latent/` (from Stage 1)
- **Output:** `../data/gnn_stage/` (graphs with edge_scores)
- **Checkpoints:** `../saved_models/` (trained GNN models)

## GPU Training

For GPU cluster training via HTCondor:

```bash
cd /data/alice/bkuipers/low_pt_gnn_pipeline/jobs
condor_submit train_gnn.sub
```
