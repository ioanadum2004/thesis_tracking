# Stage 3: Track Building and Evaluation

Build candidate tracks from classified edges and evaluate reconstruction performance.

## Scripts

- **`track_build_and_evaluate.py`** - All-in-one: build tracks + evaluate + plot metrics
- **`build_tracks.py`** - Build tracks using Connected Components clustering on edge scores
- **`custom_loop_track_builder.py`** - Custom track building algorithm for looping particles

## Configuration

Config files in `../acorn_configs/track_building_stage_(3)/`:
- `track_build_and_evaluate.yaml` - Combined config for track building, evaluation, and plotting

## Usage

All scripts should be run from this stage directory:

```bash
cd /data/alice/bkuipers/low_pt_gnn_pipeline/track_building_stage_\(3\)

# Full pipeline: build + evaluate + plot
python track_build_and_evaluate.py testset

# Re-evaluate without rebuilding tracks
python track_build_and_evaluate.py testset --skip-build

# Use custom loop track builder
python custom_loop_track_builder.py testset
```

## Track Building Algorithm

1. **Connected Components** (default) - Cluster hits connected by high-scoring edges (score > threshold)
2. **Custom Loop Builder** - Specialized algorithm for looping particles that pass through detector layers multiple times

## Evaluation Metrics

- **Efficiency** - Fraction of true particles successfully reconstructed
- **Fake rate** - Fraction of reconstructed tracks not matching any true particle
- **Clone rate** - Fraction of true particles reconstructed multiple times
- **Purity/Completeness** - Hit-level matching quality

## Data Flow

- **Input:** `../data/gnn_stage/` (from Stage 2, with edge scores)
- **Output:**
  - `../data/track_building/` - Graphs with track_edges and hit_track_labels
  - `../data/track_evaluation/<dataset>/` - Evaluation results and metrics
  - `../data/visuals/track_metrics/<dataset>/` - Efficiency and purity plots

## Dependencies

- **low_pt_custom_utils/** - Shared evaluation and plotting utilities
  - `track_evaluation_utils.py` - Evaluation and plotting functions
  - `track_build_utils.py` - Track building helper functions
