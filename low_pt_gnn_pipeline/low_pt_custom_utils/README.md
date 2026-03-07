# Low-pT Custom Utilities

This directory contains utility modules for the low-pT GNN tracking pipeline.

## Modules

### `track_build_utils.py`

Contains helper functions specifically for the custom loop track builder algorithm:

- **Layer detection**: `assign_layers()` - Groups hits into detector layers based on radial position
- **Edge lookup utilities**: `build_adjacency()`, `build_full_adjacency()`, `get_edge_score_from_adj()`
- **Connected Components**: `initial_cc_clustering()` - Standard CC clustering baseline
- **Problem detection**: `identify_problem_clusters()` - Identifies clusters with multiple first-layer hits (looping particles)
- **Segment building**: `build_outward_segments()` - Tree exploration to build outward-going track segments
- **Conflict resolution**: `resolve_conflicts()` - Resolves shared spacepoints between competing segments
- **Loop matching**: `match_loop_segments()` - Matches inward and outward segments using 4-priority system

These utilities are specific to the custom loop track building algorithm.

### `track_evaluation_utils.py`

Contains general-purpose evaluation and plotting utilities that can be used with **any** track building algorithm:

- **Graph data loading**:
  - `safe_load_reconstruction_df()` - Load reconstruction DataFrame from graph with track labels
  - `safe_load_particles_df()` - Build particles DataFrame from ground truth
  - `safe_evaluate_labelled_graph()` - Evaluate a single graph against truth

- **Evaluation**:
  - `run_evaluation()` - Evaluate all events in a dataset
  - `save_evaluation_results()` - Save evaluation results to disk (CSV, JSON, text summary)

- **Plotting**:
  - `plot_efficiency_vs_variable()` - Plot efficiency vs pT or eta
  - `plot_clone_rate_vs_variable()` - Plot clone rate vs pT or eta
  - `plot_purity_vs_variable()` - Plot purity vs pT or eta
  - `plot_completeness_vs_variable()` - Plot completeness vs pT or eta
  - `run_plotting()` - Create all metric plots

These utilities work with any track building algorithm (Connected Components, custom loop builder, etc.) as long as the graphs contain the required attributes (`hit_track_labels`, `hit_particle_id`, etc.).

## Usage

### Custom Loop Track Builder

```python
from low_pt_custom_utils.track_build_utils import (
    assign_layers,
    build_adjacency,
    initial_cc_clustering,
    identify_problem_clusters,
    build_outward_segments,
    resolve_conflicts,
    match_loop_segments,
)
```

### Evaluation and Plotting (Any Track Builder)

```python
from low_pt_custom_utils.track_evaluation_utils import (
    run_evaluation,
    save_evaluation_results,
    run_plotting,
)
```

## Design Principles

1. **Separation of concerns**: Track building logic (`track_build_utils.py`) is separate from evaluation/plotting logic (`track_evaluation_utils.py`)

2. **Reusability**: Evaluation utilities can be used with any track building algorithm, not just the custom loop builder

3. **Single source of truth**: Both `track_build_and_evaluate.py` and `custom_loop_track_builder.py` import from the same evaluation utils module, avoiding code duplication

## Integration with Existing Scripts

- `custom_loop_track_builder.py` imports from both utils modules
- `track_build_and_evaluate.py` imports from `track_evaluation_utils.py` for evaluation and plotting
- Both scripts share the same evaluation and plotting code, ensuring consistency
