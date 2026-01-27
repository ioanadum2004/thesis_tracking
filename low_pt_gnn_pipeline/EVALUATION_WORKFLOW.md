# Track Building Evaluation Workflow

## Overview

This workflow evaluates track building performance and creates efficiency plots similar to ACTS. It consists of two scripts:

1. **`evaluate_tracks.py`** - Evaluates tracks and saves statistics
2. **`plot_track_metrics.py`** - Creates efficiency plots from evaluation results

## Workflow

### Step 1: Build Tracks

First, build tracks from graphs with edge scores:

```bash
python build_tracks.py
```

This creates graphs with `hit_track_labels` in `data/track_building/`.

### Step 2: Evaluate Tracks

Run evaluation to calculate metrics:

```bash
python evaluate_tracks.py --dataset testset
```

**What it does:**
- Loads graphs with track labels from `data/track_building/`
- Matches reconstructed tracks to truth particles
- Calculates efficiency, fake rate, clone rate, purity, completeness
- Saves results to `data/track_evaluation/`

**Output files:**
- `matching_df_testset.csv` - Detailed per-track, per-particle matching
- `particles_testset.csv` - Particle-level information with reconstruction status
- `summary_testset.json` - Summary statistics (efficiency, fake rate, etc.)
- `results_summary_ATLAS_testset.txt` - Text summary (ACORN format)

### Step 3: Create Plots

Generate efficiency plots:

```bash
python plot_track_metrics.py --dataset testset
```

**What it does:**
- Reads evaluation results from Step 2
- Creates efficiency plots vs pT and η
- Creates fake rate plots vs pT and η
- Creates clone rate plots vs pT and η

**Output files:**
- `efficiency_vs_pt_testset.png`
- `efficiency_vs_eta_testset.png`
- `fake_rate_vs_pt_testset.png`
- `fake_rate_vs_eta_testset.png`
- `clone_rate_vs_pt_testset.png`
- `clone_rate_vs_eta_testset.png`

## Configuration

Edit `acorn_configs/track_building_eval.yaml` to customize:

- **Matching parameters**: `matching_fraction`, `matching_style`
- **Particle selection**: `target_tracks` (fiducial cuts)
- **Plot settings**: `plots.tracking_efficiency.variables` (binning, ranges)

## Example Usage

```bash
# Evaluate testset
python evaluate_tracks.py --dataset testset

# Plot results
python plot_track_metrics.py --dataset testset

# Evaluate and plot valset
python evaluate_tracks.py --dataset valset
python plot_track_metrics.py --dataset valset --input-dir data/track_evaluation
```

## Metrics Calculated

### Efficiency
- **Definition**: Fraction of truth particles successfully reconstructed
- **Formula**: `N_reconstructed_particles / N_total_particles`
- **Plotted**: vs pT and η

### Fake Rate
- **Definition**: Fraction of tracks that don't match any truth particle
- **Formula**: `N_fake_tracks / N_total_tracks`
- **Plotted**: vs pT and η

### Clone Rate
- **Definition**: Fraction of particles with multiple matched tracks
- **Formula**: `N_cloned_particles / N_reconstructed_particles`
- **Plotted**: vs pT and η

### Purity (Average)
- **Definition**: Average fraction of hits in tracks belonging to matched particle
- **Formula**: `mean(n_shared_hits / track_length)`

### Completeness (Average)
- **Definition**: Average fraction of truth particle hits in matched tracks
- **Formula**: `mean(n_shared_hits / particle_nhits)`

## Comparison with ACTS

This workflow produces metrics compatible with ACTS evaluation:

- **Same matching logic**: Uses matching fraction threshold (default 0.5)
- **Same metrics**: Efficiency, fake rate, clone rate
- **Same plots**: Efficiency vs pT and η
- **Compatible format**: Can be compared directly with ACTS results

## Troubleshooting

### Error: "Input directory not found"
- Make sure you ran `build_tracks.py` first
- Check that `data/track_building/<dataset>/` contains `.pyg` files

### Error: "Matching DataFrame not found"
- Make sure you ran `evaluate_tracks.py` first
- Check that `data/track_evaluation/` exists and contains CSV files

### Plots look wrong
- Check that `particles_testset.csv` has `pt` and `eta` columns
- Verify `x_scale` in config matches your data units (MeV vs GeV)
- Adjust `x_lim` in config to match your data range
