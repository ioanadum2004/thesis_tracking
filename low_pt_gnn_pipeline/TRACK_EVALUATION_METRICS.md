# Track Evaluation Metrics: ACTS vs Our Pipeline

## Overview

This document explains how ACTS evaluates track reconstruction performance (from `perfect_spacepoints.py`) and how we can implement similar particle physics metrics for our GNN-based track building pipeline.

## ACTS Evaluation Approach

### What ACTS Does

1. **Simulation**: Generates particles → propagates through detector → creates hits
2. **Reconstruction**: Runs Kalman Filter tracking → produces reconstructed tracks
3. **Matching**: Uses `TrackTruthMatcher` to match reconstructed tracks to truth particles
4. **Metrics**: Calculates efficiency, fake rate, clone rate, etc.

### Key Components in ACTS

#### 1. TrackTruthMatcher (lines 505-516 in perfect_spacepoints.py)
```python
matchAlg = acts.examples.TrackTruthMatcher(
    inputTracks=trackFinder.config.outputTracks,
    inputParticles="particles_selected",
    inputMeasurementParticlesMap="measurement_particles_map",
    outputTrackParticleMatching="ckf_track_particle_matching",
    outputParticleTrackMatching="ckf_particle_track_matching",
    doubleMatching=True,
    matchingRatio=matching_fraction_threshold,  # Default: 0.5
)
```

**What it does:**
- Matches each reconstructed track to truth particles based on shared hits
- Uses `matchingRatio` threshold (default 0.5 = 50% of hits must match)
- Creates bidirectional mappings:
  - Track → Particle (which particle does this track match?)
  - Particle → Tracks (which tracks match this particle?)

#### 2. TrackFinderPerformanceWriter (lines 541-552)
Writes ROOT files with performance metrics including:
- **Efficiency plots**: vs pT, vs η (eta)
- **Duplicate rate plots**: vs pT, vs η
- **Matching details**: per-track, per-particle matching information

### Key Metrics ACTS Calculates

#### 1. **Track Efficiency** (Particle-level)
```
Efficiency = N_reconstructed_particles / N_total_particles
```
- **Definition**: Fraction of truth particles that are successfully reconstructed
- **Calculation**: For each truth particle, check if it has at least one matched track
- **Typical values**: 0.7-0.95 (70-95%) depending on pT range

#### 2. **Hit Efficiency** (Hit-level)
```
Hit Efficiency = N_hits_in_matched_tracks / N_total_truth_hits
```
- **Definition**: Fraction of truth hits that are included in reconstructed tracks
- **Calculation**: Count hits from truth particles that appear in matched tracks
- **Also called**: "Completeness" in some contexts

#### 3. **Fake Rate** (Track-level)
```
Fake Rate = N_fake_tracks / N_total_tracks
```
- **Definition**: Fraction of reconstructed tracks that don't match any truth particle
- **Calculation**: Tracks with no particle match (or match < threshold)
- **Typical values**: 0.01-0.1 (1-10%)

#### 4. **Clone Rate / Duplication Rate** (Particle-level)
```
Clone Rate = N_particles_with_multiple_tracks / N_reconstructed_particles
```
- **Definition**: Fraction of particles that have multiple matched tracks
- **Calculation**: Count particles matched to 2+ tracks
- **Typical values**: 0.05-0.2 (5-20%)

#### 5. **Purity** (Track-level)
```
Purity = N_matched_hits / N_total_hits_in_track
```
- **Definition**: Fraction of hits in a track that belong to the matched particle
- **Calculation**: For each matched track, count hits from matched particle
- **Typical values**: 0.8-1.0 (80-100%)

#### 6. **Completeness** (Track-level)
```
Completeness = N_matched_hits / N_total_hits_in_particle
```
- **Definition**: Fraction of truth particle hits that are in the matched track
- **Calculation**: For each matched track-particle pair, count shared hits
- **Typical values**: 0.7-1.0 (70-100%)

### Matching Logic

ACTS uses a **matching fraction threshold** (default 0.5):
- A track matches a particle if: `shared_hits / track_hits >= 0.5`
- A particle matches a track if: `shared_hits / particle_hits >= 0.5`
- For "double matching": Both conditions must be true

## Our Pipeline: What We Have

### Input Data Structure

After running `build_tracks.py`, we have:

1. **Track files** (`event000000000.txt`):
   - Each line = one track
   - Numbers = hit IDs (node indices)
   - Example: `14 17 44 47 76 74 ...` = Track with hits [14, 17, 44, ...]

2. **Graph files** (`event000000003.pyg`):
   - Contains truth information:
     - `hit_particle_id`: Which particle each hit belongs to
     - `track_edges`: Truth edges (which hits are connected in truth tracks)
     - `track_particle_id`: Particle IDs for truth tracks
     - `track_particle_pt`: Transverse momentum of particles
     - `track_particle_eta`: Pseudorapidity of particles
     - `track_particle_nhits`: Number of hits per particle
   - Contains reconstruction:
     - `hit_track_labels`: Which track each hit belongs to (-1 = no track)

### Available Truth Information

From the graph files, we can extract:
- **Truth particles**: `track_particle_id`, `track_particle_pt`, `track_particle_eta`, `track_particle_nhits`
- **Truth hits**: `hit_particle_id` (which particle each hit belongs to)
- **Truth edges**: `track_edges` (connections between hits in truth tracks)

## Implementation Plan

### Step 1: Load Data

```python
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def load_track_file(track_path):
    """Load tracks from TXT file."""
    tracks = []
    with open(track_path, 'r') as f:
        for line in f:
            hit_ids = [int(x) for x in line.strip().split()]
            if len(hit_ids) > 0:
                tracks.append(hit_ids)
    return tracks

def load_graph(graph_path):
    """Load graph with truth information."""
    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    return graph
```

### Step 2: Build Truth-Prediction Matching

```python
def match_tracks_to_particles(tracks, graph, matching_fraction=0.5):
    """
    Match reconstructed tracks to truth particles.
    
    Args:
        tracks: List of lists, each inner list is hit IDs for a track
        graph: PyG graph with truth information
        matching_fraction: Threshold for matching (default 0.5)
    
    Returns:
        matching_df: DataFrame with track-particle matches
    """
    # Extract truth information
    hit_particle_id = graph.hit_particle_id.numpy()
    
    # Build matching matrix
    matches = []
    for track_id, track_hits in enumerate(tracks):
        # Get particle IDs for hits in this track
        track_particles = hit_particle_id[track_hits]
        
        # Count hits per particle
        unique_particles, counts = np.unique(track_particles, return_counts=True)
        
        # Calculate purity for each particle
        track_length = len(track_hits)
        for particle_id, n_shared in zip(unique_particles, counts):
            if particle_id == 0:  # Skip noise hits
                continue
            
            purity = n_shared / track_length
            if purity >= matching_fraction:
                # Get truth particle info
                particle_mask = graph.track_particle_id == particle_id
                if particle_mask.any():
                    particle_pt = graph.track_particle_pt[particle_mask][0].item()
                    particle_eta = graph.track_particle_eta[particle_mask][0].item()
                    particle_nhits = graph.track_particle_nhits[particle_mask][0].item()
                    
                    completeness = n_shared / particle_nhits
                    
                    matches.append({
                        'track_id': track_id,
                        'particle_id': particle_id,
                        'n_shared': n_shared,
                        'track_length': track_length,
                        'particle_nhits': particle_nhits,
                        'purity': purity,
                        'completeness': completeness,
                        'pt': particle_pt,
                        'eta': particle_eta,
                    })
    
    return pd.DataFrame(matches)
```

### Step 3: Calculate Metrics

```python
def calculate_efficiency_metrics(matching_df, graph, tracks):
    """
    Calculate efficiency, fake rate, clone rate, etc.
    
    Returns:
        dict with all metrics
    """
    # Get unique particles and tracks
    unique_particles = graph.track_particle_id.unique().numpy()
    unique_particles = unique_particles[unique_particles > 0]  # Remove noise
    
    n_total_particles = len(unique_particles)
    n_total_tracks = len(tracks)
    
    # Track efficiency: particles with at least one match
    matched_particles = matching_df['particle_id'].unique()
    n_reconstructed_particles = len(matched_particles)
    efficiency = n_reconstructed_particles / n_total_particles if n_total_particles > 0 else 0
    
    # Fake rate: tracks with no match
    matched_tracks = matching_df['track_id'].unique()
    n_fake_tracks = n_total_tracks - len(matched_tracks)
    fake_rate = n_fake_tracks / n_total_tracks if n_total_tracks > 0 else 0
    
    # Clone rate: particles with multiple tracks
    particle_track_counts = matching_df.groupby('particle_id')['track_id'].nunique()
    n_cloned_particles = (particle_track_counts > 1).sum()
    clone_rate = n_cloned_particles / n_reconstructed_particles if n_reconstructed_particles > 0 else 0
    
    # Average purity and completeness
    avg_purity = matching_df['purity'].mean() if len(matching_df) > 0 else 0
    avg_completeness = matching_df['completeness'].mean() if len(matching_df) > 0 else 0
    
    # Hit efficiency: fraction of truth hits in matched tracks
    hit_particle_id = graph.hit_particle_id.numpy()
    total_truth_hits = (hit_particle_id > 0).sum()
    
    matched_hits = 0
    for track_id in matched_tracks:
        track_hits = tracks[track_id]
        track_particles = hit_particle_id[track_hits]
        matched_hits += (track_particles > 0).sum()
    
    hit_efficiency = matched_hits / total_truth_hits if total_truth_hits > 0 else 0
    
    return {
        'efficiency': efficiency,
        'fake_rate': fake_rate,
        'clone_rate': clone_rate,
        'avg_purity': avg_purity,
        'avg_completeness': avg_completeness,
        'hit_efficiency': hit_efficiency,
        'n_total_particles': n_total_particles,
        'n_reconstructed_particles': n_reconstructed_particles,
        'n_total_tracks': n_total_tracks,
        'n_fake_tracks': n_fake_tracks,
        'n_cloned_particles': n_cloned_particles,
    }
```

### Step 4: Plot vs pT and η

```python
def plot_efficiency_vs_pt_eta(matching_df, graph, output_dir):
    """Create efficiency plots vs pT and eta, similar to ACTS."""
    import matplotlib.pyplot as plt
    
    # Get particle information
    particles_df = pd.DataFrame({
        'particle_id': graph.track_particle_id.numpy(),
        'pt': graph.track_particle_pt.numpy(),
        'eta': graph.track_particle_eta.numpy(),
    }).drop_duplicates('particle_id')
    
    # Merge with matching to see which particles are reconstructed
    particles_df['is_reconstructed'] = particles_df['particle_id'].isin(
        matching_df['particle_id'].unique()
    )
    
    # Plot efficiency vs pT
    pt_bins = np.logspace(-1, 2, 20)  # 0.1 to 100 GeV
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    
    efficiency_pt = []
    for i in range(len(pt_bins) - 1):
        mask = (particles_df['pt'] >= pt_bins[i]) & (particles_df['pt'] < pt_bins[i+1])
        subset = particles_df[mask]
        if len(subset) > 0:
            eff = subset['is_reconstructed'].mean()
            efficiency_pt.append(eff)
        else:
            efficiency_pt.append(np.nan)
    
    plt.figure(figsize=(8, 6))
    plt.plot(pt_centers, efficiency_pt, 'o-')
    plt.xscale('log')
    plt.xlabel('pT [GeV]')
    plt.ylabel('Efficiency')
    plt.title('Track Efficiency vs pT')
    plt.grid(True)
    plt.savefig(output_dir / 'efficiency_vs_pt.png')
    
    # Similar for eta...
```

## Next Steps

1. **Create evaluation script**: `evaluate_tracks.py`
   - Load track files and graph files
   - Match tracks to particles
   - Calculate all metrics
   - Save results to CSV/JSON

2. **Create plotting script**: `plot_track_metrics.py`
   - Plot efficiency vs pT, η
   - Plot fake rate vs pT, η
   - Plot clone rate vs pT, η
   - Create summary plots

3. **Integrate with acorn**: Use acorn's existing evaluation utilities
   - `acorn.stages.track_building.utils.evaluate_labelled_graph()` already exists!
   - Can be adapted for our use case

4. **Compare with ACTS**: Run same events through both pipelines and compare metrics

## Using ACTS Functions Directly

### Option 1: TrackTruthMatcher (Requires ACTS Format)

ACTS has `acts.examples.TrackTruthMatcher` available in Python, but it requires:
- **ACTS track format**: `ConstTrackContainer` (not simple hit lists)
- **ACTS particle format**: `SimParticleContainer`
- **Measurement-particles map**: `MeasurementParticlesMap`

**Location**: `acts/Examples/Algorithms/TruthTracking/TrackTruthMatcher.hpp`

**Python binding**: `acts.examples.TrackTruthMatcher` (see `acts/Examples/Python/src/TruthTracking.cpp`)

**Issue**: Our tracks are simple hit ID lists, not ACTS track objects with measurements. We would need to:
1. Convert our hit lists to ACTS track format (complex)
2. Convert our graph truth to ACTS particle format (complex)
3. Create measurement-particles map (complex)

**Conclusion**: Not practical for our use case without significant data conversion.

### Option 2: TruthGraphMetricsHook (Graph-level, not Track-level)

ACTS has `TruthGraphMetricsHook` in `acts/Plugins/Gnn/` that calculates:
- **Efficiency**: Intersection edges / Truth edges
- **Purity**: Intersection edges / Predicted edges

**Location**: `acts/Plugins/Gnn/src/TruthGraphMetricsHook.cpp`

**Python binding**: `Acts.TruthGraphMetricsHook` (see `acts/Examples/Python/src/GnnTrackFinding.cpp`)

**Issue**: This works at the **edge/graph level**, not track level. It compares predicted edges vs truth edges, not tracks vs particles.

**Conclusion**: Useful for edge classifier evaluation, but not for track-level metrics.

### Option 3: Read ACTS ROOT Performance Files

ACTS writes performance metrics to ROOT files via `TrackFinderPerformanceWriter`:
- File: `performance_finding_ckf_matchingdetails.root`
- Contains: Efficiency, fake rate, clone rate, etc. vs pT, η

**Location**: `acts/Examples/Io/Root/src/TrackFinderPerformanceWriter.cpp`

**Python**: Can read with `uproot`:
```python
import uproot
file = uproot.open("performance_finding_ckf_matchingdetails.root")
# Access trees and branches
```

**Issue**: This requires running ACTS tracking first, which we're not doing.

**Conclusion**: Not applicable - we want to evaluate our own tracks.

### Option 4: Adapt ACTS Matching Logic (Recommended)

The matching logic in `TrackTruthMatcher.cpp` is straightforward:
1. For each track, count hits per particle
2. Calculate matching fraction: `shared_hits / track_hits`
3. Match if fraction >= threshold (default 0.5)

We can implement the same logic for our format:
- Tracks: Lists of hit IDs
- Truth: `hit_particle_id` from graph
- Matching: Same algorithm, different data structures

**Conclusion**: ✅ **Best approach** - Use ACTS's matching algorithm logic, adapted to our data format.

## Recommended Implementation

Since ACTS's `TrackTruthMatcher` requires ACTS-specific data formats, we should:

1. **Use ACORN's existing evaluation** (`acorn.stages.track_building.utils.evaluate_labelled_graph`)
   - Already works with PyG graphs
   - Already calculates efficiency, fake rate, clone rate
   - Already handles matching with configurable threshold
   - ✅ **YES - Supports efficiency vs pT and η plots!**
   
   The `tracking_efficiency()` method in `TrackBuildingStage` uses `evaluate_labelled_graph()` and then:
   - Extracts particle `pt` and `eta` from graph (`track_particle_pt`, `track_particle_eta`)
   - Creates efficiency plots vs pT and η using `plot_eff()` function
   - Configurable via YAML config with `variables: pt:` and `variables: eta:` sections
   
   Example config:
   ```yaml
   plots:
     tracking_efficiency:
       variables:
         pt:
           x_label: '$p_T [GeV]$'
           x_scale: 0.001  # Convert MeV to GeV
           x_lim: [1, 20]
         eta:
           x_label: '$\eta$'
           x_lim: [-4, 4]
   ```

2. **Or adapt ACTS's matching logic** to our format:
   - Implement the same matching algorithm
   - Use same metrics definitions
   - Ensure compatibility with ACTS conventions

3. **For plotting**: Use ACTS-style plots (efficiency vs pT, η) but with our data

## References

- ACTS TrackTruthMatcher: `acts/Examples/Algorithms/TruthTracking/TrackTruthMatcher.cpp`
- ACTS Python bindings: `acts/Examples/Python/src/TruthTracking.cpp`
- ACTS Performance Writer: `acts/Examples/Io/Root/src/TrackFinderPerformanceWriter.cpp`
- ACTS TruthGraphMetricsHook: `acts/Plugins/Gnn/src/TruthGraphMetricsHook.cpp`
- ACORN evaluation: `acorn/acorn/stages/track_building/utils.py`
- ACORN matching: `acorn/acorn/stages/track_building/models/hgnn_matching_utils.py`
