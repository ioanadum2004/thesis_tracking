#!/usr/bin/env python3
"""
Complete preprocessing pipeline: data reading + graph construction

This script runs two stages:
1. Data Reading: Uses acorn's ActsReader to convert ACTS CSV data to PyG graphs
2. Graph Construction: Builds edges between hits using simple radius + KNN

Usage:
    python build_simple_graphs.py

Configuration is automatically loaded from acorn_configs/pipeline_config.yaml
"""

import sys
import os
from pathlib import Path
import yaml
import torch
import pandas as pd
from tqdm import tqdm

# Add acorn to path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))

def load_config():
    """Load the combined pipeline configuration"""
    config_path = SCRIPT_DIR / 'acorn_configs' / 'graph_construction.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Make paths absolute relative to script directory
    for path_key in ['input_dir', 'stage_dir', 'detector_path']:
        if path_key in config:
            config[path_key] = str(SCRIPT_DIR / config[path_key])
    
    if 'graph_construction' in config:
        for path_key in ['input_dir', 'output_dir']:
            if path_key in config['graph_construction']:
                config['graph_construction'][path_key] = str(SCRIPT_DIR / config['graph_construction'][path_key])
    
    return config

def run_data_reading(config, cleanup_csv=False):
    """
    Stage 1: Run acorn's ActsReader to convert ACTS CSV to PyG graphs
    
    Args:
        config: Pipeline configuration
        cleanup_csv: If True, delete original ACTS CSV files after processing (saves disk space)
    """
    print("="*80)
    print("STAGE 1: Data Reading (ActsReader)")
    print("="*80)
    print(f"Input:  {config['input_dir']}")
    print(f"Output: {config['stage_dir']}")
    print()
    
    # Import acorn's ActsReader
    try:
        from acorn.stages.data_reading.models.acts_reader import ActsReader
    except ImportError as e:
        print(f"ERROR: Could not import acorn. Make sure you're in the acorn conda environment.")
        print(f"  {e}")
        sys.exit(1)
    
    # Run the ActsReader inference (this does CSV -> PyG conversion)
    reader = ActsReader.infer(config)
    
    # Optional: Clean up original ACTS CSV files to save disk space
    if cleanup_csv:
        print("\nCleaning up original ACTS CSV files...")
        csv_dir = Path(config['input_dir'])
        cleanup_patterns = ['*-hits.csv', '*-particles*.csv', '*-measurements.csv', 
                           '*-measurement-simhit-map.csv', '*-cells.csv']
        total_deleted = 0
        for pattern in cleanup_patterns:
            files = list(csv_dir.glob(pattern))
            for f in files:
                f.unlink()
                total_deleted += 1
        print(f"  Deleted {total_deleted} CSV files (keeping only detectors.csv)")
        print("  All data preserved in feature_store/")
    
    print("\n✓ Data reading complete!\n")

def build_edges_simple(positions, r_max=0.15, k_max=500):
    """
    Build edges between hits using radius + KNN
    
    Args:
        positions: Tensor of shape [num_hits, 3] with (r, phi, z) coordinates
        r_max: Maximum radius to connect hits
        k_max: Maximum number of neighbors per hit
    
    Returns:
        edge_index: Tensor of shape [2, num_edges] with candidate edges
    """
    from torch_geometric.nn import radius
    
    # Normalize positions
    pos_normalized = positions.clone()
    scales = torch.tensor([1000.0, 3.14, 1000.0], device=positions.device)
    pos_normalized = pos_normalized / scales
    
    # Build edges using PyG radius function
    edge_index = radius(pos_normalized, pos_normalized, r=r_max, max_num_neighbors=k_max)
    
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    return edge_index

def compute_edge_truth(edge_index, particle_id):
    """
    Label edges as true (1) or false (0) based on particle IDs
    Also compute track_to_edge_map
    
    Args:
        edge_index: Tensor [2, num_edges]
        particle_id: Tensor [num_hits] with particle ID for each hit
    
    Returns:
        edge_y: Tensor [num_edges] with 1 for true edges, 0 for fake edges
        track_to_edge_map: Tensor [num_tracks, max_edges_per_track] mapping tracks to edges
    """
    # Get particle IDs for source and target nodes
    pid_src = particle_id[edge_index[0]]
    pid_tgt = particle_id[edge_index[1]]
    
    # True edge if both hits belong to same particle (and not noise: pid > 0)
    edge_y = ((pid_src == pid_tgt) & (pid_src > 0)).long()
    
    # Build track_to_edge_map: for each track, list which edges belong to it
    unique_pids = particle_id.unique()
    unique_pids = unique_pids[unique_pids > 0]  # Remove noise
    num_tracks = len(unique_pids)
    
    # Find which edges belong to each track
    track_to_edge_list = []
    for pid in unique_pids:
        # Edges where both source and target belong to this particle
        track_edges = ((pid_src == pid) & (pid_tgt == pid)).nonzero(as_tuple=True)[0]
        track_to_edge_list.append(track_edges)
    
    # Pad to same length
    if len(track_to_edge_list) > 0:
        max_edges = max([len(te) for te in track_to_edge_list])
        track_to_edge_map = torch.full((num_tracks, max_edges), -1, dtype=torch.long)
        for i, track_edges in enumerate(track_to_edge_list):
            track_to_edge_map[i, :len(track_edges)] = track_edges
    else:
        track_to_edge_map = torch.empty((0, 0), dtype=torch.long)
    
    return edge_y, track_to_edge_map

def process_event(input_path, output_path, truth_csv_path, r_max=0.15, k_max=500):
    """Process a single event file"""
    graph = torch.load(input_path)
    
    # Load hit coordinates and particle IDs from truth CSV
    truth_df = pd.read_csv(truth_csv_path)
    hit_particle_ids = torch.tensor(truth_df['particle_id'].values, dtype=torch.long)
    
    # Extract Cartesian coordinates from CSV
    x = truth_df['x'].values
    y = truth_df['y'].values
    z_cart = truth_df['z'].values
    
    # Compute cylindrical coordinates (r, phi, z)
    import numpy as np
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Convert to tensors
    r_tensor = torch.tensor(r, dtype=torch.float32)
    phi_tensor = torch.tensor(phi, dtype=torch.float32)
    z_tensor = torch.tensor(z_cart, dtype=torch.float32)
    
    # Store coordinates on graph for later use
    graph.r = r_tensor.double()
    graph.phi = phi_tensor.double()
    graph.z = z_tensor.double()
    graph.x = torch.tensor(x, dtype=torch.float64)
    
    # Build edge positions from hit coordinates
    positions = torch.stack([r_tensor, phi_tensor, z_tensor], dim=1)
    
    # Build candidate edges
    edge_index = build_edges_simple(positions, r_max, k_max)
    
    # Compute truth labels and track-to-edge mapping
    edge_y, track_to_edge_map = compute_edge_truth(edge_index, hit_particle_ids)
    
    # Add to graph
    graph.edge_index = edge_index
    graph.edge_y = edge_y
    graph.track_to_edge_map = track_to_edge_map
    graph.particle_id = hit_particle_ids
    
    # Save
    torch.save(graph, output_path)
    
    return graph.num_nodes, edge_index.shape[1], edge_y.sum().item()

def run_graph_construction(config):
    """
    Stage 2: Build edges using simple radius + KNN approach
    """
    print("="*80)
    print("STAGE 2: Graph Construction (Radius + KNN)")
    print("="*80)
    
    gc_config = config['graph_construction']
    input_dir = Path(gc_config['input_dir'])
    output_dir = Path(gc_config['output_dir'])
    r_max = gc_config['r_max']
    k_max = gc_config['k_max']
    datasets = gc_config['datasets']
    
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Radius: {r_max}, Max neighbors: {k_max}")
    print()
    
    # Process each dataset
    for dataset_name in datasets:
        input_dataset_dir = input_dir / dataset_name
        output_dataset_dir = output_dir / dataset_name
        
        if not input_dataset_dir.exists():
            print(f"Skipping {dataset_name} - directory not found")
            continue
            
        output_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all event files
        event_files = sorted([f.name for f in input_dataset_dir.glob('*-graph.pyg')])
        
        if len(event_files) == 0:
            print(f"Skipping {dataset_name} - no graph files found")
            continue
        
        print(f"\nProcessing {dataset_name}: {len(event_files)} events")
        
        total_nodes = 0
        total_edges = 0
        total_true_edges = 0
        
        for event_file in tqdm(event_files, desc=f"  {dataset_name}"):
            input_path = input_dataset_dir / event_file
            output_path = output_dataset_dir / event_file
            # Convert Path to string for replace operation
            truth_csv_path = str(input_path).replace('-graph.pyg', '-truth.csv')
            
            num_nodes, num_edges, num_true = process_event(input_path, output_path, truth_csv_path, r_max, k_max)
            total_nodes += num_nodes
            total_edges += num_edges
            total_true_edges += num_true
        
        # Print statistics
        print(f"  ✓ Total nodes: {total_nodes}")
        print(f"  ✓ Total edges: {total_edges}")
        print(f"  ✓ True edges: {total_true_edges} ({100*total_true_edges/total_edges:.1f}%)")
        print(f"  ✓ Avg edges/node: {total_edges/total_nodes:.1f}")
    
    print("\n✓ Graph construction complete!\n")

def main():
    """Run the complete preprocessing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GNN Training Data Preprocessing Pipeline")
    parser.add_argument(
        "--cleanup-csv", 
        action="store_true",
        help="Delete original ACTS CSV files after ActsReader processes them (saves disk space)"
    )
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GNN TRAINING PIPELINE - DATA PREPROCESSING")
    print("="*80 + "\n")
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded from: acorn_configs/graph_construction.yaml\n")
    
    # Stage 1: Data Reading
    run_data_reading(config, cleanup_csv=args.cleanup_csv)
    
    # Stage 2: Graph Construction
    run_graph_construction(config)
    
    print("="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nProcessed graphs are in: {config['graph_construction']['output_dir']}")
    print(f"Ready for training with: python train_with_loss_logging.py\n")

if __name__ == "__main__":
    main()
