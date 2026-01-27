#!/usr/bin/env python3
"""
Evaluate track building performance and save statistics.

This script:
1. Loads graphs with track labels (from build_tracks.py)
2. Evaluates tracks against truth particles
3. Calculates efficiency, fake rate, clone rate, etc.
4. Saves detailed matching DataFrame and summary statistics

Usage:
    python evaluate_tracks.py [--config CONFIG_FILE] [--dataset DATASET]

Configuration:
    acorn_configs/track_building_eval.yaml (or specify with --config)
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import json
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))

from acorn.stages.track_building.utils import evaluate_labelled_graph, load_particles_df, get_matching_df, calculate_matching_fraction, apply_fiducial_sel
from acorn.stages.track_building.track_building_stage import make_result_summary
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate track building performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_tracks.py                    # Use default config
  python evaluate_tracks.py --config acorn_configs/track_building_egival.yaml
  python evaluate_tracks.py --dataset testset  # Evaluate testset instead of valset
  
Note: Make sure to run build_tracks.py first to generate tracks.
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to evaluation config file (default: acorn_configs/track_building_eval.yaml)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['trainset', 'valset', 'testset'],
        default='testset',
        help='Dataset to evaluate (default: testset)'
    )
    args = parser.parse_args()
    
    # Default config path
    if args.config is None:
        config_file = SCRIPT_DIR / 'acorn_configs' / 'track_building_eval.yaml'
    else:
        config_file = Path(args.config)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    print("="*70)
    print("TRACK BUILDING EVALUATION")
    print("="*70)
    print(f"Loading config from: {config_file}\n")
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set dataset
    config['dataset'] = args.dataset
    
    # Verify input directory exists (should contain graphs with hit_track_labels)
    input_dir = Path(config.get('input_dir', config.get('stage_dir')))
    if not input_dir.is_absolute():
        input_dir = SCRIPT_DIR / input_dir
    
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Please run build_tracks.py first to generate tracks."
        )
    
    # Set output directory
    output_dir = Path(config.get('output_dir', config.get('stage_dir', 'data/track_evaluation')))
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Matching fraction: {config.get('matching_fraction', 0.5)}")
    print(f"Matching style: {config.get('matching_style', 'ATLAS')}")
    print()
    
    # Load dataset
    from acorn.stages.track_building.track_building_stage import GraphDataset
    
    dataset_dir = input_dir / args.dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Count events
    graph_files = list(dataset_dir.glob('*.pyg'))
    num_events = len(graph_files)
    if num_events == 0:
        raise ValueError(f"No graph files found in {dataset_dir}")
    
    print(f"Found {num_events} events in {args.dataset}")
    print()
    
    # Create dataset
    dataset = GraphDataset(
        input_dir=str(input_dir),
        data_name=args.dataset,
        num_events=num_events,
        stage="test",
        hparams=config
    )
    
    # Evaluate all events
    print("="*70)
    print("EVALUATING TRACKS")
    print("="*70)
    print()
    
    evaluated_events = []
    times = []
    
    def safe_load_reconstruction_df(graph):
        """
        Load reconstruction DataFrame, using hit_particle_id if available to avoid
        track_edges/track_particle_id shape mismatch.
        """
        if hasattr(graph, "hit_id"):
            hit_id = graph.hit_id
        else:
            hit_id = torch.arange(graph.num_nodes)
        
        reco_df = pd.DataFrame({"hit_id": hit_id, "track_id": graph.hit_track_labels})
        
        # Use hit_particle_id if available (more reliable)
        if hasattr(graph, 'hit_particle_id'):
            hit_pid = graph.hit_particle_id
            if isinstance(hit_pid, torch.Tensor):
                reco_df["particle_id"] = hit_pid.numpy()
            else:
                reco_df["particle_id"] = hit_pid
        else:
            # Fallback to original method using track_edges
            try:
                node_id = graph.track_edges.reshape(-1)
                pids = graph.track_particle_id.repeat(2)
                
                # Check if lengths match
                if len(node_id) != len(pids):
                    raise ValueError(
                        f"track_edges and track_particle_id length mismatch: "
                        f"{len(node_id)} vs {len(pids)}. "
                        f"track_edges shape: {graph.track_edges.shape}, "
                        f"track_particle_id shape: {graph.track_particle_id.shape}"
                    )
                
                pid_df = pd.DataFrame({"hit_id": node_id, "particle_id": pids})
                pid_df.drop_duplicates(subset=["hit_id", "particle_id"], inplace=True)
                reco_df = reco_df.merge(pid_df, on="hit_id", how="outer")
            except (ValueError, AttributeError) as e:
                # If track_edges method fails, set particle_id to 0 (noise)
                print(f"Warning: Could not load particle_id from track_edges: {e}")
                print("  Setting all particle_id to 0 (noise)")
                reco_df["particle_id"] = 0
        
        reco_df.fillna({"track_id": -1, "particle_id": 0}, inplace=True)
        return reco_df
    
    def safe_load_particles_df(graph, sel_conf):
        """
        Custom load_particles_df that works with graphs that may not have track_particle_* attributes.
        Builds particles_df from hit_particle_id and available attributes.
        """
        # Attributes that are computed later - skip them
        computed_later = {'n_true_hits', 'nhits'}
        filtered_sel_conf = {k: v for k, v in sel_conf.items() if k not in computed_later}
        
        # Get unique particles from hit_particle_id
        if 'hit_particle_id' in graph:
            unique_pids = torch.unique(graph.hit_particle_id)
            unique_pids = unique_pids[unique_pids > 0].numpy()  # Remove noise (0)
        else:
            raise ValueError("Graph must have hit_particle_id attribute")
        
        # Build particles dataframe
        particles_data = {"particle_id": unique_pids}
        
        # Try to get pt from track_edges/pt mapping
        # pt is edge-level, need to map to particles via hits
        if 'track_edges' in graph and 'pt' in graph:
            # Map edge-level pt to particles via hit_particle_id
            edge_hits = graph.track_edges.reshape(-1)
            edge_pids = graph.hit_particle_id[edge_hits].numpy()
            edge_pt = graph.pt.repeat(2).numpy()  # pt is per edge, repeat for both endpoints
            
            # Create mapping: particle_id -> pt (take first occurrence)
            pid_pt_map = {}
            for pid, pval in zip(edge_pids, edge_pt):
                if pid > 0 and pid not in pid_pt_map:
                    pid_pt_map[pid] = pval
            
            particles_data["pt"] = [pid_pt_map.get(pid, 0.0) for pid in unique_pids]
        elif 'track_particle_pt' in graph:
            # Direct particle-level pt
            particles_data["pt"] = graph.track_particle_pt.numpy()[:len(unique_pids)]
        else:
            # No pt available - set to 0 (may cause issues with pt selection)
            particles_data["pt"] = [0.0] * len(unique_pids)
        
        # Try to get eta
        if 'track_edges' in graph and 'eta_particle' in graph:
            edge_hits = graph.track_edges.reshape(-1)
            edge_pids = graph.hit_particle_id[edge_hits].numpy()
            edge_eta = graph.eta_particle.repeat(2).numpy()
            
            pid_eta_map = {}
            for pid, eval in zip(edge_pids, edge_eta):
                if pid > 0 and pid not in pid_eta_map:
                    pid_eta_map[pid] = eval
            
            particles_data["eta"] = [pid_eta_map.get(pid, 0.0) for pid in unique_pids]
        elif 'track_particle_eta' in graph:
            particles_data["eta"] = graph.track_particle_eta.numpy()[:len(unique_pids)]
        
        particles_df = pd.DataFrame(particles_data)
        particles_df = particles_df.drop_duplicates(subset=["particle_id"])
        
        return particles_df
    
    def safe_evaluate_labelled_graph(graph, sel_conf, matching_fraction, matching_style, min_track_length):
        """
        Wrapper around evaluate_labelled_graph that uses safe_load_reconstruction_df.
        """
        # Load reconstruction DataFrame with safe method
        reco_df = safe_load_reconstruction_df(graph)
        
        # Continue with rest of evaluation
        # For particles_df, skip nhits (computed later)
        particles_sel_conf = {k: v for k, v in sel_conf.items() if k != 'nhits'}
        particles_df = safe_load_particles_df(graph, particles_sel_conf)
        
        # For matching_df, map nhits -> n_true_hits (which is computed in get_matching_df)
        matching_sel_conf = {}
        for k, v in sel_conf.items():
            if k == 'nhits':
                matching_sel_conf['n_true_hits'] = v  # Map nhits to n_true_hits
            else:
                matching_sel_conf[k] = v
        
        matching_df = get_matching_df(reco_df, particles_df, matching_sel_conf, min_track_length=min_track_length)
        
        # Flatten event_id if it's a list
        event_id = graph.event_id
        while type(event_id) == list:
            event_id = event_id[0]
        matching_df["event_id"] = int(event_id)
        
        matching_df = calculate_matching_fraction(matching_df)
        
        # Run matching depending on the matching style
        if matching_style == "ATLAS":
            matching_df["is_matched"] = matching_df["is_reconstructed"] = (
                matching_df.purity_reco >= matching_fraction
            )
        elif matching_style == "one_way":
            matching_df["is_matched"] = matching_df.purity_reco >= matching_fraction
            matching_df["is_reconstructed"] = matching_df.eff_true >= matching_fraction
        elif matching_style == "two_way":
            matching_df["is_matched"] = matching_df["is_reconstructed"] = (
                matching_df.purity_reco >= matching_fraction
            ) & (matching_df.eff_true >= matching_fraction)
        
        return matching_df
    
    for event in tqdm(dataset, desc=f"Evaluating {args.dataset}"):
        try:
            matching_df = safe_evaluate_labelled_graph(
                event,
                sel_conf=config.get("target_tracks", {}),
                matching_fraction=config.get("matching_fraction", 0.5),
                matching_style=config.get("matching_style", "ATLAS"),
                min_track_length=config.get("min_track_length", 1),
            )
            evaluated_events.append(matching_df)
            
            if hasattr(event, 'time_taken'):
                times.append(event.time_taken)
        except Exception as e:
            event_id = getattr(event, 'event_id', 'unknown')
            if isinstance(event_id, list):
                event_id = event_id[0] if len(event_id) > 0 else 'unknown'
            print(f"\nError evaluating event {event_id}: {e}")
            import traceback
            traceback.print_exc()
            # Try to diagnose the issue
            if hasattr(event, 'track_edges') and hasattr(event, 'track_particle_id'):
                print(f"  track_edges shape: {event.track_edges.shape}")
                print(f"  track_particle_id shape: {event.track_particle_id.shape}")
                print(f"  hit_particle_id available: {hasattr(event, 'hit_particle_id')}")
                if hasattr(event, 'hit_particle_id'):
                    print(f"  hit_particle_id shape: {event.hit_particle_id.shape}")
            raise
    
    # Concatenate all events
    evaluated_events = pd.concat(evaluated_events, ignore_index=True)
    
    # Calculate times
    if times:
        times = np.array(times)
        time_avg = np.mean(times)
        time_std = np.std(times)
    else:
        time_avg = 0.0
        time_std = 0.0
    
    # Filter particles and tracks
    particles = evaluated_events[evaluated_events["is_reconstructable"]]
    reconstructed_particles = particles[
        particles["is_reconstructed"] & particles["is_matchable"]
    ]
    tracks = evaluated_events[evaluated_events["is_matchable"]]
    matched_tracks = tracks[tracks["is_matched"]]
    
    # Calculate metrics
    n_particles = len(particles.drop_duplicates(subset=["event_id", "particle_id"]))
    n_reconstructed_particles = len(
        reconstructed_particles.drop_duplicates(subset=["event_id", "particle_id"])
    )
    
    n_tracks = len(tracks.drop_duplicates(subset=["event_id", "track_id"]))
    n_matched_tracks = len(
        matched_tracks.drop_duplicates(subset=["event_id", "track_id"])
    )
    
    n_dup_reconstructed_particles = (
        len(reconstructed_particles) - n_reconstructed_particles
    )
    
    efficiency = n_reconstructed_particles / n_particles if n_particles > 0 else 0.0
    fake_rate = 1 - (n_matched_tracks / n_tracks) if n_tracks > 0 else 0.0
    clone_rate = n_dup_reconstructed_particles / n_reconstructed_particles if n_reconstructed_particles > 0 else 0.0
    
    # Calculate average purity and completeness
    avg_purity = evaluated_events['purity_reco'].mean() if 'purity_reco' in evaluated_events.columns else 0.0
    avg_completeness = evaluated_events['eff_true'].mean() if 'eff_true' in evaluated_events.columns else 0.0
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # 1. Save detailed matching DataFrame
    matching_df_path = output_dir / f"matching_df_{args.dataset}.csv"
    evaluated_events.to_csv(matching_df_path, index=False)
    print(f"Saved matching DataFrame: {matching_df_path}")
    
    # 2. Save summary statistics
    summary = {
        'dataset': args.dataset,
        'n_events': num_events,
        'n_particles': int(n_particles),
        'n_reconstructed_particles': int(n_reconstructed_particles),
        'n_tracks': int(n_tracks),
        'n_matched_tracks': int(n_matched_tracks),
        'n_duplicate_particles': int(n_dup_reconstructed_particles),
        'efficiency': float(efficiency),
        'fake_rate': float(fake_rate),
        'clone_rate': float(clone_rate),
        'avg_purity': float(avg_purity),
        'avg_completeness': float(avg_completeness),
        'time_avg_seconds': float(time_avg),
        'time_std_seconds': float(time_std),
        'matching_fraction': config.get('matching_fraction', 0.5),
        'matching_style': config.get('matching_style', 'ATLAS'),
    }
    
    summary_json_path = output_dir / f"summary_{args.dataset}.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {summary_json_path}")
    
    # Also save as text summary (ACORN format)
    summary_text = make_result_summary(
        n_reconstructed_particles,
        n_particles,
        n_matched_tracks,
        n_tracks,
        n_dup_reconstructed_particles,
        efficiency,
        fake_rate,
        clone_rate,
        time_avg,
        time_std,
    )
    
    summary_txt_path = output_dir / f"results_summary_{config.get('matching_style', 'ATLAS')}_{args.dataset}.txt"
    with open(summary_txt_path, 'w') as f:
        f.write(summary_text)
    print(f"Saved summary text: {summary_txt_path}")
    
    # 3. Save binned data for plotting (efficiency vs pt and eta)
    # Prepare particles DataFrame for plotting
    particles_plot = particles.drop_duplicates(subset=["particle_id"]).copy()
    
    # Save particles DataFrame with reconstruction status
    particles_df_path = output_dir / f"particles_{args.dataset}.csv"
    particles_plot.to_csv(particles_df_path, index=False)
    print(f"Saved particles DataFrame: {particles_df_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(summary_text)
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")
    print("\nNext step: Create plots with:")
    print(f"  python plot_track_metrics.py --input-dir {output_dir} --dataset {args.dataset}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
