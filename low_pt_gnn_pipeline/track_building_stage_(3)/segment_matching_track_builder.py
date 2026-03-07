#!/usr/bin/env python3
"""
Segment Matching Track Builder — Helix-Based Segment Matching

Builds tracks by fitting helices to GNN-identified segments and matching
segments from the same particle based on helix parameter compatibility
(circle center, radius, pitch).

Algorithm:
    1. Extract segments (CC clustering on GNN edge scores, or ground truth)
    2. Fit circle (Kasa method) and pitch to each segment using ALL hits
    3. Match segments via helix parameter comparison (greedy, highest-score-first)
    4. Attach short segments (1-2 hits) by endpoint proximity
    5. Evaluate and plot

Usage:
    python segment_matching_track_builder.py testset
    python segment_matching_track_builder.py testset --use-gt-segments
    python segment_matching_track_builder.py testset --score-cut 0.8
    python segment_matching_track_builder.py testset --skip-build
"""

import argparse
import sys
from pathlib import Path
from time import process_time, perf_counter

import torch
import yaml
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))
sys.path.insert(0, str(PIPELINE_ROOT))

from acorn.utils.loading_utils import load_datafiles_in_dir

# Import evaluation/plotting from utils module
from low_pt_custom_utils.track_evaluation_utils import (
    run_evaluation,
    save_evaluation_results,
    run_plotting,
)

# Import segment matching logic
from low_pt_custom_utils.segment_matching import build_tracks_for_event


# ─── Main Algorithm ─────────────────────────────────────────────────────────

def run_segment_matching(dataset_name, config, score_cut=None, use_gt_segments=None):
    """Build tracks for all events in a dataset using helix-based segment matching."""
    input_dir = Path(config.get('input_dir', 'data/gnn_stage'))
    if not input_dir.is_absolute():
        input_dir = PIPELINE_ROOT / input_dir

    output_dir = PIPELINE_ROOT / 'data' / 'track_building'
    dataset_output = output_dir / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    # Override config with CLI args if provided
    if score_cut is not None:
        config['score_cut'] = score_cut
    if use_gt_segments is not None:
        config['use_gt_segments'] = use_gt_segments

    actual_score_cut = config.get('score_cut', 0.5)
    actual_use_gt = config.get('use_gt_segments', False)
    B_field = config.get('B_field', 2.0)

    # Load event files
    data_split = config.get('data_split', [0, 1000, 1000])
    split_map = {'trainset': data_split[0], 'valset': data_split[1], 'testset': data_split[2]}
    num_events = split_map.get(dataset_name, 1000)

    input_paths = load_datafiles_in_dir(str(input_dir), dataset_name, num_events)
    input_paths.sort()

    segment_mode = "ground truth (hit_segment_id)" if actual_use_gt else f"CC clusters (score > {actual_score_cut})"

    print(f"Segment Matching Track Builder (Helix-Based)")
    print(f"  Input:              {input_dir / dataset_name}")
    print(f"  Output:             {dataset_output}")
    print(f"  Events:             {len(input_paths)}")
    print(f"  Segment mode:       {segment_mode}")
    print(f"  B field:            {B_field} T")
    matching = config.get('matching', {})
    print(f"  Max center dist:    {matching.get('max_center_distance', 100.0)} mm")
    print(f"  Sigma center:       {matching.get('sigma_center', 30.0)} mm")
    print(f"  Sigma R:            {matching.get('sigma_R', 0.1)}")
    print()

    # Accumulate statistics
    total_stats = {
        "n_segments": 0,
        "n_good_fits": 0,
        "n_poor_fits": 0,
        "n_no_fits": 0,
        "n_matched_pairs": 0,
        "n_complete_tracks": 0,
        "n_no_match": 0,
        "n_standalone": 0,
        "n_total_tracks": 0,
        "n_assigned_hits": 0,
        "n_unassigned_hits": 0,
    }
    total_time = 0.0

    for event_idx, event_path in enumerate(tqdm(input_paths, desc=f"Building tracks for {dataset_name}")):
        t_event = perf_counter()
        graph = torch.load(event_path, map_location="cpu", weights_only=False)

        labels, event_stats = build_tracks_for_event(graph, config)

        graph.hit_track_labels = labels
        graph.time_taken = perf_counter() - t_event
        total_time += graph.time_taken

        # Accumulate statistics
        for key in total_stats:
            total_stats[key] += event_stats[key]

        # Save
        event_id = graph.event_id
        if isinstance(event_id, list):
            event_id = event_id[0]
        torch.save(graph, dataset_output / f"event{event_id}.pyg")

    # Print summary
    n_events = len(input_paths)
    print(f"\n{'='*70}")
    print(f"SEGMENT MATCHING SUMMARY - {dataset_name.upper()}")
    print(f"{'='*70}")

    print(f"\nSegment Statistics (totals across {n_events} events):")
    print(f"  Total segments:        {total_stats['n_segments']:6d}")
    print(f"  Good fits (3+ hits):   {total_stats['n_good_fits']:6d}")
    print(f"  Poor fits (2 hits):    {total_stats['n_poor_fits']:6d}")
    print(f"  No fits (1 hit):       {total_stats['n_no_fits']:6d}")

    print(f"\nMatching Results:")
    print(f"  Matched pairs:         {total_stats['n_matched_pairs']:6d}  (2 segments → 1 track)")
    print(f"  Standalone (exit):     {total_stats['n_complete_tracks']:6d}  (outer_r ≥ threshold, complete tracks)")
    print(f"  Standalone (no match): {total_stats['n_no_match']:6d}  (loop segment, no partner found)")
    print(f"  Total tracks:          {total_stats['n_total_tracks']:6d}")

    print(f"\nHit Assignment:")
    print(f"  Assigned:              {total_stats['n_assigned_hits']:6d}")
    print(f"  Unassigned:            {total_stats['n_unassigned_hits']:6d}")

    print(f"\nTiming:")
    print(f"  Total time:            {total_time:.2f}s")
    if n_events > 0:
        print(f"  Average per event:     {total_time/n_events:.4f}s")
        print(f"  Events per second:     {n_events/total_time:.1f}" if total_time > 0 else "  Events per second:     N/A")

    print(f"\n{'='*70}")
    print(f"Tracks saved to: {dataset_output}")
    print(f"{'='*70}")


# ─── Script Entry Point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Helix-based segment matching track builder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python segment_matching_track_builder.py testset
  python segment_matching_track_builder.py testset --use-gt-segments
  python segment_matching_track_builder.py testset --score-cut 0.8
  python segment_matching_track_builder.py testset --skip-build
        """
    )
    parser.add_argument(
        'dataset',
        type=str,
        choices=['trainset', 'valset', 'testset'],
        help='Dataset to process'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: acorn_configs/track_building_stage_(3)/segment_matching.yaml)'
    )
    parser.add_argument(
        '--score-cut',
        type=float,
        default=None,
        help='Override score cut threshold for CC clustering'
    )
    parser.add_argument(
        '--use-gt-segments',
        action='store_true',
        default=None,
        help='Use ground truth segment labels instead of CC clusters'
    )
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip track building, only re-evaluate'
    )

    args = parser.parse_args()

    # Load config
    if args.config is None:
        config_file = PIPELINE_ROOT / 'acorn_configs' / 'track_building_stage_(3)' / 'segment_matching.yaml'
    else:
        config_file = Path(args.config)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config['dataset'] = args.dataset

    eval_output_dir = PIPELINE_ROOT / 'data' / 'track_evaluation' / args.dataset
    plot_output_dir = PIPELINE_ROOT / 'data' / 'visuals' / 'track_metrics' / args.dataset

    # ── Step 1: Track Building ──────────────────────────────────────────
    if not args.skip_build:
        print("=" * 70)
        print("STEP 1: HELIX-BASED SEGMENT MATCHING")
        print("=" * 70)
        print()

        run_segment_matching(
            args.dataset, config,
            score_cut=args.score_cut,
            use_gt_segments=args.use_gt_segments,
        )
        print()
    else:
        print("Skipping track building (--skip-build)")
        print()

    # ── Step 2: Evaluation ──────────────────────────────────────────────
    print("=" * 70)
    print(f"STEP 2: EVALUATING {args.dataset.upper()}")
    print("=" * 70)
    print()

    config['input_dir'] = str(PIPELINE_ROOT / 'data' / 'track_building')
    evaluated_events, summary, summary_text = run_evaluation(args.dataset, config)
    save_evaluation_results(evaluated_events, summary, summary_text, args.dataset, eval_output_dir)

    print()
    print(summary_text)

    # ── Step 3: Plotting ────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 3: PLOTTING METRICS")
    print("=" * 70)
    print()

    run_plotting(evaluated_events, summary, args.dataset, plot_output_dir, config.get('plots', {}))

    # ── Done ────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"  Tracks:     data/track_building/")
    print(f"  Evaluation: {eval_output_dir.relative_to(PIPELINE_ROOT)}/")
    print(f"  Plots:      {plot_output_dir.relative_to(PIPELINE_ROOT)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
