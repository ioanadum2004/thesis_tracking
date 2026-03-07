#!/usr/bin/env python3
"""
GNN Segment Matching Track Builder

Builds tracks using a pretrained mini-GNN segment embedder. Segments from the
same particle should have high cosine similarity in the learned embedding space.

Algorithm:
    1. Extract segments (CC clustering on GNN edge scores, or ground truth)
    2. Accept segments that reach the outer detector as complete tracks
    3. Embed remaining segments with the pretrained mini-GNN
    4. Greedy matching: connect segment pairs with highest cosine similarity
       down to a configurable lower bound
    5. Evaluate and plot

Usage:
    python GNN_segment_matching_track_builder.py testset
    python GNN_segment_matching_track_builder.py testset --use-gt-segments
    python GNN_segment_matching_track_builder.py testset --score-cut 0.8
    python GNN_segment_matching_track_builder.py testset --skip-build
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch
import yaml
from torch_geometric.data import Batch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))
sys.path.insert(0, str(PIPELINE_ROOT))

from acorn.utils.loading_utils import load_datafiles_in_dir

from low_pt_custom_utils.track_evaluation_utils import (
    run_evaluation,
    save_evaluation_results,
    run_plotting,
)
from low_pt_custom_utils.segment_matching import (
    extract_segments_from_cc,
    extract_segments_from_ground_truth,
    segments_to_track_labels,
)
from low_pt_custom_utils.mini_gnn_segment_embedding import (
    load_segment_gnn,
    segment_to_pyg,
)


# ─── GNN-Based Segment Matching ─────────────────────────────────────────────


def match_segments_gnn(segments, graph, model, config, device="cpu"):
    """
    Match segments using cosine similarity of GNN embeddings.

    Args:
        segments:  List[SegmentInfo] from CC or GT extraction.
        graph:     PyG Data object for the full event (provides hit coords).
        model:     Pretrained SegmentGNN in eval mode (has .node_scales attribute).
        config:    Dict with 'cos_sim_threshold' and 'outer_r_threshold'.
        device:    Torch device string.

    Returns:
        matched_tracks: List of [SegmentInfo, SegmentInfo] pairs.
        unmatched:      List of SegmentInfo not in any matched pair
                        (includes complete tracks).
    """
    cos_sim_threshold = config.get("cos_sim_threshold", 0.5)
    outer_r_threshold = config.get("outer_r_threshold", 1000.0)
    node_scales = model.node_scales

    # Separate complete tracks (reach outer detector)
    to_match = []
    complete = []
    for seg in segments:
        if seg.outer_r >= outer_r_threshold:
            complete.append(seg)
        else:
            to_match.append(seg)

    # Build per-segment PyG graphs and embed
    seg_graphs = [segment_to_pyg(seg, graph, node_scales=tuple(node_scales))
                  for seg in to_match]
    batch_data = Batch.from_data_list(seg_graphs).to(device)

    with torch.no_grad():
        embeddings = model(batch_data.x, batch_data.edge_index, batch_data.batch)
        # embeddings: (n_segments, emb_dim), L2-normalized

    # Cosine similarity = dot product (unit vectors)
    sim_matrix = embeddings @ embeddings.T  # (N, N)

    # Extract upper-triangle pairs and sort by similarity descending
    n = len(to_match)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            s = sim_matrix[i, j].item()
            if s >= cos_sim_threshold:
                pairs.append((s, i, j))

    pairs.sort(key=lambda x: x[0], reverse=True)

    # Greedy matching
    matched_tracks = []
    matched_set = set()
    for score, i, j in pairs:
        if i not in matched_set and j not in matched_set:
            matched_tracks.append([to_match[i], to_match[j]])
            matched_set.add(i)
            matched_set.add(j)

    # Unmatched = complete tracks + segments that didn't get paired
    unmatched = complete + [to_match[i] for i in range(n) if i not in matched_set]

    return matched_tracks, unmatched


# ─── Per-Event Track Building ───────────────────────────────────────────────


def build_tracks_for_event_gnn(graph, config, model, device):
    """
    Build tracks for one event using GNN-scored segment matching.

    Returns:
        hit_track_labels: Tensor of shape (num_nodes,), -1 for unassigned hits.
        stats: Dict with per-event counts.
    """
    use_gt = config.get("use_gt_segments", False)
    score_cut = config.get("score_cut", 0.5)
    gnn_config = config.get("gnn", {})

    # Step 1: Extract segments
    if use_gt:
        segments = extract_segments_from_ground_truth(graph)
    else:
        segments = extract_segments_from_cc(graph, score_cut)

    n_segments = len(segments)

    # Step 2+3: GNN-based matching (includes complete track separation)
    matched_tracks, unmatched = match_segments_gnn(
        segments, graph, model, gnn_config, device=device,
    )

    # Step 4: Convert to hit labels
    num_nodes = graph.hit_x.size(0)
    hit_t = graph.hit_t.cpu().numpy() if hasattr(graph, "hit_t") else None
    labels = segments_to_track_labels(matched_tracks, unmatched, num_nodes, hit_t=hit_t)

    # Statistics
    outer_r_threshold = gnn_config.get("outer_r_threshold", 1000.0)
    n_matched_pairs = len(matched_tracks)
    n_complete_tracks = sum(1 for s in unmatched if s.outer_r >= outer_r_threshold)
    n_no_match = len(unmatched) - n_complete_tracks
    n_standalone = len(unmatched)
    n_total_tracks = n_matched_pairs + n_standalone
    n_assigned = (labels >= 0).sum().item()
    n_unassigned = num_nodes - n_assigned

    stats = {
        "n_segments": n_segments,
        "n_matched_pairs": n_matched_pairs,
        "n_complete_tracks": n_complete_tracks,
        "n_no_match": n_no_match,
        "n_standalone": n_standalone,
        "n_total_tracks": n_total_tracks,
        "n_assigned_hits": n_assigned,
        "n_unassigned_hits": n_unassigned,
    }

    return labels, stats


# ─── Main Algorithm ─────────────────────────────────────────────────────────


def run_gnn_segment_matching(dataset_name, config, model, device,
                              score_cut=None, use_gt_segments=None):
    """Build tracks for all events in a dataset using GNN-scored segment matching."""
    input_dir = Path(config.get("input_dir", "data/gnn_stage"))
    if not input_dir.is_absolute():
        input_dir = PIPELINE_ROOT / input_dir

    output_dir = PIPELINE_ROOT / "data" / "track_building"
    dataset_output = output_dir / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    if score_cut is not None:
        config["score_cut"] = score_cut
    if use_gt_segments is not None:
        config["use_gt_segments"] = use_gt_segments

    actual_score_cut = config.get("score_cut", 0.5)
    actual_use_gt = config.get("use_gt_segments", False)
    gnn_config = config.get("gnn", {})

    data_split = config.get("data_split", [0, 1000, 1000])
    split_map = {"trainset": data_split[0], "valset": data_split[1], "testset": data_split[2]}
    num_events = split_map.get(dataset_name, 1000)

    input_paths = load_datafiles_in_dir(str(input_dir), dataset_name, num_events)
    input_paths.sort()

    segment_mode = "ground truth (hit_segment_id)" if actual_use_gt else f"CC clusters (score > {actual_score_cut})"

    print(f"GNN Segment Matching Track Builder")
    print(f"  Input:              {input_dir / dataset_name}")
    print(f"  Output:             {dataset_output}")
    print(f"  Events:             {len(input_paths)}")
    print(f"  Segment mode:       {segment_mode}")
    print(f"  Cos sim threshold:  {gnn_config.get('cos_sim_threshold', 0.5)}")
    print(f"  Outer R threshold:  {gnn_config.get('outer_r_threshold', 1000.0)} mm")
    print(f"  Device:             {device}")
    print()

    total_stats = {
        "n_segments": 0,
        "n_matched_pairs": 0,
        "n_complete_tracks": 0,
        "n_no_match": 0,
        "n_standalone": 0,
        "n_total_tracks": 0,
        "n_assigned_hits": 0,
        "n_unassigned_hits": 0,
    }
    total_time = 0.0

    for event_path in tqdm(input_paths, desc=f"Building tracks for {dataset_name}"):
        t_event = perf_counter()
        graph = torch.load(event_path, map_location="cpu", weights_only=False)

        labels, event_stats = build_tracks_for_event_gnn(graph, config, model, device)

        graph.hit_track_labels = labels
        graph.time_taken = perf_counter() - t_event
        total_time += graph.time_taken

        for key in total_stats:
            total_stats[key] += event_stats[key]

        event_id = graph.event_id
        if isinstance(event_id, list):
            event_id = event_id[0]
        torch.save(graph, dataset_output / f"event{event_id}.pyg")

    # Summary
    n_events = len(input_paths)
    print(f"\n{'='*70}")
    print(f"GNN SEGMENT MATCHING SUMMARY - {dataset_name.upper()}")
    print(f"{'='*70}")

    print(f"\nSegment Statistics (totals across {n_events} events):")
    print(f"  Total segments:        {total_stats['n_segments']:6d}")

    print(f"\nMatching Results:")
    print(f"  Matched pairs:         {total_stats['n_matched_pairs']:6d}  (2 segments → 1 track)")
    print(f"  Standalone (exit):     {total_stats['n_complete_tracks']:6d}  (outer_r >= threshold)")
    print(f"  Standalone (no match): {total_stats['n_no_match']:6d}  (no GNN match found)")
    print(f"  Total tracks:          {total_stats['n_total_tracks']:6d}")

    print(f"\nHit Assignment:")
    print(f"  Assigned:              {total_stats['n_assigned_hits']:6d}")
    print(f"  Unassigned:            {total_stats['n_unassigned_hits']:6d}")

    print(f"\nTiming:")
    print(f"  Total time:            {total_time:.2f}s")
    print(f"  Average per event:     {total_time/n_events:.4f}s")
    print(f"  Events per second:     {n_events/total_time:.1f}")

    print(f"\n{'='*70}")
    print(f"Tracks saved to: {dataset_output}")
    print(f"{'='*70}")


# ─── Script Entry Point ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="GNN segment embedding track builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python GNN_segment_matching_track_builder.py testset
  python GNN_segment_matching_track_builder.py testset --use-gt-segments
  python GNN_segment_matching_track_builder.py testset --score-cut 0.8
  python GNN_segment_matching_track_builder.py testset --skip-build
        """,
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["trainset", "valset", "testset"],
        help="Dataset to process",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: acorn_configs/track_building_stage_(3)/gnn_segment_matching.yaml)",
    )
    parser.add_argument(
        "--score-cut",
        type=float,
        default=None,
        help="Override CC edge score cut threshold",
    )
    parser.add_argument(
        "--use-gt-segments",
        action="store_true",
        default=None,
        help="Use ground truth segment labels instead of CC clusters",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip track building, only re-evaluate existing output",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available, else cpu)",
    )

    args = parser.parse_args()

    # Load config
    if args.config is None:
        config_file = (
            PIPELINE_ROOT
            / "acorn_configs"
            / "track_building_stage_(3)"
            / "gnn_segment_matching.yaml"
        )
    else:
        config_file = Path(args.config)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    config["dataset"] = args.dataset

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    eval_output_dir = PIPELINE_ROOT / "data" / "track_evaluation" / args.dataset
    plot_output_dir = PIPELINE_ROOT / "data" / "visuals" / "track_metrics" / args.dataset

    # ── Step 1: Track Building ──────────────────────────────────────────────
    if not args.skip_build:
        print("=" * 70)
        print("STEP 1: GNN SEGMENT MATCHING")
        print("=" * 70)
        print()

        # Load model
        gnn_config = config.get("gnn", {})
        model_path = gnn_config.get("model_path", "saved_models/mini_gnn_segment_embedder.pt")
        if not Path(model_path).is_absolute():
            model_path = str(PIPELINE_ROOT / model_path)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Trained GNN not found at: {model_path}\n"
                f"Run train_mini_GNN_to_match.py first."
            )

        print(f"Loading mini-GNN from: {model_path}")
        model = load_segment_gnn(model_path, device=device)
        print()

        run_gnn_segment_matching(
            args.dataset,
            config,
            model,
            device,
            score_cut=args.score_cut,
            use_gt_segments=args.use_gt_segments,
        )
        print()
    else:
        print("Skipping track building (--skip-build)")
        print()

    # ── Step 2: Evaluation ─────────────────────────────────────────────────
    print("=" * 70)
    print(f"STEP 2: EVALUATING {args.dataset.upper()}")
    print("=" * 70)
    print()

    config["input_dir"] = str(PIPELINE_ROOT / "data" / "track_building")
    evaluated_events, summary, summary_text = run_evaluation(args.dataset, config)
    save_evaluation_results(evaluated_events, summary, summary_text, args.dataset, eval_output_dir)

    print()
    print(summary_text)

    # ── Step 3: Plotting ───────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 3: PLOTTING METRICS")
    print("=" * 70)
    print()

    run_plotting(evaluated_events, summary, args.dataset, plot_output_dir, config.get("plots", {}))

    # ── Done ───────────────────────────────────────────────────────────────
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
