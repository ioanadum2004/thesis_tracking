#!/usr/bin/env python3
"""
Run MLP segment pair data mining for a specific chunk of events.

Each job processes a disjoint subset of events from each split and saves
partial tensors to data/track_building/MLP_segments_chunk{N:02d}/.

Usage:
    python run_mlp_mining_chunk.py --chunk 0 --total-chunks 50
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "acorn"))
sys.path.insert(0, str(PIPELINE_ROOT))

from acorn.utils.loading_utils import load_datafiles_in_dir
from low_pt_custom_utils.segment_matching import (
    extract_segments_from_cc,
    fit_helices_to_segments,
)
from low_pt_custom_utils.mlp_segment_matching import build_event_training_pairs


def get_chunk_paths(all_paths, chunk, total_chunks):
    """Return the disjoint slice of paths assigned to this chunk."""
    n = len(all_paths)
    events_per_chunk = n // total_chunks
    remainder = n % total_chunks

    if chunk < remainder:
        start = chunk * (events_per_chunk + 1)
        end = start + events_per_chunk + 1
    else:
        start = remainder * (events_per_chunk + 1) + (chunk - remainder) * events_per_chunk
        end = start + events_per_chunk

    return all_paths[start:end]


def mine_paths(paths, config, split_name, rng):
    """Mine segment pairs for a list of event paths. Returns dict or None."""
    score_cut = config.get("score_cut", 0.85)
    B_field = config.get("B_field", 2.0)
    outlier_rejection = config.get("outlier_rejection", False)
    neg_ratio = config.get("neg_ratio", 5)
    feature_scales = config.get("feature_scales")

    all_features = []
    all_labels = []
    n_pos_total = 0
    n_neg_total = 0

    for event_path in tqdm(paths, desc=f"  Mining {split_name}"):
        graph = torch.load(event_path, map_location="cpu", weights_only=False)

        segments = extract_segments_from_cc(graph, score_cut)
        if not segments:
            continue

        segments = fit_helices_to_segments(
            segments, graph, B_field=B_field, outlier_rejection=outlier_rejection
        )

        pos_feats, neg_feats = build_event_training_pairs(
            graph, segments,
            neg_ratio=neg_ratio,
            rng=rng,
            feature_scales=feature_scales,
        )

        for f in pos_feats:
            all_features.append(f)
            all_labels.append(1.0)
        for f in neg_feats:
            all_features.append(f)
            all_labels.append(0.0)

        n_pos_total += len(pos_feats)
        n_neg_total += len(neg_feats)

    if not all_features:
        print(f"  WARNING: No pairs found for {split_name} in this chunk — skipping save.")
        return None

    print(f"  {split_name}: {n_pos_total} pos, {n_neg_total} neg ({len(all_features)} total pairs)")

    X = torch.tensor(np.array(all_features), dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)
    return {"X": X, "y": y, "feature_scales": feature_scales}


def main():
    parser = argparse.ArgumentParser(description="Mine MLP pairs for one chunk of events")
    parser.add_argument("--chunk", type=int, required=True, help="Chunk index (0-indexed)")
    parser.add_argument("--total-chunks", type=int, required=True, help="Total number of chunks")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    if args.config is None:
        config_file = (
            PIPELINE_ROOT
            / "acorn_configs"
            / "track_building_stage_(3)"
            / "mlp_data_mining.yaml"
        )
    else:
        config_file = Path(args.config)

    with open(config_file) as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print(f"MLP DATA MINING — Chunk {args.chunk}/{args.total_chunks}")
    print("=" * 65)
    print(f"Config: {config_file}")
    print(f"Node:   {__import__('socket').gethostname()}")
    print()

    input_dir = Path(config.get("input_dir", "data/gnn_stage"))
    if not input_dir.is_absolute():
        input_dir = PIPELINE_ROOT / input_dir

    chunk_output_dir = PIPELINE_ROOT / f"data/track_building/MLP_segments_chunk{args.chunk:02d}"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {input_dir}")
    print(f"Output: {chunk_output_dir}")
    print()

    data_split = config.get("data_split", [30000, 900, 900])
    n_train = data_split[0]
    n_val   = data_split[1]
    n_test  = data_split[2] if len(data_split) > 2 else 0

    splits = [("trainset", n_train), ("valset", n_val)]
    if n_test > 0:
        splits.append(("testset", n_test))

    # Different seed per chunk so negative sampling isn't identical across chunks
    rng = np.random.default_rng(42 + args.chunk)

    t0 = perf_counter()
    for split_name, n_events in splits:
        if n_events == 0:
            continue

        all_paths = load_datafiles_in_dir(str(input_dir), split_name, n_events)
        all_paths.sort()
        chunk_paths = get_chunk_paths(all_paths, args.chunk, args.total_chunks)

        print(f"{split_name}: {len(chunk_paths)} / {len(all_paths)} events in this chunk")
        if not chunk_paths:
            continue

        data = mine_paths(chunk_paths, config, split_name, rng)
        if data is not None:
            out_path = chunk_output_dir / f"{split_name}.pt"
            torch.save(data, out_path)
            print(f"  Saved → {out_path}  (X={tuple(data['X'].shape)}, y={tuple(data['y'].shape)})\n")

    print(f"\n✓ Chunk {args.chunk} complete in {perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()
