#!/usr/bin/env python3
"""
MLP Segment Pair Data Mining

Mines positive/negative segment pairs from GNN-stage graphs and saves them to
disk as pre-built tensors. 

For each event:
  1. Extract segments via CC clustering on GNN edge scores.
  2. Fit Kasa circle + pitch to each segment.
  3. Mine positive pairs (same majority hit_particle_id) and negative pairs
     (different particle), subsampled to neg_ratio × n_pos.
  4. Accumulate features and labels.


Usage:
    python mlp_data_mining.py
    python mlp_data_mining.py --config path/to/config.yaml
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


# ─── Per-split mining ────────────────────────────────────────────────────────

def mine_split(input_paths, config, split_name: str, rng) -> dict:
    """
    Mine segment pairs for one data split.

    Returns a dict {"X": tensor (N,19), "y": tensor (N,1)} ready to save.
    """
    score_cut = config.get("score_cut", 0.85)
    B_field = config.get("B_field", 2.0)
    outlier_rejection = config.get("outlier_rejection", False)
    neg_ratio = config.get("neg_ratio", 5)
    feature_scales = config.get("feature_scales")

    all_features = []
    all_labels = []
    n_pos_total = 0
    n_neg_total = 0

    for event_path in tqdm(input_paths, desc=f"  Mining {split_name}"):
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
        raise RuntimeError(f"No pairs found for {split_name}. Check input data.")

    print(f"  {split_name}: {n_pos_total} positives, {n_neg_total} negatives "
          f"({len(all_features)} total pairs)")

    X = torch.tensor(np.array(all_features), dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)
    return {"X": X, "y": y, "feature_scales": feature_scales}


# ─── Main ────────────────────────────────────────────────────────────────────


def run_mining(config):
    input_dir = Path(config.get("input_dir", "data/gnn_stage"))
    if not input_dir.is_absolute():
        input_dir = PIPELINE_ROOT / input_dir

    output_dir = Path(config.get("output_dir", "data/track_building/MLP_segments"))
    if not output_dir.is_absolute():
        output_dir = PIPELINE_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_split = config.get("data_split")
    n_train, n_val, n_test = data_split[0], data_split[1], data_split[2] if len(data_split) > 2 else 0

    rng = np.random.default_rng(42)

    splits = [("trainset", n_train), ("valset", n_val)]
    if n_test > 0:
        splits.append(("testset", n_test))

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    t0 = perf_counter()
    for split_name, n_events in splits:
        if n_events == 0:
            print(f"  Skipping {split_name} (n=0)")
            continue

        paths = load_datafiles_in_dir(str(input_dir), split_name, n_events)
        paths.sort()
        print(f"{split_name}: {len(paths)} events")

        data = mine_split(paths, config, split_name, rng)

        out_path = output_dir / f"{split_name}.pt"
        torch.save(data, out_path)
        print(f"  Saved → {out_path}  (shape X={tuple(data['X'].shape)}, y={tuple(data['y'].shape)})\n")

    print(f"Mining complete in {perf_counter() - t0:.1f}s.")


def main():
    parser = argparse.ArgumentParser(
        description="Mine MLP segment pair training data from GNN graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: acorn_configs/track_building_stage_(3)/mlp_data_mining.yaml)",
    )
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

    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("MLP SEGMENT PAIR DATA MINING")
    print("=" * 65)
    print(f"Config: {config_file}")
    print()

    run_mining(config)


if __name__ == "__main__":
    main()
