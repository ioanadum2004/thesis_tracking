#!/usr/bin/env python3
"""
Build tracks, evaluate performance, and plot metrics in one step.

This script combines the track building, evaluation, and plotting stages:
1. Builds candidate tracks from inferred graphs (score cut + connected components)
2. Evaluates tracks against truth particles (efficiency, fake rate, clone rate)
3. Plots efficiency and clone rate vs pT and eta

Usage:
    python track_build_and_evaluate.py <dataset>

Examples:
    python track_build_and_evaluate.py testset
    python track_build_and_evaluate.py valset
    python track_build_and_evaluate.py testset --skip-build    # Re-evaluate without rebuilding
    python track_build_and_evaluate.py testset --config acorn_configs/track_build_and_evaluate.yaml

Config:
    Default config: acorn_configs/track_build_and_evaluate.yaml
    Contains both track building and evaluation parameters in one file
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import json
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))
sys.path.insert(0, str(PIPELINE_ROOT))

from acorn.core.infer_stage import infer
from acorn.stages.track_building.utils import get_matching_df, calculate_matching_fraction
from acorn.stages.track_building.track_building_stage import make_result_summary, GraphDataset
from tqdm import tqdm

# Import evaluation and plotting utilities from shared module
from low_pt_custom_utils.track_evaluation_utils import (
    run_evaluation,
    save_evaluation_results,
    run_plotting,
)


# ─── Track Building ─────────────────────────────────────────────────────────

def run_track_building(build_config_file):
    """Run track building using ACORN's infer stage."""
    with open(build_config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Make paths absolute relative to pipeline root directory
    for path_key in ['input_dir', 'stage_dir', 'output_dir']:
        if path_key in config and not Path(config[path_key]).is_absolute():
            config[path_key] = str(PIPELINE_ROOT / config[path_key])

    # Save the updated config temporarily
    temp_config_file = SCRIPT_DIR / '.temp_track_build_config.yaml'
    with open(temp_config_file, 'w') as f:
        yaml.safe_dump(config, f)

    input_dir = Path(config['input_dir'])

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Please run infer_gnn.py first to generate graphs with edge_scores."
        )

    print(f"Track building method: {config.get('model', 'ConnectedComponents')}")
    print(f"Score cut: {config.get('score_cut', 0.5)}")
    print(f"Output: {config['stage_dir']}")
    print()

    # Use the temporary config file with absolute paths
    infer(str(temp_config_file), verbose=False, checkpoint=None)

    # Clean up temporary config file
    temp_config_file.unlink()

    print(f"Tracks saved to: {config['stage_dir']}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build tracks, evaluate performance, and plot metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python track_build_and_evaluate.py testset
    # Full pipeline: build + evaluate + plot for testset

  python track_build_and_evaluate.py valset --skip-build
    # Skip track building, only re-evaluate and plot

  python track_build_and_evaluate.py testset --config acorn_configs/track_building_stage_(3)/track_build_and_evaluate.yaml
    # Use custom evaluation config
        """
    )
    parser.add_argument(
        'dataset',
        type=str,
        choices=['trainset', 'valset', 'testset'],
        help='Dataset to evaluate: trainset, valset, or testset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to combined config file (default: acorn_configs/track_building_stage_(3)/track_build_and_evaluate.yaml)'
    )
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip track building step (use existing tracks in data/track_building/)'
    )

    args = parser.parse_args()

    # Load combined config
    if args.config is None:
        config_file = PIPELINE_ROOT / 'acorn_configs' / 'track_building_stage_(3)' / 'track_build_and_evaluate.yaml'
    else:
        config_file = Path(args.config)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        eval_config = yaml.safe_load(f)
    eval_config['dataset'] = args.dataset

    # Output directories
    eval_output_dir = PIPELINE_ROOT / 'data' / 'track_evaluation' / args.dataset
    plot_output_dir = PIPELINE_ROOT / 'data' / 'visuals' / 'track_metrics' / args.dataset

    # ── Step 1: Track Building ────────────────────────────────────────────
    if not args.skip_build:
        print("=" * 70)
        print("STEP 1: TRACK BUILDING")
        print("=" * 70)
        print()

        # Use the same config file for track building
        run_track_building(config_file)
        print()
    else:
        print("Skipping track building (--skip-build)")
        print()

    # ── Step 2: Evaluation ────────────────────────────────────────────────
    print("=" * 70)
    print(f"STEP 2: EVALUATING {args.dataset.upper()}")
    print("=" * 70)
    print()

    # Override input_dir to read from track_building output
    eval_config['input_dir'] = str(PIPELINE_ROOT / 'data' / 'track_building')

    evaluated_events, summary, summary_text = run_evaluation(args.dataset, eval_config)
    save_evaluation_results(evaluated_events, summary, summary_text, args.dataset, eval_output_dir)

    print()
    print(summary_text)

    # ── Step 3: Plotting ──────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 3: PLOTTING METRICS")
    print("=" * 70)
    print()

    run_plotting(evaluated_events, summary, args.dataset, plot_output_dir, eval_config.get('plots', {}))

    # ── Done ──────────────────────────────────────────────────────────────
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
