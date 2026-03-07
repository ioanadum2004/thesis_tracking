#!/usr/bin/env python3
"""
Build candidate tracks from graphs with classified edges.

This script:
1. Loads graphs with edge_scores (from infer_gnn.py)
2. Applies score cut to filter edges
3. Builds tracks using connected components or other algorithms
4. Saves tracks as CSV/TXT files

Usage:
    python build_tracks.py [--config CONFIG_FILE]
    
Configuration:
    acorn_configs/track_building_stage_(3)/track_build_and_evaluate.yaml (or specify with --config)
"""

import argparse
import sys
from pathlib import Path
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))
sys.path.insert(0, str(PIPELINE_ROOT))

from acorn.core.infer_stage import infer


def main():
    parser = argparse.ArgumentParser(
        description='Build tracks from graphs with edge scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_tracks.py                    # Use default config
  python build_tracks.py --config acorn_configs/track_building_stage_(3)/track_build_and_evaluate.yaml
  
Note: Make sure to run infer_gnn.py first to generate edge_scores on graphs.
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to track building config file (default: acorn_configs/track_building_stage_(3)/track_build_and_evaluate.yaml)'
    )
    args = parser.parse_args()
    
    # Default config path
    if args.config is None:
        config_file = PIPELINE_ROOT / 'acorn_configs' / 'track_building_stage_(3)' / 'track_build_and_evaluate.yaml'
    else:
        config_file = Path(args.config)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    print("="*70)
    print("TRACK BUILDING")
    print("="*70)
    print(f"Loading config from: {config_file}\n")
    
    # Load and display config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Make paths absolute relative to pipeline root directory
    for path_key in ['input_dir', 'stage_dir', 'output_dir']:
        if path_key in config and not Path(config[path_key]).is_absolute():
            config[path_key] = str(PIPELINE_ROOT / config[path_key])

    print("="*70)
    print("TRACK BUILDING CONFIGURATION")
    print("="*70)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("="*70)
    print()

    # Verify input directory exists
    input_dir = Path(config['input_dir'])
    
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Please run infer_gnn.py first to generate graphs with edge_scores."
        )
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {config['stage_dir']}")
    print(f"Track building method: {config.get('model', 'ConnectedComponents')}")
    print(f"Score cut: {config.get('score_cut', 0.5)}")
    print()
    
    # Run track building using acorn's infer_stage
    print("="*70)
    print("BUILDING TRACKS")
    print("="*70)
    print()
    
    infer(str(config_file), verbose=False, checkpoint=None)
    
    print("\n" + "="*70)
    print("TRACK BUILDING COMPLETE!")
    print("="*70)
    print(f"Tracks saved to: {config['stage_dir']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
