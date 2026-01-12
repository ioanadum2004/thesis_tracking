#!/usr/bin/env python3
"""
Convert ACTS CSV files to PyG format using ActsReader

This script takes the raw ACTS simulation CSV files and converts them to
PyG graph format with train/val/test splits. The resulting graphs contain:
- Hit coordinates (r, φ, z) 
- Particle IDs
- No edges (those are built later by metric learning or simple graphs)

Usage:
    python convert_csv_to_pyg_sets.py
    
Configuration:
    acorn_configs/convert_csv_to_pyg_sets.yaml
"""

import sys
from pathlib import Path
import yaml

# Add acorn to path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))

from acorn.stages.data_reading.models.acts_reader import ActsReader


def main():
    """Convert ACTS CSV files to PyG graphs using ActsReader"""
    
    print("="*80)
    print("CONVERT ACTS CSV TO PYG FORMAT")
    print("="*80)
    print()
    
    # Load configuration
    config_path = SCRIPT_DIR / 'acorn_configs' / 'convert_csv_to_pyg_sets.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration: {config_path.name}")
    print(f"Input:  {config['input_dir']}")
    print(f"Output: {config['stage_dir']}")
    print(f"Data split: {config['data_split']}")
    print()
    
    # Create reader and convert
    print("Starting conversion...")
    reader = ActsReader.infer(config)
    
    print()
    print("="*80)
    print("✓ CSV TO PYG CONVERSION COMPLETE!")
    print("="*80)
    print(f"\nPyG graphs saved to: {config['stage_dir']}")
    print(f"  - trainset: {config['data_split'][0]} events")
    print(f"  - valset:   {config['data_split'][1]} events")
    print(f"  - testset:  {config['data_split'][2]} events")
    print()
    print("Next step:")
    print("  python train_latent_cluster_learning.py")
    print()


if __name__ == "__main__":
    main()
