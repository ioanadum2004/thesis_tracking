#!/usr/bin/env python3
"""
Run edge classifier inference to generate edge scores on graphs.

This script:
1. Loads a trained GNN edge classifier checkpoint
2. Runs inference on graphs (trainset, valset, testset) to generate edge_scores
3. Saves graphs with edge_scores to output directory (preserving train/val/test split)

Usage:
    python infer_gnn.py [--checkpoint PATH] [--config CONFIG_FILE]
    
Configuration:
    acorn_configs/gnn_train.yaml (or specify with --config)
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
from pytorch_lightning import Trainer

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))

from acorn.core.core_utils import find_latest_checkpoint, str_to_class
from acorn.core.infer_stage import infer as acorn_infer
from acorn.utils.loading_utils import add_variable_name_prefix_in_config


def main():
    parser = argparse.ArgumentParser(
        description='Run edge classifier inference to generate edge scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_gnn.py                    # Use checkpoint from config file
  python infer_gnn.py --checkpoint data/gnn_stage/checkpoints/gnn_best_val_loss_val_loss=0.0026.ckpt  # Override config
  python infer_gnn.py --config acorn_configs/gnn_train.yaml
        """
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file (default: use checkpoint from config file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: acorn_configs/gnn_train.yaml)'
    )
    args = parser.parse_args()
    
    # Default config path (use inference config if available, otherwise training config)
    if args.config is None:
        infer_config = SCRIPT_DIR / 'acorn_configs' / 'gnn_infer.yaml'
        train_config = SCRIPT_DIR / 'acorn_configs' / 'gnn_train.yaml'
        config_file = infer_config if infer_config.exists() else train_config
    else:
        config_file = Path(args.config)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    print("="*70)
    print("EDGE CLASSIFIER INFERENCE")
    print("="*70)
    print(f"Loading config from: {config_file}\n")
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Add required stage and model fields if not present
    if "stage" not in config:
        config["stage"] = "edge_classifier"
    if "model" not in config:
        config["model"] = "InteractionGNN"
    
    if not config.get("variable_with_prefix"):
        config = add_variable_name_prefix_in_config(config)
    
    config["skip_existing"] = False
    
    # Find checkpoint: priority: command line arg > config file > auto-find (fallback)
    if args.checkpoint is not None:
        # Command line argument takes highest priority
        checkpoint_path = Path(args.checkpoint)
    elif config.get("checkpoint") is not None:
        # Use checkpoint from config file (default behavior)
        checkpoint_path = Path(config["checkpoint"])
        # If relative path, make it relative to script directory
        if not checkpoint_path.is_absolute():
            checkpoint_path = SCRIPT_DIR / checkpoint_path
    else:
        # Fallback: auto-find latest checkpoint (only if not in config)
        checkpoint_path = find_latest_checkpoint(
            config["stage_dir"],
            templates=["best*.ckpt", "*.ckpt"]
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint specified in config file and none found in {config['stage_dir']}. "
                "Please set 'checkpoint' in config file or specify --checkpoint"
            )
    
    # Verify checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print("\n" + "="*70)
    print("RUNNING INFERENCE")
    print("="*70)
    print(f"Input directory: {config['input_dir']}")
    print(f"Output directory: {config['stage_dir']}")
    print(f"Data split: {config.get('data_split', 'N/A')}")
    print(f"Checkpoint: {checkpoint_path}")
    print("="*70 + "\n")
    
    # Run inference using acorn's infer_stage
    acorn_infer(str(config_file), verbose=False, checkpoint=str(checkpoint_path))
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE!")
    print("="*70)
    print(f"Graphs with edge_scores saved to: {config['stage_dir']}")
    print("\nNext step: Run track building:")
    print(f"  python build_tracks.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
