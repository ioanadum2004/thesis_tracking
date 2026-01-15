#!/usr/bin/env python3
"""
Train metric learning model on PyG feature store

Loads PyG graphs with spacepoints (r, φ, z) and particle IDs,
trains embedding model using contrastive learning.

Usage:
    python train_latent_cluster_learning.py
    
Configuration:
    acorn_configs/latent_cluster_learning_train.yaml
"""

import sys
from pathlib import Path
import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

# Enable Tensor Cores for L40S GPU
torch.set_float32_matmul_precision('medium')

# Add acorn to path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))

from acorn.stages.graph_construction.models.metric_learning import MetricLearning


class LossPrinterCallback(Callback):
    """Print clean one-line summary per epoch"""
    
    def __init__(self):
        self.train_loss = None
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Store training loss"""
        metrics = trainer.callback_metrics
        if 'train_loss' in metrics:
            self.train_loss = metrics['train_loss'].item()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Print epoch summary"""
        metrics = trainer.callback_metrics
        
        # Extract metrics
        epoch = trainer.current_epoch
        train_loss = self.train_loss if self.train_loss is not None else float('nan')
        val_loss = metrics.get('val_loss', torch.tensor(float('nan'))).item()
        f1 = metrics.get('f1', torch.tensor(float('nan'))).item()
        signal_eff = metrics.get('signal_eff', torch.tensor(float('nan'))).item()
        signal_pur = metrics.get('signal_pur', torch.tensor(float('nan'))).item()
        
        # Print clean one-liner
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"f1={f1:.4f}, eff={signal_eff:.4f}, pur={signal_pur:.4f}")


def main():
    """Train metric learning model"""
    
    # Load configuration
    config_path = SCRIPT_DIR / 'acorn_configs' / 'latent_cluster_learning_train.yaml'
    
    print("="*80)
    print("METRIC LEARNING TRAINING - LATENT SPACE EMBEDDING")
    print("="*80)
    print()
    print(f"Loading config from: {config_path.name}")
    print()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("="*80)
    print()
    
    # Create model
    stage_module = MetricLearning(config)
    
    # Setup output directory
    output_dir = SCRIPT_DIR / config['stage_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get accelerator setting
    accelerator = config.get('accelerator', 'gpu')
    if accelerator == 'gpu' and not torch.cuda.is_available():
        print("WARNING: GPU requested but not available, falling back to CPU")
        accelerator = 'cpu'
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename=config.get('checkpoint_filename', 'metric-{epoch:02d}-{val_loss:.4f}'),
        monitor=config.get('metric_to_monitor', 'val_loss'),
        mode=config.get('metric_mode', 'min'),
        save_top_k=config.get('save_top_k', 1),
        save_last=config.get('save_last', False),
    )
    
    early_stop_callback = EarlyStopping(
        monitor=config.get('metric_to_monitor', 'val_loss'),
        patience=config.get('early_stopping_patience', 10),
        mode=config.get('metric_mode', 'min'),
        verbose=True,
    )
    
    loss_printer = LossPrinterCallback()
    
    print("="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    print(f"Input features: {config['node_features']}")
    print(f"Hidden layers: {config['nb_layer']} x {config['emb_hidden']} neurons")
    print(f"Embedding dimension: {config['emb_dim']}D latent space")
    print(f"Activation: {config['activation']}")
    print("="*80)
    print()
    
    print("="*80)
    print("TRAINING STRATEGY")
    print("="*80)
    print(f"Training approach: Contrastive learning with margin={config['margin']}")
    print(f"Points per batch: {config['points_per_batch']}")
    print(f"Training radius (normalized): {config['r_train']}")
    print(f"Max neighbors (train/val): {config['knn']}/{config['knn_val']}")
    print(f"Randomisation: {config['randomisation']}x random negative pairs")
    print("="*80)
    print()
    
    # Setup trainer
    trainer = Trainer(
        max_epochs=config['max_epochs'],
        accelerator=accelerator,
        devices=config.get('devices', 1),
        callbacks=[checkpoint_callback, early_stop_callback, loss_printer],
        log_every_n_steps=config.get('log_every_n_steps', 50),
        check_val_every_n_epoch=config.get('check_val_every_n_epoch', 1),
        enable_progress_bar=False,  # Clean output
        enable_model_summary=True,
    )
    
    # Train
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print()
    trainer.fit(stage_module)
    
    print("="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Best {config['metric_to_monitor']}: {checkpoint_callback.best_model_score:.4f}")
    print()
    print("Next steps:")
    print(f"  python build_latent_graphs.py {Path(checkpoint_callback.best_model_path).stem}")
    print()


if __name__ == "__main__":
    main()
