
import argparse
import os
import sys
import yaml
from pathlib import Path
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))
sys.path.insert(0, str(PIPELINE_ROOT))

from acorn.core.core_utils import get_stage_module
from acorn.utils.loading_utils import add_variable_name_prefix_in_config
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


class WeightedInteractionGNN(InteractionGNN):

    def train_dataloader(self):
        """Override to enable shuffling for better gradient accumulation."""
        if self.trainset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[0]
        return DataLoader(self.trainset, batch_size=1, num_workers=num_workers, shuffle=True)
    
    def training_step(self, batch, batch_idx):
        """Override to enable batch-level logging (on_step=True) for W&B."""
        output = self(batch)
        loss, pos_loss, neg_loss = self.loss_function(output, batch)

        # Scale loss for gradient accumulation (maintains same effective LR)
        accum = self.trainer.accumulate_grad_batches
        scaled_loss = loss / accum

        # Log unscaled loss so values are comparable across different accumulation settings
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log("train_pos_loss", pos_loss, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log("train_neg_loss", neg_loss, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return scaled_loss

    def loss_function(self, output, batch, balance="proportional"):
        """Override to apply pos_weight to positive loss, otherwise same as parent."""
        # Compute losses same as parent
        negative_mask = ((batch.edge_y == 0) & (batch.edge_weights != 0)) | (
            batch.edge_weights < 0
        )
        positive_mask = (batch.edge_y == 1) & (batch.edge_weights > 0)

        negative_loss = F.binary_cross_entropy_with_logits(
            output[negative_mask],
            torch.zeros_like(output[negative_mask]),
            weight=batch.edge_weights[negative_mask].abs(),
            reduction="sum",
        )

        positive_loss = F.binary_cross_entropy_with_logits(
            output[positive_mask],
            torch.ones_like(output[positive_mask]),
            weight=batch.edge_weights[positive_mask].abs(),
            reduction="sum",
        )
        
        # Apply pos_weight to positive loss if set in config
        pos_weight = self.hparams.get("pos_weight")
        positive_loss = positive_loss * pos_weight

        # Rest is identical to parent
        if balance == "proportional":
            sow = batch.edge_weights.abs().sum()
            return (
                (positive_loss + negative_loss) / sow,
                positive_loss.detach() / sow,
                negative_loss.detach() / sow,
            )
        else:
            n_pos, n_neg = positive_mask.sum(), negative_mask.sum()
            sow = (
                batch.edge_weights[positive_mask].abs().sum() / n_pos
                + batch.edge_weights[negative_mask].abs().sum() / n_neg
            )
            return (
                (positive_loss / n_pos + negative_loss / n_neg) / sow,
                positive_loss.detach() / n_pos / sow,
                negative_loss.detach() / n_neg / sow,
            )


class LossPrinterCallback(Callback):
    
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Print learning rate at the start of each epoch."""
        if trainer.optimizers:
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            print(f" Learning Rate: {current_lr:.6f}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Get the training loss from the logged metrics
        if 'train_loss' in trainer.callback_metrics:
            loss = trainer.callback_metrics['train_loss'].item()
            self.train_losses.append(loss)
            
            # Report GPU memory usage for finding optimal settings 
            # if torch.cuda.is_available():
            #     mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            #     mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            #     mem_max = torch.cuda.max_memory_allocated(0) / 1024**3
            # else:
            #     mem_allocated = mem_reserved = mem_max = 0
            
            print(f"\n{'='*70}")
            print(f"Epoch {trainer.current_epoch} Training Loss: {loss:.6f}")
            # if torch.cuda.is_available():
            #     print(f"GPU Memory: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB, Peak={mem_max:.2f}GB")
            # print(f"{'='*70}\n")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the validation loss
        if 'val_loss' in trainer.callback_metrics:
            loss = trainer.callback_metrics['val_loss'].item()
            self.val_losses.append(loss)
            
            # GPU memory usage
            # if torch.cuda.is_available():
            #     mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            #     mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            # else:
            #     mem_allocated = mem_reserved = 0
                
            print(f"\n{'='*70}")
            print(f"Epoch {trainer.current_epoch} Validation Loss: {loss:.6f}")
            # if torch.cuda.is_available():
            #     print(f"GPU Memory: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB")
            # print(f"{'='*70}\n")
    
    def on_train_end(self, trainer, pl_module):
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15}")
        print(f"{'-'*70}")
        for i, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses)):
            print(f"{i:<10} {train_loss:<15.6f} {val_loss:<15.6f}")
        print(f"{'='*70}\n")


def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train GNN edge classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_myGNN.py        # Train from scratch               
  python train_myGNN.py --resume data/gnn_stage/checkpoints/gnn_best_val_loss_val_loss=0.0026.ckpt         # Resume training from checkpoint
        """
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: acorn_configs/gnn_stage_(2)/gnn_train.yaml)'
    )
    args = parser.parse_args()
    
    # Enable Tensor Cores for faster matrix operations on L40S GPU if available.    (GPU optimization)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # or 'high' for even more speed

    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # Use provided config or default
    if args.config:
        config_file = Path(args.config)
        if not config_file.is_absolute():
            # If path starts with acorn_configs, make it relative to pipeline root
            if str(config_file).startswith('acorn_configs'):
                config_file = PIPELINE_ROOT / config_file
            else:
                config_file = SCRIPT_DIR / config_file
    else:
        config_file = PIPELINE_ROOT / 'acorn_configs' / 'gnn_stage_(2)' / 'gnn_train.yaml'
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        print(f"Resuming training from: {resume_path}\n")
    
    print(f"Loading config from: {config_file}\n")
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Make paths absolute relative to pipeline root directory
    for path_key in ['input_dir', 'stage_dir']:
        if path_key in config and not Path(config[path_key]).is_absolute():
            config[path_key] = str(PIPELINE_ROOT / config[path_key])

    if not config.get("variable_with_prefix"):
        config = add_variable_name_prefix_in_config(config)
    
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(yaml.dump(config))
    print("="*70)
    

    stage_module_class = WeightedInteractionGNN
    
    # Setup stage directory
    os.makedirs(config["stage_dir"], exist_ok=True)
    
    # Get stage module
    stage_module, ckpt_config, default_root_dir, checkpoint = get_stage_module(
        config, stage_module_class, checkpoint_path=None, checkpoint_resume_dir=None
    )
    
    if (not config.get("variable_with_prefix")) or config.get(
        "add_variable_name_prefix_in_ckpt"
    ):
        stage_module._hparams = add_variable_name_prefix_in_config(
            stage_module._hparams
        )
    
    # Create custom trainer with loss printer
    loss_printer = LossPrinterCallback()
    
    # Create checkpoint callback to save best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config["stage_dir"]) / "checkpoints",
        filename="gnn_best_val_loss_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # Only save the single best model
        save_last=False,  # Don't save last.ckpt
        verbose=True,
    )

    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.get("early_stopping_patience", 2),
        mode="min",
        verbose=True,
        strict=True,  # Crash if monitored metric is not found
    )

    # Setup loggers
    loggers = [CSVLogger(save_dir=config["stage_dir"], name="logs")]
    
    # Add W&B logger if enabled
    if config.get("use_wandb", False):
        loggers.append(WandbLogger(
            project=config.get("project", "GNN_Training"),
            entity=config.get("wandb_entity"),
            config=config,
        ))
    
    trainer = Trainer(
        accelerator=config.get("accelerator", "cpu"),
        devices=config.get("devices", 1),
        num_nodes=config.get("nodes", 1),
        max_epochs=config["max_epochs"],
        callbacks=[loss_printer, checkpoint_callback, early_stopping],
        logger=loggers,
        log_every_n_steps=config.get("wandb_log_every_n_batches", 1),  # Control logging frequency
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
        val_check_interval=config.get("val_check_interval"),  # Check validation within epochs (e.g., 0.5 = every 50%)
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Train (with optional resume)
    trainer.fit(stage_module, ckpt_path=args.resume)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")
    print(f"Checkpoints saved in: {config['stage_dir']}/checkpoints/")
    print(f"Logs saved in: {config['stage_dir']}/logs/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


