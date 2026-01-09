
import os
import sys
import yaml
from pathlib import Path
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))

from acorn.core.core_utils import get_stage_module
from acorn.utils.loading_utils import add_variable_name_prefix_in_config
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN


class LossPrinterCallback(Callback):
    
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Get the training loss from the logged metrics
        if 'train_loss' in trainer.callback_metrics:
            loss = trainer.callback_metrics['train_loss'].item()
            self.train_losses.append(loss)
            
            # rapport GPU memory usage for finsing optimal settings 
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
            print(f"{'='*70}\n")
    
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
    
    # Enable Tensor Cores for faster matrix operations on L40S GPU if available.    (GPU optimization)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # or 'high' for even more speed

    SCRIPT_DIR = Path(__file__).resolve().parent
    config_file = SCRIPT_DIR / 'acorn_configs' / 'minimal_gnn_train.yaml'
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    print(f"Loading config from: {config_file}\n")
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if not config.get("variable_with_prefix"):
        config = add_variable_name_prefix_in_config(config)
    
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(yaml.dump(config))
    print("="*70)
    

    stage_module_class = InteractionGNN
    
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
    
    logger = CSVLogger(save_dir=config["stage_dir"], name="logs")
    
    trainer = Trainer(
        accelerator=config.get("accelerator", "cpu"),
        devices=config.get("devices", 1),
        num_nodes=config.get("nodes", 1),
        max_epochs=config["max_epochs"],
        callbacks=[loss_printer],
        logger=logger,
        log_every_n_steps=1,  # Log every batch
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Train
    trainer.fit(stage_module)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best model saved in: {config['stage_dir']}/artifacts/")
    print(f"Logs saved in: {config['stage_dir']}/logs/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


