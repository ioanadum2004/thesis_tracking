#!/usr/bin/env python3
"""
Train metric learning model with Supervised Contrastive (SupCon) loss.

Replaces the pairwise hinge loss in train_latent_cluster_learning.py with SupCon,
which uses ALL positive pairs per anchor simultaneously against ALL negatives in
the batch — no dead zone, no margin to collapse.

Key differences from train_latent_cluster_learning.py:
  - Loss: SupCon with temperature τ instead of hinge loss with margin
  - Training step: sample hits directly, no hard-negative edge building
  - No randomisation / r_train / knn (training) config params needed
  - Validation metrics (efficiency, purity, F1 via KNN) are unchanged

Usage:
    python train_supcon_metric_learning.py
    python train_supcon_metric_learning.py --config path/to/config.yaml

Configuration:
    acorn_configs/latent_stage_(1)/supcon_cluster_learning_train.yaml
"""

import sys
import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import CSVLogger, WandbLogger

torch.set_float32_matmul_precision('medium')

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))

from acorn.stages.graph_construction.models.metric_learning import MetricLearning
import torch.optim.lr_scheduler as lr_scheduler

sys.path.insert(0, str(PIPELINE_ROOT))
from low_pt_custom_utils.graph_utils import patch_acorn_build_edges, build_edges
patch_acorn_build_edges()


class SupConMetricLearning(MetricLearning):
    """
    MetricLearning with Supervised Contrastive (SupCon) loss.

    The hinge loss trains on sampled pairs one at a time. SupCon instead
    treats each event as a classification problem: given an anchor hit,
    can the model rank all same-particle hits above all other-particle hits
    in the batch? Every forward pass sees all positives and all negatives
    simultaneously.

    Loss formula for anchor u with positives P(u):
        L(u) = -1/|P(u)| * Σ_{v+∈P(u)} log[ exp(u·v+/τ) / Σ_j exp(u·vj/τ) ]

    τ (temperature) controls cluster tightness: smaller = tighter.
    Noise hits (particle_id == 0) are excluded from the anchor set but
    included in the denominator as negatives.
    """

    def supcon_loss(self, embeddings, labels, temperature):
        """
        Supervised Contrastive loss on L2-normalised embeddings.

        Args:
            embeddings: (N, D) unit-sphere embeddings from self.forward()
            labels:     (N,)  particle IDs — 0 = noise (excluded as anchors)
            temperature: τ scalar

        Returns:
            scalar loss
        """
        N = embeddings.shape[0]
        device = embeddings.device

        # --- Similarity matrix (N, N) ---
        sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

        # Numerical stability: subtract row-wise max before exp
        # (does not change the softmax output, prevents overflow)
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

        # --- Positive mask: same particle, excluding self ---
        labels_col = labels.unsqueeze(1)   # (N, 1)
        labels_row = labels.unsqueeze(0)   # (1, N)
        pos_mask = (labels_col == labels_row).float()   # (N, N)
        pos_mask.fill_diagonal_(0)

        # --- Denominator: sum over all non-self pairs ---
        exp_sim = torch.exp(sim_matrix)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        exp_sim = exp_sim.masked_fill(self_mask, 0)
        denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)   # (N, 1)

        # --- Log-probability of each pair ---
        log_prob = sim_matrix - torch.log(denom)   # (N, N)

        # --- Mean log-prob over positives per anchor ---
        n_positives = pos_mask.sum(dim=1)   # (N,)

        # Valid anchors: not noise AND has at least one same-particle hit in this batch
        valid = (labels != 0) & (n_positives > 0)

        if valid.sum() == 0:
            # Can happen if the sampled batch contains only noise or single-hit particles
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / n_positives.clamp(min=1)
        loss = -mean_log_prob_pos[valid].mean()

        return loss

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        input_data = self.get_input_data(batch)
        n_hits = input_data.shape[0]

        # Random sample of points_per_batch hits per event
        sample_size = min(self.hparams["points_per_batch"], n_hits)
        sample_idx = torch.randperm(n_hits, device=self.device)[:sample_size]

        # Embed sampled hits onto unit sphere
        embedding = self(input_data[sample_idx])

        # Particle labels for sampled hits
        labels = batch.hit_particle_id[sample_idx].long()

        temperature = self.hparams.get("temperature", 0.1)
        loss = self.supcon_loss(embedding, labels, temperature)

        self.log("train_loss", loss, batch_size=1)
        return loss

    # ------------------------------------------------------------------
    # Validation / shared evaluation
    # Keeps KNN-based efficiency/purity/F1 metrics identical to the
    # hinge-loss model. Only the logged loss value changes.
    # ------------------------------------------------------------------

    def shared_evaluation(self, batch, knn_radius, knn_num):
        embedding = self.apply_embedding(batch)

        # Build KNN graph for efficiency/purity metrics (same as parent)
        batch.edge_index = build_edges(
            query=embedding,
            database=embedding,
            indices=None,
            r_max=knn_radius,
            k_max=knn_num,
        )

        (
            batch.edge_index,
            batch.edge_y,
            batch.track_to_edge_map,
            true_edges,
        ) = self.get_truth(
            batch, batch.edge_index, self.hparams.get("undirected", False)
        )

        weights = self.get_weights(batch)

        # SupCon loss on a sampled subset of hits
        # (full event can be large — the N×N similarity matrix scales quadratically)
        n_hits = embedding.shape[0]
        sample_size = min(self.hparams["points_per_batch"], n_hits)
        sample_idx = torch.randperm(n_hits, device=self.device)[:sample_size]
        labels = batch.hit_particle_id[sample_idx].long()

        temperature = self.hparams.get("temperature", 0.1)
        loss = self.supcon_loss(embedding[sample_idx], labels, temperature)

        if hasattr(self, "trainer") and self.trainer.state.stage in ["train", "validate"]:
            self.log_metrics(
                batch, loss, batch.edge_index, true_edges, batch.edge_y, weights
            )

        return {
            "loss": loss,
            "preds": embedding,
            "truth_graph": true_edges,
        }

    def validation_step(self, batch, batch_idx):
        knn_val = self.hparams.get("knn_val", 50)
        outputs = self.shared_evaluation(batch, self.hparams["r_train"], knn_val)
        return outputs["loss"]

    # ------------------------------------------------------------------
    # Logging (prog_bar=True so W&B picks up metrics prominently)
    # ------------------------------------------------------------------

    def log_metrics(self, batch, loss, pred_edges, true_edges, truth, weights):
        from acorn.stages.graph_construction.utils import build_signal_edges

        signal_true_edges = build_signal_edges(
            batch, self.hparams["weighting"], true_edges
        )
        true_pred_edges = pred_edges[:, truth == 1]
        signal_true_pred_edges = pred_edges[:, (truth == 1) & (weights > 0)]

        total_eff = true_pred_edges.shape[1] / true_edges.shape[1]
        signal_eff = signal_true_pred_edges.shape[1] / signal_true_edges.shape[1]
        total_pur = true_pred_edges.shape[1] / pred_edges.shape[1]
        signal_pur = signal_true_pred_edges.shape[1] / pred_edges.shape[1]
        f1 = 2 * (signal_eff * signal_pur) / (signal_eff + signal_pur + 1e-8)

        current_lr = self.optimizers().param_groups[0]["lr"]

        self.log_dict(
            {
                "val_loss": loss,
                "lr": current_lr,
                "total_eff": total_eff,
                "total_pur": total_pur,
                "signal_eff": signal_eff,
                "signal_pur": signal_pur,
                "f1": f1,
            },
            batch_size=1,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )

    # ------------------------------------------------------------------
    # Dataloader: shuffle=True for better training
    # ------------------------------------------------------------------

    def train_dataloader(self):
        if self.trainset is None:
            return None
        num_workers = (
            16
            if "num_workers" not in self.hparams or self.hparams["num_workers"] is None
            else self.hparams["num_workers"][0]
        )
        return DataLoader(
            self.trainset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=lambda lst: lst[0],
        )

    # ------------------------------------------------------------------
    # Optimizer: ReduceLROnPlateau
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams["lr"],
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]

        scheduler_type = self.hparams.get("scheduler", "ReduceLROnPlateau")

        if scheduler_type == "ReduceLROnPlateau":
            scheduler = [
                {
                    "scheduler": lr_scheduler.ReduceLROnPlateau(
                        optimizer[0],
                        mode=self.hparams.get("metric_mode", "max"),
                        factor=self.hparams["factor"],
                        patience=self.hparams["patience"],
                        verbose=True,
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": self.hparams.get("metric_to_monitor", "f1"),
                }
            ]
        elif scheduler_type == "StepLR":
            scheduler = [
                {
                    "scheduler": lr_scheduler.StepLR(
                        optimizer[0],
                        step_size=self.hparams["patience"],
                        gamma=self.hparams["factor"],
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                }
            ]
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        return optimizer, scheduler


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------

class LossPrinterCallback(Callback):
    """Print clean one-line summary per epoch."""

    def __init__(self):
        self.train_loss = None

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'train_loss' in metrics:
            self.train_loss = metrics['train_loss'].item()

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        train_loss = self.train_loss if self.train_loss is not None else float('nan')
        val_loss = metrics.get('val_loss', torch.tensor(float('nan'))).item()
        f1 = metrics.get('f1', torch.tensor(float('nan'))).item()
        signal_eff = metrics.get('signal_eff', torch.tensor(float('nan'))).item()
        signal_pur = metrics.get('signal_pur', torch.tensor(float('nan'))).item()
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"f1={f1:.4f}, eff={signal_eff:.4f}, pur={signal_pur:.4f}"
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train SupCon metric learning model'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config file (default: acorn_configs/latent_stage_(1)/supcon_cluster_learning_train.yaml)'
    )
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = WORKSPACE_ROOT / config_path
    else:
        config_path = (
            WORKSPACE_ROOT
            / 'acorn_configs'
            / 'latent_stage_(1)'
            / 'supcon_cluster_learning_train.yaml'
        )

    print("=" * 80)
    print("SUPCON METRIC LEARNING TRAINING")
    print("=" * 80)
    print()
    print(f"Config: {config_path.relative_to(WORKSPACE_ROOT) if config_path.is_relative_to(WORKSPACE_ROOT) else config_path}")
    print()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'input_dir' in config and not Path(config['input_dir']).is_absolute():
        config['input_dir'] = str(PIPELINE_ROOT / config['input_dir'])
    if 'stage_dir' in config and not Path(config['stage_dir']).is_absolute():
        config['stage_dir'] = str(PIPELINE_ROOT / config['stage_dir'])

    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("=" * 80)
    print()

    stage_module = SupConMetricLearning(config)

    output_dir = Path(config['stage_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = config.get('accelerator', 'gpu')
    if accelerator == 'gpu' and not torch.cuda.is_available():
        print("WARNING: GPU requested but not available, falling back to CPU")
        accelerator = 'cpu'

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename=config.get('checkpoint_filename', 'supcon-{epoch:02d}-{f1:.4f}'),
        monitor=config.get('metric_to_monitor', 'f1'),
        mode=config.get('metric_mode', 'max'),
        save_top_k=config.get('save_top_k', 1),
        save_last=config.get('save_last', False),
    )

    early_stop_callback = EarlyStopping(
        monitor=config.get('metric_to_monitor', 'f1'),
        patience=config.get('early_stopping_patience', 5),
        mode=config.get('metric_mode', 'max'),
        verbose=True,
    )

    loss_printer = LossPrinterCallback()

    loggers = [CSVLogger(save_dir=config["stage_dir"], name="logs")]

    if config.get("use_wandb", False):
        wandb_kwargs = {
            "project": config.get("project", "Low_pt_supcon_MLP"),
            "entity": config.get("wandb_entity"),
            "config": config,
        }
        if "run_name" in config:
            wandb_kwargs["name"] = config["run_name"]
        loggers.append(WandbLogger(**wandb_kwargs))

    print("=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    print(f"Input features: {config['node_features']}")
    print(f"Hidden layers:  {config['nb_layer']} x {config['emb_hidden']} neurons")
    print(f"Embedding dim:  {config['emb_dim']}D on unit sphere")
    print(f"Activation:     {config['activation']}")
    print("=" * 80)
    print("LOSS")
    print("=" * 80)
    print(f"Loss:           SupCon (temperature τ = {config.get('temperature', 0.1)})")
    print(f"Points/batch:   {config['points_per_batch']} hits sampled per event")
    print(f"Validation KNN: r={config['r_train']}, k={config.get('knn_val', 50)}")
    print("=" * 80)
    print()

    trainer = Trainer(
        max_epochs=config['max_epochs'],
        accelerator=accelerator,
        devices=config.get('devices', 1),
        callbacks=[checkpoint_callback, early_stop_callback, loss_printer],
        logger=loggers,
        log_every_n_steps=config.get('wandb_log_every_n_batches', 50),
        check_val_every_n_epoch=config.get('check_val_every_n_epoch', 1),
        val_check_interval=config.get('val_check_interval'),
        enable_progress_bar=True,
        enable_model_summary=True,
        precision=config.get('precision', '32'),
    )

    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()
    trainer.fit(stage_module)

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Best {config['metric_to_monitor']}: {checkpoint_callback.best_model_score:.4f}")
    print()
    print("Next steps:")
    print(f"  python build_latent_graphs.py {Path(checkpoint_callback.best_model_path).stem}")
    print()


if __name__ == "__main__":
    main()
