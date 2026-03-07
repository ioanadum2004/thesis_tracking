#!/usr/bin/env python3
"""
Train MLP Segment Matcher

Trains a small MLP to score helix segment pairs as match / no-match.
Expects pre-mined pair tensors produced by mlp_data_mining.py.

Loss:   Weighted BCE (pos_weight derived from data class ratio)
Opt:    Adam with ReduceLROnPlateau scheduler
Stop:   Early stopping on validation loss (configurable patience)
Save:   Best model (lowest val_loss) saved with val_loss in filename.

Usage:
    python train_mlp_segment_matcher.py
    python train_mlp_segment_matcher.py --config path/to/config.yaml
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "acorn"))
sys.path.insert(0, str(PIPELINE_ROOT))

from low_pt_custom_utils.mlp_segment_matching import SegmentPairMLP


# ─── Training Loop ──────────────────────────────────────────────────────────


def run_training(config):
    """Load pre-mined data and run the training loop."""

    mined_data_dir = Path(config.get("mined_data_dir", "data/track_building/MLP_segments"))
    if not mined_data_dir.is_absolute():
        mined_data_dir = PIPELINE_ROOT / mined_data_dir

    model_save_base = Path(config.get("model_save_path", "data/track_building/MLP_checkpoints/mlp_segment_matcher.pt"))
    if not model_save_base.is_absolute():
        model_save_base = PIPELINE_ROOT / model_save_base
    model_save_base.parent.mkdir(parents=True, exist_ok=True)

    mlp_config = config.get("mlp", {})
    hidden_dims = mlp_config.get("hidden_dims", [64, 64])
    dropout = mlp_config.get("dropout", 0.1)

    lr = config.get("lr", 1e-3)
    max_epochs = config.get("max_epochs", 100)
    patience = config.get("patience", 10)
    batch_size = config.get("batch_size", 512)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load pre-mined tensors ────────────────────────────────────────────────
    print(f"\nLoading mined pairs from {mined_data_dir}")
    train_data = torch.load(mined_data_dir / "trainset.pt", weights_only=True)
    val_data   = torch.load(mined_data_dir / "valset.pt",   weights_only=True)

    X_train, y_train = train_data["X"], train_data["y"]
    X_val,   y_val   = val_data["X"],   val_data["y"]
    feature_scales = train_data.get("feature_scales")

    data_split = config.get("data_split", [None, None])
    n_train_max = data_split[0] if data_split and data_split[0] else None
    n_val_max   = data_split[1] if data_split and len(data_split) > 1 and data_split[1] else None

    if n_train_max is not None:
        X_train, y_train = X_train[:n_train_max], y_train[:n_train_max]
    if n_val_max is not None:
        X_val, y_val = X_val[:n_val_max], y_val[:n_val_max]

    n_pos = int(y_train.sum().item())
    n_neg = int((1 - y_train).sum().item())
    pos_weight_val = float(n_neg) / max(n_pos, 1)
    print(f"  trainset: {n_pos} positives, {n_neg} negatives  (pos_weight={pos_weight_val:.2f})")
    print(f"  valset:   {int(y_val.sum())} positives, {int((1-y_val).sum())} negatives")

    # Shuffle training set
    perm = torch.randperm(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size * 4,
        shuffle=False,
    )

    # ── Model, loss, optimizer ───────────────────────────────────────────────
    model = SegmentPairMLP(hidden_dims=hidden_dims, dropout=dropout).to(device)
    pos_weight = torch.tensor([pos_weight_val], device=device)          # adjust for class inbalance

    def weighted_bce(pred, target):
        weight = torch.where(target > 0.5, pos_weight, torch.ones_like(pos_weight))
        loss = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
        return (loss * weight).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: hidden_dims={hidden_dims}, dropout={dropout}, params={n_params:,}")
    print(f"Training: lr={lr}, batch_size={batch_size}, max_epochs={max_epochs}, patience={patience}")
    print(f"Saving to: {model_save_base.parent}/<stem>_val_loss-<val>.pt\n")
    print(f"{'Epoch':>6}  {'train_loss':>12}  {'val_loss':>12}  {'lr':>10}  status")
    print("-" * 65)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    current_save_path = None

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = weighted_bce(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch)
                loss = weighted_bce(pred, y_batch)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Save best model (val_loss in filename; delete previous best)
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            new_save_path = model_save_base.parent / (
                f"{model_save_base.stem}_val_loss-{val_loss:.6f}{model_save_base.suffix}"
            )
            torch.save({
                "state_dict": model.state_dict(),
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "feature_scales": feature_scales,
            }, new_save_path)
            if current_save_path is not None and current_save_path.exists():
                current_save_path.unlink()
            current_save_path = new_save_path
            patience_counter = 0
            status = f"* saved ({new_save_path.name})"
        else:
            patience_counter += 1

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {current_lr:>10.2e}  {status}")

        if patience_counter >= patience:
            print(f"\nEarly stopping: val_loss did not improve for {patience} epochs.")
            break

    print(f"\nTraining complete.")
    print(f"  Best val_loss: {best_val_loss:.6f}")
    print(f"  Model saved:   {current_save_path}")


# ─── Entry Point ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP segment matcher from pre-mined pair tensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: acorn_configs/track_building_stage_(3)/mlp_segment_matching_train.yaml)",
    )
    args = parser.parse_args()

    if args.config is None:
        config_file = (
            PIPELINE_ROOT
            / "acorn_configs"
            / "track_building_stage_(3)"
            / "mlp_segment_matching_train.yaml"
        )
    else:
        config_file = Path(args.config)

    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("MLP SEGMENT MATCHER — TRAINING")
    print("=" * 65)
    print(f"Config: {config_file}")
    print()

    run_training(config)


if __name__ == "__main__":
    main()
