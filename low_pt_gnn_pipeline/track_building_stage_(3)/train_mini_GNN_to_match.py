#!/usr/bin/env python3
"""
Train Mini-GNN Segment Embedder

Trains a small GNN to embed detector hit segments into a latent space where
segments from the same particle have high cosine similarity and segments from
different particles are pushed apart.

Requires mined segment data (run mini_gnn_data_mining.py first).

Usage:              
    python train_mini_GNN_to_match.py           
    python train_mini_GNN_to_match.py --config path/to/config.yaml
"""

import argparse
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "acorn"))
sys.path.insert(0, str(PIPELINE_ROOT))

from low_pt_custom_utils.mini_gnn_segment_embedding import (
    SegmentGNN,
    supcon_loss,
    NODE_FEATURE_DIM,
)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_precomputed(precomputed_dir, split, n_events):
    """
    Load precomputed segment data from a consolidated .pt file.

    Expects <precomputed_dir>/<split>.pt produced by combine_mini_gnn_chunks.py.

    Args:
        precomputed_dir: Path to precomputed_segments/ directory.
        split:           "trainset" or "valset".
        n_events:        Max number of events to load.

    Returns:
        List of (seg_data_list, particle_ids) tuples.
    """
    precomputed_dir = Path(precomputed_dir)
    consolidated_file = precomputed_dir / f"{split}.pt"

    if not consolidated_file.exists():
        raise FileNotFoundError(
            f"Consolidated data not found: {consolidated_file}\n"
            f"Run combine_mini_gnn_chunks.py first."
        )

    print(f"  Loading {split} from {consolidated_file.name}...")
    cached = torch.load(consolidated_file, map_location="cpu", weights_only=False)
    results = cached["results"][:n_events]

    total_segs = sum(len(segs) for segs, _ in results)
    print(f"  {split}: {len(results)} events, {total_segs} segments "
          f"(avg {total_segs / max(1, len(results)):.1f}/event)")
    return results


def run_epoch(model, precomputed, config, optimizer, device, train: bool, epoch: int = 0):
    """
    Run one full pass (train or validate) over precomputed segment data.

    Args:
        model:        SegmentGNN model.
        precomputed:  List of (seg_data_list, particle_ids) from precompute_segments().
        config:       Training config dict.
        optimizer:    Adam optimizer 
        device:       Torch device string.
        train:        Whether to update model weights.

    Returns:
        mean_loss: Average per-batch contrastive loss.
    """
    temperature = config.get("temperature", 0.1)
    batch_events = config.get("batch_events", 1)

    model.train(train)
    losses = []

    # Filter to events with ≥2 segments
    valid_indices = [i for i, (segs, _) in enumerate(precomputed) if len(segs) >= 2]

    if train:
        random.shuffle(valid_indices)

    # Chunk into multi-event batches
    batches = [valid_indices[i:i + batch_events]
               for i in range(0, len(valid_indices), batch_events)]

    phase = "train" if train else "val"
    pbar = tqdm(batches, desc=f"Epoch {epoch:>3} {phase}", unit="batch", leave=False)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx_list in pbar:
            # Gather segments from all events in this batch.
            # Offset particle IDs per event so that pid=7 in event A
            # doesn't collide with pid=7 in event B (they're different particles).
            all_seg_data = []
            all_pids = []
            pid_offset = 0
            for idx in batch_idx_list:
                seg_data_list, particle_ids = precomputed[idx]
                all_seg_data.extend(seg_data_list)
                for pid in particle_ids:
                    all_pids.append(pid + pid_offset if pid > 0 else 0)
                if particle_ids:
                    pid_offset += max(particle_ids) + 1

            if len(all_seg_data) < 2:
                continue

            batch_data = Batch.from_data_list(all_seg_data).to(device)
            pid_tensor = torch.tensor(all_pids, dtype=torch.long, device=device)

            if train:
                optimizer.zero_grad()

            embeddings = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = supcon_loss(embeddings, pid_tensor, temperature=temperature)

            if train and loss.requires_grad:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

    return float(np.mean(losses)) if losses else 0.0


# ─── Training Entry Point ────────────────────────────────────────────────────


def run_training(config):
    """Full training loop: load mined data → build model → train → validate → save."""

    model_save_dir = Path(config.get("model_save_dir", "saved_models"))
    if not model_save_dir.is_absolute():
        model_save_dir = PIPELINE_ROOT / model_save_dir
    model_save_dir.mkdir(parents=True, exist_ok=True)

    data_split = config.get("data_split", [500, 100, 0])
    n_train = data_split[0]
    n_val = data_split[1]

    gnn_config = config.get("gnn", {})
    hidden_dim = gnn_config.get("hidden_dim", 64)
    emb_dim = gnn_config.get("emb_dim", 32)
    n_layers = gnn_config.get("n_layers", 3)
    dropout = gnn_config.get("dropout", 0.0)
    proj_dim = gnn_config.get("proj_dim", None)
    proj_layers = gnn_config.get("proj_layers", 1)

    lr = config.get("lr", 1e-3)
    max_epochs = config.get("max_epochs", 100)
    patience = config.get("patience", 10)
    scheduler_patience = config.get("scheduler_patience", 5)
    scheduler_factor = config.get("scheduler_factor", 0.5)
    val_check_interval = config.get("val_check_interval", 1.0)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load precomputed segment data (from mini_gnn_data_mining.py) ────────
    precomputed_dir = Path(config.get("precomputed_dir", "data/track_building/mini_gnn_segments"))
    if not precomputed_dir.is_absolute():
        precomputed_dir = PIPELINE_ROOT / precomputed_dir

    print(f"\nLoading mined segments from {precomputed_dir}")
    t0 = perf_counter()
    train_precomputed = load_precomputed(precomputed_dir, "trainset", n_train)
    val_precomputed = load_precomputed(precomputed_dir, "valset", n_val)
    print(f"  Load time: {perf_counter() - t0:.1f}s")

    # ── Build model ──────────────────────────────────────────────────────────
    model = SegmentGNN(
        node_in_dim=NODE_FEATURE_DIM,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        n_layers=n_layers,
        dropout=dropout,
        proj_dim=proj_dim,
        proj_layers=proj_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: hidden_dim={hidden_dim}, emb_dim={emb_dim}, n_layers={n_layers}, "
          f"proj_dim={proj_dim or hidden_dim}, proj_layers={proj_layers}, "
          f"dropout={dropout}, params={n_params:,}")
    batch_events = config.get("batch_events", 1)
    n_valid_train = sum(1 for segs, _ in train_precomputed if len(segs) >= 2)
    n_steps_per_epoch = max(1, (n_valid_train + batch_events - 1) // batch_events)

    print(f"Loss:  SupCon (temperature τ={config.get('temperature', 0.1)})")
    print(f"Training: lr={lr}, max_epochs={max_epochs}, patience={patience}")
    print(f"  Batch: {batch_events} events/step → ~{n_steps_per_epoch} optimizer steps/epoch")
    print(f"  val_check_interval: {val_check_interval} "
          f"({int(1 / val_check_interval)} val checks per epoch, "
          f"every {int(n_steps_per_epoch * val_check_interval)} steps)")
    print(f"Saving best model to: {model_save_dir}/")
    print()
    print(f"{'Step':>8}  {'train_loss':>12}  {'val_loss':>12}  {'lr':>10}  {'status'}")
    print("-" * 70)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor, verbose=False
    )

    best_val_loss = float("inf")
    patience_counter = 0
    model_save_path = None

    # ── Training loop ────────────────────────────────────────────────────────
    import math
    # For intra-epoch validation, split precomputed data into chunks
    valid_train_indices = [i for i, (segs, _) in enumerate(train_precomputed) if len(segs) >= 2]
    chunk_size_events = max(1, math.ceil(len(valid_train_indices) * val_check_interval))
    early_stopped = False

    for epoch in range(1, max_epochs + 1):
        # Shuffle which events go into which chunk (run_epoch also shuffles within)
        random.shuffle(valid_train_indices)

        # Split into chunks for intra-epoch validation
        chunks = [valid_train_indices[i:i + chunk_size_events]
                  for i in range(0, len(valid_train_indices), chunk_size_events)]

        for chunk_idx, chunk_indices in enumerate(chunks):
            frac = (chunk_idx + 1) / len(chunks)
            step_label = f"{epoch - 1 + frac:.2f}" if len(chunks) > 1 else f"{epoch}"

            # Build chunk-specific precomputed sublist
            chunk_precomputed = [train_precomputed[i] for i in chunk_indices]

            train_loss = run_epoch(model, chunk_precomputed, config, optimizer, device, train=True, epoch=epoch)
            val_loss = run_epoch(model, val_precomputed, config, optimizer, device, train=False, epoch=epoch)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            status = ""
            if val_loss < best_val_loss:
                # Remove previous best checkpoint
                if model_save_path is not None and model_save_path.exists():
                    model_save_path.unlink()

                best_val_loss = val_loss
                model_save_path = model_save_dir / f"mini_gnn_segment_embedder_val_loss={best_val_loss:.4f}.pt"
                torch.save({
                    "state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim,
                    "emb_dim": emb_dim,
                    "n_layers": n_layers,
                    "dropout": dropout,
                    "proj_dim": proj_dim,
                    "proj_layers": proj_layers,
                    "node_scales": gnn_config.get("node_scales", [1000.0, 1000.0, 500.0, 1000.0]),
                }, model_save_path)
                patience_counter = 0
                status = f"* saved → {model_save_path.name}"
            else:
                patience_counter += 1

            print(f"{step_label:>8}  {train_loss:>12.6f}  {val_loss:>12.6f}  {current_lr:>10.2e}  {status}")

            if patience_counter >= patience:
                print(f"\nEarly stopping: val_loss did not improve for {patience} val checks.")
                early_stopped = True
                break

        if early_stopped:
            break

    print(f"\nTraining complete.")
    print(f"  Best val_loss: {best_val_loss:.6f}")
    print(f"  Model saved:   {model_save_path}")
    print()
    print("Next step: use the trained embeddings as an additional cosine similarity")
    print("feature in the MLP segment matcher (mlp_segment_matching.py).")


# ─── Entry Point ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train mini-GNN for cosine-contrastive segment embedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to config YAML "
            "(default: acorn_configs/track_building_stage_(3)/mini_gnn_segment_matching_train.yaml)"
        ),
    )
    args = parser.parse_args()

    if args.config is None:
        config_file = (
            PIPELINE_ROOT
            / "acorn_configs"
            / "track_building_stage_(3)"
            / "mini_gnn_segment_matching_train.yaml"
        )
    else:
        config_file = Path(args.config)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("MINI-GNN SEGMENT EMBEDDER — TRAINING")
    print("=" * 65)
    print(f"Config: {config_file}")
    print()

    run_training(config)


if __name__ == "__main__":
    main()
