#!/usr/bin/env python3
"""
Combine chunked MLP mining tensors into final split files.

Scans data/track_building/MLP_segments_chunk*/ for partial tensors,
concatenates them per split, saves to data/track_building/MLP_segments/,
then removes the chunk directories.

Usage:
    python combine_mlp_chunks.py
    python combine_mlp_chunks.py --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent


def main():
    parser = argparse.ArgumentParser(description="Combine chunked MLP mining tensors")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing or deleting")
    args = parser.parse_args()

    track_building_dir = PIPELINE_ROOT / "data" / "track_building"
    chunk_dirs = sorted(track_building_dir.glob("MLP_segments_chunk*"))

    if not chunk_dirs:
        print("ERROR: No chunk directories found in data/track_building/MLP_segments_chunk*/")
        sys.exit(1)

    print(f"Found {len(chunk_dirs)} chunk directories:")
    for d in chunk_dirs:
        files = list(d.glob("*.pt"))
        print(f"  {d.name}: {[f.name for f in files]}")

    output_dir = track_building_dir / "MLP_segments"
    print(f"\nOutput: {output_dir}")
    print()

    for split_name in ["trainset", "valset", "testset"]:
        chunk_files = sorted(
            f for d in chunk_dirs for f in [d / f"{split_name}.pt"] if f.exists()
        )

        if not chunk_files:
            print(f"{split_name}: no chunks found — skipping")
            continue

        print(f"{split_name}: combining {len(chunk_files)} chunks...")

        all_X = []
        all_y = []
        feature_scales = None

        for pt_file in chunk_files:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            all_X.append(data["X"])
            all_y.append(data["y"])
            if feature_scales is None:
                feature_scales = data.get("feature_scales")

        X = torch.cat(all_X, dim=0)
        y = torch.cat(all_y, dim=0)

        print(f"  Combined: X={tuple(X.shape)}, y={tuple(y.shape)}")

        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{split_name}.pt"
            torch.save({"X": X, "y": y, "feature_scales": feature_scales}, out_path)
            print(f"  Saved → {out_path}")
        else:
            print(f"  [dry-run] Would save → {output_dir / f'{split_name}.pt'}")
        print()

    if args.dry_run:
        print("[dry-run] Would remove chunk directories:")
        for d in chunk_dirs:
            print(f"  {d}")
    else:
        print("Removing chunk directories...")
        for d in chunk_dirs:
            shutil.rmtree(d)
            print(f"  Removed {d.name}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
