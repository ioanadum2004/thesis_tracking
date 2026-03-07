#!/usr/bin/env python3
"""
Combine chunked mini-GNN mining output into consolidated .pt files.

Scans data/track_building/mini_gnn_segments_chunk*/<split>/ for per-event .pyg
files, loads them all, and saves a single consolidated .pt file per split at
data/track_building/mini_gnn_segments/<split>.pt.

This avoids the 30k-file loading bottleneck that causes GPU jobs to be held
on the cluster (GPU sits idle during slow per-file loading).

Usage:
    python combine_mini_gnn_chunks.py
    python combine_mini_gnn_chunks.py --dry-run
    python combine_mini_gnn_chunks.py --keep-chunks   # don't delete chunk dirs
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent


def main():
    parser = argparse.ArgumentParser(description="Combine chunked mini-GNN mining output")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing or deleting")
    parser.add_argument("--keep-chunks", action="store_true", help="Keep chunk directories after combining")
    args = parser.parse_args()

    track_building_dir = PIPELINE_ROOT / "data" / "track_building"
    chunk_dirs = sorted(track_building_dir.glob("mini_gnn_segments_chunk*"))

    if not chunk_dirs:
        print("ERROR: No chunk directories found in data/track_building/mini_gnn_segments_chunk*/")
        sys.exit(1)

    output_dir = track_building_dir / "mini_gnn_segments"

    print(f"Found {len(chunk_dirs)} chunk directories")
    print(f"Output: {output_dir}")
    print()

    for split_name in ["trainset", "valset", "testset"]:
        # Gather all .pyg files for this split across chunks
        all_files = []
        for chunk_dir in chunk_dirs:
            split_dir = chunk_dir / split_name
            if split_dir.exists():
                all_files.extend(split_dir.glob("*.pyg"))

        if not all_files:
            print(f"{split_name}: no files found — skipping")
            continue

        all_files.sort(key=lambda f: f.name)
        print(f"{split_name}: {len(all_files)} event files")

        if args.dry_run:
            print(f"  [dry-run] Would consolidate → {output_dir / f'{split_name}.pt'}")
            print()
            continue

        # Load all per-event files and consolidate into a single list
        results = []
        total_segs = 0
        pbar = tqdm(all_files, desc=f"  Loading {split_name}", unit="ev")
        for f in pbar:
            data = torch.load(f, map_location="cpu", weights_only=False)
            seg_data_list = data["seg_data_list"]
            particle_ids = data["particle_ids"]
            results.append((seg_data_list, particle_ids))
            total_segs += len(seg_data_list)

        # Save as single consolidated file
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{split_name}.pt"
        torch.save({"results": results}, out_path)
        size_gb = out_path.stat().st_size / 1e9
        print(f"  Saved: {out_path} ({size_gb:.2f} GB)")
        print(f"  {len(results)} events, {total_segs} segments "
              f"(avg {total_segs / max(1, len(results)):.1f}/event)")
        print()

    if not args.dry_run and not args.keep_chunks:
        print("Removing chunk directories...")
        for d in chunk_dirs:
            shutil.rmtree(d)
            print(f"  Removed {d.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
