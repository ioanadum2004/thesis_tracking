#!/usr/bin/env python3
"""
Calculate expected number of geometric loops per particle in an event.

Each small time interval dt, the particle completes dN = ds/C(t) loops,
where ds = v*dt is the path increment and C(t) = 2*pi*pT(t)/(qB) is
the instantaneous circumference. So dN ∝ dt/pT(t).

Integrating and using the path budget L = f * 2*pi*pT_init/(qB):

    N_loops = f * pT_init * sum(dt_i / pT_i) / T_total
            = f * pT_init / pT_harmonic

where pT_harmonic = T_total / sum(dt_i / pT_i) is the time-weighted
harmonic mean of pT.

Usage:
    python loop_calculation_per_event.py <dataset> <event_index> [--loop-fraction F]

Examples:
    python loop_calculation_per_event.py valset 0
    python loop_calculation_per_event.py testset 5 --loop-fraction 0.5
"""

import sys
import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent

PDG_NAMES = {
    13: "mu-", -13: "mu+",
    11: "e-", -11: "e+",
    211: "pi+", -211: "pi-",
    2212: "p", -2212: "p-bar",
    321: "K+", -321: "K-",
    22: "gamma",
}


def main():
    parser = argparse.ArgumentParser(description="Calculate expected geometric loops per particle")
    parser.add_argument("dataset", help="Dataset name (trainset, valset, testset)")
    parser.add_argument("event_index", type=int, help="Event index within the dataset")
    parser.add_argument("--loop-fraction", "-f", type=float, default=1.0,
                        help="Loop fraction parameter (default: 1.0)")
    args = parser.parse_args()

    # Find the PyG file to get the event ID
    pyg_dir = PIPELINE_ROOT / "data" / "feature_store" / args.dataset
    pyg_files = sorted(glob.glob(str(pyg_dir / "*.pyg")))
    if not pyg_files:
        print(f"No PyG files found in {pyg_dir}")
        sys.exit(1)
    if args.event_index >= len(pyg_files):
        print(f"Event index {args.event_index} out of range (0-{len(pyg_files)-1})")
        sys.exit(1)

    pyg_path = pyg_files[args.event_index]
    # Extract event ID from filename: event000100000-graph.pyg -> 000100000
    event_id = Path(pyg_path).stem.replace("-graph", "").replace("event", "")
    csv_prefix = PIPELINE_ROOT / "data" / "csv" / f"event{event_id}"

    # Load CSV data
    hits_path = f"{csv_prefix}-hits.csv"
    particles_path = f"{csv_prefix}-particles_initial.csv"

    hits = pd.read_csv(hits_path)
    particles = pd.read_csv(particles_path)

    # Compute pT at each hit from truth momentum
    hits["hit_pt"] = np.sqrt(hits["tpx"]**2 + hits["tpy"]**2)

    # Sort hits by particle and time
    hits = hits.sort_values(["particle_id", "tt"])

    # Build per-particle summary
    results = []
    for pid, group in hits.groupby("particle_id"):
        if pid == 0:
            continue  # skip noise
        nhits = len(group)
        if nhits < 2:
            continue

        pt_vals = group["hit_pt"].values
        t_vals = group["tt"].values
        pt_first = pt_vals[0]
        pt_last = pt_vals[-1]
        pt_loss_frac = (pt_first - pt_last) / pt_first

        # Time-weighted harmonic mean of pT:
        # Use midpoint pT between consecutive hits, weighted by dt
        dt = np.diff(t_vals)
        pt_mid = 0.5 * (pt_vals[:-1] + pt_vals[1:])  # pT at midpoint of each interval
        t_total = t_vals[-1] - t_vals[0]
        if t_total > 0 and np.all(pt_mid > 0):
            pt_harmonic = t_total / np.sum(dt / pt_mid)
        else:
            pt_harmonic = pt_first  # fallback

        n_loops = args.loop_fraction * pt_first / pt_harmonic

        # Look up particle type
        particle_row = particles[particles["particle_id"] == pid]
        if len(particle_row) > 0:
            pdg = int(particle_row["particle_type"].iloc[0])
            species = PDG_NAMES.get(pdg, f"PDG={pdg}")
        else:
            species = "?"
            pdg = 0

        results.append({
            "particle_id": pid,
            "species": species,
            "pdg": pdg,
            "nhits": nhits,
            "pT_initial": pt_first * 1000,  # GeV -> MeV
            "pT_harmonic": pt_harmonic * 1000,
            "pT_final": pt_last * 1000,
            "pT_loss_%": pt_loss_frac * 100,
            "N_loops": n_loops,
        })

    results.sort(key=lambda r: r["N_loops"], reverse=True)

    # Print
    f = args.loop_fraction
    print(f"Event {event_id} ({args.dataset}[{args.event_index}])  |  loop_fraction = {f}")
    print(f"Formula: N_loops = {f} x pT_initial / pT_harmonic   (time-weighted harmonic mean)")
    print()
    print(f"{'Species':<8} {'particle_id':>20} {'pT_init (MeV)':>13} {'pT_harm (MeV)':>13} {'pT_loss':>8} {'Hits':>5} {'N_loops':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['species']:<8} {r['particle_id']:>20} {r['pT_initial']:>13.1f} {r['pT_harmonic']:>13.1f} {r['pT_loss_%']:>7.1f}% {r['nhits']:>5} {r['N_loops']:>8.2f}")


if __name__ == "__main__":
    main()
