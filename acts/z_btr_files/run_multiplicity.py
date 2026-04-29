#!/usr/bin/env python3
"""
run_multiplicity.py

Sweeps over multiplicity values, running both the baseline and seed-filter
pipelines for each, saving all ROOT outputs into separate directories.

Usage:
    python run_multiplicity.py --dry-run
    python run_multiplicity.py --multiplicities 1 2 3
"""

import argparse
import json
import copy
import subprocess
from pathlib import Path

# ── Edit these if your paths ever change ─────────────────────────────────────
BASELINE_SCRIPT   = "Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py"
SEEDFILTER_MLP    = "Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py"
SEEDFILTER_BDT    = "Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py"
SHARED_CONFIG     = Path("Examples/Configs/perfect-spacepoints-multigen-config.json")
OUTPUT_ROOT       = Path("z_btr_files/multiplicity_sweep")
# ─────────────────────────────────────────────────────────────────────────────

FILES_TO_COLLECT = [
    "particles.root",
    "tracksummary_ckf.root",
    "estimatedparams.root",
]


def overwrite_config(base_config: dict, multiplicity: int):
    #write a modified copy of the config with the given multiplicity
    cfg = copy.deepcopy(base_config)
    cfg["simulation"]["eventGenerator"]["multiplicity"] = multiplicity
    with open(SHARED_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Config written: multiplicity={multiplicity}")


def restore_config(original_config: dict):
    #restore the config to its original value
    with open(SHARED_CONFIG, "w") as f:
        json.dump(original_config, f, indent=2)
    print(f"  Config restored.")


def run_script_old(script: str, run_dir: Path, dry_run: bool) -> bool: # returns a bool
    #run a pipeline script, directing its output to run_dir
    cmd = ["python", script, str(run_dir)]
    print(f"  cmd : {' '.join(cmd)}")
    if dry_run:
        print("  [dry-run] skipping execution")
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR (exit code {result.returncode})")
        print(result.stderr[-3000:])
        return False
    print("  OK")
    return True

def run_script(script: str, run_dir: Path, dry_run: bool, model_arg: str = "") -> bool:
    cmd = ["python", script, str(run_dir)]
    if model_arg:
        cmd += model_arg.split()  # adds ["--model", "none"] etc.
    print(f"  cmd : {' '.join(cmd)}")
    if dry_run:
        print("  [dry-run] skipping execution")
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR (exit code {result.returncode})")
        print(result.stderr[-3000:])
        return False
    print("  OK")
    return True

def check_outputs(run_dir: Path, label: str):
    #verify expected ROOT files exist in run_dir and print their sizes
    for fname in FILES_TO_COLLECT:
        f = run_dir / fname
        if f.exists():
            size_mb = f.stat().st_size / 1e6
            print(f"  [{label}] {fname} — {size_mb:.1f} MB")
        else:
            print(f"  [{label}] WARNING: missing {fname}")

def run_one_multiplicity(base_config: dict, mult: int, dry_run: bool):
    print(f"\n{'='*55}")
    print(f"  Multiplicity = {mult}")
    print(f"{'='*55}")

    pipelines = [
        ("baseline",   BASELINE_SCRIPT,  "--model none"),
        ("mlp",        SEEDFILTER_MLP,   "--model mlp"),
        ("tree",        SEEDFILTER_BDT,   "--model tree"),
    ]

    for label, script, model_arg in pipelines:
        run_dir = OUTPUT_ROOT / f"mult_{mult}" / label
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{label}]")
        try:
            overwrite_config(base_config, mult)
            success = run_script(script, run_dir, dry_run, model_arg=model_arg)
            if success and not dry_run:
                check_outputs(run_dir, label)
        finally:
            restore_config(base_config)
            
def run_one_multiplicity_old(base_config: dict, mult: int, dry_run: bool):
    #run all pipelines for a single multiplicity value."""
    print(f"\n{'='*55}")
    print(f"  Multiplicity = {mult}")
    print(f"{'='*55}")

    pipelines = [
        ("baseline",   BASELINE_SCRIPT),
        ("mlp",        SEEDFILTER_MLP),
        ("tree",        SEEDFILTER_BDT),
    ]

    for label, script in pipelines:
        run_dir = OUTPUT_ROOT / f"mult_{mult}" / label
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{label}]")
        try:
            overwrite_config(base_config, mult)
            success = run_script(script, run_dir, dry_run)
            if success and not dry_run:
                check_outputs(run_dir, label)
        finally:
            #runs even if the script crashes so config is always restored
            restore_config(base_config)


def main():
    p = argparse.ArgumentParser(
        description="Sweep multiplicity values across baseline and seed-filter pipelines."
    )
    p.add_argument(
        "--multiplicities", nargs="+", type=int,
        default=[1, 2, 5, 10, 20],
        help="List of multiplicity values to sweep (default: 1 2 5 10 20)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them"
    )
    args = p.parse_args()

    #read original config once, used to restore after every run
    with open(SHARED_CONFIG) as f:
        original_config = json.load(f)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for mult in args.multiplicities:
        run_one_multiplicity(original_config, mult, args.dry_run)

    #summary 
    print(f"\n{'='*55}")
    print("SWEEP COMPLETE")
    print(f"{'='*55}")
    for mult in args.multiplicities:
        for label in ("baseline", "mlp", "tree"):
            d = OUTPUT_ROOT / f"mult_{mult}" / label
            roots = list(d.glob("*.root")) if d.exists() else []
            status = f"{len(roots)} ROOT files" if roots else "NO OUTPUT"
            print(f"  mult={mult:3d} / {label:12s}: {status}")

    print(f"\nAll outputs under: {OUTPUT_ROOT.resolve()}/")


if __name__ == "__main__":
    main()
