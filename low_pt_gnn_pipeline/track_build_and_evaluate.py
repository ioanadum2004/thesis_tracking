#!/usr/bin/env python3
"""
Build tracks, evaluate performance, and plot metrics in one step.

This script combines the track building, evaluation, and plotting stages:
1. Builds candidate tracks from inferred graphs (score cut + connected components)
2. Evaluates tracks against truth particles (efficiency, fake rate, clone rate)
3. Plots efficiency and clone rate vs pT and eta

Usage:
    python track_build_and_evaluate.py <dataset>

Examples:
    python track_build_and_evaluate.py testset
    python track_build_and_evaluate.py valset
    python track_build_and_evaluate.py testset --skip-build    # Re-evaluate without rebuilding
    python track_build_and_evaluate.py testset --config acorn_configs/track_build_and_evaluate.yaml

Config:
    Default config: acorn_configs/track_build_and_evaluate.yaml
    Contains both track building and evaluation parameters in one file
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import json
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))

from acorn.core.infer_stage import infer
from acorn.stages.track_building.utils import get_matching_df, calculate_matching_fraction
from acorn.stages.track_building.track_building_stage import make_result_summary, GraphDataset
from tqdm import tqdm

try:
    from atlasify import atlasify
    HAS_ATLASIFY = True
except ImportError:
    HAS_ATLASIFY = False


# ─── Track Building ─────────────────────────────────────────────────────────

def run_track_building(build_config_file):
    """Run track building using ACORN's infer stage."""
    with open(build_config_file, 'r') as f:
        config = yaml.safe_load(f)

    input_dir = Path(config['input_dir'])
    if not input_dir.is_absolute():
        input_dir = SCRIPT_DIR / input_dir

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Please run infer_gnn.py first to generate graphs with edge_scores."
        )

    print(f"Track building method: {config.get('model', 'ConnectedComponents')}")
    print(f"Score cut: {config.get('score_cut', 0.5)}")
    print(f"Output: {config['stage_dir']}")
    print()

    infer(str(build_config_file), verbose=False, checkpoint=None)

    print(f"Tracks saved to: {config['stage_dir']}")


# ─── Evaluation ──────────────────────────────────────────────────────────────

def safe_load_reconstruction_df(graph):
    """Load reconstruction DataFrame, handling tensor conversion."""
    if hasattr(graph, "hit_id"):
        hit_id = graph.hit_id
        if isinstance(hit_id, torch.Tensor):
            hit_id = hit_id.cpu().numpy()
    else:
        hit_id = torch.arange(graph.num_nodes).numpy()

    track_labels = graph.hit_track_labels
    if isinstance(track_labels, torch.Tensor):
        track_labels = track_labels.cpu().numpy()

    reco_df = pd.DataFrame({"hit_id": hit_id, "track_id": track_labels})

    if hasattr(graph, 'hit_particle_id'):
        hit_pid = graph.hit_particle_id
        if isinstance(hit_pid, torch.Tensor):
            reco_df["particle_id"] = hit_pid.cpu().numpy()
        else:
            reco_df["particle_id"] = hit_pid
    else:
        try:
            node_id = graph.track_edges.reshape(-1)
            pids = graph.track_particle_id.repeat(2)
            if len(node_id) != len(pids):
                raise ValueError(
                    f"track_edges and track_particle_id length mismatch: "
                    f"{len(node_id)} vs {len(pids)}"
                )
            pid_df = pd.DataFrame({"hit_id": node_id, "particle_id": pids})
            pid_df.drop_duplicates(subset=["hit_id", "particle_id"], inplace=True)
            reco_df = reco_df.merge(pid_df, on="hit_id", how="outer")
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not load particle_id from track_edges: {e}")
            reco_df["particle_id"] = 0

    reco_df.fillna({"track_id": -1, "particle_id": 0}, inplace=True)
    return reco_df


def safe_load_particles_df(graph, sel_conf):
    """Build particles DataFrame from hit_particle_id and available attributes."""
    if 'hit_particle_id' in graph:
        unique_pids = torch.unique(graph.hit_particle_id)
        unique_pids = unique_pids[unique_pids > 0].numpy()
    else:
        raise ValueError("Graph must have hit_particle_id attribute")

    particles_data = {"particle_id": unique_pids}

    # Map pt to particles
    if 'track_edges' in graph and 'pt' in graph:
        edge_hits = graph.track_edges.reshape(-1)
        edge_pids = graph.hit_particle_id[edge_hits].numpy()
        edge_pt = graph.pt.repeat(2).numpy()
        pid_pt_map = {}
        for pid, pval in zip(edge_pids, edge_pt):
            if pid > 0 and pid not in pid_pt_map:
                pid_pt_map[pid] = pval
        particles_data["pt"] = [pid_pt_map.get(pid, 0.0) for pid in unique_pids]
    elif 'track_particle_pt' in graph:
        particles_data["pt"] = graph.track_particle_pt.numpy()[:len(unique_pids)]
    else:
        particles_data["pt"] = [0.0] * len(unique_pids)

    # Map eta to particles
    if 'track_edges' in graph and 'eta_particle' in graph:
        edge_hits = graph.track_edges.reshape(-1)
        edge_pids = graph.hit_particle_id[edge_hits].numpy()
        edge_eta = graph.eta_particle.repeat(2).numpy()
        pid_eta_map = {}
        for pid, eval in zip(edge_pids, edge_eta):
            if pid > 0 and pid not in pid_eta_map:
                pid_eta_map[pid] = eval
        particles_data["eta"] = [pid_eta_map.get(pid, 0.0) for pid in unique_pids]
    elif 'track_particle_eta' in graph:
        particles_data["eta"] = graph.track_particle_eta.numpy()[:len(unique_pids)]

    particles_df = pd.DataFrame(particles_data)
    particles_df = particles_df.drop_duplicates(subset=["particle_id"])
    return particles_df


def safe_evaluate_labelled_graph(graph, sel_conf, matching_fraction, matching_style, min_track_length):
    """Evaluate a single graph with track labels against truth."""
    reco_df = safe_load_reconstruction_df(graph)

    particles_sel_conf = {k: v for k, v in sel_conf.items() if k != 'nhits'}
    particles_df = safe_load_particles_df(graph, particles_sel_conf)

    matching_sel_conf = {}
    for k, v in sel_conf.items():
        if k == 'nhits':
            matching_sel_conf['n_true_hits'] = v
        else:
            matching_sel_conf[k] = v

    matching_df = get_matching_df(reco_df, particles_df, matching_sel_conf, min_track_length=min_track_length)

    event_id = graph.event_id
    while type(event_id) == list:
        event_id = event_id[0]
    matching_df["event_id"] = int(event_id)

    matching_df = calculate_matching_fraction(matching_df)

    if matching_style == "ATLAS":
        matching_df["is_matched"] = matching_df["is_reconstructed"] = (
            matching_df.purity_reco >= matching_fraction
        )
    elif matching_style == "one_way":
        matching_df["is_matched"] = matching_df.purity_reco >= matching_fraction
        matching_df["is_reconstructed"] = matching_df.eff_true >= matching_fraction
    elif matching_style == "two_way":
        matching_df["is_matched"] = matching_df["is_reconstructed"] = (
            matching_df.purity_reco >= matching_fraction
        ) & (matching_df.eff_true >= matching_fraction)

    return matching_df


def run_evaluation(dataset_name, eval_config):
    """
    Evaluate all events in a dataset and return results.

    Returns:
        evaluated_events: concatenated matching DataFrame
        summary: dict with summary statistics
        summary_text: text summary string
    """
    input_dir = Path(eval_config.get('input_dir', eval_config.get('stage_dir')))
    if not input_dir.is_absolute():
        input_dir = SCRIPT_DIR / input_dir

    dataset_dir = input_dir / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    graph_files = list(dataset_dir.glob('*.pyg'))
    num_events = len(graph_files)
    if num_events == 0:
        raise ValueError(f"No graph files found in {dataset_dir}")

    print(f"Found {num_events} events in {dataset_name}")
    print(f"Matching fraction: {eval_config.get('matching_fraction', 0.5)}")
    print(f"Matching style: {eval_config.get('matching_style', 'ATLAS')}")
    print()

    dataset = GraphDataset(
        input_dir=str(input_dir),
        data_name=dataset_name,
        num_events=num_events,
        stage="test",
        hparams=eval_config
    )

    evaluated_events = []
    times = []

    for event in tqdm(dataset, desc=f"Evaluating {dataset_name}"):
        try:
            matching_df = safe_evaluate_labelled_graph(
                event,
                sel_conf=eval_config.get("target_tracks", {}),
                matching_fraction=eval_config.get("matching_fraction", 0.5),
                matching_style=eval_config.get("matching_style", "ATLAS"),
                min_track_length=eval_config.get("min_track_length", 1),
            )
            evaluated_events.append(matching_df)
            if hasattr(event, 'time_taken'):
                times.append(event.time_taken)
        except Exception as e:
            event_id = getattr(event, 'event_id', 'unknown')
            if isinstance(event_id, list):
                event_id = event_id[0] if len(event_id) > 0 else 'unknown'
            print(f"\nError evaluating event {event_id}: {e}")
            import traceback
            traceback.print_exc()
            raise

    evaluated_events = pd.concat(evaluated_events, ignore_index=True)

    if times:
        times = np.array(times)
        time_avg, time_std = np.mean(times), np.std(times)
    else:
        time_avg, time_std = 0.0, 0.0

    # Calculate metrics
    particles = evaluated_events[evaluated_events["is_reconstructable"]]
    reconstructed_particles = particles[particles["is_reconstructed"] & particles["is_matchable"]]
    tracks = evaluated_events[evaluated_events["is_matchable"]]
    matched_tracks = tracks[tracks["is_matched"]]

    n_particles = len(particles.drop_duplicates(subset=["event_id", "particle_id"]))
    n_reconstructed = len(reconstructed_particles.drop_duplicates(subset=["event_id", "particle_id"]))
    n_tracks = len(tracks.drop_duplicates(subset=["event_id", "track_id"]))
    n_matched = len(matched_tracks.drop_duplicates(subset=["event_id", "track_id"]))
    n_duplicates = len(reconstructed_particles) - n_reconstructed

    efficiency = n_reconstructed / n_particles if n_particles > 0 else 0.0
    fake_rate = 1 - (n_matched / n_tracks) if n_tracks > 0 else 0.0
    clone_rate = n_duplicates / n_reconstructed if n_reconstructed > 0 else 0.0

    summary = {
        'dataset': dataset_name,
        'n_events': num_events,
        'n_particles': int(n_particles),
        'n_reconstructed_particles': int(n_reconstructed),
        'n_tracks': int(n_tracks),
        'n_matched_tracks': int(n_matched),
        'n_duplicate_particles': int(n_duplicates),
        'efficiency': float(efficiency),
        'fake_rate': float(fake_rate),
        'clone_rate': float(clone_rate),
        'avg_purity': float(evaluated_events['purity_reco'].mean() if 'purity_reco' in evaluated_events.columns else 0.0),
        'avg_completeness': float(evaluated_events['eff_true'].mean() if 'eff_true' in evaluated_events.columns else 0.0),
        'time_avg_seconds': float(time_avg),
        'time_std_seconds': float(time_std),
        'matching_fraction': eval_config.get('matching_fraction', 0.5),
        'matching_style': eval_config.get('matching_style', 'ATLAS'),
    }

    summary_text = make_result_summary(
        n_reconstructed, n_particles, n_matched, n_tracks, n_duplicates,
        efficiency, fake_rate, clone_rate, time_avg, time_std,
    )

    return evaluated_events, summary, summary_text


def save_evaluation_results(evaluated_events, summary, summary_text, dataset_name, output_dir):
    """Save evaluation results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Matching DataFrame
    evaluated_events.to_csv(output_dir / f"matching_df_{dataset_name}.csv", index=False)

    # Summary JSON
    with open(output_dir / f"summary_{dataset_name}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Summary text
    with open(output_dir / f"results_summary_{summary.get('matching_style', 'ATLAS')}_{dataset_name}.txt", 'w') as f:
        f.write(summary_text)

    # Particles DataFrame for plotting
    particles = evaluated_events[evaluated_events["is_reconstructable"]]
    particles_plot = particles.drop_duplicates(subset=["particle_id"]).copy()
    particles_plot.to_csv(output_dir / f"particles_{dataset_name}.csv", index=False)

    print(f"Evaluation results saved to: {output_dir}")


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_efficiency_vs_variable(particles_df, var, varconf, output_path, summary=None):
    """Plot efficiency vs pT or eta."""
    x = particles_df[var].values
    if 'x_scale' in varconf:
        x = x * float(varconf['x_scale'])

    reconstructable = particles_df['is_reconstructable']
    reconstructed = particles_df['is_reconstructable'] & particles_df['is_reconstructed']
    true_x = x[reconstructable]
    reco_x = x[reconstructed]

    if len(true_x) == 0:
        print(f"Warning: No reconstructable particles with {var} data. Skipping.")
        return
    if var == 'pt' and np.all(true_x <= 0):
        print(f"Warning: All {var} values are <= 0. Skipping log-scale plot.")
        return

    if 'x_lim' in varconf:
        if var == 'pt':
            x_bins = np.logspace(np.log10(varconf['x_lim'][0]), np.log10(varconf['x_lim'][1]), 20)
        else:
            x_bins = np.linspace(varconf['x_lim'][0], varconf['x_lim'][1], 20)
    else:
        x_bins = np.logspace(-1, 2, 20) if var == 'pt' else np.linspace(-4, 4, 20)

    true_vals, true_bins = np.histogram(true_x, bins=x_bins)
    reco_vals, _ = np.histogram(reco_x, bins=x_bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        eff = np.true_divide(reco_vals, true_vals)
        err = np.sqrt(eff * (1 - eff) / true_vals)
        err[true_vals == 0] = 0
        eff[true_vals == 0] = np.nan

    xvals = (true_bins[1:] + true_bins[:-1]) / 2
    xerrs = (true_bins[1:] - true_bins[:-1]) / 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(xvals, eff, xerr=xerrs, yerr=err, fmt='o', color='black',
                label='Track efficiency', capsize=3, capthick=1.5)
    ax.set_xlabel(varconf.get('x_label', var), fontsize=14)
    ax.set_ylabel('Efficiency', fontsize=14)
    ax.set_ylim(varconf.get('y_lim', [0, 1.1]))
    if var == 'pt':
        ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    if summary:
        title = f"Track Efficiency vs {var.upper()}"
        if 'efficiency' in summary:
            title += f" (Overall: {summary['efficiency']:.3f})"
        ax.set_title(title, fontsize=14)
    if HAS_ATLASIFY:
        atlasify(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_clone_rate_vs_variable(matching_df, var, varconf, output_path, summary=None):
    """Plot clone rate vs pT or eta."""
    particles = matching_df[matching_df['is_reconstructable'] & matching_df['is_reconstructed']].copy()
    particle_track_counts = particles.groupby(['event_id', 'particle_id']).size().reset_index(name='n_tracks')
    particle_track_counts['is_cloned'] = particle_track_counts['n_tracks'] > 1

    particles_unique = particles.drop_duplicates(subset=['event_id', 'particle_id'])
    particle_track_counts = particle_track_counts.merge(
        particles_unique[['event_id', 'particle_id', var]],
        on=['event_id', 'particle_id']
    )

    x = particle_track_counts[var].values
    if 'x_scale' in varconf:
        x = x * float(varconf['x_scale'])

    all_x = x
    cloned_x = x[particle_track_counts['is_cloned']]

    if 'x_lim' in varconf:
        if var == 'pt':
            x_bins = np.logspace(np.log10(varconf['x_lim'][0]), np.log10(varconf['x_lim'][1]), 20)
        else:
            x_bins = np.linspace(varconf['x_lim'][0], varconf['x_lim'][1], 20)
    else:
        x_bins = np.logspace(-1, 2, 20) if var == 'pt' else np.linspace(-4, 4, 20)

    all_vals, all_bins = np.histogram(all_x, bins=x_bins)
    cloned_vals, _ = np.histogram(cloned_x, bins=x_bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        clone_rate = np.true_divide(cloned_vals, all_vals)
        err = np.sqrt(clone_rate * (1 - clone_rate) / all_vals)
        err[all_vals == 0] = 0
        clone_rate[all_vals == 0] = np.nan

    xvals = (all_bins[1:] + all_bins[:-1]) / 2
    xerrs = (all_bins[1:] - all_bins[:-1]) / 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(xvals, clone_rate, xerr=xerrs, yerr=err, fmt='o', color='orange',
                label='Clone rate', capsize=3, capthick=1.5)
    ax.set_xlabel(varconf.get('x_label', var), fontsize=14)
    ax.set_ylabel('Clone Rate', fontsize=14)
    ax.set_ylim([0, 1.1])
    if var == 'pt':
        ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    if summary and 'clone_rate' in summary:
        ax.set_title(f"Clone Rate vs {var.upper()} (Overall: {summary['clone_rate']:.3f})", fontsize=14)
    if HAS_ATLASIFY:
        atlasify(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def run_plotting(evaluated_events, summary, dataset_name, output_dir):
    """Create all metric plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load particles for efficiency plots
    particles_df = evaluated_events[evaluated_events["is_reconstructable"]].drop_duplicates(subset=["particle_id"]).copy()

    plot_config = {
        'pt': {'x_label': '$p_T$ [GeV]', 'x_lim': [0.05, 1.0], 'y_lim': [0, 1.1]},
        'eta': {'x_label': '$\\eta$', 'x_lim': [-4, 4], 'y_lim': [0, 1.1]},
    }

    for var in ['pt', 'eta']:
        varconf = plot_config[var]

        eff_path = output_dir / f"efficiency_vs_{var}_{dataset_name}.png"
        plot_efficiency_vs_variable(particles_df, var, varconf, eff_path, summary)

        clone_path = output_dir / f"clone_rate_vs_{var}_{dataset_name}.png"
        plot_clone_rate_vs_variable(evaluated_events, var, varconf, clone_path, summary)

    print(f"Plots saved to: {output_dir}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build tracks, evaluate performance, and plot metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python track_build_and_evaluate.py testset
    # Full pipeline: build + evaluate + plot for testset

  python track_build_and_evaluate.py valset --skip-build
    # Skip track building, only re-evaluate and plot

  python track_build_and_evaluate.py testset --config acorn_configs/track_building_eval.yaml
    # Use custom evaluation config
        """
    )
    parser.add_argument(
        'dataset',
        type=str,
        choices=['trainset', 'valset', 'testset'],
        help='Dataset to evaluate: trainset, valset, or testset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to combined config file (default: acorn_configs/track_build_and_evaluate.yaml)'
    )
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip track building step (use existing tracks in data/track_building/)'
    )

    args = parser.parse_args()

    # Load combined config
    if args.config is None:
        config_file = SCRIPT_DIR / 'acorn_configs' / 'track_build_and_evaluate.yaml'
    else:
        config_file = Path(args.config)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        eval_config = yaml.safe_load(f)
    eval_config['dataset'] = args.dataset

    # Output directories
    eval_output_dir = SCRIPT_DIR / 'data' / 'track_evaluation' / args.dataset
    plot_output_dir = SCRIPT_DIR / 'data' / 'visuals' / 'track_metrics' / args.dataset

    # ── Step 1: Track Building ────────────────────────────────────────────
    if not args.skip_build:
        print("=" * 70)
        print("STEP 1: TRACK BUILDING")
        print("=" * 70)
        print()

        # Use the same config file for track building
        run_track_building(config_file)
        print()
    else:
        print("Skipping track building (--skip-build)")
        print()

    # ── Step 2: Evaluation ────────────────────────────────────────────────
    print("=" * 70)
    print(f"STEP 2: EVALUATING {args.dataset.upper()}")
    print("=" * 70)
    print()

    # Override input_dir to read from track_building output
    eval_config['input_dir'] = str(SCRIPT_DIR / 'data' / 'track_building')

    evaluated_events, summary, summary_text = run_evaluation(args.dataset, eval_config)
    save_evaluation_results(evaluated_events, summary, summary_text, args.dataset, eval_output_dir)

    print()
    print(summary_text)

    # ── Step 3: Plotting ──────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 3: PLOTTING METRICS")
    print("=" * 70)
    print()

    run_plotting(evaluated_events, summary, args.dataset, plot_output_dir)

    # ── Done ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"  Tracks:     data/track_building/")
    print(f"  Evaluation: {eval_output_dir.relative_to(SCRIPT_DIR)}/")
    print(f"  Plots:      {plot_output_dir.relative_to(SCRIPT_DIR)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
