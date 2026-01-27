#!/usr/bin/env python3
"""
Plot track building performance metrics from evaluation results.

This script reads the matching DataFrame and summary statistics from evaluate_tracks.py
and creates efficiency plots vs pT and η, similar to ACTS performance plots.

Usage:
    python plot_track_metrics.py [--input-dir DIR] [--dataset DATASET] [--output-dir DIR]

Examples:
    python plot_track_metrics.py
        # Uses default: data/track_evaluation/testset
    
    python plot_track_metrics.py --input-dir results/eval --dataset testset
        # Plot from custom evaluation directory
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

try:
    from atlasify import atlasify
    HAS_ATLASIFY = True
except ImportError:
    HAS_ATLASIFY = False
    print("Warning: atlasify not available. Plots will not have ATLAS styling.")


def load_evaluation_data(input_dir, dataset):
    """Load evaluation data from CSV files."""
    input_dir = Path(input_dir)
    
    # Load matching DataFrame
    matching_df_path = input_dir / f"matching_df_{dataset}.csv"
    if not matching_df_path.exists():
        raise FileNotFoundError(f"Matching DataFrame not found: {matching_df_path}")
    
    matching_df = pd.read_csv(matching_df_path)
    
    # Load particles DataFrame
    particles_df_path = input_dir / f"particles_{dataset}.csv"
    if particles_df_path.exists():
        particles_df = pd.read_csv(particles_df_path)
    else:
        # Fallback: extract from matching_df
        particles_df = matching_df.drop_duplicates(subset=["particle_id"]).copy()
    
    # Load summary
    summary_path = input_dir / f"summary_{dataset}.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = None
    
    return matching_df, particles_df, summary


def plot_efficiency_vs_variable(particles_df, var, varconf, output_path, summary=None):
    """
    Plot efficiency vs a variable (pt or eta).
    
    Args:
        particles_df: DataFrame with particle information and is_reconstructed flag
        var: Variable name ('pt' or 'eta')
        varconf: Configuration dict with x_label, x_lim, x_scale, y_lim, etc.
        output_path: Path to save the plot
        summary: Optional summary dict for title
    """
    if var not in ['pt', 'eta']:
        raise ValueError(f"Variable must be 'pt' or 'eta', got '{var}'")
    
    # Get variable values
    x = particles_df[var].values
    
    # Apply scaling if specified
    if 'x_scale' in varconf:
        x = x * float(varconf['x_scale'])
    
    # Filter reconstructable particles
    reconstructable = particles_df['is_reconstructable']
    reconstructed = particles_df['is_reconstructable'] & particles_df['is_reconstructed']
    
    true_x = x[reconstructable]
    reco_x = x[reconstructed]
    
    # Determine bins
    if 'x_bins' in varconf:
        x_bins = varconf['x_bins']
    elif 'x_lim' in varconf:
        if var == 'pt':
            # Log scale for pT
            x_bins = np.logspace(
                np.log10(varconf['x_lim'][0]),
                np.log10(varconf['x_lim'][1]),
                20
            )
        else:
            # Linear scale for eta
            x_bins = np.linspace(varconf['x_lim'][0], varconf['x_lim'][1], 20)
    else:
        if var == 'pt':
            x_bins = np.logspace(-1, 2, 20)  # 0.1 to 100 GeV
        else:
            x_bins = np.linspace(-4, 4, 20)
    
    # Calculate histograms
    true_vals, true_bins = np.histogram(true_x, bins=x_bins)
    reco_vals, reco_bins = np.histogram(reco_x, bins=x_bins)
    
    # Calculate efficiency and errors
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = np.true_divide(reco_vals, true_vals)
        # Binomial errors
        err = np.sqrt(eff * (1 - eff) / true_vals)
        err[true_vals == 0] = 0
        eff[true_vals == 0] = np.nan
    
    # Bin centers and widths
    xvals = (true_bins[1:] + true_bins[:-1]) / 2
    xerrs = (true_bins[1:] - true_bins[:-1]) / 2
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(
        xvals,
        eff,
        xerr=xerrs,
        yerr=err,
        fmt='o',
        color='black',
        label='Track efficiency',
        capsize=3,
        capthick=1.5,
    )
    
    ax.set_xlabel(varconf.get('x_label', f'{var}'), fontsize=14)
    ax.set_ylabel('Efficiency', fontsize=14)
    
    if 'y_lim' in varconf:
        ax.set_ylim(varconf['y_lim'])
    else:
        ax.set_ylim([0, 1.1])
    
    if var == 'pt':
        ax.set_xscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add summary info to title if available
    if summary:
        title = f"Track Efficiency vs {var.upper()}"
        if 'efficiency' in summary:
            title += f" (Overall: {summary['efficiency']:.3f})"
        ax.set_title(title, fontsize=14)
    
    if HAS_ATLASIFY:
        atlasify(ax, subtext_size=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_fake_rate_vs_variable(matching_df, var, varconf, output_path, summary=None):
    """
    Plot fake rate vs pt or eta.
    
    Note: Fake tracks don't have particle pt/eta in matching_df, so we can't plot
    fake rate vs particle kinematics. This plot shows the overall fake rate.
    For track-level fake rate vs kinematics, we'd need track reconstruction parameters.
    """
    if var not in ['pt', 'eta']:
        raise ValueError(f"Variable must be 'pt' or 'eta', got '{var}'")
    
    # Get tracks
    tracks = matching_df[matching_df['is_matchable']].copy()
    
    # Get unique tracks per event
    tracks_unique = tracks.drop_duplicates(subset=['event_id', 'track_id'])
    
    # Check if we have the variable
    if var not in tracks_unique.columns:
        print(f"Warning: Cannot plot fake rate vs {var}: variable not in matching_df.")
        print("  Fake tracks don't have particle kinematics. Skipping this plot.")
        return
    
    # Get variable values (particle pt/eta - only available for matched tracks)
    x = tracks_unique[var].values
    
    # Apply scaling
    if 'x_scale' in varconf:
        x = x * float(varconf['x_scale'])
    
    # Separate matched and fake tracks
    matched = tracks_unique['is_matched']
    fake = ~matched
    
    # Note: Fake tracks don't appear in matching_df (they have no particle match)
    # So we can't plot fake rate vs particle kinematics
    # Instead, we'll create a simple plot showing overall fake rate
    # For proper fake rate vs kinematics, we'd need track reconstruction parameters
    
    # For now, skip this plot and note the limitation
    print(f"Note: Fake rate vs {var} plot skipped.")
    print("  Reason: Fake tracks don't have particle kinematics in matching_df.")
    print("  Overall fake rate is available in summary.json")
    return
    
    # Determine bins
    if 'x_bins' in varconf:
        x_bins = varconf['x_bins']
    elif 'x_lim' in varconf:
        if var == 'pt':
            x_bins = np.logspace(
                np.log10(varconf['x_lim'][0]),
                np.log10(varconf['x_lim'][1]),
                20
            )
        else:
            x_bins = np.linspace(varconf['x_lim'][0], varconf['x_lim'][1], 20)
    else:
        if var == 'pt':
            x_bins = np.logspace(-1, 2, 20)
        else:
            x_bins = np.linspace(-4, 4, 20)
    
    # Calculate histograms
    all_vals, all_bins = np.histogram(all_x, bins=x_bins)
    fake_vals, _ = np.histogram(fake_x, bins=x_bins)
    
    # Calculate fake rate
    with np.errstate(divide='ignore', invalid='ignore'):
        fake_rate = np.true_divide(fake_vals, all_vals)
        err = np.sqrt(fake_rate * (1 - fake_rate) / all_vals)
        err[all_vals == 0] = 0
        fake_rate[all_vals == 0] = np.nan
    
    xvals = (all_bins[1:] + all_bins[:-1]) / 2
    xerrs = (all_bins[1:] - all_bins[:-1]) / 2
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(
        xvals,
        fake_rate,
        xerr=xerrs,
        yerr=err,
        fmt='o',
        color='red',
        label='Fake rate',
        capsize=3,
        capthick=1.5,
    )
    
    ax.set_xlabel(varconf.get('x_label', f'{var}'), fontsize=14)
    ax.set_ylabel('Fake Rate', fontsize=14)
    ax.set_ylim([0, 1.1])
    
    if var == 'pt':
        ax.set_xscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    if summary and 'fake_rate' in summary:
        ax.set_title(f"Fake Rate vs {var.upper()} (Overall: {summary['fake_rate']:.3f})", fontsize=14)
    
    if HAS_ATLASIFY:
        atlasify(ax, subtext_size=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_clone_rate_vs_variable(matching_df, var, varconf, output_path, summary=None):
    """Plot clone rate vs pt or eta."""
    if var not in ['pt', 'eta']:
        raise ValueError(f"Variable must be 'pt' or 'eta', got '{var}'")
    
    # Get reconstructed particles (may have duplicates)
    particles = matching_df[matching_df['is_reconstructable'] & matching_df['is_reconstructed']].copy()
    
    # Count tracks per particle
    particle_track_counts = particles.groupby(['event_id', 'particle_id']).size().reset_index(name='n_tracks')
    particle_track_counts['is_cloned'] = particle_track_counts['n_tracks'] > 1
    
    # Merge back to get pt/eta
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
    
    # Determine bins
    if 'x_bins' in varconf:
        x_bins = varconf['x_bins']
    elif 'x_lim' in varconf:
        if var == 'pt':
            x_bins = np.logspace(
                np.log10(varconf['x_lim'][0]),
                np.log10(varconf['x_lim'][1]),
                20
            )
        else:
            x_bins = np.linspace(varconf['x_lim'][0], varconf['x_lim'][1], 20)
    else:
        if var == 'pt':
            x_bins = np.logspace(-1, 2, 20)
        else:
            x_bins = np.linspace(-4, 4, 20)
    
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
    
    ax.errorbar(
        xvals,
        clone_rate,
        xerr=xerrs,
        yerr=err,
        fmt='o',
        color='orange',
        label='Clone rate',
        capsize=3,
        capthick=1.5,
    )
    
    ax.set_xlabel(varconf.get('x_label', f'{var}'), fontsize=14)
    ax.set_ylabel('Clone Rate', fontsize=14)
    ax.set_ylim([0, 1.1])
    
    if var == 'pt':
        ax.set_xscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    if summary and 'clone_rate' in summary:
        ax.set_title(f"Clone Rate vs {var.upper()} (Overall: {summary['clone_rate']:.3f})", fontsize=14)
    
    if HAS_ATLASIFY:
        atlasify(ax, subtext_size=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot track building performance metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_track_metrics.py
    # Plot from default directory: data/track_evaluation/testset
  
  python plot_track_metrics.py --input-dir results/eval --dataset testset
    # Plot from custom directory
  
  python plot_track_metrics.py --output-dir plots/
    # Save plots to custom directory
        """
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory with evaluation results (default: data/track_evaluation)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['trainset', 'valset', 'testset'],
        default='testset',
        help='Dataset to plot (default: testset)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as input-dir)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file with plot settings (optional)'
    )
    
    args = parser.parse_args()
    
    # Determine input directory
    script_dir = Path(__file__).resolve().parent
    if args.input_dir is None:
        input_dir = script_dir / 'data' / 'track_evaluation'
    else:
        input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config if provided
    plot_config = {}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if 'plots' in config and 'tracking_efficiency' in config['plots']:
                plot_config = config['plots']['tracking_efficiency'].get('variables', {})
    
    # Default plot config
    default_config = {
        'pt': {
            'x_label': '$p_T$ [GeV]',
            'x_scale': 0.001,  # Convert MeV to GeV if needed
            'x_lim': [0.1, 100],
            'y_lim': [0, 1.1],
        },
        'eta': {
            'x_label': '$\eta$',
            'x_lim': [-4, 4],
            'y_lim': [0, 1.1],
        },
    }
    
    # Merge configs
    for var in ['pt', 'eta']:
        if var in plot_config:
            default_config[var].update(plot_config[var])
    
    print("="*70)
    print("PLOTTING TRACK METRICS")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {args.dataset}")
    print()
    
    # Load data
    print("Loading evaluation data...")
    matching_df, particles_df, summary = load_evaluation_data(input_dir, args.dataset)
    
    print(f"Loaded {len(matching_df)} matching entries")
    print(f"Loaded {len(particles_df)} particles")
    if summary:
        print(f"Overall efficiency: {summary.get('efficiency', 0):.3f}")
        print(f"Overall fake rate: {summary.get('fake_rate', 0):.3f}")
        print(f"Overall clone rate: {summary.get('clone_rate', 0):.3f}")
    print()
    
    # Create plots
    print("Creating plots...")
    
    for var in ['pt', 'eta']:
        varconf = default_config[var]
        
        # Efficiency vs variable
        eff_path = output_dir / f"efficiency_vs_{var}_{args.dataset}.png"
        plot_efficiency_vs_variable(particles_df, var, varconf, eff_path, summary)
        
        # Fake rate vs variable (skipped - fake tracks don't have particle kinematics)
        # Uncomment if you have track-level kinematics available
        # fake_path = output_dir / f"fake_rate_vs_{var}_{args.dataset}.png"
        # plot_fake_rate_vs_variable(matching_df, var, varconf, fake_path, summary)
        
        # Clone rate vs variable
        clone_path = output_dir / f"clone_rate_vs_{var}_{args.dataset}.png"
        plot_clone_rate_vs_variable(matching_df, var, varconf, clone_path, summary)
    
    print("\n" + "="*70)
    print("✓ Plotting complete!")
    print("="*70)
    print(f"\nPlots saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
