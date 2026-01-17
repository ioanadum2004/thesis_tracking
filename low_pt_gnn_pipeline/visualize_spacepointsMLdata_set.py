#!/usr/bin/env python3
"""
visualize_spacepoints.py <event_prefix> [--output OUTPUT] [--color-by COLOR_BY] [--max-points MAX]

Creates an interactive 3D HTML visualization of spacepoints from CSV files.
Reads hits from <event_prefix>-hits.csv and optionally truth/particles data.

Usage examples:
  python visualize_spacepoints.py ML_data_trainSamples_100_events/event000001000
    # Visualizes hits from event000001000-hits.csv
  
  python visualize_spacepoints.py ML_data_trainSamples_100_events/event000001000 --color-by particle
    # Colors hits by particle_id (requires truth CSV)
  
  python visualize_spacepoints.py ML_data_trainSamples_100_events/event000001000 --color-by volume --max-points 50000
    # Colors by volume_id and limits to 50000 points

Dependencies: plotly, pandas, numpy
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("Error: plotly is required. Install with: pip install plotly pandas")
    sys.exit(1)


def read_hits(hits_path):
    """Read hits CSV file and return DataFrame."""
    try:
        df = pd.read_csv(hits_path)
        required_cols = ['x', 'y', 'z']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns: {missing}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading hits file: {e}")


def read_truth(truth_path):
    """Read truth CSV file and return DataFrame."""
    try:
        df = pd.read_csv(truth_path)
        return df
    except Exception as e:
        print(f"Warning: Could not read truth file: {e}")
        return None


def read_particles(particles_path):
    """Read particles CSV file and return DataFrame."""
    try:
        df = pd.read_csv(particles_path)
        return df
    except Exception as e:
        print(f"Warning: Could not read particles file: {e}")
        return None


def calculate_pt_eta(px, py, pz):
    """Calculate transverse momentum (pt) and pseudorapidity (eta) from momentum components."""
    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    # Avoid division by zero
    eta = np.where(p != pz, 0.5 * np.log((p + pz) / (p - pz)), 0.0)
    return pt, eta


def create_visualization(df_hits, df_truth=None, df_particles=None, color_by='none', max_points=None):
    """
    Create an interactive Plotly figure of the spacepoints.
    
    Args:
        df_hits: DataFrame with x, y, z columns
        df_truth: Optional DataFrame with hit_id and particle_id columns
        df_particles: Optional DataFrame with particle_id and momentum columns
        color_by: 'none', 'particle', 'volume', or 'layer'
        max_points: Maximum number of points to display (for performance)
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Merge truth data if available
    if df_truth is not None and 'hit_id' in df_truth.columns and 'particle_id' in df_truth.columns:
        df_hits = df_hits.merge(df_truth[['hit_id', 'particle_id']], on='hit_id', how='left')
        # Fill NaN particle_ids with 0 (noise hits)
        df_hits['particle_id'] = df_hits['particle_id'].fillna(0)
    
    # Prepare data for plotting
    x = df_hits['x'].values
    y = df_hits['y'].values
    z = df_hits['z'].values
    
    num_points = len(x)
    
    # Downsample if needed
    if max_points is not None and num_points > max_points:
        idx = np.random.default_rng(seed=0).choice(num_points, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
        z = z[idx]
        df_hits = df_hits.iloc[idx].reset_index(drop=True)
        info_msg = f" (downsampled from {num_points} to {max_points} points)"
    else:
        info_msg = ""
    
    # Determine color column and title
    color_col = None
    title_suffix = ""
    
    if color_by == 'particle':
        if 'particle_id' in df_hits.columns:
            color_col = df_hits['particle_id'].astype(str)
            unique_particles = df_hits['particle_id'].nunique()
            title_suffix = f" — {unique_particles} unique particles"
        else:
            print("Warning: particle_id not available, coloring by volume instead")
            color_by = 'volume'
    
    if color_by == 'volume':
        if 'volume_id' in df_hits.columns:
            color_col = df_hits['volume_id'].astype(str)
            title_suffix = f" — {df_hits['volume_id'].nunique()} volumes"
        else:
            print("Warning: volume_id not available, using no coloring")
            color_by = 'none'
    
    if color_by == 'layer':
        if 'layer_id' in df_hits.columns:
            color_col = df_hits['layer_id'].astype(str)
            title_suffix = f" — {df_hits['layer_id'].nunique()} layers"
        else:
            print("Warning: layer_id not available, using no coloring")
            color_by = 'none'
    
    # Create hover text
    hover_text = []
    for i in range(len(df_hits)):
        hover_info = f"Hit {df_hits.iloc[i]['hit_id']}<br>"
        hover_info += f"x: {x[i]:.2f} mm<br>y: {y[i]:.2f} mm<br>z: {z[i]:.2f} mm<br>"
        if 'volume_id' in df_hits.columns:
            hover_info += f"volume_id: {df_hits.iloc[i]['volume_id']}<br>"
        if 'layer_id' in df_hits.columns:
            hover_info += f"layer_id: {df_hits.iloc[i]['layer_id']}<br>"
        if 'module_id' in df_hits.columns:
            hover_info += f"module_id: {df_hits.iloc[i]['module_id']}<br>"
        if 'particle_id' in df_hits.columns:
            pid = df_hits.iloc[i]['particle_id']
            if pid != 0:
                hover_info += f"particle_id: {int(pid)}<br>"
            else:
                hover_info += f"particle_id: noise<br>"
        hover_text.append(hover_info)
    
    # Create figure
    if color_col is not None:
        # Use plotly express for automatic color handling
        df_plot = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'color': color_col,
            'hover': hover_text
        })
        
        fig = px.scatter_3d(
            df_plot,
            x='x', y='y', z='z',
            color='color',
            title=f"Spacepoints ({len(x)} hits){title_suffix}{info_msg}",
            labels={'color': color_by.title()}
        )
        
        # Update marker properties
        fig.update_traces(
            marker=dict(size=3, opacity=0.8),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text
        )
    else:
        # Simple scatter without coloring
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=3,
                color='steelblue',
                opacity=0.8
            ),
            text=hover_text,
            hoverinfo='text',
            name=f'Hits ({len(x)})'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Spacepoints ({len(x)} hits){info_msg}",
                x=0.5,
                xanchor='center'
            )
        )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='x (mm)', backgroundcolor="white", gridcolor="lightgray"),
            yaxis=dict(title='y (mm)', backgroundcolor="white", gridcolor="lightgray"),
            zaxis=dict(title='z (mm)', backgroundcolor="white", gridcolor="lightgray"),
            aspectmode='data'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        hovermode='closest'
    )
    
    # Try to add particle info to title if available
    if df_particles is not None and 'particle_id' in df_hits.columns:
        # Find primary particle (most hits)
        particle_counts = df_hits['particle_id'].value_counts()
        if len(particle_counts) > 0 and particle_counts.index[0] != 0:
            primary_pid = particle_counts.index[0]
            if 'particle_id' in df_particles.columns:
                particle_row = df_particles[df_particles['particle_id'] == primary_pid]
                if len(particle_row) > 0:
                    px_val = particle_row.iloc[0].get('px', None)
                    py_val = particle_row.iloc[0].get('py', None)
                    pz_val = particle_row.iloc[0].get('pz', None)
                    if px_val is not None and py_val is not None and pz_val is not None:
                        pt, eta = calculate_pt_eta(px_val, py_val, pz_val)
                        current_title = fig.layout.title.text
                        fig.update_layout(
                            title=dict(
                                text=f"{current_title} — Primary: Pt={pt:.3f}, eta={eta:.3f}",
                                x=0.5,
                                xanchor='center'
                            )
                        )
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive 3D HTML visualization of spacepoints from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_spacepoints.py ML_data_trainSamples_100_events/event000001000
    # Visualizes hits from event000001000-hits.csv
  
  python visualize_spacepoints.py ML_data_trainSamples_100_events/event000001000 --color-by particle
    # Colors hits by particle_id (requires truth CSV)
  
  python visualize_spacepoints.py ML_data_trainSamples_100_events/event000001000 --color-by volume --max-points 50000
    # Colors by volume_id and limits to 50000 points
        """
    )
    parser.add_argument(
        'event_prefix',
        type=str,
        help='Event prefix (e.g., "ML_data_trainSamples_100_events/event000001000") - will look for <prefix>-hits.csv'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output HTML file path (default: <event_prefix>-spacepoints.html)'
    )
    parser.add_argument(
        '--color-by',
        type=str,
        choices=['none', 'particle', 'volume', 'layer'],
        default='none',
        help='Color points by: none, particle (requires truth CSV), volume, or layer (default: none)'
    )
    parser.add_argument(
        '--max-points',
        type=int,
        default=None,
        help='Maximum number of points to display (for performance with large datasets)'
    )
    
    args = parser.parse_args()
    
    # Determine file paths
    event_prefix = Path(args.event_prefix)
    if not event_prefix.is_absolute():
        # Try relative to current directory
        event_prefix = Path.cwd() / event_prefix
    
    hits_path = event_prefix.parent / f"{event_prefix.name}-hits.csv"
    truth_path = event_prefix.parent / f"{event_prefix.name}-truth.csv"
    particles_path = event_prefix.parent / f"{event_prefix.name}-particles.csv"
    
    # Validate hits file
    if not hits_path.exists():
        print(f"Error: Hits file not found: {hits_path}")
        sys.exit(1)
    
    print(f"Loading hits from: {hits_path}")
    try:
        df_hits = read_hits(hits_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Hits loaded: {len(df_hits)} points")
    
    # Load truth and particles if available
    df_truth = None
    if truth_path.exists():
        print(f"Loading truth from: {truth_path}")
        df_truth = read_truth(truth_path)
        if df_truth is not None:
            print(f"Truth loaded: {len(df_truth)} entries")
    
    df_particles = None
    if particles_path.exists():
        print(f"Loading particles from: {particles_path}")
        df_particles = read_particles(particles_path)
        if df_particles is not None:
            print(f"Particles loaded: {len(df_particles)} entries")
    
    # Determine output path
    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = event_prefix.parent / f"{event_prefix.name}-spacepoints.html"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating visualization (color_by={args.color_by})...")
    try:
        fig = create_visualization(
            df_hits,
            df_truth=df_truth,
            df_particles=df_particles,
            color_by=args.color_by,
            max_points=args.max_points
        )
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Writing interactive HTML to: {output_path}")
    try:
        fig.write_html(str(output_path), include_plotlyjs='cdn')
    except Exception as e:
        print(f"Error writing HTML file: {e}")
        sys.exit(1)
    
    print("\n✓ Visualization complete!")
    print(f"\nOpen in browser: {output_path.absolute()}")


if __name__ == '__main__':
    main()
