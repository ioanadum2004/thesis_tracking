#!/usr/bin/env python3
"""
create_visual_raw_csv_spacepoints.py <event_prefix> [--output OUTPUT] [--color-by COLOR_BY] [--max-points MAX]

Creates an interactive 3D HTML visualization of spacepoints from CSV files.
Supports two data formats:
  1. ML_data format: x,y,z columns, separate truth.csv for particle_id
  2. ACTS format: tx,ty,tz columns, particle_id already in hits file

Usage examples:
  python create_visual_raw_csv_spacepoints.py ML_data_trainSamples_100_events/event000001000
    # Visualizes hits from event000001000-hits.csv (colors by particle_id by default)
  
  python create_visual_raw_csv_spacepoints.py data/csv/event000000000
    # Visualizes hits from event000000000-hits.csv (ACTS format, colors by particle_id by default)
  
  python create_visual_raw_csv_spacepoints.py data/csv/event000000000 --color-by none
    # No coloring (single color)
  
  python create_visual_raw_csv_spacepoints.py ML_data_trainSamples_100_events/event000001000 --color-by volume --max-points 50000
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
    print("Error: plotly is required. Install with: pip install plotly pandas numpy")
    sys.exit(1)


def detect_format(df_hits):
    """
    Detect the data format based on column names.
    Returns: 'ml_data' or 'acts'
    """
    if 'x' in df_hits.columns and 'y' in df_hits.columns and 'z' in df_hits.columns:
        return 'ml_data'
    elif 'tx' in df_hits.columns and 'ty' in df_hits.columns and 'tz' in df_hits.columns:
        return 'acts'
    else:
        raise RuntimeError("Could not detect data format. Expected either (x,y,z) or (tx,ty,tz) columns")


def read_hits(hits_path):
    """Read hits CSV file and return DataFrame with normalized column names."""
    try:
        df = pd.read_csv(hits_path)
        format_type = detect_format(df)
        
        # Normalize column names for consistent processing
        if format_type == 'acts':
            # Rename tx,ty,tz to x,y,z for consistent processing
            df = df.rename(columns={'tx': 'x', 'ty': 'y', 'tz': 'z'})
            # Use index column as hit_id if hit_id doesn't exist
            if 'hit_id' not in df.columns and 'index' in df.columns:
                df['hit_id'] = df['index']
        
        # Verify we have x, y, z now
        required_cols = ['x', 'y', 'z']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns: {missing}")
        
        return df, format_type
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


def read_particles(particles_path, particles_simulated_path=None):
    """
    Read particles CSV file and return DataFrame.
    Tries particles.csv first, then particles_simulated.csv
    """
    # Try particles.csv first (ML_data format)
    if particles_path.exists():
        try:
            df = pd.read_csv(particles_path)
            return df
        except Exception as e:
            print(f"Warning: Could not read particles file: {e}")
    
    # Try particles_simulated.csv (ACTS format)
    if particles_simulated_path is not None and particles_simulated_path.exists():
        try:
            df = pd.read_csv(particles_simulated_path)
            return df
        except Exception as e:
            print(f"Warning: Could not read particles_simulated file: {e}")
    
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
        df_hits: DataFrame with x, y, z columns (normalized)
        df_truth: Optional DataFrame with hit_id and particle_id columns (for ML_data format)
        df_particles: Optional DataFrame with particle_id and momentum columns
        color_by: 'none', 'particle', 'volume', or 'layer'
        max_points: Maximum number of points to display (for performance)
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Merge truth data if available (for ML_data format)
    if df_truth is not None and 'hit_id' in df_truth.columns and 'particle_id' in df_truth.columns:
        if 'hit_id' in df_hits.columns:
            df_hits = df_hits.merge(df_truth[['hit_id', 'particle_id']], on='hit_id', how='left')
            # Fill NaN particle_ids with 0 (noise hits)
            df_hits['particle_id'] = df_hits['particle_id'].fillna(0)
    
    # Build particle info dictionary (particle_id -> (pt, eta))
    particle_info = {}
    if df_particles is not None and 'particle_id' in df_particles.columns:
        for _, row in df_particles.iterrows():
            pid = row['particle_id']
            px_val = row.get('px', None)
            py_val = row.get('py', None)
            pz_val = row.get('pz', None)
            if px_val is not None and py_val is not None and pz_val is not None:
                pt, eta = calculate_pt_eta(px_val, py_val, pz_val)
                particle_info[int(pid)] = (pt, eta)
    
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
    
    # Always show particle count in title if available
    if 'particle_id' in df_hits.columns:
        unique_particles = len(df_hits[df_hits['particle_id'] > 0]['particle_id'].unique())
        title_suffix = f" — {unique_particles} unique particles"
    
    if color_by == 'particle':
        if 'particle_id' in df_hits.columns:
            color_col = df_hits['particle_id'].astype(str)
        else:
            print("Warning: particle_id not available, coloring by volume instead")
            color_by = 'volume'
    
    if color_by == 'volume':
        if 'volume_id' in df_hits.columns:
            color_col = df_hits['volume_id'].astype(str)
            title_suffix = f" — {df_hits['volume_id'].nunique()} volumes"
        elif 'geometry_id' in df_hits.columns:
            # For ACTS format, might need to extract volume_id from geometry_id
            print("Warning: volume_id not directly available (geometry_id present), using no coloring")
            color_by = 'none'
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
    
    # Create hover text - will use DataFrame values after df_plot is created to ensure exact match
    hover_text = []
    for i in range(len(df_hits)):
        hit_id = df_hits.iloc[i].get('hit_id', i)
        hover_info = f"Hit {hit_id}<br>"
        hover_text.append(hover_info)
        if 'volume_id' in df_hits.columns:
            hover_info += f"volume_id: {df_hits.iloc[i]['volume_id']}<br>"
        if 'geometry_id' in df_hits.columns:
            hover_info += f"geometry_id: {df_hits.iloc[i]['geometry_id']}<br>"
        if 'layer_id' in df_hits.columns:
            hover_info += f"layer_id: {df_hits.iloc[i]['layer_id']}<br>"
        if 'module_id' in df_hits.columns:
            hover_info += f"module_id: {df_hits.iloc[i]['module_id']}<br>"
        if 'particle_id' in df_hits.columns:
            pid = int(df_hits.iloc[i]['particle_id'])
            if pid != 0:
                hover_info += f"particle_id: {pid}"
                # Add pt and eta if available
                if pid in particle_info:
                    pt, eta = particle_info[pid]
                    hover_info += f" (Pt={pt:.3f} GeV, η={eta:.3f})"
                hover_info += "<br>"
            else:
                hover_info += f"particle_id: noise<br>"
        hover_text.append(hover_info)
    
    # Create hover text using the exact numpy arrays that will be plotted
    hover_text_final = []
    for i in range(len(x)):
        hit_id = df_hits.iloc[i].get('hit_id', i)
        # Use numpy arrays directly (same as what's plotted) to ensure exact match
        x_val = float(x[i])
        y_val = float(y[i])
        z_val = float(z[i])
        r_val = np.sqrt(x_val**2 + y_val**2)
        
        hover_info = f"Hit {hit_id}<br>"
        hover_info += f"x: {x_val:.2f} mm<br>y: {y_val:.2f} mm<br>z: {z_val:.2f} mm<br>r: {r_val:.2f} mm<br>"
        
        if 'volume_id' in df_hits.columns:
            hover_info += f"volume_id: {df_hits.iloc[i]['volume_id']}<br>"
        if 'geometry_id' in df_hits.columns:
            hover_info += f"geometry_id: {df_hits.iloc[i]['geometry_id']}<br>"
        if 'layer_id' in df_hits.columns:
            hover_info += f"layer_id: {df_hits.iloc[i]['layer_id']}<br>"
        if 'module_id' in df_hits.columns:
            hover_info += f"module_id: {df_hits.iloc[i]['module_id']}<br>"
        if 'particle_id' in df_hits.columns:
            pid = int(df_hits.iloc[i]['particle_id'])
            if pid != 0:
                hover_info += f"particle_id: {pid}"
                # Add pt and eta if available
                if pid in particle_info:
                    pt, eta = particle_info[pid]
                    hover_info += f" (Pt={pt:.3f} GeV, η={eta:.3f})"
                hover_info += "<br>"
            else:
                hover_info += f"particle_id: noise<br>"
        
        hover_text_final.append(hover_info)
    
    # Create figure using go.Scatter3d directly (like other visualization scripts) for exact coordinate control
    fig = go.Figure()
    
    if color_col is not None:
        # Group points by color and create separate traces
        if color_by == 'particle' and 'particle_id' in df_hits.columns:
            # Group by particle ID for better legend control
            unique_particles = sorted(df_hits['particle_id'].unique())
            colors = px.colors.qualitative.Set3
            
            for i, pid in enumerate(unique_particles):
                mask = (df_hits['particle_id'] == pid).values  # Convert to numpy array
                pid_x = x[mask]
                pid_y = y[mask]
                pid_z = z[mask]
                pid_hover = [hover_text_final[j] for j in range(len(x)) if mask[j]]
                
                if pid == 0:
                    name = 'Noise'
                    color = 'gray'
                else:
                    if pid in particle_info:
                        pt, eta = particle_info[int(pid)]
                        name = f"Particle {int(pid)} (Pt={pt:.3f} GeV, η={eta:.3f})"
                    else:
                        name = f"Particle {int(pid)}"
                    color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter3d(
                    x=pid_x,
                    y=pid_y,
                    z=pid_z,
                    mode='markers',
                    marker=dict(size=3, color=color, opacity=0.8),
                    text=pid_hover,
                    hoverinfo='text',
                    name=name,
                    showlegend=True
                ))
        else:
            # For volume/layer coloring, use plotly express but then extract traces
            df_plot = pd.DataFrame({
                'x': x,
                'y': y,
                'z': z,
                'color': color_col
            })
            
            temp_fig = px.scatter_3d(
                df_plot,
                x='x', y='y', z='z',
                color='color',
                title=f"Spacepoints ({len(x)} hits){title_suffix}{info_msg}"
            )
            
            # Copy traces to main figure with hover text
            for i, trace in enumerate(temp_fig.data):
                trace.text = hover_text_final
                trace.hoverinfo = 'text'
                fig.add_trace(trace)
    else:
        # No coloring - single trace
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=3, color='steelblue', opacity=0.8),
            text=hover_text_final,
            hoverinfo='text',
            name=f'Hits ({len(x)})'
        ))
    
    # Set title
    fig.update_layout(
        title=dict(
            text=f"Spacepoints ({len(x)} hits){title_suffix}{info_msg}",
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
    
    # Title already set correctly - just show spacepoints count and particle count
    # PT/eta info is shown in legend, not title
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive 3D HTML visualization of spacepoints from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_visual_raw_csv_spacepoints.py ML_data_trainSamples_100_events/event000001000
    # Visualizes hits from event000001000-hits.csv (ML_data format)
  
  python create_visual_raw_csv_spacepoints.py data/csv/event000000000
    # Visualizes hits from event000000000-hits.csv (ACTS format)
  
  python create_visual_raw_csv_spacepoints.py data/csv/event000000000 --color-by particle
    # Colors hits by particle_id
  
  python create_visual_raw_csv_spacepoints.py ML_data_trainSamples_100_events/event000001000 --color-by volume --max-points 50000
    # Colors by volume_id and limits to 50000 points
        """
    )
    parser.add_argument(
        'event_prefix',
        type=str,
        help='Event prefix (e.g., "data/csv/event000000000" or "ML_data_trainSamples_100_events/event000001000")'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output HTML file path (default: data/visuals/raw_spacepoints/<dataset>/<event_prefix>-spacepoints.html)'
    )
    parser.add_argument(
        '--color-by',
        type=str,
        choices=['none', 'particle', 'volume', 'layer'],
        default='particle',
        help='Color points by: none, particle (requires particle_id), volume, or layer (default: particle)'
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
    
    # Determine parent directory and base name
    # Handle both cases: full path or just prefix
    if event_prefix.suffix:  # Has a file extension
        # Treat as file path, get parent and stem
        parent_dir = event_prefix.parent
        base_name = event_prefix.stem
    else:
        # Treat as directory/prefix path
        # Extract the last component as base_name
        base_name = event_prefix.name
        parent_dir = event_prefix.parent
    
    hits_path = parent_dir / f"{base_name}-hits.csv"
    truth_path = parent_dir / f"{base_name}-truth.csv"
    particles_path = parent_dir / f"{base_name}-particles.csv"
    particles_simulated_path = parent_dir / f"{base_name}-particles_simulated.csv"
    
    # Validate hits file
    if not hits_path.exists():
        print(f"Error: Hits file not found: {hits_path}")
        sys.exit(1)
    
    print(f"Loading hits from: {hits_path}")
    try:
        df_hits, format_type = read_hits(hits_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Hits loaded: {len(df_hits)} points (format: {format_type})")
    
    # Load truth and particles if available
    df_truth = None
    if format_type == 'ml_data' and truth_path.exists():
        print(f"Loading truth from: {truth_path}")
        df_truth = read_truth(truth_path)
        if df_truth is not None:
            print(f"Truth loaded: {len(df_truth)} entries")
    
    df_particles = None
    if particles_path.exists() or particles_simulated_path.exists():
        print(f"Loading particles...")
        df_particles = read_particles(particles_path, particles_simulated_path)
        if df_particles is not None:
            print(f"Particles loaded: {len(df_particles)} entries")
    
    # Determine output path
    if args.output is not None:
        output_path = Path(args.output)
    else:
        # Default: save to data/visuals/raw_spacepoints/<dataset>/
        # Try to detect dataset from path structure
        script_dir = Path(__file__).resolve().parent
        visuals_dir = script_dir / 'data' / 'visuals' / 'raw_spacepoints'
        
        dataset_name = None
        # Check if any parent directory is trainset/valset/testset
        for parent in hits_path.parents:
            if parent.name in ['trainset', 'valset', 'testset']:
                dataset_name = parent.name
                break
        
        if dataset_name:
            output_path = visuals_dir / dataset_name / f"{base_name}-spacepoints.html"
        else:
            # Fallback: save to raw_spacepoints root
            output_path = visuals_dir / f"{base_name}-spacepoints.html"
    
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
