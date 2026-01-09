#!/usr/bin/env python3
"""
create_graph_visual.py <dataset> <index> [--output OUTPUT]
create_graph_visual.py <graph_file.pyg> [--output OUTPUT]

Creates an interactive 3D HTML visualization of a graph from the GNN pipeline.
Displays all nodes (hits) and edges, with edges colored by their truth labels:
  - Red: true edges (hits from same particle)
  - Black: false edges (candidate connections)

Usage examples:
  python create_graph_visual.py trainset 1
    # Visualizes the 1st graph in trainset
  
  python create_graph_visual.py valset 3
    # Visualizes the 3rd graph in valset
  
  python create_graph_visual.py data/graph_constructed/trainset/event000000001-graph.pyg
    # Can still use full path if needed
  
Dependencies: torch, torch_geometric, plotly, pandas, numpy
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

try:
    import plotly.graph_objects as go
    import pandas as pd
except ImportError:
    print("Error: plotly and pandas are required. Install with: pip install plotly pandas")
    sys.exit(1)


def load_graph(graph_path):
    """Load a PyTorch Geometric graph from file."""
    try:
        graph = torch.load(graph_path, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Error loading graph file: {e}")
    
    
    return graph


def cylindrical_to_cartesian(r, phi, z):
    """Convert cylindrical coordinates (r, phi, z) to Cartesian (x, y, z)."""
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z


def create_visualization(graph, r_max=None, k_max=None):
    """
    Create an interactive Plotly figure of the graph.
    Visualizes the graph exactly as constructed - showing all nodes (hits) and edges
    that will be used as input to the GNN model.
    
    Args:
        graph: PyTorch Geometric Data object with nodes and edges
        r_max: Radius parameter used in graph construction (for title)
        k_max: Max neighbors parameter used in graph construction (for title)
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Extract node positions (convert to numpy)
    r = graph.r.numpy()
    phi = graph.phi.numpy()
    z_cyl = graph.z.numpy()
    
    # Convert cylindrical to Cartesian coordinates for plotting
    x, y, z = cylindrical_to_cartesian(r, phi, z_cyl)
    
    num_nodes = len(x)
    
    # Extract edge information
    edge_index = graph.edge_index.numpy()
    edge_y = graph.edge_y.numpy()
    
    # Separate true and false edges
    true_edges_mask = (edge_y == 1)
    false_edges_mask = (edge_y == 0)
    
    true_edges = edge_index[:, true_edges_mask]
    false_edges = edge_index[:, false_edges_mask]
    
    num_true = true_edges.shape[1]
    num_false = false_edges.shape[1]
    
    # Create figure
    fig = go.Figure()
    
    # Add false edges (plot first so they're in background)
    edge_x_false = []
    edge_y_false = []
    edge_z_false = []
    
    for i in range(false_edges.shape[1]):
        src = false_edges[0, i]
        dst = false_edges[1, i]
        edge_x_false.extend([x[src], x[dst], None])
        edge_y_false.extend([y[src], y[dst], None])
        edge_z_false.extend([z[src], z[dst], None])
    
    fig.add_trace(go.Scatter3d(
        x=edge_x_false,
        y=edge_y_false,
        z=edge_z_false,
        mode='lines',
        line=dict(color='black', width=1),
        opacity=0.1,
        name=f'False edges ({num_false})',
        hoverinfo='skip',
        showlegend=True
    ))
    
    # Add true edges (on top)
    edge_x_true = []
    edge_y_true = []
    edge_z_true = []
    
    for i in range(true_edges.shape[1]):
        src = true_edges[0, i]
        dst = true_edges[1, i]
        edge_x_true.extend([x[src], x[dst], None])
        edge_y_true.extend([y[src], y[dst], None])
        edge_z_true.extend([z[src], z[dst], None])
    
    fig.add_trace(go.Scatter3d(
        x=edge_x_true,
        y=edge_y_true,
        z=edge_z_true,
        mode='lines',
        line=dict(color='red', width=2),
        opacity=0.6,
        name=f'True edges ({num_true})',
        hoverinfo='skip',
        showlegend=True
    ))
    
    # Add nodes
    # Try to get particle_id if available (but only if it matches num_nodes)
    particle_ids = None
    if hasattr(graph, 'particle_id') and graph.particle_id is not None:
        pid_tensor = graph.particle_id
        if len(pid_tensor) == num_nodes:
            particle_ids = pid_tensor.numpy()
        else:
            print(f"Warning: particle_id length ({len(pid_tensor)}) doesn't match num_nodes ({num_nodes}), skipping")
    
    hover_text = []
    for i in range(num_nodes):
        hover_info = f"Node {i}<br>x: {x[i]:.1f} mm<br>y: {y[i]:.1f} mm<br>z: {z[i]:.1f} mm<br>"
        hover_info += f"r: {r[i]:.1f} mm<br>phi: {phi[i]:.3f} rad"
        if particle_ids is not None:
            hover_info += f"<br>particle_id: {particle_ids[i]}"
        hover_text.append(hover_info)
    
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=4,
            color='steelblue',
            opacity=0.8,
            line=dict(width=0.5, color='white')
        ),
        name=f'Nodes ({num_nodes})',
        text=hover_text,
        hoverinfo='text',
        showlegend=True
    ))
    
    # Update layout
    title_text = f"Graph: {num_nodes} nodes, {num_true} true edges, {num_false} false edges"
    if r_max is not None and k_max is not None:
        title_text += f"<br>Graph Construction: r_max={r_max}, k_max={k_max}"
    
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center'
        ),
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
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive 3D HTML visualization of a graph from the GNN pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_graph_visual.py trainset 1
    # Visualizes the 1st graph in data/graph_constructed/trainset/
  
  python create_graph_visual.py valset 3
    # Visualizes the 3rd graph in data/graph_constructed/valset/
  
  python create_graph_visual.py data/graph_constructed/trainset/event000000003-graph.pyg
    # Can still use full path if needed
        """
    )
    parser.add_argument(
        'input',
        type=str,
        help='Either: (1) "<dataset> <index>" like "trainset 1" or "valset 3", or (2) full path to .pyg file'
    )
    parser.add_argument(
        'index',
        type=str,
        nargs='?',
        default=None,
        help='Index number (if first argument is dataset name). Ignored if first argument is a file path.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output HTML file path (default: data/visuals/<dataset>/<filename>.html)'
    )
    
    args = parser.parse_args()
    
    # Determine graph file path
    script_dir = Path(__file__).resolve().parent
    graph_constructed_dir = script_dir / 'data' / 'graph_constructed'
    
    # Check if input is "<dataset> <index>" format
    if args.index is not None:
        # Format: "trainset 1" or "valset 3"
        dataset_name = args.input
        try:
            index = int(args.index)
        except ValueError:
            print(f"Error: Index must be a number, got '{args.index}'")
            sys.exit(1)
        
        dataset_dir = graph_constructed_dir / dataset_name
        if not dataset_dir.exists():
            print(f"Error: Dataset directory not found: {dataset_dir}")
            print(f"Available datasets: {[d.name for d in graph_constructed_dir.iterdir() if d.is_dir()]}")
            sys.exit(1)
        
        # Get all .pyg files sorted
        graph_files = sorted([f for f in dataset_dir.glob('*.pyg')])
        if len(graph_files) == 0:
            print(f"Error: No graph files found in {dataset_dir}")
            sys.exit(1)
        
        if index < 1 or index > len(graph_files):
            print(f"Error: Index {index} out of range. Available: 1-{len(graph_files)}")
            sys.exit(1)
        
        graph_path = graph_files[index - 1]  # Convert to 0-indexed
        print(f"Selected: {graph_path.name} (index {index} of {len(graph_files)} in {dataset_name})")
    else:
        # Format: full path to file
        graph_path = Path(args.input)
        if not graph_path.is_absolute():
            # Try relative to script directory
            graph_path = script_dir / graph_path
    
    # Validate input file
    if not graph_path.exists():
        print(f"Error: Graph file not found: {graph_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output is not None:
        output_path = Path(args.output)
    else:
        # Default: save to data/visuals/ preserving dataset structure
        # e.g., data/graph_constructed/trainset/event000000001-graph.pyg 
        #   -> data/visuals/trainset/event000000001-graph.html
        visuals_dir = script_dir / 'data' / 'visuals'
        
        # Try to preserve dataset structure (trainset/valset/testset)
        # Check if graph_path is inside data/graph_constructed/
        try:
            relative_path = graph_path.relative_to(graph_constructed_dir)
            # relative_path will be like: trainset/event000000001-graph.pyg
            output_path = visuals_dir / relative_path.with_suffix('.html')
        except ValueError:
            # If not in expected structure, just use filename
            output_path = visuals_dir / graph_path.name.replace('.pyg', '.html')
        
        # Create visuals directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading graph from: {graph_path}")
    try:
        graph = load_graph(graph_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Graph loaded successfully:")
    print(f"  • Nodes: {graph.num_nodes}")
    print(f"  • Edges: {graph.edge_index.shape[1]}")
    print(f"  • True edges: {graph.edge_y.sum().item()} ({100*graph.edge_y.sum().item()/graph.edge_y.shape[0]:.1f}%)")
    
    # Load graph construction parameters from config
    config_path = script_dir / 'acorn_configs' / 'graph_construction.yaml'
    r_max = None
    k_max = None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'graph_construction' in config:
                r_max = config['graph_construction'].get('r_max')
                k_max = config['graph_construction'].get('k_max')
                print(f"  • Graph construction params: r_max={r_max}, k_max={k_max}")
    except Exception as e:
        print(f"  • Warning: Could not load graph construction params from config: {e}")
    
    print("\nCreating visualization...")
    try:
        fig = create_visualization(graph, r_max=r_max, k_max=k_max)
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


