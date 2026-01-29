#!/usr/bin/env python3
"""
create_visual_track_building.py - Visualize proposed tracks from track building stage

Creates an interactive 3D HTML visualization showing the final candidate tracks built
from the track building stage (after applying score cuts to GNN-classified edges).
Tracks are colored based on their correctness according to truth information:
  - Green: Correct track edges (truth_map > 0)
  - Red: Incorrect track edges (truth_map == 0)
  - Orange: Missed track edges (truth_map == -1)

This script loads graphs that have track_edges from build_tracks.py (in data/track_building/).

Usage:
  python create_visual_track_building.py <dataset> <index> [--output FILE]

Examples:
  python create_visual_track_building.py trainset 1
    # Visualizes 1st graph from data/track_building/trainset/

  python create_visual_track_building.py valset 3
    # Visualizes 3rd validation graph

  python create_visual_track_building.py testset 5
    # Visualizes 5th test graph

  python create_visual_track_building.py trainset 1 --output custom/path.html
    # Custom output location
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

try:
    import plotly.graph_objects as go
except ImportError:
    print("Error: plotly is required. Install with: pip install plotly")
    sys.exit(1)


def cylindrical_to_cartesian(r, phi, z):
    """Convert cylindrical coordinates (r, phi, z) to Cartesian (x, y, z)."""
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z


def load_track_building_graph(dataset_name, index, input_dir=None):
    """
    Load a graph from the track building output directory.

    Args:
        dataset_name: 'trainset', 'valset', or 'testset'
        index: Index of graph (1-based)
        input_dir: Directory containing track_building data (default: ../data/track_building)

    Returns:
        graph: PyTorch Geometric Data object with track_edges and truth_map
        graph_path: Path to the loaded graph file
    """
    script_dir = Path(__file__).resolve().parent

    if input_dir is None:
        input_dir = script_dir.parent / 'data' / 'track_building'
    else:
        input_dir = Path(input_dir)

    # Validate dataset name
    if dataset_name not in ['trainset', 'valset', 'testset']:
        raise ValueError(f"Dataset must be 'trainset', 'valset', or 'testset', got '{dataset_name}'")

    dataset_dir = input_dir / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Get all .pyg files sorted
    graph_files = sorted([f for f in dataset_dir.glob('*.pyg')])
    if len(graph_files) == 0:
        raise ValueError(f"No graph files found in {dataset_dir}")

    if index < 1 or index > len(graph_files):
        raise ValueError(f"Index {index} out of range. Available: 1-{len(graph_files)}")

    # Select the graph file at the specified index (convert to 0-indexed)
    graph_path = graph_files[index - 1]
    print(f"Loading {dataset_name} graph #{index} from: {graph_path.name}")

    # Load the graph file
    try:
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load graph file {graph_path}: {e}")

    # Check if graph has track_edges
    if not hasattr(graph, 'track_edges'):
        # Provide helpful error message with available attributes
        available_attrs = [attr for attr in dir(graph) if not attr.startswith('_') and not callable(getattr(graph, attr, None))]
        attrs_str = ', '.join(available_attrs[:15])  # Show first 15
        if len(available_attrs) > 15:
            attrs_str += f", ... ({len(available_attrs)} total)"

        raise ValueError(
            f"\nGraph does not have track_edges attribute.\n"
            f"  Graph file: {graph_path}\n"
            f"  Graph data attributes: {attrs_str}\n"
            f"\n  Solution: Re-run track building:\n"
            f"    python build_tracks.py\n"
        )

    # Check if graph has truth_map
    if not hasattr(graph, 'truth_map'):
        print(f"Warning: Graph does not have truth_map attribute. Track correctness colors will not be available.")

    return graph, graph_path


def visualize_track_building_graph(dataset_name, index, input_dir=None, output_path=None):
    """
    Load and visualize a graph with built tracks from track building stage.

    Args:
        dataset_name: Name of dataset ('trainset', 'valset', or 'testset')
        index: Index of graph in dataset (1-based)
        input_dir: Directory containing track_building data
        output_path: Output HTML file path
    """
    print("=" * 70)
    print(f"VISUALIZING TRACK BUILDING: {dataset_name} graph #{index}")
    print("=" * 70)
    print()

    # Load graph
    graph, graph_path = load_track_building_graph(dataset_name, index, input_dir)

    print(f"Graph loaded:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Track edges: {graph.track_edges.shape[1]}")
    print(f"  Has truth_map: {hasattr(graph, 'truth_map')}")
    print()

    # Create visualization
    print("Creating visualization...")
    fig = create_track_visualization(graph, dataset_name, index)

    # Determine output path
    script_dir = Path(__file__).resolve().parent
    if output_path is None:
        visuals_dir = script_dir.parent / 'data' / 'visuals' / 'track_building' / dataset_name
        visuals_dir.mkdir(parents=True, exist_ok=True)
        output_path = visuals_dir / f"{dataset_name}{index:03d}_tracks.html"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing interactive HTML to: {output_path}")
    fig.write_html(str(output_path), include_plotlyjs='cdn')

    print()
    print("=" * 70)
    print("✓ Visualization complete!")
    print("=" * 70)
    print(f"\nOpen in browser: {output_path.absolute()}")

    return output_path


def create_track_visualization(graph, dataset_name, index):
    """
    Create interactive Plotly visualization with tracks colored by correctness.

    Args:
        graph: PyTorch Geometric Data object with track_edges and truth_map
        dataset_name: Name of dataset ('trainset', 'valset', or 'testset')
        index: Index of graph in dataset
    """
    # Extract node positions
    # Handle both prefixed (hit_r) and non-prefixed (r) names
    if hasattr(graph, 'hit_r'):
        r = graph.hit_r.cpu().numpy()
        phi = graph.hit_phi.cpu().numpy()
        z = graph.hit_z.cpu().numpy()
    elif hasattr(graph, 'r'):
        r = graph.r.cpu().numpy()
        phi = graph.phi.cpu().numpy()
        z = graph.z.cpu().numpy()
    else:
        # Fallback to x, y, z if available (though original script used x, hit_y, z)
        if hasattr(graph, 'x') and hasattr(graph, 'z'):
            # If we have Cartesian coordinates, use them directly
            if hasattr(graph, 'hit_y'):
                x = graph.x.cpu().numpy()
                y = graph.hit_y.cpu().numpy()
                z_cart = graph.z.cpu().numpy()
                # Calculate r and phi for hover info
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                # Convert for consistency
                x, y, z_cart = x, y, z_cart
            else:
                raise ValueError("Graph does not have recognizable coordinate attributes (r/phi/z or x/y/z)")
        else:
            raise ValueError("Graph does not have r, phi, z coordinates (checked both prefixed and non-prefixed)")

    # Convert to Cartesian if we have cylindrical coordinates
    if 'x' not in locals():
        x, y, z_cart = cylindrical_to_cartesian(r, phi, z)

    num_nodes = len(x)

    # Extract track edges
    track_edges = graph.track_edges.cpu().numpy()

    # Extract truth map if available
    has_truth = hasattr(graph, 'truth_map')
    if has_truth:
        truth_map = graph.truth_map.cpu().numpy()

        # Separate edges by correctness
        # Positive integers indicate correct tracks
        correct_mask = truth_map > 0
        # -1 indicates missed tracks
        missed_mask = truth_map == -1
        # 0 or other values indicate incorrect tracks
        incorrect_mask = (truth_map == 0) | ((truth_map < 0) & (truth_map != -1))

        num_correct = correct_mask.sum()
        num_missed = missed_mask.sum()
        num_incorrect = incorrect_mask.sum()

        print(f"Track edge statistics:")
        print(f"  Correct edges:   {num_correct:>6} (green)")
        print(f"  Incorrect edges: {num_incorrect:>6} (red)")
        print(f"  Missed edges:    {num_missed:>6} (orange)")
        print()

    # Create figure
    fig = go.Figure()

    # Add edges by category if truth_map is available
    if has_truth:
        # 1. Correct edges (green)
        if correct_mask.sum() > 0:
            edge_x, edge_y, edge_z = [], [], []
            for i in np.where(correct_mask)[0]:
                src, dst = track_edges[0, i], track_edges[1, i]
                edge_x.extend([x[src], x[dst], None])
                edge_y.extend([y[src], y[dst], None])
                edge_z.extend([z_cart[src], z_cart[dst], None])

            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='green', width=2.5),
                opacity=0.8,
                name=f'Correct Tracks ({num_correct})',
                hoverinfo='skip',
                showlegend=True
            ))

        # 2. Incorrect edges (red)
        if incorrect_mask.sum() > 0:
            edge_x, edge_y, edge_z = [], [], []
            for i in np.where(incorrect_mask)[0]:
                src, dst = track_edges[0, i], track_edges[1, i]
                edge_x.extend([x[src], x[dst], None])
                edge_y.extend([y[src], y[dst], None])
                edge_z.extend([z_cart[src], z_cart[dst], None])

            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='red', width=2),
                opacity=0.7,
                name=f'Incorrect Tracks ({num_incorrect})',
                hoverinfo='skip',
                showlegend=True
            ))

        # 3. Missed edges (orange)
        if missed_mask.sum() > 0:
            edge_x, edge_y, edge_z = [], [], []
            for i in np.where(missed_mask)[0]:
                src, dst = track_edges[0, i], track_edges[1, i]
                edge_x.extend([x[src], x[dst], None])
                edge_y.extend([y[src], y[dst], None])
                edge_z.extend([z_cart[src], z_cart[dst], None])

            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='orange', width=2),
                opacity=0.7,
                name=f'Missed Tracks ({num_missed})',
                hoverinfo='skip',
                showlegend=True
            ))
    else:
        # No truth information - just plot all edges in blue
        edge_x, edge_y, edge_z = [], [], []
        for i in range(track_edges.shape[1]):
            src, dst = track_edges[0, i], track_edges[1, i]
            edge_x.extend([x[src], x[dst], None])
            edge_y.extend([y[src], y[dst], None])
            edge_z.extend([z_cart[src], z_cart[dst], None])

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='blue', width=2),
            opacity=0.7,
            name=f'Track Edges ({track_edges.shape[1]})',
            hoverinfo='skip',
            showlegend=True
        ))

    # Add nodes
    hover_text = []
    for i in range(num_nodes):
        hover_info = f"Node {i}<br>x: {x[i]:.1f} mm<br>y: {y[i]:.1f} mm<br>z: {z_cart[i]:.1f} mm<br>"
        hover_info += f"r: {r[i]:.1f} mm<br>phi: {phi[i]:.3f} rad"
        hover_text.append(hover_info)

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z_cart,
        mode='markers',
        marker=dict(
            size=3,
            color='steelblue',
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        name=f'Nodes ({num_nodes})',
        text=hover_text,
        hoverinfo='text',
        showlegend=True
    ))

    # Create title
    title_text = f"Track Building: {dataset_name.capitalize()} Graph #{index}<br>"
    if has_truth:
        total_edges = track_edges.shape[1]
        title_text += f"Total Track Edges: {total_edges} | Correct: {num_correct} | Incorrect: {num_incorrect} | Missed: {num_missed}"
    else:
        title_text += f"Total Track Edges: {track_edges.shape[1]}"

    # Update layout
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
            bgcolor="rgba(255, 255, 255, 0.9)"
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        hovermode='closest'
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize proposed tracks from track building stage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_visual_track_building.py trainset 1
    # Visualizes 1st graph from data/track_building/trainset/

  python create_visual_track_building.py valset 3
    # Visualizes 3rd validation graph

  python create_visual_track_building.py testset 5
    # Visualizes 5th test graph

  python create_visual_track_building.py trainset 1 --output custom/path.html
    # Save to custom location

Track edge colors (when truth_map is available):
  Green  - Correct track edges (truth_map > 0)
  Red    - Incorrect track edges (truth_map == 0)
  Orange - Missed track edges (truth_map == -1)
        """
    )
    parser.add_argument(
        'dataset',
        type=str,
        choices=['trainset', 'valset', 'testset'],
        help='Dataset to visualize: trainset, valset, or testset'
    )
    parser.add_argument(
        'index',
        type=int,
        help='Index of graph to visualize (1-based, e.g., 1 for first graph)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory containing track_building data (default: ../data/track_building)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output HTML file path (default: ../data/visuals/track_building/<dataset>/<dataset><N>_tracks.html)'
    )

    args = parser.parse_args()

    try:
        visualize_track_building_graph(
            args.dataset,
            args.index,
            args.input_dir,
            args.output
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
