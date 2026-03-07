#!/usr/bin/env python3
"""
create_visual_inferred_graph.py - Visualize GNN inference results

Creates an interactive 3D HTML visualization showing how the GNN model classified
edges in graphs from the inference stage. Edges are colored by classification correctness:
  - Dark green: Correctly classified true edges (True Positives)
  - Light gray: Correctly classified false edges (True Negatives)
  - Red: Incorrectly classified true edges (False Negatives - missed tracks)
  - Orange: Incorrectly classified false edges (False Positives - false connections)

This script loads graphs that already have edge_scores from infer_gnn.py (in data/gnn_stage/).

Usage:
  python create_visual_inferred_graph.py <dataset> <index> [--edge-cut THRESHOLD] [--output FILE]

Examples:
  python create_visual_inferred_graph.py trainset 1
    # Visualizes 1st graph from data/gnn_stage/trainset/
  
  python create_visual_inferred_graph.py valset 3
    # Visualizes 3rd validation graph
  
  python create_visual_inferred_graph.py testset 5 --edge-cut 0.3
    # Visualizes 5th test graph with threshold 0.3
  
  python create_visual_inferred_graph.py trainset 1 --output custom/path.html
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

from visual_utils import (
    validate_dataset_name,
    extract_node_coordinates,
    get_standard_scene_layout,
    build_edge_coordinates,
    create_node_hover_text
)


def load_inferred_graph(dataset_name, index, input_dir=None):
    """
    Load a graph from the inference output directory (gnn_stage).
    
    Args:
        dataset_name: 'trainset', 'valset', or 'testset'
        index: Index of graph (1-based)
        input_dir: Directory containing gnn_stage data (default: data/gnn_stage)
    
    Returns:
        graph: PyTorch Geometric Data object with edge_scores
    """
    script_dir = Path(__file__).resolve().parent
    
    if input_dir is None:
        input_dir = script_dir.parent / 'data' / 'gnn_stage'
    else:
        input_dir = Path(input_dir)
    
    # Validate dataset name
    validate_dataset_name(dataset_name)
    
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
    
    # Check if graph has edge_scores (or scores if prefixes were removed)
    # Acorn may save as 'scores' if variable_with_prefix is False
    if not hasattr(graph, 'edge_scores'):
        # Check if it has 'scores' instead (unprefixed version)
        if hasattr(graph, 'scores'):
            # Rename scores to edge_scores for consistency
            graph.edge_scores = graph.scores
            print(f"Note: Found 'scores' attribute, using it as 'edge_scores'")
        else:
            # Check a few other files to see if any have edge_scores or scores
            print(f"\nWarning: Graph {graph_path.name} does not have edge_scores or scores.")
            print(f"Checking other files in {dataset_name}...")
            files_with_scores = []
            files_without_scores = []
            for test_file in graph_files[:min(10, len(graph_files))]:
                try:
                    test_graph = torch.load(test_file, map_location='cpu', weights_only=False)
                    if hasattr(test_graph, 'edge_scores') or hasattr(test_graph, 'scores'):
                        files_with_scores.append(test_file.name)
                    else:
                        files_without_scores.append(test_file.name)
                except Exception as e:
                    print(f"  Warning: Could not check {test_file.name}: {e}")
            
            if files_with_scores:
                print(f"  ✓ Found {len(files_with_scores)} files WITH edge_scores/scores: {files_with_scores[:5]}")
                print(f"  ✗ Found {len(files_without_scores)} files WITHOUT edge_scores/scores: {files_without_scores[:5]}")
                print(f"\n  Suggestion: The files without edge_scores may have been created before inference ran.")
                print(f"  Try deleting files in {dataset_dir} and re-running infer_gnn.py")
            else:
                print(f"  ✗ No files found with edge_scores or scores in {dataset_name}.")
                print(f"  Inference may not have completed successfully or files were not saved correctly.")
                print(f"\n  Suggestion: Re-run infer_gnn.py to generate edge scores.")
            
            # Provide helpful error message with available attributes
            available_attrs = [attr for attr in dir(graph) if not attr.startswith('_') and not callable(getattr(graph, attr, None))]
            attrs_str = ', '.join(available_attrs[:15])  # Show first 15
            if len(available_attrs) > 15:
                attrs_str += f", ... ({len(available_attrs)} total)"
            
            raise ValueError(
                f"\nGraph does not have edge_scores or scores attribute.\n"
                f"  Graph file: {graph_path}\n"
                f"  Graph data attributes: {attrs_str}\n"
                f"\n  Solution: Delete existing files in {dataset_dir} and re-run:\n"
                f"    rm -rf {dataset_dir}/*.pyg\n"
                f"    python infer_gnn.py\n"
            )
    
    return graph, graph_path


def visualize_inferred_graph(dataset_name, index, edge_cut=0.5, input_dir=None, output_path=None):
    """
    Load and visualize an inferred graph with edge classification results.
    
    Args:
        dataset_name: Name of dataset ('trainset', 'valset', or 'testset')
        index: Index of graph in dataset (1-based)
        edge_cut: Classification threshold (default: 0.5)
        input_dir: Directory containing gnn_stage data
        output_path: Output HTML file path
    """
    print("=" * 70)
    print(f"VISUALIZING INFERRED GRAPH: {dataset_name} graph #{index}")
    print("=" * 70)
    print()
    
    # Load graph
    graph, graph_path = load_inferred_graph(dataset_name, index, input_dir)
    
    print(f"Graph loaded:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Has edge_scores: {hasattr(graph, 'edge_scores')}")
    if hasattr(graph, 'edge_scores'):
        edge_scores = graph.edge_scores
        print(f"  Edge scores shape: {edge_scores.shape}")
        print(f"  Edge scores range: [{edge_scores.min():.3f}, {edge_scores.max():.3f}]")
        print(f"  Edge scores mean: {edge_scores.mean():.3f}")
    print()
    
    # Get edge scores and truth
    edge_scores = graph.edge_scores.cpu().numpy()
    predictions = (edge_scores > edge_cut).astype(bool)
    
    # Get ground truth (edge_y)
    if not hasattr(graph, 'edge_y'):
        raise ValueError("Graph does not have edge_y (truth labels). Cannot evaluate classification.")
    
    truth = graph.edge_y.cpu().numpy().astype(bool)
    
    # Classify edges into 4 categories
    true_positives = predictions & truth  # Correctly classified true edges
    true_negatives = ~predictions & ~truth  # Correctly classified false edges
    false_negatives = ~predictions & truth  # Missed true edges
    false_positives = predictions & ~truth  # False connections
    
    num_tp = true_positives.sum()
    num_tn = true_negatives.sum()
    num_fn = false_negatives.sum()
    num_fp = false_positives.sum()
    
    print(f"Classification results (threshold={edge_cut}):")
    print(f"  True Positives (TP):  {num_tp:>6} - Correctly found tracks (dark green)")
    print(f"  True Negatives (TN):  {num_tn:>6} - Correctly rejected false edges (light gray)")
    print(f"  False Negatives (FN): {num_fn:>6} - Missed tracks (red)")
    print(f"  False Positives (FP): {num_fp:>6} - False connections (orange)")
    print()
    
    accuracy = (num_tp + num_tn) / len(truth)
    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0
    
    print(f"Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print()
    
    # Create visualization
    print("Creating visualization...")
    fig = create_classification_visualization(
        graph, 
        true_positives, 
        true_negatives, 
        false_negatives, 
        false_positives,
        dataset_name,
        index,
        edge_cut
    )
    
    # Determine output path
    script_dir = Path(__file__).resolve().parent
    if output_path is None:
        visuals_dir = script_dir.parent / 'data' / 'visuals' / 'gnn_inferred_data' / dataset_name
        visuals_dir.mkdir(parents=True, exist_ok=True)
        output_path = visuals_dir / f"{dataset_name}{index:03d}_threshold{edge_cut}.html"
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


def create_classification_visualization(graph, tp_mask, tn_mask, fn_mask, fp_mask, dataset_name, index, edge_cut):
    """
    Create interactive Plotly visualization with edges colored by classification.
    
    Args:
        graph: PyTorch Geometric Data object
        tp_mask: Boolean mask for true positives
        tn_mask: Boolean mask for true negatives
        fn_mask: Boolean mask for false negatives
        fp_mask: Boolean mask for false positives
        dataset_name: Name of dataset ('trainset', 'valset', or 'testset')
        index: Index of graph in dataset
        edge_cut: Classification threshold
    """
    # Extract node positions and convert to Cartesian
    coords = extract_node_coordinates(graph, return_cylindrical=True, return_cartesian=True)
    x, y, z_cart = coords['x'], coords['y'], coords['z_cart']
    r, phi, z = coords['r'], coords['phi'], coords['z']
    num_nodes = len(x)
    
    # Build edges by category using visual_utils
    edge_index = graph.edge_index.cpu().numpy()
    edge_data = build_edge_coordinates(
        edge_index,
        x, y, z_cart,
        masks={
            'tn': tn_mask,
            'fp': fp_mask,
            'fn': fn_mask,
            'tp': tp_mask
        }
    )

    # Create figure
    fig = go.Figure()

    # Add edges for each category (order matters for layering)
    # 1. True Negatives (light gray, bottom layer, very transparent)
    if edge_data['tn'][3] > 0:  # count > 0
        edge_x, edge_y, edge_z, count = edge_data['tn']
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='#A9A9A9', width=0.5),
            opacity=0.05,
            name=f'True Negatives ({count})',
            hoverinfo='skip',
            showlegend=True
        ))

    # 2. False Positives (orange)
    if edge_data['fp'][3] > 0:
        edge_x, edge_y, edge_z, count = edge_data['fp']
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='orange', width=2),
            opacity=0.7,
            name=f'False Positives ({count})',
            hoverinfo='skip',
            showlegend=True
        ))

    # 3. False Negatives (red)
    if edge_data['fn'][3] > 0:
        edge_x, edge_y, edge_z, count = edge_data['fn']
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='red', width=2),
            opacity=0.7,
            name=f'False Negatives ({count})',
            hoverinfo='skip',
            showlegend=True
        ))

    # 4. True Positives (dark green, top layer)
    if edge_data['tp'][3] > 0:
        edge_x, edge_y, edge_z, count = edge_data['tp']
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='darkgreen', width=2.5),
            opacity=0.8,
            name=f'True Positives ({count})',
            hoverinfo='skip',
            showlegend=True
        ))
    
    # Add nodes with hover text
    hover_text = [
        create_node_hover_text(i, {
            'x': x[i], 'y': y[i], 'z_cart': z_cart[i],
            'r': r[i], 'phi': phi[i]
        })
        for i in range(num_nodes)
    ]
    
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
    
    # Calculate metrics for title
    total_edges = len(tp_mask)
    accuracy = (tp_mask.sum() + tn_mask.sum()) / total_edges
    precision = tp_mask.sum() / (tp_mask.sum() + fp_mask.sum()) if (tp_mask.sum() + fp_mask.sum()) > 0 else 0
    recall = tp_mask.sum() / (tp_mask.sum() + fn_mask.sum()) if (tp_mask.sum() + fn_mask.sum()) > 0 else 0
    
    # Update layout with standard scene configuration
    title_text = f"Inferred Graph: {dataset_name.capitalize()} #{index} | Threshold: {edge_cut}<br>"
    title_text += f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}"

    fig.update_layout(**get_standard_scene_layout(title_text, margin_t=60))
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GNN inference results on graphs with edge_scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_visual_inferred_graph.py trainset 1
    # Visualizes 1st graph from data/gnn_stage/trainset/
  
  python create_visual_inferred_graph.py valset 3
    # Visualizes 3rd validation graph
  
  python create_visual_inferred_graph.py testset 5 --edge-cut 0.3
    # Visualizes 5th test graph with threshold 0.3
  
  python create_visual_inferred_graph.py trainset 1 --output custom/path.html
    # Save to custom location

Edge colors:
  Dark green  - True Positives (correctly found tracks)
  Light gray  - True Negatives (correctly rejected false edges)
  Red         - False Negatives (missed tracks)
  Orange      - False Positives (false connections)
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
        '--edge-cut',
        type=float,
        default=0.5,
        help='Score threshold for edge classification (default: 0.5)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory containing gnn_stage data (default: data/gnn_stage)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output HTML file path (default: data/visuals/gnn_inferred_data/<dataset>/<dataset><N>_threshold<T>.html)'
    )
    
    args = parser.parse_args()
    
    try:
        visualize_inferred_graph(
            args.dataset,
            args.index,
            args.edge_cut,
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
