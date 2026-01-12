#!/usr/bin/env python3
"""
create_evaluated_visual.py - Visualize GNN edge classification results

Creates an interactive 3D HTML visualization showing how the GNN model classified
edges in a test graph. Edges are colored by classification correctness:
  - Dark green: Correctly classified true edges (True Positives)
  - Light gray: Correctly classified false edges (True Negatives)
  - Red: Incorrectly classified true edges (False Negatives - missed tracks)
  - Orange: Incorrectly classified false edges (False Positives - false connections)

Usage:
  python create_evaluated_visual.py <model_name> <test_index> [--edge-cut THRESHOLD] [--output FILE]

Examples:
  python create_evaluated_visual.py epoch9_900 1
    # Evaluates 1st test graph with model epoch9_900
  
  python create_evaluated_visual.py epoch23_900 5 --edge-cut 0.3
    # Evaluates 5th test graph with threshold 0.3
  
  python create_evaluated_visual.py epoch9_900 1 --output results/graph1.html
    # Custom output location
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

try:
    import plotly.graph_objects as go
except ImportError:
    print("Error: plotly is required. Install with: pip install plotly")
    sys.exit(1)

from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN


def find_checkpoint(model_name, base_dir=None):
    """Find checkpoint file in saved_models directory."""
    if base_dir is None:
        base_dir = Path(__file__).parent / "saved_models"
    
    checkpoint_path = base_dir / f"{model_name}.ckpt"
    if checkpoint_path.exists():
        return checkpoint_path
    
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def cylindrical_to_cartesian(r, phi, z):
    """Convert cylindrical coordinates (r, phi, z) to Cartesian (x, y, z)."""
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z


def evaluate_and_visualize(model_name, test_index, edge_cut=0.5, config_file='acorn_configs/minimal_gnn_train.yaml', output_path=None):
    """
    Load model, evaluate on a specific test graph, and create visualization.
    
    Args:
        model_name: Name of model checkpoint
        test_index: Index of test graph (1-based)
        edge_cut: Classification threshold
        config_file: Path to config file
        output_path: Output HTML file path
    """
    print("=" * 70)
    print(f"EVALUATING MODEL: {model_name} on test graph #{test_index}")
    print("=" * 70)
    print()
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find checkpoint
    checkpoint_path = find_checkpoint(model_name)
    print(f"Found checkpoint: {checkpoint_path}")
    
    # Load model
    print("Loading model from checkpoint...")
    model = InteractionGNN.load_from_checkpoint(
        str(checkpoint_path),
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Setup data
    test_data_dir = Path(__file__).parent / "data" / "graph_constructed"
    model.hparams['input_dir'] = str(test_data_dir)
    model.hparams['stage_dir'] = str(test_data_dir)
    model.hparams['data_split'] = config['data_split']
    
    print(f"Loading test data from: {test_data_dir}/testset")
    model.setup(stage='test')
    
    # Get test dataset
    test_loader = model.test_dataloader()
    test_dataset = test_loader.dataset
    
    # Validate index
    if test_index < 1 or test_index > len(test_dataset):
        raise ValueError(f"Test index {test_index} out of range. Available: 1-{len(test_dataset)}")
    
    # Get specific graph (convert to 0-indexed)
    graph = test_dataset[test_index - 1]
    print(f"Loaded test graph #{test_index}")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print()
    
    # Run inference
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Running inference on {device}...")
    with torch.no_grad():
        graph = graph.to(device)
        output = model(graph)
        scores = torch.sigmoid(output)
        predictions = (scores > edge_cut).cpu().numpy()
    
    # Get ground truth
    truth = graph.edge_y.cpu().numpy().astype(bool)
    predictions = predictions.astype(bool)
    
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
        model_name,
        test_index,
        edge_cut
    )
    
    # Determine output path
    if output_path is None:
        visuals_dir = Path(__file__).parent / 'data' / 'visuals' / 'evaluated' / 'testset'
        visuals_dir.mkdir(parents=True, exist_ok=True)
        output_path = visuals_dir / f"{model_name}_test{test_index:03d}_threshold{edge_cut}.html"
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


def create_classification_visualization(graph, tp_mask, tn_mask, fn_mask, fp_mask, model_name, test_index, edge_cut):
    """
    Create interactive Plotly visualization with edges colored by classification.
    
    Args:
        graph: PyTorch Geometric Data object
        tp_mask: Boolean mask for true positives
        tn_mask: Boolean mask for true negatives
        fn_mask: Boolean mask for false negatives
        fp_mask: Boolean mask for false positives
        model_name: Name of the model
        test_index: Index of test graph
        edge_cut: Classification threshold
    """
    # Extract node positions
    r = graph.hit_r.cpu().numpy()
    phi = graph.hit_phi.cpu().numpy()
    z = graph.hit_z.cpu().numpy()
    
    # Convert to Cartesian
    x, y, z_cart = cylindrical_to_cartesian(r, phi, z)
    num_nodes = len(x)
    
    # Extract edges
    edge_index = graph.edge_index.cpu().numpy()
    
    # Create figure
    fig = go.Figure()
    
    # Add edges for each category (order matters for layering)
    # 1. True Negatives (light gray, bottom layer, very transparent)
    if tn_mask.sum() > 0:
        edge_x, edge_y, edge_z = [], [], []
        for i in np.where(tn_mask)[0]:
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_x.extend([x[src], x[dst], None])
            edge_y.extend([y[src], y[dst], None])
            edge_z.extend([z_cart[src], z_cart[dst], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='lightgray', width=0.5),
            opacity=0.05,
            name=f'True Negatives ({tn_mask.sum()})',
            hoverinfo='skip',
            showlegend=True
        ))
    
    # 2. False Positives (orange)
    if fp_mask.sum() > 0:
        edge_x, edge_y, edge_z = [], [], []
        for i in np.where(fp_mask)[0]:
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_x.extend([x[src], x[dst], None])
            edge_y.extend([y[src], y[dst], None])
            edge_z.extend([z_cart[src], z_cart[dst], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='orange', width=2),
            opacity=0.7,
            name=f'False Positives ({fp_mask.sum()})',
            hoverinfo='skip',
            showlegend=True
        ))
    
    # 3. False Negatives (red)
    if fn_mask.sum() > 0:
        edge_x, edge_y, edge_z = [], [], []
        for i in np.where(fn_mask)[0]:
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_x.extend([x[src], x[dst], None])
            edge_y.extend([y[src], y[dst], None])
            edge_z.extend([z_cart[src], z_cart[dst], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='red', width=2),
            opacity=0.7,
            name=f'False Negatives ({fn_mask.sum()})',
            hoverinfo='skip',
            showlegend=True
        ))
    
    # 4. True Positives (dark green, top layer)
    if tp_mask.sum() > 0:
        edge_x, edge_y, edge_z = [], [], []
        for i in np.where(tp_mask)[0]:
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_x.extend([x[src], x[dst], None])
            edge_y.extend([y[src], y[dst], None])
            edge_z.extend([z_cart[src], z_cart[dst], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='darkgreen', width=2.5),
            opacity=0.8,
            name=f'True Positives ({tp_mask.sum()})',
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
    
    # Calculate metrics for title
    total_edges = len(tp_mask)
    accuracy = (tp_mask.sum() + tn_mask.sum()) / total_edges
    precision = tp_mask.sum() / (tp_mask.sum() + fp_mask.sum()) if (tp_mask.sum() + fp_mask.sum()) > 0 else 0
    recall = tp_mask.sum() / (tp_mask.sum() + fn_mask.sum()) if (tp_mask.sum() + fn_mask.sum()) > 0 else 0
    
    # Update layout
    title_text = f"Model: {model_name} | Test Graph #{test_index} | Threshold: {edge_cut}<br>"
    title_text += f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}"
    
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
        description='Visualize GNN edge classification results on a test graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_evaluated_visual.py epoch9_900 1
    # Evaluate 1st test graph with model epoch9_900
  
  python create_evaluated_visual.py epoch23_900 5 --edge-cut 0.3
    # Evaluate 5th test graph with threshold 0.3
  
  python create_evaluated_visual.py epoch9_900 1 --output results/graph1.html
    # Save to custom location

Edge colors:
  Dark green  - True Positives (correctly found tracks)
  Light gray  - True Negatives (correctly rejected false edges)
  Red         - False Negatives (missed tracks)
  Orange      - False Positives (false connections)
        """
    )
    parser.add_argument(
        'model_name',
        type=str,
        help='Name of the model checkpoint (e.g., epoch9_900)'
    )
    parser.add_argument(
        'test_index',
        type=int,
        help='Index of test graph to visualize (1-based, e.g., 1 for first test graph)'
    )
    parser.add_argument(
        '--edge-cut',
        type=float,
        default=0.5,
        help='Score threshold for edge classification (default: 0.5)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='acorn_configs/minimal_gnn_train.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output HTML file path (default: data/visuals/evaluated/testset/<model>_test<N>_threshold<T>.html)'
    )
    
    args = parser.parse_args()
    
    try:
        evaluate_and_visualize(
            args.model_name,
            args.test_index,
            args.edge_cut,
            args.config,
            args.output
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
