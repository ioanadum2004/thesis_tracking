#!/usr/bin/env python3
"""
Evaluate metric learning (latent cluster learning) model on test set

Computes TP, FP, TN, FN and other metrics for graph construction performance.

Usage:
    python test_my_latent_model.py <checkpoint_path> [--knn KNN] [--r-max R_MAX]
    
Example:
    python test_my_latent_model.py saved_models/low_pt_latent_f1=0.0149.ckpt
    python test_my_latent_model.py saved_models/low_pt_latent_f1=0.0149.ckpt --knn 500 --r-max 0.15
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Add acorn to path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))

from acorn.stages.graph_construction.models.metric_learning import MetricLearning
from acorn.stages.graph_construction.models.utils import build_edges, graph_intersection
from acorn.stages.graph_construction.graph_construction_stage import EventDataset


def load_model(checkpoint_path):
    """Load trained metric learning model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract hyperparameters
    hparams = checkpoint['hyper_parameters']
    
    print(f"  Model architecture:")
    print(f"    Input features: {hparams['node_features']}")
    print(f"    Hidden layers: {hparams['nb_layer']} x {hparams['emb_hidden']}")
    print(f"    Embedding dim: {hparams['emb_dim']}D")
    print(f"    Activation: {hparams['activation']}")
    print()
    
    # Create model instance
    model = MetricLearning(hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, hparams


def evaluate_model(model, hparams, testset, knn_max=50, r_max=0.15, device='cpu'):
    """
    Evaluate metric learning model on test set
    
    Args:
        model: Trained MetricLearning model
        hparams: Model hyperparameters
        testset: Test dataset
        knn_max: Maximum number of neighbors for KNN graph construction
        r_max: Maximum radius for graph construction
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    # Create data loader
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
    
    # Accumulate metrics across all events
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    total_pred_edges = 0
    total_true_edges = 0
    total_nodes = 0
    
    # Per-event metrics for analysis
    event_metrics = []
    
    print(f"Evaluating on {len(testset)} test events...")
    print(f"Graph construction parameters: r_max={r_max}, k_max={knn_max}")
    print()
    print("NOTE: For graph construction stage, high recall (low FN) is critical.")
    print(f"      Current k_max={knn_max} may be too low. Try k_max=1000-2000 for better recall.")
    print()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing events")):
            batch = batch.to(device)
            
            # Get embeddings
            embedding = model.apply_embedding(batch)
            
            # Build graph using KNN in latent space
            pred_edges = build_edges(
                query=embedding,
                database=embedding,
                indices=None,
                r_max=r_max,
                k_max=knn_max,
                backend="FRNN",
            )
            
            # Get truth edges from batch
            true_edges = batch.track_edges
            
            # Count unique true edges first (handle undirected case where edges might be duplicated)
            # True edges might be stored as undirected (each edge appears twice: (i,j) and (j,i))
            true_edges_unique = torch.unique(true_edges, dim=1)
            num_unique_true_edges = len(true_edges_unique[0])
            
            # Compute intersection to get truth labels for predicted edges
            pred_edges, edge_y, track_to_edge_map = graph_intersection(
                pred_edges,
                true_edges,
                return_y_pred=True,
                return_truth_to_pred=True,
                unique_pred=False,
                undirected=hparams.get("undirected", False),
            )
            
            # Convert to numpy for easier computation
            edge_y_np = edge_y.cpu().numpy().astype(bool)
            
            # For graph construction, we predict edges exist if they're in the graph
            # So: predicted = True for all edges in pred_edges
            predictions = np.ones(len(edge_y_np), dtype=bool)
            
            # Compute confusion matrix
            # TP: predicted edges that are true
            tp_raw = np.sum(predictions & edge_y_np)
            
            # Count how many unique true edges we found
            # If graph is undirected, pred_edges might contain duplicates of the same true edge
            # So we need to count unique true edges in our predictions
            true_pred_edges = pred_edges[:, edge_y_np]
            if true_pred_edges.shape[1] > 0:
                # Make edges canonical (smaller node first) to handle undirected duplicates
                true_pred_edges_canonical = torch.stack([
                    torch.min(true_pred_edges, dim=0)[0],
                    torch.max(true_pred_edges, dim=0)[0]
                ])
                true_pred_edges_unique = torch.unique(true_pred_edges_canonical, dim=1)
                tp = len(true_pred_edges_unique[0])
            else:
                tp = 0
            
            fp = np.sum(predictions & ~edge_y_np)
            
            # Calculate True Negatives: pairs of hits we correctly did NOT connect
            # Total possible edges in undirected graph: N*(N-1)/2
            num_nodes_val = batch.num_nodes
            if hasattr(num_nodes_val, 'item'):
                num_nodes_val = num_nodes_val.item()
            elif isinstance(num_nodes_val, (list, tuple)):
                num_nodes_val = num_nodes_val[0] if len(num_nodes_val) > 0 else 0
            
            total_possible_edges = num_nodes_val * (num_nodes_val - 1) // 2
            
            # Edges we created (unique)
            pred_edges_unique = torch.unique(pred_edges, dim=1)
            num_pred_edges_unique = len(pred_edges_unique[0])
            
            # False edges that exist (should not be connected)
            # Total false edges = total_possible - true_edges
            total_false_edges = total_possible_edges - num_unique_true_edges
            
            # TN = False edges we correctly did NOT create
            # = False edges that exist AND we didn't create
            # = total_false_edges - fp
            tn = max(0, total_false_edges - fp)  # Ensure non-negative
            
            fn = num_unique_true_edges - tp  # True edges not in our prediction
            
            # Accumulate
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            total_pred_edges += len(pred_edges[0])
            total_true_edges += num_unique_true_edges  # Count unique true edges, not duplicates
            
            # Handle num_nodes - could be int or tensor
            num_nodes = batch.num_nodes
            if hasattr(num_nodes, 'item'):
                num_nodes = num_nodes.item()
            elif isinstance(num_nodes, (list, tuple)):
                num_nodes = num_nodes[0] if len(num_nodes) > 0 else 0
            total_nodes += num_nodes
            
            # Store per-event metrics
            # Handle num_nodes - could be int or tensor
            num_nodes = batch.num_nodes
            if hasattr(num_nodes, 'item'):
                num_nodes = num_nodes.item()
            elif isinstance(num_nodes, (list, tuple)):
                num_nodes = num_nodes[0] if len(num_nodes) > 0 else 0
            
            event_metrics.append({
                'event_id': batch.event_id[0] if hasattr(batch, 'event_id') and hasattr(batch.event_id, '__getitem__') else batch_idx,
                'num_nodes': num_nodes,
                'num_pred_edges': len(pred_edges[0]),
                'num_true_edges': len(true_edges[0]),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            })
    
    # Compute aggregate metrics
    total_edges_evaluated = total_tp + total_fp + total_fn  # Note: TN=0 for graph construction
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Efficiency and purity (as defined in the paper)
    efficiency = total_tp / total_true_edges if total_true_edges > 0 else 0.0
    purity = total_tp / total_pred_edges if total_pred_edges > 0 else 0.0
    
    return {
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_tn': total_tn,
        'total_fn': total_fn,
        'total_pred_edges': total_pred_edges,
        'total_true_edges': total_true_edges,
        'total_nodes': total_nodes,
        'num_events': len(testset),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'efficiency': efficiency,
        'purity': purity,
        'event_metrics': event_metrics,
    }


def print_results(metrics, knn_max, r_max):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Graph construction parameters:")
    print(f"  r_max: {r_max}")
    print(f"  k_max: {knn_max}")
    print()
    print(f"Dataset:")
    print(f"  Number of events: {metrics['num_events']}")
    print(f"  Total nodes: {metrics['total_nodes']:,}")
    print(f"  Total true edges: {metrics['total_true_edges']:,}")
    print(f"  Total predicted edges: {metrics['total_pred_edges']:,}")
    print()
    print("="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    print(f"  True Positives (TP):  {metrics['total_tp']:>12,} - Correctly identified true edges")
    print(f"  False Positives (FP): {metrics['total_fp']:>12,} - False edges incorrectly included")
    print(f"  True Negatives (TN):  {metrics['total_tn']:>12,} - (N/A for graph construction)")
    print(f"  False Negatives (FN): {metrics['total_fn']:>12,} - True edges missed")
    print()
    print("="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"  Precision:  {metrics['precision']:.6f} ({metrics['precision']*100:.2f}%)")
    print(f"              TP / (TP + FP) = {metrics['total_tp']:,} / {metrics['total_tp'] + metrics['total_fp']:,}")
    print()
    print(f"  Recall:     {metrics['recall']:.6f} ({metrics['recall']*100:.2f}%)")
    print(f"              TP / (TP + FN) = {metrics['total_tp']:,} / {metrics['total_tp'] + metrics['total_fn']:,}")
    print()
    print(f"  F1 Score:   {metrics['f1_score']:.6f}")
    print()
    print(f"  Efficiency: {metrics['efficiency']:.6f} ({metrics['efficiency']*100:.2f}%)")
    print(f"              Fraction of true edges found")
    print()
    print(f"  Purity:     {metrics['purity']:.6f} ({metrics['purity']*100:.2f}%)")
    print(f"              Fraction of predicted edges that are true")
    print()
    print("="*80)
    
    # Per-event statistics
    if metrics['event_metrics']:
        precisions = [m['precision'] for m in metrics['event_metrics']]
        recalls = [m['recall'] for m in metrics['event_metrics']]
        print("PER-EVENT STATISTICS")
        print("="*80)
        print(f"  Precision: mean={np.mean(precisions):.4f}, std={np.std(precisions):.4f}")
        print(f"             min={np.min(precisions):.4f}, max={np.max(precisions):.4f}")
        print(f"  Recall:    mean={np.mean(recalls):.4f}, std={np.std(recalls):.4f}")
        print(f"             min={np.min(recalls):.4f}, max={np.max(recalls):.4f}")
        print("="*80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate metric learning model on test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_my_latent_model.py saved_models/low_pt_latent_f1=0.0149.ckpt
  python test_my_latent_model.py saved_models/low_pt_latent_f1=0.0149.ckpt --knn 1000 --r-max 0.15
  python test_my_latent_model.py saved_models/low_pt_latent_f1=0.0149.ckpt --knn 2000 --r-max 0.15
  
Note: For graph construction, use high k_max (1000-2000) to achieve high recall (low FN).
      Precision will be lower but that's OK - the GNN will filter false positives later.
        """
    )
    parser.add_argument(
        'checkpoint_path',
        type=str,
        help='Path to model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--knn', '--k-max',
        type=int,
        default=1000,
        help='Maximum number of neighbors for KNN graph construction (default: 1000, recommended: 500-2000 for high recall)'
    )
    parser.add_argument(
        '--r-max',
        type=float,
        default=0.15,
        help='Maximum radius for graph construction (default: 0.15)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: acorn_configs/latent_cluster_learning_train.yaml)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: auto (try GPU, fallback to CPU), cpu, or cuda (default: auto)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("METRIC LEARNING MODEL EVALUATION")
    print("="*80)
    print()
    
    # Determine device - try GPU, fallback to CPU
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  Using CUDA for evaluation")
        else:
            device = 'cpu'
            print("⚠ GPU not available, falling back to CPU")
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠ WARNING: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        print(f"Using device: {device}")
    print()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = SCRIPT_DIR / checkpoint_path
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    model, hparams = load_model(checkpoint_path)
    
    # Load config
    if args.config is None:
        config_path = SCRIPT_DIR / 'acorn_configs' / 'latent_cluster_learning_train.yaml'
    else:
        config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup dataset - try multiple possible paths
    input_dir_config = config['input_dir']
    
    # Try the path as specified in config
    input_dir = SCRIPT_DIR / input_dir_config
    testset_dir = input_dir / 'testset'
    
    # If not found, try with an extra 'data/' prefix (common structure issue)
    if not testset_dir.exists():
        alt_input_dir = SCRIPT_DIR / 'data' / input_dir_config
        alt_testset_dir = alt_input_dir / 'testset'
        if alt_testset_dir.exists():
            print(f"Note: Found testset at alternative path: {alt_testset_dir}")
            input_dir = alt_input_dir
            testset_dir = alt_testset_dir
        else:
            print(f"ERROR: Test set directory not found at either:")
            print(f"  {input_dir / 'testset'}")
            print(f"  {alt_input_dir / 'testset'}")
            print(f"\nPlease check your config file 'input_dir' setting or directory structure.")
            sys.exit(1)
    
    print(f"Loading test set from: {testset_dir}")
    print(f"Using input_dir: {input_dir}")
    print()
    
    # Create dataset
    # Get number of test events from config
    data_split = config.get('data_split', [9000, 500, 500])
    num_test_events = data_split[2] if len(data_split) > 2 else None
    
    testset = EventDataset(
        input_dir=str(input_dir),
        data_name='testset',
        num_events=num_test_events,
        hparams=hparams,
    )
    
    print(f"Loaded {len(testset)} test events")
    print()
    
    # Evaluate
    metrics = evaluate_model(
        model,
        hparams,
        testset,
        knn_max=args.knn,
        r_max=args.r_max,
        device=device,
    )
    
    # Print results
    print_results(metrics, args.knn, args.r_max)
    
    print("✓ Evaluation complete!")


if __name__ == '__main__':
    main()
