"""
Simple script to evaluate model accuracy on test data.
Uses the model's data loading pipeline to ensure proper preprocessing.
"""

import argparse
from pathlib import Path
import yaml
import torch
from sklearn.metrics import roc_auc_score
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN


def find_checkpoint(model_name, base_dir=None):
    """Find checkpoint file in saved_models directory."""
    if base_dir is None:
        base_dir = Path(__file__).parent / "saved_models"
    
    checkpoint_path = base_dir / f"{model_name}.ckpt"
    if checkpoint_path.exists():
        return checkpoint_path
    
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def evaluate_accuracy(model_name, config_file='acorn_configs/gnn_train.yaml', edge_cut=0.5, use_validation=False):
    """
    Evaluate model accuracy on test or validation data.
    
    Args:
        model_name: Name of the model checkpoint (e.g., 'epoch9_900')
        config_file: Path to config file
        edge_cut: Score threshold for classifying edges as positive (default: 0.5)
        use_validation: If True, evaluate on validation set; if False, evaluate on test set
    """
    dataset_name = "validation" if use_validation else "test"
    print("=" * 70)
    print(f"EVALUATING MODEL ACCURACY: {model_name}")
    print(f"Dataset: {dataset_name.upper()} set")
    print("=" * 70)
    print()
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find checkpoint
    checkpoint_path = find_checkpoint(model_name)

    # Load model
    print("Loading model from checkpoint...")
    model = InteractionGNN.load_from_checkpoint(
        str(checkpoint_path),
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Override paths to use current config
    data_dir = Path(__file__).parent / "data" / "graph_constructed_latent"
    model.hparams['input_dir'] = str(data_dir)
    model.hparams['stage_dir'] = str(data_dir)
    model.hparams['data_split'] = config['data_split']
    
    # CRITICAL: Ensure preprocessing is enabled for test stage to match validation preprocessing
    # (including feature scaling/normalization). Without this, test data won't be scaled
    # and will perform poorly since the model was trained on scaled features.
    model.hparams['reprocess_classifier'] = True
    
    # Setup data (this will preprocess it)
    if use_validation:
        model.setup(stage='fit')  # 'fit' stage loads train/val/test sets with preprocessing
        data_loader = model.val_dataloader()
    else:
        model.setup(stage='test')  # Now with reprocess_classifier=True, this will also preprocess
        data_loader = model.test_dataloader()
    
    num_samples = len(data_loader.dataset) if hasattr(data_loader.dataset, '__len__') else 'unknown'
    print(f"Loaded {num_samples} {dataset_name} graphs")
    print()
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    
    # Evaluate
    print("=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)
    
    # Metrics for ALL edges
    total_edges = 0
    all_tp = 0
    all_fp = 0
    all_tn = 0
    all_fn = 0
    
    # For AUC computation
    all_scores = []
    all_truth_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            scores = torch.sigmoid(output)
            predictions = (scores > edge_cut).cpu()
            
            # Get truth labels
            all_truth = batch.edge_y.bool().cpu()
            
            # Compute metrics for ALL edges
            tp = (predictions & all_truth).sum().item()
            fp = (predictions & ~all_truth).sum().item()
            tn = (~predictions & ~all_truth).sum().item()
            fn = (~predictions & all_truth).sum().item()
            
            all_tp += tp
            all_fp += fp
            all_tn += tn
            all_fn += fn
            total_edges += len(all_truth)
            
            # Collect scores for AUC
            all_scores.append(scores.cpu().detach())
            all_truth_labels.append(all_truth.float())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches...")
    
    # Compute AUC
    all_scores_tensor = torch.cat(all_scores)
    all_truth_tensor = torch.cat(all_truth_labels)
    auc = roc_auc_score(all_truth_tensor.numpy(), all_scores_tensor.numpy())
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Edge classification threshold: {edge_cut}")
    print()
    print(f"Total edges evaluated: {total_edges:,}")
    print()
    print("Classification breakdown:")
    print(f"  True Positives (TP):  {all_tp:>10,} - Correctly identified true edges")
    print(f"  False Positives (FP): {all_fp:>10,} - False edges incorrectly called true")
    print(f"  True Negatives (TN):  {all_tn:>10,} - Correctly identified false edges")
    print(f"  False Negatives (FN): {all_fn:>10,} - True edges incorrectly called false")
    print()
    
    accuracy = (all_tp + all_tn) / total_edges if total_edges > 0 else 0
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("Performance metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%) - TP / (TP + FP)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%) - TP / (TP + FN)")
    print(f"  F1 Score:  {f1_score:.4f}")
    print(f"  AUC:       {auc:.6f}")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy on test data")
    parser.add_argument(
        'model_name',
        type=str,
        help='Name of the model checkpoint (e.g., epoch9_900)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='acorn_configs/gnn_train.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--edge-cut',
        type=float,
        default=0.5,
        help='Score threshold for edge classification (default: 0.5)'
    )
    parser.add_argument(
        '--validation',
        action='store_true',
        help='Evaluate on validation set instead of test set'
    )
    
    args = parser.parse_args()
    evaluate_accuracy(args.model_name, args.config, args.edge_cut, args.validation)


if __name__ == '__main__':
    main()
