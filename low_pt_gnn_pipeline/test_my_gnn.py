"""
Simple script to evaluate model accuracy on test data.
Uses the model's data loading pipeline to ensure proper preprocessing.
"""

import argparse
from pathlib import Path
import yaml
import torch
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN


def find_checkpoint(model_name, base_dir=None):
    """Find checkpoint file in saved_models directory."""
    if base_dir is None:
        base_dir = Path(__file__).parent / "saved_models"
    
    checkpoint_path = base_dir / f"{model_name}.ckpt"
    if checkpoint_path.exists():
        return checkpoint_path
    
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def evaluate_accuracy(model_name, config_file='acorn_configs/minimal_gnn_train.yaml', edge_cut=0.5):
    """
    Evaluate model accuracy on test data.
    
    Args:
        model_name: Name of the model checkpoint (e.g., 'epoch9_900')
        config_file: Path to config file
        edge_cut: Score threshold for classifying edges as positive (default: 0.5)
    """
    print("=" * 70)
    print(f"EVALUATING MODEL ACCURACY: {model_name}")
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
    test_data_dir = Path(__file__).parent / "data" / "graph_constructed"
    model.hparams['input_dir'] = str(test_data_dir)
    model.hparams['stage_dir'] = str(test_data_dir)
    model.hparams['data_split'] = config['data_split']
    
    # Setup test data (this will preprocess it)
    model.setup(stage='test')
    
    # Get test dataloader
    test_loader = model.test_dataloader()
    num_samples = len(test_loader.dataset) if hasattr(test_loader.dataset, '__len__') else 'unknown'
    print(f"Loaded {num_samples} test graphs")
    print()
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    
    # Evaluate
    print("=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)
    
    total_edges = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            scores = torch.sigmoid(output)
            predictions = (scores > edge_cut).cpu()
            
            # Get truth labels
            truth = batch.edge_y.bool().cpu()
            
            # Compute metrics
            tp = (predictions & truth).sum().item()
            fp = (predictions & ~truth).sum().item()
            tn = (~predictions & ~truth).sum().item()
            fn = (~predictions & truth).sum().item()
            
            true_positives += tp
            false_positives += fp
            true_negatives += tn
            false_negatives += fn
            total_edges += len(truth)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches...")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Edge classification threshold: {edge_cut}")
    print(f"Total edges evaluated: {total_edges:,}")
    print()
    print("Classification breakdown:")
    print(f"  True Positives (TP):  {true_positives:>10,} - Correctly identified true edges")
    print(f"  False Positives (FP): {false_positives:>10,} - False edges incorrectly called true")
    print(f"  True Negatives (TN):  {true_negatives:>10,} - Correctly identified false edges")
    print(f"  False Negatives (FN): {false_negatives:>10,} - True edges incorrectly called false")
    print()
    
    # Compute metrics
    accuracy = (true_positives + true_negatives) / total_edges if total_edges > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("Performance metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%) - TP / (TP + FP)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%) - TP / (TP + FN)")
    print(f"  F1 Score:  {f1_score:.4f}")
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
        default='acorn_configs/minimal_gnn_train.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--edge-cut',
        type=float,
        default=0.5,
        help='Score threshold for edge classification (default: 0.5)'
    )
    
    args = parser.parse_args()
    evaluate_accuracy(args.model_name, args.config, args.edge_cut)


if __name__ == '__main__':
    main()
