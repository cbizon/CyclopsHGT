"""
Evaluate predictions against ground truth (Indications List).

Calculates accuracy, precision, recall, F1 for holdout predictions.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score


def load_ground_truth(csv_path, head_col, tail_col):
    """Load ground truth positives from Indications List."""
    df = pd.read_csv(csv_path)
    positives = set(zip(df[head_col], df[tail_col]))
    print(f"Loaded {len(positives)} positive pairs from ground truth")
    return positives


def evaluate_at_threshold(predictions_df, ground_truth, threshold=0.5):
    """Calculate metrics at a specific threshold."""
    # Classify each prediction
    predictions_df['is_positive'] = predictions_df.apply(
        lambda row: (row['drug_id'], row['disease_id']) in ground_truth, axis=1
    )
    predictions_df['predicted_positive'] = predictions_df['score'] >= threshold

    # Calculate metrics
    y_true = predictions_df['is_positive'].values
    y_pred = predictions_df['predicted_positive'].values

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total': len(y_true),
        'n_positives': np.sum(y_true),
        'n_negatives': np.sum(~y_true),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate predictions against ground truth'
    )
    parser.add_argument('--predictions', type=str, required=True,
                        help='Predictions TSV file (e.g., holdout_predictions.tsv)')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Ground truth CSV (Indications List)')
    parser.add_argument('--head_col', type=str, required=True,
                        help='Column name for head entities in ground truth')
    parser.add_argument('--tail_col', type=str, required=True,
                        help='Column name for tail entities in ground truth')
    parser.add_argument('--thresholds', type=str, default='0.5,0.7,0.9,0.95',
                        help='Comma-separated thresholds to evaluate')

    args = parser.parse_args()

    # Load data
    print(f"Loading ground truth from {args.ground_truth}...")
    ground_truth = load_ground_truth(args.ground_truth, args.head_col, args.tail_col)

    print(f"\nLoading predictions from {args.predictions}...")
    predictions = pd.read_csv(args.predictions, sep='\t')
    print(f"Loaded {len(predictions)} predictions")

    # Calculate metrics at various thresholds
    thresholds = [float(t) for t in args.thresholds.split(',')]

    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")

    results = []
    for threshold in thresholds:
        metrics = evaluate_at_threshold(predictions, ground_truth, threshold)
        results.append(metrics)

        print(f"Threshold = {threshold:.2f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {metrics['tp']:6,}  FN: {metrics['fn']:6,}")
        print(f"    FP: {metrics['fp']:6,}  TN: {metrics['tn']:6,}")
        print(f"  Dataset:")
        print(f"    Positives: {metrics['n_positives']:,} ({100*metrics['n_positives']/metrics['total']:.1f}%)")
        print(f"    Negatives: {metrics['n_negatives']:,} ({100*metrics['n_negatives']/metrics['total']:.1f}%)")
        print()

    # Calculate AUC metrics (threshold-independent)
    predictions['is_positive'] = predictions.apply(
        lambda row: (row['drug_id'], row['disease_id']) in ground_truth, axis=1
    )
    y_true = predictions['is_positive'].values
    y_score = predictions['score'].values

    try:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

        print(f"{'='*80}")
        print("THRESHOLD-INDEPENDENT METRICS")
        print(f"{'='*80}")
        print(f"AUROC (ROC-AUC): {auroc:.4f}")
        print(f"AUPRC (PR-AUC):  {auprc:.4f}")
        print()
    except Exception as e:
        print(f"Could not calculate AUC metrics: {e}")

    # Summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total predictions: {len(predictions):,}")
    print(f"Positives (in ground truth): {np.sum(y_true):,} ({100*np.mean(y_true):.1f}%)")
    print(f"Negatives (not in ground truth): {np.sum(~y_true):,} ({100*np.mean(~y_true):.1f}%)")
    print(f"\nScore distribution:")
    print(f"  Mean: {y_score.mean():.4f}")
    print(f"  Median: {np.median(y_score):.4f}")
    print(f"  Positives mean: {y_score[y_true].mean():.4f}")
    print(f"  Negatives mean: {y_score[~y_true].mean():.4f}")


if __name__ == "__main__":
    main()
