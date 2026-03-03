"""
Analyze false positives in holdout predictions.
"""

import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description='Analyze false positives')
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--ground_truth', type=str, required=True)
    parser.add_argument('--head_col', type=str, required=True)
    parser.add_argument('--tail_col', type=str, required=True)
    parser.add_argument('--non_indication_treats', type=str, required=True,
                        help='File with non-indication treats edges')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    # Load ground truth
    gt_df = pd.read_csv(args.ground_truth)
    ground_truth = set(zip(gt_df[args.head_col], gt_df[args.tail_col]))
    print(f"Ground truth positives: {len(ground_truth)}")

    # Load non-indication treats edges
    non_indication_df = pd.read_csv(args.non_indication_treats, sep='\t')
    non_indication_treats = set(zip(non_indication_df['subject'], non_indication_df['object']))
    print(f"Non-indication treats edges in full graph: {len(non_indication_treats)}")

    # Load predictions
    pred_df = pd.read_csv(args.predictions, sep='\t')
    print(f"\nTotal predictions: {len(pred_df)}")

    # Label each prediction
    pred_df['is_positive'] = pred_df.apply(
        lambda row: (row['drug_id'], row['disease_id']) in ground_truth, axis=1
    )
    pred_df['predicted_positive'] = pred_df['score'] >= args.threshold
    pred_df['is_non_indication_treats'] = pred_df.apply(
        lambda row: (row['drug_id'], row['disease_id']) in non_indication_treats, axis=1
    )

    # Identify false positives
    false_positives = pred_df[
        (~pred_df['is_positive']) & pred_df['predicted_positive']
    ].copy()

    print(f"\nAt threshold {args.threshold}:")
    print(f"False positives: {len(false_positives)}")

    # Categorize false positives
    fp_non_indication_treats = false_positives['is_non_indication_treats'].sum()
    fp_other = len(false_positives) - fp_non_indication_treats

    print(f"\nFalse positive breakdown:")
    print(f"  Non-indication treats edges: {fp_non_indication_treats} ({100*fp_non_indication_treats/len(false_positives):.1f}%)")
    print(f"  Other (not in graph): {fp_other} ({100*fp_other/len(false_positives):.1f}%)")

    # Show examples of each type
    print(f"\nExamples of non-indication treats false positives:")
    fp_treats = false_positives[false_positives['is_non_indication_treats']].head(10)
    print(fp_treats[['drug_id', 'disease_id', 'score']].to_string(index=False))

    print(f"\nExamples of other false positives (not in graph):")
    fp_other_examples = false_positives[~false_positives['is_non_indication_treats']].head(10)
    print(fp_other_examples[['drug_id', 'disease_id', 'score']].to_string(index=False))

    # Save all false positives with categorization
    false_positives[['drug_id', 'disease_id', 'score', 'is_non_indication_treats']].to_csv(
        args.output, sep='\t', index=False
    )
    print(f"\nSaved all false positives to {args.output}")


if __name__ == "__main__":
    main()
