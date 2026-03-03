"""
Find 'treats' edges in the graph that are NOT in the Indications List.
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Find treats edges not in Indications List'
    )
    parser.add_argument('--graph_edges', type=str, required=True,
                        help='Path to edges.tsv from graph')
    parser.add_argument('--indications_list', type=str, required=True,
                        help='Path to Indications List CSV')
    parser.add_argument('--head_col', type=str, required=True,
                        help='Column name for head entities in Indications List')
    parser.add_argument('--tail_col', type=str, required=True,
                        help='Column name for tail entities in Indications List')
    parser.add_argument('--output', type=str, required=True,
                        help='Output TSV file')
    parser.add_argument('--predicate', type=str, default='treats',
                        help='Predicate to filter (default: treats)')

    args = parser.parse_args()

    # Load Indications List
    print(f"Loading Indications List from {args.indications_list}...")
    indications_df = pd.read_csv(args.indications_list)
    indications_set = set(zip(
        indications_df[args.head_col],
        indications_df[args.tail_col]
    ))
    print(f"Loaded {len(indications_set)} indication pairs")

    # Load graph edges
    print(f"\nLoading graph edges from {args.graph_edges}...")
    edges_df = pd.read_csv(args.graph_edges, sep='\t')
    print(f"Total edges in graph: {len(edges_df)}")

    # Filter to 'treats' predicate
    treats_edges = edges_df[edges_df['predicate'] == args.predicate].copy()
    print(f"'treats' edges in graph: {len(treats_edges)}")

    # Find treats edges NOT in Indications List
    treats_edges['in_indications'] = treats_edges.apply(
        lambda row: (row['subject'], row['object']) in indications_set,
        axis=1
    )

    in_indications = treats_edges['in_indications'].sum()
    not_in_indications = (~treats_edges['in_indications']).sum()

    print(f"\n'treats' edges IN Indications List: {in_indications}")
    print(f"'treats' edges NOT IN Indications List: {not_in_indications}")

    # Save edges NOT in Indications List
    non_indication_treats = treats_edges[~treats_edges['in_indications']]
    non_indication_treats = non_indication_treats[['subject', 'predicate', 'object']]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    non_indication_treats.to_csv(output_path, sep='\t', index=False)

    print(f"\nWrote {len(non_indication_treats)} non-indication 'treats' edges to {output_path}")

    # Show some examples
    print(f"\nFirst 20 examples of 'treats' edges NOT in Indications List:")
    print(non_indication_treats.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
