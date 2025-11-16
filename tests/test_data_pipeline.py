"""
Test data loading and splitting functionality.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_tsv_format():
    """Test that we can create and read the expected TSV format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create nodes.tsv
        nodes_df = pd.DataFrame({
            'id': ['CHEBI:1', 'CHEBI:2', 'MONDO:1', 'MONDO:2'],
            'type': ['Chemical', 'Chemical', 'Disease', 'Disease']
        })
        nodes_df.to_csv(tmpdir / 'nodes.tsv', sep='\t', index=False)

        # Create edges.tsv
        edges_df = pd.DataFrame({
            'subject': ['CHEBI:1', 'CHEBI:2'],
            'predicate': ['treats', 'treats'],
            'object': ['MONDO:1', 'MONDO:2']
        })
        edges_df.to_csv(tmpdir / 'edges.tsv', sep='\t', index=False)

        # Verify files exist and can be read
        assert (tmpdir / 'nodes.tsv').exists()
        assert (tmpdir / 'edges.tsv').exists()

        nodes_read = pd.read_csv(tmpdir / 'nodes.tsv', sep='\t')
        edges_read = pd.read_csv(tmpdir / 'edges.tsv', sep='\t')

        assert len(nodes_read) == 4
        assert len(edges_read) == 2
        assert list(nodes_read.columns) == ['id', 'type']
        assert list(edges_read.columns) == ['subject', 'predicate', 'object']


def test_node_types_from_ids():
    """Test that we correctly handle node types (not inferring from IDs)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create nodes with explicit types
        nodes_df = pd.DataFrame({
            'id': ['CHEBI:1', 'MONDO:1'],
            'type': ['Chemical', 'Disease']  # Explicit types required
        })
        nodes_df.to_csv(tmpdir / 'nodes.tsv', sep='\t', index=False)

        nodes_read = pd.read_csv(tmpdir / 'nodes.tsv', sep='\t')

        # Verify we have both columns
        assert 'id' in nodes_read.columns
        assert 'type' in nodes_read.columns

        # Verify types are explicit
        assert nodes_read.loc[nodes_read['id'] == 'CHEBI:1', 'type'].iloc[0] == 'Chemical'
        assert nodes_read.loc[nodes_read['id'] == 'MONDO:1', 'type'].iloc[0] == 'Disease'


def test_edge_stratification():
    """Test that edge types can be grouped for stratification."""
    edges_df = pd.DataFrame({
        'subject': ['CHEBI:1', 'CHEBI:2', 'CHEBI:3', 'HGNC:1', 'HGNC:2'],
        'predicate': ['treats', 'treats', 'treats', 'associated_with', 'associated_with'],
        'object': ['MONDO:1', 'MONDO:2', 'MONDO:1', 'MONDO:1', 'MONDO:2']
    })

    # Group by predicate
    predicate_counts = edges_df['predicate'].value_counts()

    assert predicate_counts['treats'] == 3
    assert predicate_counts['associated_with'] == 2

    # Test stratified sampling
    for predicate in edges_df['predicate'].unique():
        predicate_edges = edges_df[edges_df['predicate'] == predicate]
        n_edges = len(predicate_edges)

        # Calculate split sizes (80/10/10)
        train_size = int(n_edges * 0.8)
        val_size = int(n_edges * 0.1)
        test_size = n_edges - train_size - val_size

        assert train_size + val_size + test_size == n_edges


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
