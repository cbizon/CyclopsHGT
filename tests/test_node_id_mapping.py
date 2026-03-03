"""
Unit test to isolate the node ID mapping problem between heterogeneous and homogeneous graphs.

The issue: Training batches use heterogeneous node IDs, but the model operates on a homogeneous graph
with different node numbering. We need to verify the mapping works correctly.
"""

import pytest
import torch
import dgl


def test_node_id_changes_after_to_homogeneous():
    """
    Test that node IDs change when converting to homogeneous.

    This is the ROOT CAUSE of the CUDA error: batches use het IDs, model uses homog IDs.
    """
    # Create heterogeneous graph
    graph_data = {
        ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 0])),
        ('user', 'likes', 'item'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
    }
    g_het = dgl.heterograph(graph_data)

    print(f"\n=== Heterogeneous Graph ===")
    print(f"User nodes: {g_het.num_nodes('user')} (IDs: 0-{g_het.num_nodes('user')-1})")
    print(f"Item nodes: {g_het.num_nodes('item')} (IDs: 0-{g_het.num_nodes('item')-1})")

    # Convert to homogeneous
    g_homogg = dgl.to_homogeneous(g_het, store_type=True)

    print(f"\n=== Homogeneous Graph ===")
    print(f"Total nodes: {g_homogg.num_nodes()}")
    print(f"Node types: {g_homogg.ndata[dgl.NTYPE]}")

    # The problem: heterogeneous node ID 0 of type 'user' is NOT the same as
    # homogeneous node ID 0!

    # In heterogeneous: user 0, user 1, item 0, item 1
    # In homogeneous: might be node 0, 1, 2, 3 (renumbered)

    # If we try to access embeddings for user 0 using het ID, we'll get the wrong node!
    print("\n❌ PROBLEM: Cannot use heterogeneous node IDs directly with homogeneous graph!")


def test_correct_approach_store_mappings():
    """
    Test the CORRECT approach: Use the mapping stored by to_homogeneous.
    """
    # Create heterogeneous graph
    graph_data = {
        ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 0])),
        ('user', 'likes', 'item'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
    }
    g_het = dgl.heterograph(graph_data)

    # Convert to homogeneous with store_type=True
    g_homogg = dgl.to_homogeneous(g_het, store_type=True)

    # DGL stores the mapping! Check what's available
    print(f"\n=== Available Attributes ===")
    print(f"ndata keys: {list(g_homogg.ndata.keys())}")
    print(f"edata keys: {list(g_homogg.edata.keys())}")

    # The _ID field contains the mapping back to heterogeneous IDs
    if '_ID' in g_homogg.ndata:
        print(f"\n✓ Node ID mapping available: {g_homogg.ndata['_ID']}")
        print(f"  These are the original heterogeneous node IDs")

    # The NTYPE field tells us which type each node belongs to
    print(f"\n✓ Node types: {g_homogg.ndata[dgl.NTYPE]}")

    # To go from het ID to homog ID, we need to reverse this mapping
    # For a given node type and het ID, find the homog ID

    node_type_id = 0  # 'user'
    het_node_id = 0   # user 0 in heterogeneous graph

    # Find where in homogeneous graph this node is
    mask = (g_homogg.ndata[dgl.NTYPE] == node_type_id) & (g_homogg.ndata['_ID'] == het_node_id)
    homo_node_id = torch.where(mask)[0]

    print(f"\n✓ Mapping: user {het_node_id} (het) -> node {homo_node_id.item()} (homog)")


def test_our_approach_is_wrong():
    """
    Test that our current approach (returning embeddings as dict) won't work
    because batches use heterogeneous IDs but we don't have the full heterogeneous graph.
    """
    # Create heterogeneous graph
    graph_data = {
        ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 0])),
        ('user', 'likes', 'item'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
    }
    g_het = dgl.heterograph(graph_data)

    # Convert to homogeneous
    g_homogg = dgl.to_homogeneous(g_het, store_type=True)

    # Simulate what our model does: extract embeddings by type
    # Fake embeddings (one per node)
    all_embeddings = torch.randn(g_homogg.num_nodes(), 8)

    # Split back by type (what our model returns)
    embeddings_dict = {}
    for ntype_id, ntype in enumerate(['user', 'item']):
        mask = g_homogg.ndata[dgl.NTYPE] == ntype_id
        embeddings_dict[ntype] = all_embeddings[mask]

    print(f"\n=== Our Current Approach ===")
    print(f"user embeddings shape: {embeddings_dict['user'].shape}")  # [2, 8]
    print(f"item embeddings shape: {embeddings_dict['item'].shape}")  # [2, 8]

    # Now the batch says: "get user 0 embedding"
    het_user_id = 0

    # We do: embeddings_dict['user'][het_user_id]
    retrieved_emb = embeddings_dict['user'][het_user_id]

    print(f"\nRetrieved embedding for 'user 0': {retrieved_emb[:3]}...")

    # But wait! The order in embeddings_dict['user'] might not match the original order!
    # The mask extraction preserves the homo order, not the het order!

    # Check the actual IDs
    user_mask = g_homogg.ndata[dgl.NTYPE] == 0
    user_homog_ids = torch.where(user_mask)[0]
    user_het_ids = g_homogg.ndata['_ID'][user_mask]

    print(f"\n=== Order Check ===")
    print(f"Homogeneous IDs: {user_homog_ids}")
    print(f"Original het IDs: {user_het_ids}")

    # If user_het_ids is [0, 1] then we're OK
    # If user_het_ids is [1, 0] then embeddings_dict['user'][0] is actually user 1!

    if not torch.equal(user_het_ids, torch.arange(len(user_het_ids))):
        print("\n❌ PROBLEM: Order is not preserved! embeddings_dict is wrong!")
    else:
        print("\n✓ Order preserved in this case (but not guaranteed in general)")


def test_solution_reorder_embeddings():
    """
    Test the SOLUTION: Reorder embeddings to match heterogeneous IDs.
    """
    # Create heterogeneous graph
    graph_data = {
        ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 0])),
        ('user', 'likes', 'item'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
    }
    g_het = dgl.heterograph(graph_data)

    # Convert to homogeneous
    g_homogg = dgl.to_homogeneous(g_het, store_type=True)

    # Fake embeddings
    all_embeddings = torch.randn(g_homogg.num_nodes(), 8)

    # Extract embeddings by type AND reorder to match het IDs
    embeddings_dict = {}
    for ntype_id, ntype in enumerate(['user', 'item']):
        mask = g_homogg.ndata[dgl.NTYPE] == ntype_id
        emb_in_homog_order = all_embeddings[mask]
        het_ids = g_homogg.ndata['_ID'][mask]

        # Reorder to match het IDs
        num_nodes_of_type = g_het.num_nodes(ntype)
        emb_in_het_order = torch.zeros(num_nodes_of_type, 8)
        emb_in_het_order[het_ids] = emb_in_homog_order

        embeddings_dict[ntype] = emb_in_het_order

    print(f"\n=== SOLUTION: Reordered Embeddings ===")
    print(f"user embeddings shape: {embeddings_dict['user'].shape}")
    print(f"Now embeddings_dict['user'][0] correctly corresponds to het user 0")
    print("\n✓ This is the fix we need!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
