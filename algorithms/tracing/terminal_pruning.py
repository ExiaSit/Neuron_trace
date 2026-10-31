"""Length-based terminal removal on the final, soma-reconnected SWC tree."""
from algorithms.tracing.gcut_adapter import (
    SHORT_LEAF_MIN_LENGTH_UM, prune_short_leaves_by_length,
)


def prune_final_short_leaves(nodes, min_length_um=SHORT_LEAF_MIN_LENGTH_UM):
    from gcut.neuron_tree import NeuronTree, NeuronNode

    before = [list(n) for n in nodes]
    tree = NeuronTree()
    for row in before:
        node = NeuronNode(*row)
        if node.node_id in tree.tree:
            raise ValueError('Duplicate SWC node ID')
        tree.add_node(node)
    protected = {n[0] for n in before if n[1] == 1 or n[6] == -1}
    removed = prune_short_leaves_by_length(tree, protected, min_length_um)
    after = [n for n in before if n[0] in tree.tree]
    audit = dict(
        stage='final_pruning', capture_point='after all other pruning rules',
        before_nodes=before, after_nodes=after, soma_ids=sorted(protected),
        removed_ids=sorted({n[0] for n in before} - set(tree.tree)),
        removed_branches=removed,
        removed_length_um=sum(branch['length_um'] for branch in removed),
        min_leaf_branch_length_um=min_length_um,
        length_metric='3-D path length including attachment edge; 1-um image XYZ',
        comparison='strictly_less_than', leaf_order='ascending_node_id',
        rule='iterative terminal path length; attachment node and somas protected',
    )
    return after, audit
