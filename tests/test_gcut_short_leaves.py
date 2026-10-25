"""Exercise physical terminal pruning through the bundled G-Cut preprocessor."""
import math
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import bootstrap
from algorithms.tracing.gcut_adapter import make_segmentor, prune_short_leaves_by_length


class GCutShortLeavesTests(unittest.TestCase):
    def preprocess(self, nodes, somas=(1,)):
        with tempfile.TemporaryDirectory() as tmp:
            swc = Path(tmp) / 'input.swc'
            soma = Path(tmp) / 'somas.txt'
            swc.write_text(''.join(
                f'{nid} {1 if nid in somas else 3} {x} {y} {z} 0.01 {parent}\n'
                for nid, x, y, z, parent in nodes))
            soma.write_text(''.join(f'{nid}\n' for nid in somas))
            segmentor = make_segmentor(str(swc), str(soma))
            for node in segmentor.tree_nodes.tree.values():
                self.assertTrue(node.parent_id == -1 or
                                node.parent_id in segmentor.tree_nodes.tree)
            return segmentor

    def test_physical_length_retains_long_one_node_and_removes_short_multi_node(self):
        result = self.preprocess([
            (1, 0, 0, 0, -1), (2, 10, 0, 0, 1), (3, 100, 0, 0, 2),
            (4, 10, .2, 0, 2), (5, 10, .4, 0, 4),
        ])
        self.assertEqual(set(result.tree_nodes.tree), {1, 2, 3})
        audit = result.short_leaf_audit
        self.assertEqual(audit['removed_ids'], [4, 5])
        self.assertEqual(audit['min_leaf_branch_length_um'], 7.75)
        self.assertNotIn('min_leaf_branch_length', audit)
        self.assertEqual(audit['removed_branches'][0]['anchor_id'], 2)
        self.assertAlmostEqual(audit['removed_length_um'], .4)

    def test_strict_threshold_includes_attachment_edge(self):
        for length, retained in [(7.749, False), (7.75, True), (7.751, True)]:
            with self.subTest(length=length):
                result = self.preprocess([(1, 0, 0, 0, -1), (2, length, 0, 0, 1)])
                self.assertEqual(2 in result.tree_nodes.tree, retained)
                if not retained:
                    self.assertAlmostEqual(result.short_leaf_audit['removed_length_um'], length)

    def test_sums_curved_path_not_tip_to_anchor_distance(self):
        result = self.preprocess([(1, 0, 0, 0, -1), (2, 4, 0, 0, 1), (3, 4, 4, 0, 2)])
        # 4 + 4 = 8 um, although the tip is only sqrt(32) um from the soma.
        self.assertEqual(set(result.tree_nodes.tree), {1, 2, 3})

    def test_distance_is_three_dimensional(self):
        result = self.preprocess([
            (1, 0, 0, 0, -1), (2, 10, 0, 0, 1),
            (3, 10, 5, 6, 2), (4, 10, -3, -4, 2),
        ])
        self.assertEqual(set(result.tree_nodes.tree), {1, 2, 3})
        self.assertAlmostEqual(result.short_leaf_audit['removed_length_um'], 5)

    def test_deletion_updates_sibling_path_immediately(self):
        result = self.preprocess([
            (1, 0, 0, 0, -1), (2, 10, 0, 0, 1),
            (3, 10, 1, 0, 2), (4, 10, -1, 0, 2),
        ])
        # First remove 3 (1 um). Then 4 -> 2 -> soma is 11 um and survives.
        self.assertEqual(set(result.tree_nodes.tree), {1, 2, 4})

    def test_new_terminal_paths_are_rechecked_until_stable(self):
        result = self.preprocess([
            (1, 0, 0, 0, -1), (2, 1, 0, 0, 1),
            (3, 1, 1, 0, 2), (4, 1, -1, 0, 2),
        ])
        self.assertEqual(set(result.tree_nodes.tree), {1})
        self.assertEqual(result.short_leaf_audit['removed_ids'], [2, 3, 4])
        self.assertEqual(result.short_leaf_audit['removed_branches'][1]['node_ids'], [4, 2])
        self.assertAlmostEqual(result.short_leaf_audit['removed_length_um'], 3)

    def test_known_soma_tip_and_its_connection_are_preserved(self):
        result = self.preprocess([
            (1, 0, 0, 0, -1), (2, 10, 0, 0, 1),
            (3, 10, 1, 0, 2), (4, 10, -1, 0, 2),
        ], somas=(1, 3))
        self.assertEqual(set(result.tree_nodes.tree), {1, 2, 3})
        self.assertEqual(result.soma_ids, {1, 3})

    def test_single_soma_is_preserved(self):
        result = self.preprocess([(1, 0, 0, 0, -1)])
        self.assertEqual(set(result.tree_nodes.tree), {1})

    def test_invalid_threshold_rejected_before_mutating_tree(self):
        for threshold in [0, -1, math.nan, math.inf]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                prune_short_leaves_by_length(None, set(), threshold)


if __name__ == '__main__':
    unittest.main()
