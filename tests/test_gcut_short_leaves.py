"""Exercise short-leaf pruning through the real bundled G-Cut preprocessor."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import bootstrap
from algorithms.tracing.gcut_adapter import make_segmentor


class GCutShortLeavesTests(unittest.TestCase):
    def preprocess(self, nodes, somas=(1,)):
        with tempfile.TemporaryDirectory() as tmp:
            swc = Path(tmp) / 'input.swc'
            soma = Path(tmp) / 'somas.txt'
            swc.write_text(''.join(
                f'{nid} {1 if nid in somas else 3} {x} {y} 0 0.01 {parent}\n'
                for nid, x, y, parent in nodes))
            soma.write_text(''.join(f'{nid}\n' for nid in somas))
            segmentor = make_segmentor(str(swc), str(soma))
            for nid, node in segmentor.tree_nodes.tree.items():
                self.assertTrue(node.parent_id == -1 or
                                node.parent_id in segmentor.tree_nodes.tree)
            return segmentor

    def test_threshold_counts_nodes_not_physical_length(self):
        # Delete a 90-um one-node tip; retain a 0.4-um two-node branch.
        result = self.preprocess([
            (1, 0, 0, -1), (2, 10, 0, 1), (3, 100, 0, 2),
            (4, 10, .2, 2), (5, 10, .4, 4),
        ])
        self.assertEqual(set(result.tree_nodes.tree), {1, 2, 4, 5})

    def test_remaining_sibling_forms_two_node_path_after_first_tip_removed(self):
        result = self.preprocess([
            (1, 0, 0, -1), (2, 10, 0, 1),
            (3, 20, 10, 2), (4, 20, -10, 2),
        ])
        # The legacy implementation updates child counts after each deletion.
        # The other tip then forms a two-node path with node 2 and survives.
        remaining = set(result.tree_nodes.tree)
        self.assertEqual(len(remaining), 3)
        self.assertTrue({1, 2}.issubset(remaining))
        self.assertEqual(len(remaining & {3, 4}), 1)

    def test_known_soma_tip_and_its_connection_are_preserved(self):
        result = self.preprocess([
            (1, 0, 0, -1), (2, 10, 0, 1),
            (3, 20, 10, 2), (4, 20, -10, 2),
        ], somas=(1, 3))
        self.assertEqual(set(result.tree_nodes.tree), {1, 2, 3})
        self.assertEqual(result.soma_ids, {1, 3})

    def test_single_soma_is_preserved(self):
        result = self.preprocess([(1, 0, 0, -1)])
        self.assertEqual(set(result.tree_nodes.tree), {1})


if __name__ == '__main__':
    unittest.main()
