import tempfile
import unittest
from pathlib import Path
from utils import bootstrap
from algorithms.tracing.gcut_adapter import make_segmentor
from algorithms.tracing.terminal_pruning import prune_final_short_leaves


class FinalShortLeavesTests(unittest.TestCase):
    def test_gcut_retains_short_terminal_until_final_pruning(self):
        nodes = [[1, 1, 0, 0, 0, .01, -1], [2, 3, 10, 0, 0, .01, 1],
                 [3, 3, 20, 0, 0, .01, 2], [4, 3, 10, 1, 0, .01, 2]]
        with tempfile.TemporaryDirectory() as tmp:
            swc, soma = Path(tmp)/'in.swc', Path(tmp)/'soma.txt'
            swc.write_text(''.join(' '.join(map(str, n))+'\n' for n in nodes))
            soma.write_text('1\n')
            segmentor = make_segmentor(str(swc), str(soma))
            self.assertEqual(set(segmentor.tree_nodes.tree), {1, 2, 3, 4})
            self.assertIsNone(segmentor.short_leaf_audit)
        after, audit = prune_final_short_leaves(nodes)
        self.assertEqual([n[0] for n in after], [1, 2, 3])
        self.assertEqual(audit['removed_ids'], [4])
        self.assertEqual(audit['stage'], 'final_pruning')

    def test_threshold_and_soma_protection(self):
        for length in [7.749, 7.75, 7.751]:
            for kind in [1, 3]:
                with self.subTest(length=length, kind=kind):
                    nodes = [[8, 1, 0, 0, 0, 1, -1], [20, kind, 0, 0, length, 1, 8]]
                    after, _ = prune_final_short_leaves(nodes)
                    self.assertEqual(len(after), 2 if kind == 1 or length >= 7.75 else 1)

    def test_iterative_path_and_no_dangling_parent(self):
        nodes = [[1, 1, 0, 0, 0, 1, -1], [2, 3, 1, 0, 0, 1, 1],
                 [3, 3, 1, 1, 0, 1, 2], [4, 3, 1, -1, 0, 1, 2]]
        after, audit = prune_final_short_leaves(nodes)
        self.assertEqual(after, nodes[:1])
        self.assertEqual(audit['removed_branches'][1]['node_ids'], [4, 2])
        self.assertEqual(audit['removed_length_um'], 3)
        self.assertEqual(len(nodes), 4)  # Caller input remains available for audit.

    def test_curved_path_length_and_idempotence(self):
        nodes = [[1, 1, 0, 0, 0, 1, -1], [2, 3, 4, 0, 0, 1, 1],
                 [3, 3, 4, 4, 0, 1, 2]]
        after, audit = prune_final_short_leaves(nodes)
        self.assertEqual(after, nodes)
        self.assertEqual(audit['removed_ids'], [])
        self.assertEqual(prune_final_short_leaves(after)[0], after)
