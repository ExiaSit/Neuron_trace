from pathlib import Path
import tempfile
import unittest
from tools.compare_mip_repair import swc_metrics


class ComparisonMetricTests(unittest.TestCase):
    def test_mean_branch_length_counts_uninterrupted_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'tree.swc'
            path.write_text('1 1 0 0 0 1 -1\n2 3 1 0 0 1 1\n3 3 2 0 0 1 2\n'
                            '4 3 2 3 0 1 3\n5 3 2 -4 0 1 3\n')
            result = swc_metrics(path)
            self.assertEqual(result['nodes'],5)
            self.assertEqual(result['length_um'],9.)
            self.assertEqual(result['branches'],3)
            self.assertEqual(result['mean_branch_length_um'],3.)
            self.assertEqual(result['tips'],2)

    def test_cycle_and_missing_parent_are_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'invalid.swc'
            for text in ['1 1 0 0 0 1 2\n2 3 1 0 0 1 1\n',
                         '1 1 0 0 0 1 -1\n2 3 1 0 0 1 8\n']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    swc_metrics(path)
