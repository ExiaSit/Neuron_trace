from pathlib import Path
import tempfile
import unittest
from tools.compare_mip_repair import swc_metrics, saved_records, valid_reconstruction_pair


class ComparisonMetricTests(unittest.TestCase):
    def test_partial_metrics_after_invalid_old_swc_are_not_a_valid_pair(self):
        complete=dict(status='complete',baseline_swc={'final':{}},candidate_swc={'final':{}},
                      baseline_final_sha256='a',candidate_final_sha256='b')
        self.assertTrue(valid_reconstruction_pair(complete))
        partial=dict(complete,baseline_swc={'app2':{}},status='failed')
        partial.pop('baseline_final_sha256')
        self.assertFalse(valid_reconstruction_pair(partial))
        partial['status']='complete'
        self.assertFalse(valid_reconstruction_pair(partial))

    def test_interrupted_atomic_file_is_not_counted_as_an_extra_case(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'cases').mkdir()
            (root/'cases/17.json').write_text('{"id":"17","status":"complete"}')
            (root/'cases/17.writing.json').write_text('{')
            self.assertEqual(saved_records(root),[dict(id='17',status='complete')])

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
