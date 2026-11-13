import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools import run_gemma_soma_seeds as runner


class RunnerOutputTests(unittest.TestCase):
    def test_explicit_reject_is_normalized_without_mutating_model_response(self):
        candidate = dict(candidate_id='C001', xyz_um=[6, 6, 6], zyx=[6, 6, 6])
        target = dict(candidate_id='T', xyz_um=[4, 4, 4], zyx=[4, 4, 4])
        views = {name: dict(raw=True, plane=plane, xyz_bounds=[[0, 0, 0], [9, 9, 9]],
                            landmarks={'C001': dict(in_depth=True)})
                 for name, plane in [('local_XY', 'XY'), ('local_XZ', 'XZ')]}
        judgment = dict(candidate_id='C001', is_soma='no', identity='distinct',
                        decision='REJECT', duplicate_of=None, reason='Non-soma morphology',
                        evidence=[dict(view_id=name, observation='Test observation') for name in views])
        call = dict(judgment=judgment, seconds=1)
        args = SimpleNamespace(prepare_only=False, endpoint='unused', limit=1)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(runner, 'runtime_identity', return_value={}), \
                patch.object(runner, 'render', return_value=([], views)), \
                patch.object(runner, 'model_call', return_value=call), \
                contextlib.redirect_stdout(io.StringIO()):
            out = Path(temp)
            runner.review_all(np.zeros((10, 10, 10)), dict(target=target, candidates=[candidate]),
                              out, args, {}, 'test prompt')
            result = json.loads((out / 'cases/C001/result.json').read_text())
            self.assertEqual(result['decision']['decision'], 'REJECT')
            self.assertEqual(result['decision']['identity'], 'not_applicable')
            self.assertEqual(result['errors'], [])
            self.assertEqual(len(result['normalization']), 1)
            self.assertEqual(result['call']['judgment']['identity'], 'distinct')
            self.assertEqual(judgment['identity'], 'distinct')
            self.assertFalse((out / 'soma_catalogue.json').exists())


if __name__ == '__main__':
    unittest.main()
