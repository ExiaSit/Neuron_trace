"""Batch contract tests; algorithm regression uses the frozen source separately."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from utils import mask_batch as run
class RunnerTests(unittest.TestCase):
    def test_group_sampling_including_tail_and_resume(self):
        pairs = [(Path(f'/raw/{i}'), Path(f'/mask/{i}'), Path(f'/out/{i}')) for i in range(1001)]
        cfg = dict(output_mask_dir=Path('/out'), output_mip_dir=Path('/mip'), mip_every=500, random_seed=17)
        choices = []
        for status in ['完成', '已存在，校验后跳过']:
            with patch.object(run, 'prepare', return_value=(cfg, pairs)), patch.object(run, 'process', return_value=status), patch.object(run, 'make_mip') as mip, patch('sys.argv', ['run.py']), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(), 0)
                selected = [int(c.args[0][1].name) for c in mip.call_args_list]
                self.assertEqual(len(selected), 3)
                self.assertTrue(0 <= selected[0] < 500)
                self.assertTrue(500 <= selected[1] < 1000)
                self.assertEqual(selected[2], 1000)
                choices.append(selected)
        self.assertEqual(choices[0], choices[1])

    def test_failure_does_not_stop_next_sample(self):
        pairs = [(Path('/raw/a'), Path('/mask/a'), Path('/out/a'))] * 2
        cfg = dict(output_mask_dir=Path('/out'), output_mip_dir=Path('/mip'), mip_every=500)
        with patch.object(run, 'prepare', return_value=(cfg, pairs)), patch.object(run, 'process', side_effect=[ValueError('bad input'), '完成']) as process, patch.object(run, 'make_mip') as mip, patch('sys.argv', ['run.py']), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run.main(), 1)
            self.assertEqual(process.call_count, 2)
            self.assertEqual(mip.call_count, 1)

    def test_result_preserves_foreground(self):
        original = np.zeros((3, 3, 3), bool); original[1, 1, 1] = True
        with self.assertRaises(ValueError):
            run.validate_result(np.zeros(original.shape, np.uint8), original)
        run.validate_result(original.astype('uint8') * 255, original)

    def test_output_cannot_overlap_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base/'raw').mkdir(); (base/'mask').mkdir()
            config = dict(raw_dir='raw', mask_dir='mask', output_mask_dir='mask/out', output_mip_dir='mip')
            (base/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, '输出目录'):
                run.prepare(base/'config.json')

    def test_slow_stage_prints_heartbeat(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            raw = np.zeros((3, 3, 3), np.uint8)
            child = Mock(pid=123)
            child.wait.side_effect = [run.subprocess.TimeoutExpired('stage', 30), 1]
            output = io.StringIO()
            pair = (Path('/raw/a'), Path('/mask/a'), Path(tmp)/'out.tif')
            with patch.object(run, 'load_inputs', return_value=(raw, raw > 0)), patch.object(run.subprocess, 'Popen', return_value=child), patch.object(run.time, 'monotonic', side_effect=[0, 31]), contextlib.redirect_stdout(output):
                with self.assertRaises(RuntimeError):
                    run.process(pair, dict(temp_dir=tmp))
            self.assertIn('PID 123', output.getvalue())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_parallel_order_and_worker_limit(self):
        import threading, time
        pairs = [(Path('/raw'), Path(str(i)), Path('/out')) for i in range(4)]
        active = 0
        maximum = 0
        lock = threading.Lock()
        def job(pair, cfg, stop):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.04 if pair[1].name == '0' else .01)
            with lock:
                active -= 1
            return '完成'
        cfg = dict(workers=2, memory_budget_gb=4, memory_reserve_gb=1)
        with patch.object(run, 'process', side_effect=job), patch.object(run, 'estimate_memory_gb', return_value=1), patch.object(run, 'available_memory_gb', return_value=100), contextlib.redirect_stdout(io.StringIO()):
            successful, failures = run.process_group(pairs, cfg, 0, 4)
        self.assertEqual(successful, pairs)
        self.assertEqual(failures, 0)
        self.assertEqual(maximum, 2)

    def test_memory_budget_limits_concurrency(self):
        import threading, time
        pairs = [(Path('/raw'), Path(str(i)), Path('/out')) for i in range(3)]
        active = 0
        maximum = 0
        lock = threading.Lock()
        def job(pair, cfg, stop):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.01)
            with lock:
                active -= 1
            if pair[1].name == '1':
                raise ValueError('bad sample')
            return '完成'
        cfg = dict(workers=4, memory_budget_gb=1.5, memory_reserve_gb=1)
        with patch.object(run, 'process', side_effect=job), patch.object(run, 'estimate_memory_gb', return_value=1), patch.object(run, 'available_memory_gb', return_value=100), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            successful, failures = run.process_group(pairs, cfg, 0, 3)
        self.assertEqual(maximum, 1)
        self.assertEqual(successful, [pairs[0], pairs[2]])
        self.assertEqual(failures, 1)

    def test_oversized_sample_reports_failure(self):
        cfg = dict(workers=4, memory_budget_gb=1, memory_reserve_gb=1)
        with patch.object(run, 'estimate_memory_gb', return_value=2), patch.object(run, 'available_memory_gb', return_value=100), contextlib.redirect_stderr(io.StringIO()):
            successes, failures = run.process_group([(Path('/raw'), Path('/mask'), Path('/out'))], cfg, 0, 1)
        self.assertEqual(successes, [])
        self.assertEqual(failures, 1)

    def test_cancel_terminates_child_and_cleans_work(self):
        import threading
        from unittest.mock import Mock
        stop = threading.Event()
        child = Mock(pid=123)
        def wait_once(timeout=None):
            stop.set()
            child.wait.side_effect = None
            child.wait.return_value = 0
            raise run.subprocess.TimeoutExpired('stage', 1)
        child.wait.side_effect = wait_once
        with tempfile.TemporaryDirectory() as tmp:
            raw = np.zeros((3, 3, 3), np.uint8)
            pair = (Path('/raw/a'), Path('/mask/a'), Path(tmp)/'out.tif')
            with patch.object(run, 'load_inputs', return_value=(raw, raw > 0)), patch.object(run.subprocess, 'Popen', return_value=child), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(InterruptedError):
                    run.process(pair, dict(temp_dir=tmp), stop)
            child.terminate.assert_called_once()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_stage_failure_cleans_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(temp_dir=tmp)
            raw = np.zeros((3, 3, 3), np.uint8)
            pair = (Path('/raw/a'), Path('/mask/a'), Path(tmp)/'out.tif')
            with patch.object(run, 'load_inputs', return_value=(raw, raw > 0)), patch.object(run.subprocess, 'Popen', return_value=type('Child', (), {'wait': lambda self, timeout=None: 1})()):
                with self.assertRaises(RuntimeError):
                    run.process(pair, cfg)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
