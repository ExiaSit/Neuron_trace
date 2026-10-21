"""Regression checks for archived v0.81 runtime wiring (no model inference)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Imports should not create output in a user's configured dataset.
_SANDBOX = tempfile.TemporaryDirectory()
_config = Path(_SANDBOX.name) / 'config.json'
_config.write_text(json.dumps({'output_root': './out', 'neuron_ids': [123]}))
os.environ['NEURON_V082_CONFIG'] = str(_config)
import run_pipeline as p

SWC = '1 1 3 4 5 2 -1\n2 3 6 4 5 1 1\n'

class RuntimeAdapters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = {k: self.root/k for k in (
            'soma_crop_dir', 'soma_reader_json_dir', 'gcut_resampled_swc_dir',
            'prune_mip_dir', 'prune_swc_dir', 'image_1um_dir', 'soma_seg_dir',
            'merged_mask_dir', 'meta_file', 'prune_error_log', 'pipeline_skip_log')}
        self.patch = patch.dict(p.PATHS, self.paths)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.paths['soma_crop_dir'].mkdir()
        self.crop = self.paths['soma_crop_dir'] / 'image_123.json'
        self.crop.write_text(json.dumps({'bounds_1um_zxy': [1, 11, 20, 40, 50, 80], 'bounds_order': 'ZYX'}))

    def test_asymmetric_bounds_preserve_source_and_no_double_swap(self):
        before = self.crop.read_bytes()
        first = p.prepare_soma_reader_json(123)
        self.assertEqual(json.loads(first.read_text())['bounds_1um_zxy'], [1, 11, 50, 80, 20, 40])
        self.assertEqual(self.crop.read_bytes(), before)
        self.crop.write_text(first.read_text())
        self.assertEqual(json.loads(p.prepare_soma_reader_json(123).read_text())['bounds_1um_zxy'], [1, 11, 50, 80, 20, 40])

    def test_unknown_bounds_order_rejected(self):
        self.crop.write_text(json.dumps({'bounds_order':'XYZ','bounds_1um_zxy':[1]*6}))
        with self.assertRaises(ValueError): p.prepare_soma_reader_json(123)

    def test_marker_and_native_swc_conversion(self):
        mod = p.load_stage_module('test_app2', '4_pre_trace_app2.py')
        self.assertEqual(mod.image_to_app2_marker([3, 4, 5], (10, 20, 30)), [5, 17, 7])
        with self.assertRaises(ValueError): mod.image_to_app2_marker([30, 4, 5], (10,20,30))
        path = self.root/'native.swc';path.write_text('1 1 3 15 5 2 -1\n')
        self.assertTrue(mod.flip_swc_y_to_image_coordinates(str(path), 20))
        self.assertEqual(float(path.read_text().splitlines()[-1].split()[3]), 4)
        before=path.read_bytes()
        self.assertFalse(mod.flip_swc_y_to_image_coordinates(str(path), 20))
        self.assertEqual(before,path.read_bytes())

    def setup_swc(self):
        path = p.gcut_resampled_swc_path(123)
        path.parent.mkdir(parents=True);path.write_text(SWC)
        return path

    def test_pruning_receives_swc_and_required_keys(self):
        source=self.setup_swc()
        def worker(path, config):
            self.assertEqual(Path(path), source)
            self.assertEqual(config['selected_swc_dir'], str(source.parent))
            self.assertEqual(config['mask_dir'], str(self.paths['merged_mask_dir']))
            self.assertEqual(Path(config['soma_img_dir']), self.paths['soma_reader_json_dir'])
            p.prune_output_swc_path(123).write_text(SWC)
            return {'status':'success','file':source.name}
        with patch.object(p,'load_stage_module',return_value=SimpleNamespace(worker_task=worker)) as load:
            p.run_prune(123, None)
            self.assertEqual(load.call_args.args[1], '6_gcut_pruning_no_radius_jump_no_long_edge_copy.py')

    def test_empty_pruning_result_cannot_be_success(self):
        self.setup_swc()
        with patch.object(p,'load_stage_module',return_value=SimpleNamespace(worker_task=lambda *a: {'status':'success'})):
            with self.assertRaises(RuntimeError):p.run_prune(123,None)

    def test_failed_stage_returns_failure_to_caller(self):
        def fail(_):raise ValueError('injected failure')
        with self.assertRaises(RuntimeError):p.run_neuron_mode([('rescale',fail)],[123])
        with self.assertRaises(RuntimeError):p.run_stage_for_neurons('gcut',lambda _:False,[123],1)

    def test_config_paths_independent_of_working_directory(self):
        script = f"import sys; sys.path.insert(0,{str(ROOT)!r}); import pipeline_config as c; print(c.BASE_DIR); print(c.PATHS['pylib'])"
        out=subprocess.check_output([sys.executable,'-c',script],cwd='/tmp',text=True).splitlines()
        self.assertEqual(out[0], str(_config.parent/'out'))
        self.assertEqual(out[1], str(ROOT/'vendor/pylib'))

if __name__ == '__main__':unittest.main()
