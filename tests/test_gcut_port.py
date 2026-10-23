"""Behavioral checks for the v0.81 to current-G-Cut adapter."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def stage():
    spec = importlib.util.spec_from_file_location('test_gcut_stage_adapter', ROOT/'5.gcut_pipeline.py')
    mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


class GCutPort(unittest.TestCase):
    def test_interrupted_swc_write_preserves_previous_checkpoint(self):
        spec=importlib.util.spec_from_file_location('atomic_gcut_writer',ROOT/'algorithms/tracing/gcut_stage.py')
        engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'target.swc';before=b'1 1 0 0 0 1 -1\n';path.write_bytes(before)
            def interrupted():
                yield (1,1,0,0,0,1,-1)
                raise RuntimeError('interrupted output')
            with self.assertRaisesRegex(RuntimeError,'interrupted output'):
                engine.write_swc(interrupted(),path)
            self.assertEqual(path.read_bytes(),before)
            engine.write_swc([(1,1,0,0,0,1,-1),(2,3,1,0,0,1,1)],path)
            self.assertFalse(path.with_name(path.name+'.writing').exists())
            self.assertEqual(len([line for line in path.read_text().splitlines() if not line.startswith('#')]),2)

    def test_metadata_and_original_mask_are_sent_to_isolated_worker(self):
        import pipeline_config as cfg
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mod=stage()
            inputs=[root/n for n in ['image_7_0000.tif','image_7.tif','image_7.swc']]
            inputs[1].write_bytes(b'original merged mask')
            args=(*inputs,root/'gcut',55.,5.,root/'errors.log',root/'soma.json',root/'soma.tif',
                  pd.DataFrame([dict(soma_x=30,soma_y=40,soma_z=5,xy_resolution=460,z_resolution=1000)],index=[7]))
            def run(command,**kwargs):
                job=json.loads(Path(command[-1]).read_text());folder=Path(job['paths'][3])
                self.assertEqual(job['paths'][1],str(inputs[1]))
                self.assertEqual(job['metadata']['soma_x'],30)
                self.assertEqual(job['gcut']['candidate_method'],'app2_gsdt')
                self.assertEqual(kwargs['env']['RECONNECT_COMPUTE_BACKEND'],'cpu')
                (folder/'worker-result.json').write_text(json.dumps(dict(success=True,result=['image_7', [90, []], 'ok'])))
                (folder/'gcut-timing.json').write_text(json.dumps({'seconds':{'primary_app2_gsdt':.25}}))
                return SimpleNamespace(returncode=0)
            with patch.dict(cfg.PATHS,{'gcut_selected_swc_dir':root/'selected'}),patch.object(mod.subprocess,'run',side_effect=run),patch.dict(os.environ,{'RECONNECT_COMPUTE_BACKEND':'parent-setting'}):
                result=mod.worker_task(args)
                self.assertEqual(os.environ['RECONNECT_COMPUTE_BACKEND'],'parent-setting')
            self.assertEqual(result[2],'ok')
            self.assertEqual(inputs[1].read_bytes(),b'original merged mask')

    def test_stage_sequence_preserves_v081_and_has_no_mask_repair(self):
        import run_pipeline as p
        names=[name for name,_ in p.build_stages(None)]
        self.assertEqual(names,['rescale','soma_crop','soma_infer','neurite_infer','merge','app2_trace','gcut','gcut_resample','prune'])
        self.assertFalse((ROOT/'algorithms/mask_repair').exists())

    def test_extracted_geometry_preserves_raw_and_uses_xyz_target(self):
        from algorithms.tracing.soma_geometry import target_body,soma_volumes
        raw=np.zeros((30,40,50),np.uint8);raw[5:10,7:14,9:18]=180
        before=raw.copy()
        slices,body,_,_=target_body(raw,[13,10,7])
        local=np.array([7,10,13])-[s.start for s in slices]
        self.assertTrue(body[tuple(local)])
        soma_volumes(raw)
        np.testing.assert_array_equal(raw,before)


if __name__=='__main__':unittest.main()
