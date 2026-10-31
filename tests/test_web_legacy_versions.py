"""Native version dispatch, original-mask semantics and live-code checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tifffile
from web_console.runner import parse_request,preflight,base_config,case_paths,run_job,run_legacy_stage
from web_console.versions import method_root,runner_entry,stage_names,ROOT
from web_console.legacy_stage import bind,crop_for_native
from utils.case_worker import save_json


class LegacyVersionTests(unittest.TestCase):
    def test_registry_routes_legacy_and_existing_versions_to_their_own_engines(self):
        from web_console.runner import STAGES
        for version in ['v0.81','v0.82','v0.82a']:
            self.assertEqual(parse_request(dict(ids='42',method_version=version))['method_version'],version)
            self.assertEqual(method_root(version),ROOT.parent/version)
            self.assertEqual(runner_entry(version),ROOT/'web_console/runner.py')
            self.assertIn('不修复',stage_names(version,STAGES)[6])
            self.assertIn('不替换',stage_names(version,STAGES)[11])
        self.assertEqual(runner_entry('v0.83'),ROOT.parent/'v0.83/web_console/runner.py')
        self.assertEqual(stage_names('v0.83a',STAGES),STAGES)

    def fixture(self,folder,version='v0.81'):
        root=Path(folder);source=root/'source';source.mkdir()
        image=np.zeros((12,16,18),np.uint8);image[3:9,4:12,4:14]=170
        tifffile.imwrite(source/'42.tif',image,photometric='minisblack')
        meta=root/'metadata.csv';meta.write_text('cell_id,v3dpbd_file,soma_x,soma_y,soma_z,xy_resolution,z_resolution\n42,Cell_Image/42.v3draw,8,8,6,1000,1000\n')
        req=parse_request(dict(ids='42',name='native test',method_version=version,start=2,stop=3,gpu_id=0,
            metadata_file=str(meta),source_tif_dir=str(source)))
        plan=preflight(req);job=root/'job';job.mkdir()
        save_json(job/'job.json',dict(request=req,plan=plan,created_at=0))
        return req,plan,job,image

    def test_native_cpu_preparation_runs_real_rescale_and_crop_scripts(self):
        for version in ['v0.81','v0.82']:
            with self.subTest(version=version),tempfile.TemporaryDirectory() as tmp:
                req,plan,job,image=self.fixture(tmp,version)
                self.assertIn('native_code_digest',plan)
                run_job(job)
                p=json.loads((job/'progress.json').read_text())
                self.assertEqual(p['status'],'complete')
                np.testing.assert_array_equal(tifffile.imread(job/'prepared/img/42_0000.tif'),image)
                self.assertEqual(tifffile.imread(job/'prepared/soma_img/42_0000.tif').shape,(128,128,128))
                audit=json.loads((job/'reconstruction/logs/42/native-stage-03.json').read_text())
                self.assertEqual(audit['source_root'],str(ROOT.parent/version))
                self.assertEqual(audit['source_code_digest'],plan['native_code_digest'])
                stamp=(job/'prepared/img/42_0000.tif').stat().st_mtime_ns
                run_job(job)
                self.assertEqual((job/'prepared/img/42_0000.tif').stat().st_mtime_ns,stamp)

    @unittest.skipUnless(os.environ.get('NEURON_TEST_NATIVE_CUDA')=='1','optional real CUDA preprocessing check')
    def test_v082a_preserves_native_gamma_gpu_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            req,plan,job,image=self.fixture(tmp,'v0.82a')
            run_job(job)
            expected=(np.sqrt(image.astype(float)/(170.+1e-7))*170.).astype(np.uint8)
            np.testing.assert_array_equal(tifffile.imread(job/'prepared/img/42_0000.tif'),expected)
            self.assertEqual(json.loads((job/'progress.json').read_text())['status'],'complete')

    def test_resampling_can_start_at_10_without_raw_or_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);up=root/'upstream/gcut_output/selected_for_pruning';up.mkdir(parents=True)
            (up/'image_42.swc').write_text('1 1 8 8 6 1 -1\n2 3 12 8 6 1 1\n3 3 12 12 6 1 2\n')
            meta=root/'meta.csv';meta.write_text('cell_id,soma_x,soma_y,soma_z,xy_resolution,z_resolution\n42,8,8,6,1000,1000\n')
            req=parse_request(dict(ids='42',method_version='v0.82',start=10,stop=10,gpu_id=0,
                metadata_file=str(meta),upstream_root=str(root/'upstream')))
            plan=preflight(req);job=root/'job';job.mkdir()
            self.assertTrue(plan['cases'][0]['inputs']['selected'].endswith('image_42.swc'))
            save_json(job/'job.json',dict(request=req,plan=plan,created_at=0))
            run_job(job)
            self.assertEqual(json.loads((job/'progress.json').read_text())['status'],'complete')
            from utils.case_worker import check_swc
            self.assertGreater(len(check_swc(job/'reconstruction/gcut_output/selected_resampled_step1/42.swc')),3)

    def test_changed_native_code_rejects_run_before_any_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            req,plan,job,image=self.fixture(tmp)
            with patch('web_console.legacy_stage.code_digest',return_value='changed'):
                with self.assertRaisesRegex(ValueError,'所选版本代码已变化'):run_job(job)
            self.assertFalse((job/'prepared/img/42_0000.tif').exists())

    def test_legacy_mask_slot_is_a_byte_identical_passthrough_not_v083_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'mask.tif';source.write_bytes(b'original mask bytes')
            output=root/'out/mask/42.tif'
            spec=dict(id='42',raw='/unused',mask=str(source),repaired=str(output),
                config=dict(output_root=str(root/'out'),method_version='v0.82'))
            with patch('utils.mask_batch.make_mip') as preview, \
                 patch('utils.case_worker.run_worker',side_effect=AssertionError('must not repair')):
                run_legacy_stage(root/'spec.json',spec,7)
            self.assertEqual(output.read_bytes(),source.read_bytes())
            audit=json.loads((root/'out/logs/42/continuity.json').read_text())
            self.assertFalse(audit['mask_repair']);preview.assert_called_once()
            self.assertFalse((root/'out/mask_steps').exists())

    def test_crop_bounds_are_adapted_only_when_explicitly_zxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'a.json';dest=root/'out/b.json'
            source.write_text(json.dumps(dict(bounds_order='ZXY',bounds_1um_zxy=[1,3,4,9,10,17])))
            crop_for_native(source,dest)
            self.assertEqual(json.loads(dest.read_text())['bounds_1um_zxy'],[1,3,10,17,4,9])
            self.assertEqual(json.loads(source.read_text())['bounds_order'],'ZXY')

    def test_input_binding_preserves_native_outputs_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';dest=root/'native/file';source.write_bytes(b'same')
            bind(source,dest);self.assertTrue(dest.is_symlink())
            bind(source,dest)
            dest.unlink();dest.write_bytes(b'same');bind(source,dest)
            dest.write_bytes(b'different')
            with self.assertRaisesRegex(ValueError,'conflict'):bind(source,dest)
            self.assertEqual(source.read_bytes(),b'same')


if __name__=='__main__':unittest.main()
