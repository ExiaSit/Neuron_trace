"""Behavior checks at the new V3D/preprocessing/package boundaries."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils import bootstrap
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tifffile
from v3dpy.loaders import Raw, PBD
from utils.v3d_io import load_image
from utils.preparation import load_step, case, fingerprint, mark, is_complete, requested_ids, inference
from algorithms.preprocessing import resample_1um
from utils.configuration import load_config


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.cfg=dict(prepared_root=str(self.root/'prepared'),v3d_dir=str(self.root/'v3d'),
                      soma_crop_shape=[8,8,8],models={},app2_temp_dir=str(self.root/'scratch'))
        Path(self.cfg['v3d_dir']).mkdir()
        self.raw=np.zeros((12,14,16),np.uint8)
        self.raw[2:10,3:12,4:13]=200
        self.raw[4,7,8]=255
        Raw().save(str(Path(self.cfg['v3d_dir'])/'image_00042.v3draw'),self.raw[None])
        self.spec=case(self.cfg,'42',dict(xy_resolution=1000,z_resolution=1000,soma_x=8,soma_y=7,soma_z=4))

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self):
        for step in [1,2,3]:
            load_step(step).run(self.spec)

    def test_raw_and_pbd_decode_identically(self):
        p=self.root/'42.v3dpbd'
        PBD().save(str(p),self.raw[None])
        self.assertTrue(np.array_equal(load_image(self.spec['source']),self.raw))
        self.assertTrue(np.array_equal(load_image(str(p)),self.raw))

    def test_native_spacing_is_identity_and_numeric_names(self):
        self.prepare()
        self.assertTrue(np.array_equal(tifffile.imread(self.spec['raw']),self.raw))
        self.assertEqual(Path(self.spec['raw']).name,'42_0000.tif')
        self.assertEqual(requested_ids(self.cfg),['42'])

    def test_constant_input_and_anisotropic_shape(self):
        out=resample_1um(np.full((7,11,13),32,np.uint8),500,2000)
        self.assertEqual(out.shape,(14,6,6))
        self.assertTrue(np.all(out==32))

    def test_crop_near_boundary_and_restore_union(self):
        self.spec['metadata'].update(soma_x=1,soma_y=2,soma_z=1)
        self.prepare()
        data=json.loads(Path(self.spec['soma_json']).read_text())
        self.assertEqual(data['bounds_1um_zxy'],[0,5,0,6,0,5])
        self.assertEqual(tifffile.imread(self.spec['crop']).shape,(8,8,8))
        soma=np.zeros((8,8,8),np.uint8);soma[1,2,3]=1
        neurite=np.zeros(self.raw.shape,np.uint8);neurite[10,12,14]=1
        for key,array in [('soma_mask',soma),('neurite_mask',neurite)]:
            p=Path(self.spec[key]);p.parent.mkdir(parents=True,exist_ok=True);tifffile.imwrite(p,array)
        load_step(6).run(self.spec)
        merged=tifffile.imread(self.spec['mask'])
        self.assertEqual(np.argwhere(merged>0).tolist(),[[1,2,3],[10,12,14]])
        self.assertEqual(set(np.unique(merged)),{0,255})

    def test_soma_override_applies_before_crop(self):
        self.cfg['soma_overrides_image_xyz']={'42':[1,2,1]}
        self.prepare()
        self.assertEqual(json.loads(Path(self.spec['soma_json']).read_text())['bounds_1um_zxy'],[0,5,0,6,0,5])

    def test_changed_inputs_refuse_stale_checkpoint(self):
        load_step(1).run(self.spec)
        key=fingerprint(self.spec,1);mark(self.spec,1,key)
        self.assertTrue(is_complete(self.spec,1,key))
        self.spec['metadata']['z_resolution']=999
        with self.assertRaisesRegex(ValueError,'fresh prepared_root'):
            is_complete(self.spec,1,fingerprint(self.spec,1))

    def test_missing_output_does_not_resume(self):
        load_step(1).run(self.spec)
        key=fingerprint(self.spec,1);mark(self.spec,1,key)
        Path(self.spec['normalized']).unlink()
        self.assertFalse(is_complete(self.spec,1,key))

    def test_model_failure_is_not_published(self):
        self.prepare()
        model_root=self.root/'models'
        self.cfg.update(nnunet_executable='/unused/nnUNetv2_predict',
                        models={'soma':dict(results=str(model_root),dataset_name='Dataset206_test',dataset_id=206)})
        folder=model_root/'Dataset206_test/nnUNetTrainer__nnUNetPlans__3d_fullres'
        (folder/'fold_0').mkdir(parents=True)
        for rel in ['fold_0/checkpoint_final.pth','plans.json','dataset.json']:(folder/rel).write_text('fixture')
        import subprocess
        with patch('utils.preparation.subprocess.run',side_effect=subprocess.CalledProcessError(1,['nnUNet'])):
            with self.assertRaises(subprocess.CalledProcessError):
                inference([self.spec],'soma')
        self.assertFalse(Path(self.spec['soma_mask']).exists())
        self.assertFalse((Path(self.cfg['prepared_root'])/'status/42_04.json').exists())
        self.assertEqual(list(Path(self.cfg['app2_temp_dir']).iterdir()),[])

    def test_config_paths_are_relative_to_config(self):
        cfg=dict(input_mode='prepared',output_root='result',raw_dir='raw',mask_dir='mask',
                 soma_crop_dir='crop',soma_seg_dir='soma',previous_swc_dirs=[])
        path=self.root/'config.json';path.write_text(json.dumps(cfg))
        resolved=load_config(path)
        self.assertEqual(resolved['output_mask_dir'],str(self.root/'result/mask'))
        self.assertEqual(resolved['raw_dir'],str(self.root/'raw'))


if __name__=='__main__':
    unittest.main()
