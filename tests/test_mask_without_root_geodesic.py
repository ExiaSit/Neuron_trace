"""The live repair stage ends at g; historical diagnostic replay stays explicit."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tifffile
from utils.case_worker import save_json, setup_config
from utils.stages import load_stage
from utils.backfill_diagnostics import replay_continuation_steps


class NoRootGeodesicTests(unittest.TestCase):
    def fixture(self, folder, review=False):
        root=Path(folder)/'output';source=Path(folder)/'source';source.mkdir()
        raw=np.ones((4,8,8),np.uint8)*80
        mask=np.zeros(raw.shape,np.uint8);mask[1,4,3:5]=255
        for name,volume in [('raw',raw),('initial',mask),('soma',mask)]:
            tifffile.imwrite(source/f'{name}.tif',volume,photometric='minisblack')
        (source/'soma.json').write_text('{}')
        spec=dict(id='42',raw=str(source/'raw.tif'),mask=str(source/'initial.tif'),
                  soma_json=str(source/'soma.json'),soma_mask=str(source/'soma.tif'),
                  metadata=dict(soma_x=8,soma_y=8,soma_z=1,xy_resolution=500,z_resolution=1000),
                  config=dict(output_root=str(root),direct_inputs=True,review_supplement_only=review,
                      output_mip_dir=str(root/'mask_mips'),comparison_dir=str(root/'comparisons'),
                      vaa3d_app2='app2',vaa3d_resample='resample'))
        p=setup_config(spec)
        tifffile.imwrite(p['mask'],mask,photometric='minisblack')
        # g consumes completed a-f references and bridge reports even when its
        # growth function is mocked. Supply the same prerequisites as a real run.
        work=root/'mask_steps/42'
        for letter in ['b','d']:
            save_json(work/letter/'report.json',dict(accepted=[]))
        points=np.argwhere(mask>0)
        save_json(work/'f/report.json',dict(completed_somas=[dict(
            center_xyz=points.mean(0)[::-1].tolist(),top_z=1,source='existing_foreground')]))
        np.savez_compressed(work/'f/completed-somas.npz',body_0=points)
        return spec,p,raw,mask>0

    def test_final_mask_is_g_and_removed_pass_cannot_be_reenabled_by_old_review_option(self):
        for review in [False,True]:
            with self.subTest(review=review), tempfile.TemporaryDirectory() as tmp:
                spec,p,raw,before=self.fixture(tmp,review)
                expected=before.copy();expected[1,4,5]=True
                original=Path(spec['mask']).read_bytes()
                with patch('algorithms.mask_repair.guarded_continuation.extend',return_value=(expected,{'traces':[]})) as extend, \
                     patch('algorithms.mask_repair.root_geodesic.supplement',side_effect=AssertionError('h must never run')) as h, \
                     patch('visualization.diagnostics.record_mask_step') as record:
                    load_stage('continuity').run(spec)
                extend.assert_called_once();h.assert_not_called()
                self.assertEqual(extend.call_args.args[2],[4.,4.,1.])
                np.testing.assert_array_equal(extend.call_args.args[1],before)
                np.testing.assert_array_equal(tifffile.imread(p['mask'])>0,expected)
                work=p['root']/'mask_steps/42'
                np.testing.assert_array_equal(tifffile.imread(work/'g/connected-mask.tif'),tifffile.imread(p['mask']))
                self.assertFalse((work/'h').exists())
                record.assert_called_once()
                self.assertEqual(record.call_args.args[1],6)
                audit=json.loads((p['root']/'logs/42/continuity.json').read_text())
                self.assertFalse(audit['root_geodesic_enabled']);self.assertEqual(audit['final_mask_step'],'g')
                self.assertEqual(audit['added_voxels'],1);self.assertTrue(audit['original_preserved'])
                self.assertNotIn('geodesic_supplement',audit)
                self.assertEqual(Path(spec['mask']).read_bytes(),original)

    def test_new_replay_only_reconstructs_g(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,p,raw,before=self.fixture(tmp)
            save_json(p['root']/'logs/42/continuity.json',dict(root_geodesic_enabled=False,final_mask_step='g'))
            with patch('algorithms.mask_repair.guarded_continuation.extend',return_value=(before,{})), \
                 patch('algorithms.mask_repair.root_geodesic.supplement',side_effect=AssertionError('h disabled')):
                steps=replay_continuation_steps(spec,raw,before)
            self.assertEqual([x[0] for x in steps],[6])

    def test_historical_replay_keeps_the_original_h_when_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,p,raw,before=self.fixture(tmp)
            save_json(p['root']/'logs/42/continuity.json',dict(geodesic_supplement={}))
            with patch('algorithms.mask_repair.guarded_continuation.extend',return_value=(before,{})) as g, \
                 patch('algorithms.mask_repair.root_geodesic.supplement',return_value=(before,{})) as h:
                steps=replay_continuation_steps(spec,raw,before)
            self.assertEqual([x[0] for x in steps],[6,7]);g.assert_called_once();h.assert_called_once()

    def test_replay_without_rule_provenance_refuses_to_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,p,raw,before=self.fixture(tmp)
            save_json(p['root']/'logs/42/continuity.json',{})
            with self.assertRaisesRegex(ValueError,'refusing to guess'):
                replay_continuation_steps(spec,raw,before)


if __name__=='__main__':unittest.main()
