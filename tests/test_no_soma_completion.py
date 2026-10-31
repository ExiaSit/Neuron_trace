import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tifffile
from algorithms.mask_repair.soma_reference import existing_bodies, segmented_target_reference
from algorithms.mask_repair import soma_completion, soma_root_repair, guarded_continuation


class NoSomaCompletion(unittest.TestCase):
    def test_c_and_f_preserve_incoming_foreground_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'b').mkdir();(root/'e').mkdir()
            mask=np.zeros((17,17,17),bool);mask[8,8,8]=True
            proposal=np.zeros_like(mask);proposal[6:11,6:11,6:11]=True
            raw=root/'raw.tif';tifffile.imwrite(raw,np.full(mask.shape,180,np.uint8))
            tifffile.imwrite(root/'b/connected-mask.tif',mask.astype('uint8')*255)
            record=dict(bbox_zyx=[[0,17]]*3,center_xyz=[8,8,8],core_radius=3,threshold=100)
            with patch.multiple(soma_completion,O=root/'c',WORK=root,RAW_PATH=str(raw)),patch.object(soma_completion,'soma_volumes',return_value=(proposal,[record])):
                soma_completion.main()
            np.testing.assert_array_equal(tifffile.imread(root/'c/connected-mask.tif')>0,mask)
            self.assertFalse(tifffile.imread(root/'c/soma-added-only.tif').any())
            self.assertTrue(np.all(mask[tifffile.imread(root/'c/soma-only.tif')>0]))
            grown=mask.copy();grown[8,8,9:14]=True
            tifffile.imwrite(root/'e/connected-mask.tif',grown.astype('uint8')*255)
            with patch.dict(os.environ,RECONNECT_WORK=str(root),RECONNECT_RAW=str(raw)),patch.object(soma_root_repair,'restore_roots',side_effect=AssertionError('Must not call growth')):
                soma_root_repair.main()
            np.testing.assert_array_equal(tifffile.imread(root/'f/connected-mask.tif')>0,grown)
            self.assertFalse(tifffile.imread(root/'f/roots-added-only.tif').any())

    def test_segmented_reference_keeps_holes_and_adds_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);seg=np.zeros((9,9,9),np.uint8);seg[2:7,2:7,2:7]=255;seg[4,4,4]=0
            tifffile.imwrite(root/'s.tif',seg);(root/'s.json').write_text(json.dumps(dict(bounds_1um_zxy=[10,19,10,19,10,19])))
            mask=np.zeros((31,31,31),bool);mask[12:17,12:17,12:17]=True
            mask[13,13,13]=False
            sl,body,_,audit=segmented_target_reference(mask,[14,14,13],root/'s.json',root/'s.tif')
            self.assertFalse(body[14,14,14]);self.assertFalse(body[13,13,13]);self.assertTrue(np.all(mask[sl][body]))
            self.assertEqual(audit['added_voxels'],0)
            with patch.object(guarded_continuation,'target_body',side_effect=AssertionError('Growth must not run')):
                final,report=guarded_continuation.extend(np.zeros(mask.shape,np.uint8),mask,[14,14,13],root/'s.json',root/'s.tif')
            np.testing.assert_array_equal(final,mask)
            self.assertFalse(report['soma']['soma_completion_enabled'])
            _,empty,_,_=segmented_target_reference(mask,[14,14,14],root/'s.json',root/'s.tif')
            self.assertFalse(empty.any())

    def test_ownership_references_never_include_absent_voxels(self):
        ref=np.ones((4,4,4),bool);mask=np.zeros_like(ref);mask[0,0,0]=True;mask[3,3,3]=True
        bodies=existing_bodies(ref,mask)
        self.assertEqual(len(bodies),2)
        self.assertEqual(sum(len(b['points_zyx']) for b in bodies),2)
        for body in bodies:self.assertTrue(mask[tuple(body['points_zyx'].T)].all())
