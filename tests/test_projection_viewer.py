import base64
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from visualization.projection_viewer import project_volume,window_image,write_site_viewer,window_defaults


class ProjectionViewerTests(unittest.TestCase):
    def test_full_and_slice_axis_orientation_matches_numpy(self):
        raw=np.arange(4*5*6,dtype=np.uint8).reshape(4,5,6)
        for axis in range(3):
            full,bounds=project_volume(raw,axis,2,'full')
            np.testing.assert_array_equal(full,raw.max(axis=axis))
            self.assertEqual(bounds,(0,raw.shape[axis]))
            single,bounds=project_volume(raw,axis,2,'slice')
            np.testing.assert_array_equal(single,np.take(raw,2,axis=axis))
            self.assertEqual(bounds,(2,3))

    def test_thin_slab_rejects_bright_structure_outside_selected_layers(self):
        raw=np.zeros((15,17,19),np.uint8);raw[7,8,9]=50;raw[:,16,:]=250
        full,_=project_volume(raw,1,8,'full')
        slab,bounds=project_volume(raw,1,8,'slab',5)
        self.assertEqual(bounds,(6,11));self.assertEqual(full[7,9],250);self.assertEqual(slab[7,9],50)

    def test_boundary_clipping_has_no_shift_or_wrap(self):
        raw=np.arange(5,dtype=np.uint8)[:,None,None]
        result,bounds=project_volume(raw,0,0,'slab',5)
        self.assertEqual(bounds,(0,3));self.assertEqual(result.item(),2)
        result,bounds=project_volume(raw,0,4,'slab',5)
        self.assertEqual(bounds,(2,5));self.assertEqual(result.item(),4)

    def test_window_changes_only_display_and_handles_constant_image(self):
        raw=np.array([0,10,20,30,255],np.uint8);saved=raw.copy()
        np.testing.assert_array_equal(window_image(raw,10,30,1),[0,0,128,255,255])
        np.testing.assert_array_equal(raw,saved)
        for value in [0,255]:
            black,white=window_defaults(np.full((4,4,4),value,np.uint8))
            self.assertLess(black,white)
        with self.assertRaises(ValueError):window_image(raw,10,10)

    def test_self_contained_payload_is_exact_unmodified_voxels(self):
        raw=np.arange(5*6*7,dtype=np.uint8).reshape(5,6,7)[:,::2,:]
        site=dict(site_id='001',kind='crossing',status='uncertain',center_xyz=[13,22,32],
                  bbox_xyz=[[10,17],[20,23],[30,35]],arms=[],context_xyz=[],image='site_001.png')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'site.html';write_site_viewer(raw,[10,20,30],site,out)
            text=out.read_text();start=text.index('<script type="application/json" id="data">')+len('<script type="application/json" id="data">')
            data=json.loads(text[start:text.index('</script>',start)])
            decoded=np.frombuffer(base64.b64decode(data['voxels_base64']),np.uint8).reshape(data['shape_zyx'])
            np.testing.assert_array_equal(raw,decoded)
            self.assertEqual(data['origin_xyz'],[10,20,30])
            self.assertNotIn('fetch(',text)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):write_site_viewer(raw.astype(np.float32),[10,20,30],site,Path(tmp)/'bad.html')

if __name__=='__main__':unittest.main()
