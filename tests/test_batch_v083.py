import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import json
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
import tifffile
from utils import reconstruction_batch as batch
from utils.case_worker import paths, setup_config, load_module


class DirectBatchTests(unittest.TestCase):
    def test_pairing_ignores_temporary_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['image_17.tif','17.tiff','image_18.tif','.image_19.tif.part','image_20.json']:
                (root/name).touch()
            found,duplicates=batch.index_files(root,{'.tif','.tiff'})
            self.assertEqual(set(found),{'18'})
            self.assertEqual(duplicates,{'17'})

    def test_tiff_truncated_transfer_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);volume=np.ones((4,8,9),np.uint8)
            for name in ['raw','mask','soma']:
                tifffile.imwrite(root/f'{name}.tif',volume,photometric='minisblack')
            p=[root/f'{name}.tif' for name in ['raw','mask','soma']]
            batch.validate_tiffs(*p)
            with tifffile.TiffFile(p[1]) as tif:
                pixel_end=max(o+n for page in tif.series[0].pages for o,n in zip(page.dataoffsets,page.databytecounts))
            p[1].write_bytes(p[1].read_bytes()[:pixel_end-7])
            with self.assertRaises(ValueError):batch.validate_tiffs(*p)

    def test_recent_files_deferred(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data';p.write_bytes(b'complete')
            self.assertFalse(batch.stable([p],120))
            self.assertTrue(batch.stable([p],120,now=time.time()+121))

    def test_direct_paths_keep_source_names_without_input_directories(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'output';src=Path(d)/'source';src.mkdir()
            spec=dict(id='17',raw=str(src/'image_17_0000.tif'),soma_json=str(src/'image_17.json'),
                soma_mask=str(src/'image_17.tif'),config=dict(direct_inputs=True,output_root=str(root),
                output_mip_dir=str(root/'mask_mips'),comparison_dir=str(root/'comparisons'),vaa3d_app2='app2',vaa3d_resample='resample'))
            p=setup_config(spec)
            self.assertEqual(p['raw'],src/'image_17_0000.tif')
            self.assertEqual(p['soma_json'],src/'image_17.json')
            self.assertFalse((root/'img').exists())
            self.assertFalse((root/'soma_img').exists())
            gcut=load_module('gcut')
            self.assertEqual(gcut.selected_swc_path(root,p['raw']).name,'17.swc')
            prune=load_module('prune')
            swc=root/'selected/17.swc';swc.parent.mkdir()
            swc.write_text('1 1 1 1 1 1 -1\n2 3 2 1 1 1 1\n')
            p['raw'].touch();(root/'mask/17.tif').touch()
            valid,result=prune.check_exist(swc,root/'mask',src,root/'final',raw_image_path=str(p['raw']))
            self.assertTrue(valid,result)
            self.assertEqual(result[2],str(p['raw']))


if __name__=='__main__':unittest.main()
