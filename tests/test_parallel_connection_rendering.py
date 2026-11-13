from pathlib import Path
import tempfile
import unittest
import numpy as np
from visualization import connection_diagnostics as renderer


class ParallelRenderingTests(unittest.TestCase):
    def test_exports_match_serial_and_gallery_waits_for_all_images(self):
        site=dict(site_id='001',kind='branchpoint',status='uncertain',center_xyz=[8,8,8],
                  bbox_xyz=[[0,16]]*3,context_xyz=[[4,8,8],[8,8,8]],image='site_001.png',
                  arms=[dict(xyz=[[8,8,8],[12,12,12]],diagnosis=dict(status='uncertain',reason='test'))])
        crop=np.arange(16**3,dtype=np.uint8).reshape(16,16,16)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);serial=root/'serial';parallel=root/'parallel';serial.mkdir();parallel.mkdir()
            renderer.render_site(crop,[0,0,0],site,serial/'site_001.png')
            renderer.configure_render_workers(2)
            try:
                renderer.render_site(crop,[0,0,0],site,parallel/'site_001.png')
                renderer.write_gallery(dict(sites=[site],counts={'uncertain':1}),parallel)
                for suffix in ['png','html']:
                    self.assertEqual((serial/f'site_001.{suffix}').read_bytes(),(parallel/f'site_001.{suffix}').read_bytes())
                self.assertTrue((parallel/'index.html').exists())
            finally:renderer.finish_rendering()

if __name__=='__main__':unittest.main()
