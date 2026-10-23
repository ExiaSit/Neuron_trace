"""Regression tests for APP2 image/marker coordinates and fresh output checks."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys
import psutil
from unittest import mock

from utils import case_worker as worker


class App2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        worker.setup_config({'id': '1', 'config': {
            'output_root': str(self.root), 'comparison_dir': str(self.root/'comparisons'),
            'output_mip_dir': str(self.root/'mask_mips'),
            'vaa3d_app2': '/unused/vaa3d', 'vaa3d_resample': '/unused/vaa3d'}})
        self.app2 = worker.load_module('app2')

    def tearDown(self):
        self.tmp.cleanup()

    def test_cropped_height_places_seed_on_exact_image_voxel(self):
        # Native initial roots measured with the installed binary, including
        # its double marker offset. Cropped height used to move Y by 32 voxels.
        self.assertEqual(self.app2.image_to_app2_marker([196,196,122], (272,360,392)), [198,165,124])
        self.assertEqual(self.app2.image_to_app2_marker([196,196,132], (272,362,392)), [198,167,134])

    def test_boundary_voxels_and_invalid_metadata(self):
        self.assertEqual(self.app2.image_to_app2_marker([0,0,0], (12,30,50)), [2,31,2])
        self.assertEqual(self.app2.image_to_app2_marker([49,29,11], (12,30,50)), [51,2,13])
        for xyz in ([50,2,2], [-1,2,2], [1,float('nan'),2]):
            with self.assertRaises(ValueError):
                self.app2.image_to_app2_marker(xyz, (12,30,50))

    def test_native_swc_hits_exact_image_y_and_conversion_is_idempotent(self):
        path=self.root/'native.swc'
        path.write_text('1 1 196 163 122 9 -1\n2 3 200 162 123 1 1\n')
        self.assertTrue(self.app2.flip_swc_y_to_image_coordinates(str(path),360))
        nodes=worker.check_swc(path)
        self.assertEqual([nodes[1].x,nodes[1].y,nodes[1].z], [196,196,122])
        before=path.read_bytes()
        self.assertFalse(self.app2.flip_swc_y_to_image_coordinates(str(path),360))
        self.assertEqual(before,path.read_bytes())

    def test_legacy_converted_swc_cannot_be_flipped_again(self):
        path=self.root/'old.swc';path.write_text('# y_axis_flipped_to_image_coordinates\n1 1 2 3 4 1 -1\n')
        before=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'regenerate'):
            self.app2.flip_swc_y_to_image_coordinates(str(path),30)
        self.assertEqual(before,path.read_bytes())

    def invoke_with_output(self, content):
        image=self.root/'input.tif';image.touch()
        marker=self.root/'seed.marker';marker.write_text('2,3,4\n')
        # Old output beside the input must never rescue an empty fresh result.
        (self.root/'input.tif_old_app2.swc').write_text('1 1 2 3 4 1 -1\n')
        output=self.root/'native'/'1.swc'
        def launch(cmd, **kwargs):
            fresh=Path(cmd[cmd.index('-i')+1])
            Path(str(fresh)+'_app2.swc').write_text(content)
            kwargs['stdout'].write('native diagnostic retained\n')
            return mock.Mock(wait=mock.Mock(return_value=0))
        with mock.patch.object(self.app2.subprocess,'Popen',side_effect=launch):
            self.app2.run_app2_locally(str(image),str(marker),str(output),5)
        return output

    def test_empty_fresh_swc_fails_despite_stale_neighbor(self):
        with self.assertRaisesRegex(ValueError,'No SWC nodes'):
            self.invoke_with_output('# header only\n')
        self.assertFalse((self.root/'native/1.swc').exists())
        self.assertIn('native diagnostic',(self.root/'native/1.app2.log').read_text())

    def test_valid_fresh_swc_published_without_temporary_volumes(self):
        output=self.invoke_with_output('1 1 2 3 4 1 -1\n2 3 3 4 5 1 1\n')
        self.assertEqual(len(worker.check_swc(output)),2)
        self.assertFalse(list(output.parent.glob('.app2-*')))

    def test_timeout_stops_descendant_and_publishes_no_swc(self):
        image=self.root/'input.tif';image.touch()
        marker=self.root/'seed.marker';marker.touch()
        child_pid=self.root/'child.pid';output=self.root/'native/1.swc'
        script=("import subprocess,sys,time; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                f"Path({str(child_pid)!r}).write_text(str(p.pid)); time.sleep(30)")
        real_popen=self.app2.subprocess.Popen
        def launch(cmd,**kwargs):
            return real_popen([sys.executable,'-c',script],**kwargs)
        with mock.patch.object(self.app2.subprocess,'Popen',side_effect=launch):
            with self.assertRaises(self.app2.TimeoutException):
                self.app2.run_app2_locally(str(image),str(marker),str(output),1)
        self.assertTrue(child_pid.exists())
        try:
            p=psutil.Process(int(child_pid.read_text()))
            self.assertEqual(p.status(),psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            pass
        self.assertFalse(output.exists())
        self.assertFalse(list(output.parent.glob('.app2-*')))


if __name__=='__main__':
    unittest.main()
