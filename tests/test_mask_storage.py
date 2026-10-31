"""Repair/resume must preserve original data independently of final outputs."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.mask_storage import preserve_original_mask
from utils.stages import load_stage


class MaskStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'input.tif'
        self.source.write_bytes(b'original TIFF bytes')
        self.out = self.root/'output'
        self.repaired = self.out/'mask/42.tif'
        self.repaired.parent.mkdir(parents=True)
        self.spec = dict(id='42',mask=str(self.source),config=dict(output_root=str(self.out)))

    def test_independent_snapshot_resume_and_changed_input(self):
        self.repaired.write_bytes(b'repaired TIFF bytes')
        saved = preserve_original_mask(self.spec)
        self.assertEqual(saved.read_bytes(),self.source.read_bytes())
        self.assertFalse(saved.samefile(self.source))
        stamp = saved.stat().st_mtime_ns
        self.assertEqual(preserve_original_mask(self.spec),saved)
        self.assertEqual(saved.stat().st_mtime_ns,stamp)
        self.source.write_bytes(b'different source bytes')
        with self.assertRaisesRegex(ValueError,'拒绝覆盖'):
            preserve_original_mask(self.spec)
        self.assertEqual(saved.read_bytes(),b'original TIFF bytes')
        self.assertEqual(self.repaired.read_bytes(),b'repaired TIFF bytes')

    def test_repair_rejects_path_symlink_and_hardlink_alias_before_any_stage(self):
        stage = load_stage('continuity')
        for kind in ('same_path','symlink','hardlink'):
            with self.subTest(kind=kind):
                self.repaired.unlink(missing_ok=True)
                if kind=='same_path':
                    self.repaired.write_bytes(b'original TIFF bytes')
                    source=self.repaired
                elif kind=='symlink':
                    self.repaired.symlink_to(self.source);source=self.source
                else:
                    os.link(self.source,self.repaired);source=self.source
                spec=dict(self.spec,mask=str(source))
                with patch.object(stage,'setup_config') as setup:
                    with self.assertRaisesRegex(ValueError,'同一文件'):
                        stage.prepare(spec,{},lambda *args: self.fail('repair ran'))
                    setup.assert_not_called()
                self.assertEqual(self.source.read_bytes(),b'original TIFF bytes')

    def test_interrupted_copy_does_not_publish_partial_original(self):
        def fail(source,destination):
            Path(destination).write_bytes(b'partial')
            raise OSError('disk full')
        with patch('utils.mask_storage.shutil.copyfile',side_effect=fail):
            with self.assertRaises(OSError):preserve_original_mask(self.spec)
        self.assertEqual(list((self.out/'mask_original').iterdir()),[])
        self.assertEqual(preserve_original_mask(self.spec).read_bytes(),self.source.read_bytes())

    def test_resume_after_mask_saved_still_keeps_original_without_rerunning(self):
        self.repaired.write_bytes(b'finished mask')
        stage=load_stage('continuity')
        with patch.object(stage,'setup_config',return_value=dict(root=self.out,mask=self.repaired)):
            stage.prepare(self.spec,{'mask_saved':True},lambda *args:self.fail('repair reran'))
        self.assertEqual((self.out/'mask_original/42.tif').read_bytes(),self.source.read_bytes())
        self.assertEqual(self.repaired.read_bytes(),b'finished mask')


if __name__=='__main__':unittest.main()
