import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

from pathlib import Path
import tempfile
import unittest
from utils.output_cleanup import cleanup

class CleanupTests(unittest.TestCase):
    def test_preserves_results_and_resume_swcs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            keep=['gcut_mips/123.png','pruning_mips/123_combined_process.png','mask/123.tif','pruning_swcs/123.swc','comparisons/123.png','mask_mips/123.png','logs/123.log','status/123.json','trace_app2/down_sampled_swcs_app2/123.swc','gcut_output/selected_for_pruning/123.swc','gcut_output/selected_resampled_step1/123.swc']
            remove=['mask_steps/123/a/connected-mask.tif','gcut_output/123/123_0000_panels/panel.png','gcut_output/123/123_0000_comprehensive_vis.png','gcut_output/resample_intermediates/123/resampled.swc','pruning_mips/process_data/123/step.tif','pruning_mips/123_old_detail.png','trace_app2/native_swcs/123.swc']
            for name in keep+remove:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('data')
            cleanup(root,'123')
            self.assertTrue(all((root/name).exists() for name in keep))
            self.assertFalse(any((root/name).exists() for name in remove))
            cleanup(root,'123')
    def test_rejects_non_numeric_id(self):
        with self.assertRaises(ValueError):cleanup('/tmp','../other')

if __name__=='__main__':unittest.main()
