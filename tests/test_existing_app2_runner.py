"""Portable regression checks for the existing-APP2 batch entry point."""
from pathlib import Path
import importlib.util,json,subprocess,sys,tempfile,unittest,csv
import numpy as np
import tifffile

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'tools/run_existing_app2.py'
spec=importlib.util.spec_from_file_location('b45_input_runner',SCRIPT)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class InvalidInputs(unittest.TestCase):
    def test_header_only_failure_is_identified(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'empty.swc';source.write_text('# APP2 failed\n# only comments\n')
            before=source.read_bytes()
            self.assertEqual(runner.check_swc_content(source)[0],'empty_swc')
            self.assertEqual(source.read_bytes(),before)

    def test_mixed_batch_continues_after_invalid_inputs_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            directories={k:root/k for k in ['app2_dir','raw_dir','mask_dir','crop_dir','soma_dir']}
            for p in directories.values():p.mkdir()
            metadata=root/'metadata.csv'
            with metadata.open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['cell_id','soma_x','soma_y','soma_z','xy_resolution','z_resolution'])
                for cid in range(100,106):writer.writerow([cid,2,2,1,1000,1000])
            for cid in range(100,106):
                (directories['app2_dir']/f'image_{cid}.swc').write_text('1 1 2 2 1 1 -1\n2 3 3 2 1 0.5 1\n')
                tifffile.imwrite(directories['raw_dir']/f'image_{cid}_0000.tif',np.ones((3,5,7),dtype=np.uint8),photometric='minisblack')
                tifffile.imwrite(directories['mask_dir']/f'image_{cid}.tif',np.ones((3,5,7),dtype=np.uint8),photometric='minisblack')
                tifffile.imwrite(directories['soma_dir']/f'image_{cid}.tif',np.ones((3,5,7),dtype=np.uint8),photometric='minisblack')
                (directories['crop_dir']/f'image_{cid}.json').write_text(json.dumps({'bounds_order':'ZYX','bounds_1um_zxy':[0,3,0,5,0,7]}))
            (directories['app2_dir']/'image_101.swc').write_text('# APP2 failed\n# only comments\n')
            with (directories['app2_dir']/'image_102.swc').open('a') as f:f.write('malformed trailing row\n')
            (directories['raw_dir']/'image_103_0000.tif').write_bytes(b'not a TIFF')
            (directories['mask_dir']/'image_105.tif').unlink()
            output=root/'output'
            cfg=dict(code_root=str(ROOT),output_root=str(output),metadata_file=str(metadata),vaa3d_resample=str(root/'unused-vaa3d'),workers=1,**{k:str(v) for k,v in directories.items()})
            config=root/'config.json';config.write_text(json.dumps(cfg))
            command=[sys.executable,str(SCRIPT),'--config',str(config),'--prepare-only']
            first=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            self.assertEqual(first.returncode,0,first.stdout)
            summary=json.loads((output/'input_preparation_summary.json').read_text())
            self.assertEqual(summary['prepared_ids'],[100,104])
            self.assertEqual(summary['preparation_failed_ids'],[102,103])
            self.assertEqual(summary['invalid_app2'],1)
            self.assertEqual(summary['missing_inputs'],1)
            pipeline=json.loads((output/'pipeline_config.json').read_text())
            self.assertEqual(pipeline['neuron_ids'],[100,104])
            original=(directories['app2_dir']/'image_104.swc').read_bytes()
            imported=(output/'trace_app2/down_sampled_swcs_app2/image_104.swc').read_bytes()
            again=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            self.assertEqual(again.returncode,0,again.stdout)
            resumed=json.loads((output/'input_preparation_summary.json').read_text())
            self.assertEqual(resumed['reused_inputs'],2)
            self.assertEqual(imported,(output/'trace_app2/down_sampled_swcs_app2/image_104.swc').read_bytes())
            self.assertEqual(original,(directories['app2_dir']/'image_104.swc').read_bytes())
            self.assertFalse((output/'trace_app2/down_sampled_swcs_app2/image_101.swc').exists())

if __name__=='__main__':unittest.main()
