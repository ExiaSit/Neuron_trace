"""Actual result paths match the same containment boundary as downloads."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from web_console.artifact_metadata import with_storage_paths
from web_console.comparison import comparison_artifacts
from web_console.server import artifacts,Handler,Manager
from utils.case_worker import save_json


class ResourcePathTests(unittest.TestCase):
    def test_skeleton_input_is_downloadable_and_cell_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'app2_input';folder.mkdir()
            for cid in ['42','420']:
                (folder/f'{cid}_skeleton_soma.tif').write_bytes(b'fixture')
            result=artifacts(root,'42')
            self.assertEqual([a['path'] for a in result],['app2_input/42_skeleton_soma.tif'])

    def test_original_and_repaired_mask_are_separate_cell_scoped_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder in ['mask_original','mask']:
                for cid in ['42','420']:
                    p=root/folder/f'{cid}.tif';p.parent.mkdir(exist_ok=True);p.write_bytes(folder.encode())
            result=artifacts(root,'42')
            self.assertEqual({a['path'] for a in result},{'mask_original/42.tif','mask/42.tif'})
            self.assertEqual({a['name'] for a in result},{'42_original_mask.tif','42_repaired_mask.tif'})
            self.assertTrue(all(not a['image'] and Path(a['storage_path']).is_file() for a in result))

    def test_imported_root_symlink_resolves_and_leaf_escape_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);real=root/'results';real.mkdir()
            (real/'42.swc').write_text('1 1 0 0 0 1 -1\n')
            alias=root/'imported';alias.symlink_to(real,target_is_directory=True)
            outside=root/'outside.json';outside.write_text('{}')
            (real/'escape.json').symlink_to(outside)
            items=[dict(path=p,name=p) for p in ['42.swc','escape.json','../outside.json',str(outside),'.']]
            result=with_storage_paths(alias,items)
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]['storage_path'],str(real/'42.swc'))
            self.assertEqual(result[0]['path'],'42.swc')
            self.assertNotIn('storage_path',items[0])

    def test_same_name_stage_files_have_distinct_actual_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folders=['pruning_swcs','trace_app2/down_sampled_swcs_app2',
                'gcut_output/selected_for_pruning','gcut_output/selected_resampled_step1']
            for folder in folders:
                p=root/folder/'60575.swc';p.parent.mkdir(parents=True);p.write_text('1 1 0 0 0 1 -1\n')
            result=artifacts(root,'60575')
            self.assertEqual(len(result),4)
            self.assertEqual(len({r['storage_path'] for r in result}),4)
            self.assertTrue(all(Path(r['storage_path']).read_bytes()==b'1 1 0 0 0 1 -1\n' for r in result))

    def test_comparison_root_paths_and_http_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);comparison=root/'comparison'
            for variant in ['baseline','candidate']:
                p=comparison/variant/'pruning_swcs/42.swc';p.parent.mkdir(parents=True)
                p.write_text('1 1 0 0 0 1 -1\n')
            record=dict(read_only=True,comparison_source=str(comparison),created_at=0,
                request=dict(name='test',ids=['42'],start=7,stop=13,method_version='v0.83a'))
            result=comparison_artifacts(record,'42')
            self.assertEqual({r['storage_path'] for r in result},
                {str(comparison/v/'pruning_swcs/42.swc') for v in ['baseline','candidate']})
            m=Manager(root/'state');job=m.jobs/'0123456789abcdef';job.mkdir()
            save_json(job/'job.json',record);save_json(job/'progress.json',dict(status='imported',completed={}))
            (job/'reconstruction').symlink_to(comparison/'candidate',target_is_directory=True)
            raw=root/'raw.tif';raw.write_bytes(b'fixture')
            save_json(comparison/'candidate/manifests/42.json',dict(id='42',raw=str(raw)))
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.manager=m
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base+'/resources.js') as r:self.assertIn(b'NeuronResources',r.read())
                with urlopen(base+'/vaa3d.js') as r:self.assertIn(b'NeuronVaa3D',r.read())
                with self.assertRaises(HTTPError) as unauthorized:urlopen(base+'/api/job?id='+job.name)
                self.assertEqual(unauthorized.exception.code,401)
                req=Request(base+'/api/job?id='+job.name+'&cell=42',headers={'X-Control-Token':m.token})
                with urlopen(req) as r:data=json.load(r)
                self.assertTrue(all(Path(a['storage_path']).is_file() for a in data['artifacts']))
                self.assertEqual(len(data['artifacts']),3) # own final + both comparison finals
                self.assertEqual(data['vaa3d']['image'],str(raw))
                self.assertEqual(len(data['vaa3d']['choices']),2) # own final deduplicated
                self.assertEqual(sum(c['default'] for c in data['vaa3d']['choices']),1)
                wrong=Request(base+'/api/job?id='+job.name+'&cell=420',headers={'X-Control-Token':m.token})
                with self.assertRaises(HTTPError) as mismatch:urlopen(wrong)
                self.assertEqual(mismatch.exception.code,400)
            finally:server.shutdown();server.server_close();thread.join();m.instance_lock.close()


if __name__=='__main__':unittest.main()
