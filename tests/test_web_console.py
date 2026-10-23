"""Exercise range execution and HTTP control against real temporary inputs."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import subprocess
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from unittest.mock import patch
import numpy as np
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from web_console.runner import parse_request, preflight, required_keys, run_job
from web_console.server import Manager, Handler, contained
from utils.case_worker import save_json


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'source';self.source.mkdir()
        image=np.zeros((12,16,18),dtype=np.uint8);image[3:9,4:12,4:14]=170
        tifffile.imwrite(self.source/'42.tif',image)
        self.meta=self.root/'meta.csv'
        self.meta.write_text('cell_id,v3dpbd_file,soma_x,soma_y,soma_z,xy_resolution,z_resolution\n42,Cell_Image/42.v3draw,8,8,6,1000,1000\n')
        self.body=dict(name='range test',ids='42',start=2,stop=3,gpu_id=0,
                       metadata_file=str(self.meta),source_tif_dir=str(self.source))
        self.manager=Manager(self.root/'state')

    def tearDown(self):
        if self.manager.active:self.manager.stop(self.manager.active)
        self.manager.instance_lock.close()
        self.tmp.cleanup()

    def wait_job(self,jid):
        deadline=time.monotonic()+30
        while self.manager.active==jid and time.monotonic()<deadline:time.sleep(.05)
        self.assertIsNone(self.manager.active)
        return self.manager.summary(self.manager.directory(jid))

    def make_job(self):
        req=parse_request(self.body);plan=preflight(req);job=self.manager.jobs/'0123456789abcdef';job.mkdir()
        save_json(job/'job.json',dict(request=req,plan=plan,created_at=time.time()))
        return job

    def test_range_rejects_shell_or_invalid_ids_and_bounds(self):
        for change in [dict(ids='42;touch /tmp/x'),dict(start=9,stop=7),dict(start=True),dict(ids='')]:
            with self.assertRaises(ValueError):parse_request(dict(self.body,**change))
        self.assertEqual(parse_request(dict(self.body,ids='042, 42，43'))['ids'],['42','43'])

    def test_dependency_graph_supports_late_starts_and_mask_only(self):
        self.assertEqual(required_keys(2,3),{'normalized'})
        self.assertEqual(required_keys(10,10),{'selected'})
        self.assertEqual(required_keys(7,7),{'raw','mask','soma_json','soma_mask'})
        self.assertIn('resampled',required_keys(11,13))
        self.assertNotIn('selected',required_keys(9,13))

    def test_preflight_requires_inputs_before_creating_job(self):
        with self.assertRaises(ValueError):self.manager.create(dict(self.body,ids='43'))
        self.assertEqual(list(self.manager.jobs.iterdir()),[])
        tifffile.imwrite(self.source/'image_42.tif',np.zeros((3,3,3),np.uint8))
        with self.assertRaisesRegex(ValueError,'重复'):preflight(parse_request(self.body))

    def test_actual_resample_crop_job_and_preview(self):
        created=self.manager.create(self.body);done=self.wait_job(created['id'])
        self.assertEqual(done['progress']['status'],'complete',done['progress'])
        job=self.manager.directory(created['id'])
        self.assertEqual(set(done['progress']['completed']),{'42:2','42:3'})
        self.assertTrue((job/'prepared/img/42_0000.tif').is_file())
        self.assertTrue((job/'prepared/soma_img/42.json').is_file())
        self.assertTrue((job/'reconstruction/preparation_mips/42_02.png').is_file())
        self.assertFalse((job/'prepared/soma_seg/42.tif').exists())
        self.assertEqual(tifffile.imread(job/'prepared/img/42_0000.tif').shape,(12,16,18))
        before=(job/'prepared/img/42_0000.tif').stat().st_mtime_ns
        run_job(job)
        self.assertEqual(before,(job/'prepared/img/42_0000.tif').stat().st_mtime_ns)

    def test_pause_then_resume_and_input_change_rejection(self):
        job=self.make_job();(job/'pause.request').touch();run_job(job)
        self.assertEqual(json.loads((job/'progress.json').read_text())['status'],'paused')
        (job/'pause.request').unlink()
        self.meta.write_text(self.meta.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'元数据已变化'):run_job(job)
        self.assertFalse((job/'prepared/img/42_0000.tif').exists())

    def test_imported_upstream_swc_is_copied_without_modifying_source(self):
        upstream=self.root/'upstream';source=upstream/'gcut_output/selected_for_pruning/42.swc'
        source.parent.mkdir(parents=True);source.write_text('1 1 8 8 6 1 -1\n2 3 9 8 6 1 1\n')
        self.body.update(start=10,stop=10,upstream_root=str(upstream))
        job=self.make_job();before=source.read_bytes()
        # Mock the native resampler only; exercise imports, range dispatch and checkpoints.
        def stage(command,**kwargs):
            spec=json.loads(Path(command[3]).read_text())
            self.assertTrue((Path(spec['config']['output_root'])/'logs'/spec['id']).is_dir())
            Path(spec['resampled']).parent.mkdir(parents=True,exist_ok=True)
            Path(spec['resampled']).write_text(before.decode())
            return type('Result',(),{'returncode':0})()
        with patch('web_console.runner.subprocess.run',side_effect=stage):run_job(job)
        self.assertEqual(source.read_bytes(),before)
        copied=job/'reconstruction/gcut_output/selected_for_pruning/42.swc'
        self.assertNotEqual(source.stat().st_ino,copied.stat().st_ino)
        self.assertEqual(copied.read_bytes(),before)

    def test_http_auth_origin_and_path_boundary(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.manager=self.manager
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(url+'/') as r:self.assertIn(b'Neuron Trace',r.read())
            with self.assertRaises(HTTPError) as cm:urlopen(url+'/api/jobs')
            self.assertEqual(cm.exception.code,401)
            request=Request(url+'/api/check',data=json.dumps(self.body).encode(),headers={'Content-Type':'application/json','X-Control-Token':self.manager.token})
            with urlopen(request) as r:self.assertTrue(json.load(r)['ok'])
            request.add_header('Origin','http://different-host.test')
            with self.assertRaises(HTTPError) as cm:urlopen(request)
            self.assertEqual(cm.exception.code,403)
            login=Request(url+'/api/login',data=json.dumps({'token':self.manager.token}).encode(),headers={'Content-Type':'application/json'})
            with urlopen(login) as r:self.assertIn('HttpOnly',r.headers['Set-Cookie'])
            with self.assertRaises(ValueError):contained(self.root,'../outside')
            link=self.root/'escape';link.symlink_to('/etc')
            with self.assertRaises(ValueError):contained(self.root,'escape/passwd')
        finally:server.shutdown();server.server_close();thread.join()

    def test_immediate_stop_terminates_nested_process_and_keeps_checkpoint(self):
        import psutil
        job=self.make_job();save_json(job/'progress.json',dict(status='running',completed={'42:2':{'seconds':1}},failed={}))
        child_file=job/'child.pid'
        script="import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True); Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
        proc=subprocess.Popen([sys.executable,'-c',script,str(child_file)])
        self.manager.active=job.name;self.manager.process=proc
        deadline=time.monotonic()+5
        while not child_file.exists() and time.monotonic()<deadline:time.sleep(.01)
        self.assertTrue(child_file.exists());pid=int(child_file.read_text())
        self.manager.stop(job.name)
        self.assertIsNotNone(proc.poll())
        self.assertTrue(not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE)
        state=json.loads((job/'progress.json').read_text())
        self.assertEqual(state['status'],'interrupted');self.assertIn('42:2',state['completed'])


if __name__=='__main__':unittest.main()
