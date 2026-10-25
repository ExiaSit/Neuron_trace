"""Version dispatch and read-only comparison file boundaries."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlencode

from web_console.runner import parse_request
from web_console.server import Manager, Handler
from web_console.versions import ROOT, preflight_request
from web_console.import_comparison import imported_progress, register
from utils.case_worker import save_json


class VersionTests(unittest.TestCase):
    def test_version_enum_and_current_default(self):
        self.assertEqual(parse_request({'ids':'42'})['method_version'], 'v0.83a')
        self.assertEqual(parse_request({'ids':'42','method_version':'v0.83'})['method_version'], 'v0.83')
        for v in ['../v0.82', '/tmp/x', '', None, [], {}]:
            with self.assertRaises(ValueError):parse_request({'ids':'42', 'method_version':v})

    def test_legacy_preflight_uses_isolated_selected_source(self):
        req={'ids':['42'], 'method_version':'v0.83'}
        with patch('web_console.versions.subprocess.run') as run:
            run.return_value=type('Result',(),dict(returncode=0,stdout='{"ok":true,"plan":{"required":[]}}'))()
            self.assertEqual(preflight_request(req),{'required':[]})
            call=run.call_args
            self.assertEqual(call.args[0][-1], 'v0.83')
            self.assertEqual(call.kwargs['cwd'],ROOT.parent/'v0.83')
            self.assertEqual(json.loads(call.kwargs['input']),req)

    def test_legacy_existing_jobs_and_new_jobs_launch_selected_runner(self):
        with tempfile.TemporaryDirectory() as temp:
            m=Manager(Path(temp))
            try:
                for version,expected in [(None,'v0.83'),('v0.83a','v0.83a')]:
                    job=m.jobs/'0123456789abcdef';job.mkdir(exist_ok=True)
                    req=dict(name='test',ids=['42'],start=7,stop=7)
                    if version:req['method_version']=version
                    save_json(job/'job.json',dict(request=req,created_at=0))
                    save_json(job/'progress.json',dict(status='paused'))
                    with patch('web_console.server.subprocess.Popen') as popen, patch('web_console.server.threading.Thread'):
                        m.start(job.name)
                        self.assertEqual(Path(popen.call_args.args[0][1]),ROOT.parent/expected/'web_console/runner.py')
                    m.active=None;m.process=None
            finally:m.instance_lock.close()


class ImportedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cohort=self.root/'cohort';self.source=self.cohort/'comparison'
        save_json(self.cohort/'cohort.json',dict(cases=[{'id':'42'}],source_metadata='/tmp/meta.csv'))
        save_json(self.source/'cases/42.json',dict(errors={},mask={},candidate_swc={'final':{'nodes':2}}))
        for v in ['baseline','candidate']:
            save_json(self.source/v/'status/42.json',dict(completed_stages=['continuity','app2'],
                status='failed',current_stage='gcut',seconds={'mask_a':10,'continuity':3,'app2':2},error='test failure'))
            swc=self.source/v/'pruning_swcs/42.swc';swc.parent.mkdir(parents=True);swc.write_text('1 1 0 0 0 1 -1\n')
        self.ids=register(self.root/'state',self.cohort)
        self.manager=Manager(self.root/'state')

    def tearDown(self):
        self.manager.instance_lock.close();self.tmp.cleanup()

    def test_import_is_idempotent_readonly_and_preserves_failures(self):
        self.assertEqual(register(self.root/'state',self.cohort),self.ids)
        job=self.manager.directory(self.ids[1]);d=self.manager.summary(job)
        self.assertTrue(d['read_only']);self.assertEqual(d['progress']['status'],'imported')
        self.assertEqual(d['progress']['completed']['42:7']['seconds'],13)
        self.assertIn('42:9',d['progress']['failed'])
        with self.assertRaisesRegex(ValueError,'只读'):self.manager.start(job.name)
        self.assertTrue((job/'reconstruction').is_symlink())

    def test_strict_swc_validation_failure_overrides_nominal_complete(self):
        save_json(self.source/'baseline/status/42.json',dict(completed_stages=['prune','comparison'],status='complete'))
        save_json(self.source/'cases/42.json',dict(errors={'baseline_full':'Disconnected SWC cycle'}))
        p=imported_progress(self.source,['42'],'baseline')
        self.assertIn('42:11',p['failed']);self.assertNotIn('42:11',p['completed'])

    def test_comparison_http_auth_scope_cell_and_symlink_boundaries(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.manager=self.manager
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        def get(path,auth=True):
            return urlopen(Request(base+path,headers={'X-Control-Token':self.manager.token} if auth else {}))
        def file(path,cell='42',jid=None,scope='comparison'):
            return '/api/file?'+urlencode(dict(id=jid or self.ids[1],scope=scope,cell=cell,path=path))
        try:
            with get('/api/job?'+urlencode(dict(id=self.ids[1],cell='42'))) as r:
                d=json.load(r);self.assertEqual(d['comparison_summary']['versions']['v0.83a']['final']['nodes'],2)
                self.assertEqual(len([a for a in d['artifacts'] if a.get('scope') and a['path'].endswith('.swc')]),2)
            with get(file('candidate/pruning_swcs/42.swc')) as r:
                self.assertEqual(r.read(),(self.source/'candidate/pruning_swcs/42.swc').read_bytes())
            for url in [file('../cohort.json'),file('candidate/pruning_swcs/42.swc',cell='43'),
                        file('candidate/pruning_swcs/421.swc'), file('summary.json',scope='anything')]:
                with self.assertRaises(HTTPError) as e:get(url)
                self.assertEqual(e.exception.code,400)
            with self.assertRaises(HTTPError) as e:get(file('candidate/pruning_swcs/42.swc'),False)
            self.assertEqual(e.exception.code,401)
            (self.source/'summary.json').symlink_to(self.cohort/'cohort.json')
            with self.assertRaises(HTTPError) as e:get(file('summary.json'))
            self.assertEqual(e.exception.code,400)
            job=self.manager.directory(self.ids[0]);record=json.loads((job/'job.json').read_text())
            record.pop('comparison_source');save_json(job/'job.json',record)
            with self.assertRaises(HTTPError):get(file('candidate/pruning_swcs/42.swc',jid=job.name))
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
