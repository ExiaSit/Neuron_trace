import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from web_console.pruning_index import pruning_index
from web_console.server import Manager, Handler


class PruningIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_counts_missing_empty_invalid_and_node_fallback(self):
        self.write(self.root/'logs/1/pruning-audit.json', {'removal_counts': {'a': 5, 'b': 0}})
        self.write(self.root/'logs/2/pruning-audit.json', {'removal_counts': {}})
        self.write(self.root/'logs/4/pruning-audit.json', {'removal_counts': {'a': -2}})
        self.write(self.root/'pruning_mips/5_pruning_reasons.json', {'removed_nodes': {
            '10': {'code': 'a', 'reason': 'A'}, '11': {'code': 'a', 'reason': 'A'}}})
        self.write(self.root/'logs/6/pruning-audit.json', {'removal_counts': {'secret': 42}})
        d = pruning_index(self.root, ['1','2','3','4','5'])
        self.assertEqual((d['recorded'],d['missing'],d['invalid']), (3,1,1))
        self.assertEqual(d['rules'], [dict(code='a',reason='a',cells=2,nodes=7)])
        self.assertEqual(d['cells'][1]['counts'], {})
        self.assertIsNone(d['cells'][2]['counts'])
        self.assertEqual(d['cells'][4]['source'], 'nodes')

    def test_scope_validation_and_symlink_escape(self):
        outside=self.root/'outside.json'
        self.write(outside, {'removal_counts': {'a': 5}})
        root=self.root/'output';p=root/'logs/1/pruning-audit.json'
        p.parent.mkdir(parents=True);p.symlink_to(outside)
        self.assertEqual(pruning_index(root,['1'])['invalid'],1)
        with self.assertRaises(ValueError):pruning_index(root,['../../other'])

    def test_http_auth_registered_job_and_readonly_output(self):
        manager=Manager(self.root/'state');job=manager.jobs/'0123456789abcdef';job.mkdir()
        self.write(job/'job.json', dict(request=dict(name='test',ids=['42'],start=7,stop=13),created_at=0,read_only=True))
        output=self.root/'imported';output.mkdir();(job/'reconstruction').symlink_to(output)
        self.write(output/'logs/42/pruning-audit.json', {'removal_counts': {'a':3}})
        self.write(output/'logs/43/pruning-audit.json', {'removal_counts': {'a':100}})
        server=ThreadingHTTPServer(('127.0.0.1',0), Handler);server.manager=manager
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}/api/pruning-index?id='
        try:
            with self.assertRaises(HTTPError) as caught:urlopen(base+job.name)
            self.assertEqual(caught.exception.code,401)
            with urlopen(Request(base+job.name,headers={'X-Control-Token':manager.token})) as r:data=json.load(r)
            self.assertEqual([c['id'] for c in data['cells']],['42'])
            self.assertEqual(data['rules'][0]['nodes'],3)
            with self.assertRaises(HTTPError):urlopen(Request(base+'invalid',headers={'X-Control-Token':manager.token}))
        finally:
            server.shutdown();server.server_close();thread.join();manager.instance_lock.close()


if __name__ == '__main__':unittest.main()
