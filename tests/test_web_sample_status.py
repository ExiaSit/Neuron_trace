import json
import os
import tempfile
import unittest
from pathlib import Path
from web_console.sample_status import sample_status
from web_console.server import Manager

class SampleStatusTests(unittest.TestCase):
 def fixture(self, root):
  (root/'reconstruction/status').mkdir(parents=True)
  progress=dict(status='running',completed={'1:7':{},'1:8':{}},failed={},current={'id':'2','stage':7})
  (root/'progress.json').write_text(json.dumps(progress));return progress
 def test_parallel_workers_and_terminal_states(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);p=self.fixture(r)
   states={'1':dict(status='complete'),'2':dict(status='running',current_stage='mask_d',started_at=1), '3':dict(status='running',current_stage='gcut',started_at=1),'4':dict(status='failed',timeout=True,error='1800 s'),'5':dict(status='paused')}
   for cid,s in states.items():(r/f'reconstruction/status/{cid}.json').write_text(json.dumps(s))
   a=sample_status(r,dict(ids=['1','2','3','4','5','6'],start=7,stop=8),p,True)
   self.assertEqual(a['counts'],dict(complete=1,running=2,timeout=1,paused=1,pending=1))
   self.assertEqual(a['samples'][2]['stage'],9)
 def test_stale_heartbeat_is_interrupted_not_running_or_failed(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);p=self.fixture(r);(r/'reconstruction/status/2.json').write_text(json.dumps(dict(status='running',current_stage='gcut')))
   os.utime(r/'progress.json',(1,1));a=sample_status(r,dict(ids=['1','2','3'],start=7,stop=8),p,True,now=200)
   self.assertTrue(a['stale']);self.assertEqual([s['status'] for s in a['samples']],['complete','interrupted','pending'])
 def test_native_stage_errors_and_missing_progress(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);p=dict(status='running',completed={'1:7':{}},failed={'2:8':'bad input'},current={'id':'3','stage':8})
   a=sample_status(r,dict(ids=['1','2','3','4'],start=7,stop=7),p,active=True)
   self.assertEqual([s['status'] for s in a['samples']],['complete','failed','running','pending'])
 def test_server_restart_preserves_external_job_progress(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);job=r/'jobs/aaaaaaaaaaaaaaaa';job.mkdir(parents=True)
   p=dict(status='running',completed={});(job/'progress.json').write_text(json.dumps(p));(job/'job.json').write_text(json.dumps(dict(read_only=True,created_at=1,request=dict(name='external',ids=['1'],start=7,stop=13))))
   m=Manager(r)
   self.assertEqual(json.loads((job/'progress.json').read_text()),p)
   m.instance_lock.close()
