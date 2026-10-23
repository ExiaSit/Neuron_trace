import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import unittest,tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
from utils import baseline_io as m
class MigrationTests(unittest.TestCase):
 def fixture(self,folder,stages):
  base=Path(folder)/'old';out=Path(folder)/'new';cid='1';cfg=dict(output_root=str(out),baseline_root=str(base),method_revision='test',vaa3d_app2='app2',vaa3d_resample='resample',python_executable='python');spec=dict(id=cid,config=cfg,metadata={'x':1},fingerprint='new');stamps=[['a',1,2]]
  fp=hashlib.sha256(json.dumps(dict(inputs=stamps,metadata=spec['metadata'],code='oldcode',tools={k:cfg[k] for k in ['vaa3d_app2','vaa3d_resample','python_executable']}),sort_keys=True).encode()).hexdigest()
  def save(p,data):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
  save(base/'status/1.json',dict(status='complete',fingerprint=fp,completed_stages=stages));save(base/'manifests/1.json',dict(fingerprint=fp))
  for sub,ext in [('mask','.tif'),('trace_app2/down_sampled_swcs_app2','.swc'),('pruning_swcs','.swc')]:
   p=base/sub/(cid+ext);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original')
  save(base/'logs/1/continuity.json',{})
  return base,out,spec,stamps,save
 @patch.object(m,'baseline_code',return_value='oldcode')
 def test_completed_upstream_only(self,_):
  with tempfile.TemporaryDirectory() as t:
   base,out,spec,stamps,save=self.fixture(t,['continuity','app2','gcut','prune']);m.seed(spec,stamps,out/'status/1.json',save,lambda p:None);state=json.loads((out/'status/1.json').read_text());self.assertIn('app2',state['completed_stages']);self.assertNotIn('gcut',state['completed_stages']);self.assertEqual(spec['guard_baseline_root'],str(base));self.assertFalse((out/'pruning_swcs/1.swc').exists())
 @patch.object(m,'baseline_code',return_value='oldcode')
 def test_changed_input_not_reused(self,_):
  with tempfile.TemporaryDirectory() as t:
   base,out,spec,stamps,save=self.fixture(t,['continuity','app2']);stamps[0][1]=3;m.seed(spec,stamps,out/'status/1.json',save,lambda p:None);self.assertFalse((out/'mask/1.tif').exists());self.assertNotIn('guard_baseline_root',spec)
 @patch.object(m,'baseline_code',return_value='oldcode')
 def test_incomplete_continuity_not_reused(self,_):
  with tempfile.TemporaryDirectory() as t:
   base,out,spec,stamps,save=self.fixture(t,['mask_f']);m.seed(spec,stamps,out/'status/1.json',save,lambda p:None);self.assertFalse((out/'status/1.json').exists())
 @patch.object(m,'baseline_code',return_value='oldcode')
 def test_resume_does_not_reseed_new_outputs(self,_):
  with tempfile.TemporaryDirectory() as t:
   base,out,spec,stamps,save=self.fixture(t,['continuity','app2']);m.seed(spec,stamps,out/'status/1.json',save,lambda p:None);p=out/'mask/1.tif';before=p.stat().st_mtime_ns;m.seed(spec,stamps,out/'status/1.json',save,lambda p:None);self.assertEqual(before,p.stat().st_mtime_ns)
 def test_atomic_replacement_does_not_mutate_baseline_hardlink(self):
  with tempfile.TemporaryDirectory() as t:
   base=Path(t)/'base';dest=Path(t)/'dest';other=Path(t)/'other';base.write_bytes(b'original');other.write_bytes(b'replacement');m.copy_output(base,dest,True);m.copy_output(other,dest);self.assertEqual(base.read_bytes(),b'original');self.assertEqual(dest.read_bytes(),b'replacement')
if __name__=='__main__':unittest.main()
