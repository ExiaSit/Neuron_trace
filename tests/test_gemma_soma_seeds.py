import importlib.util,tempfile,unittest,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np
import tifffile
from algorithms.tracing import gemma_soma_seeds as s
from visualization.gemma_soma_seeds import render
ROOT=Path(__file__).resolve().parents[1]
class SeedTests(unittest.TestCase):
 def fixture(self):
  c=dict(candidate_id='C001',xyz_um=[12,13,14],zyx=[14,13,12]);j=dict(candidate_id='C001',is_soma='yes',identity='distinct',decision='ACCEPT',duplicate_of=None,evidence=[dict(view_id=p,observation='synthetic') for p in ['xy','xz']],reason='synthetic');v={k:dict(raw=True,plane=p,landmarks={'C001':dict(in_depth=True),'T':dict(in_depth=True)}) for k,p in [('xy','XY'),('xz','XZ')]};return c,j,v
 def test_strict_threshold(self):
  d=np.zeros((5,5,9));d[2,2,2]=4.;d[2,2,6]=4.01;self.assertEqual([p['zyx'] for p in s.peaks(d,4.)],[[2,2,6]])
 def test_lower_threshold_union_preserves_old_merged_peak(self):
  d=np.zeros((5,5,9));d[2,2,2:7]=5;d[2,2,2]=7;d[2,2,6]=6
  with patch.object(s,'gsdt_volume',return_value=(1,np.ones_like(d),d)):b=s.candidates(np.ones_like(d),np.ones_like(d),[0,0,0])
  self.assertEqual(len(b['baseline_gsdt']),2);self.assertEqual(len(b['relaxed_gsdt']),1);self.assertEqual(len(b['candidates']),2)
 def test_valid_duplicate_and_close_distinct_both_allowed(self):
  c,j,v=self.fixture();self.assertEqual(s.validate(j,c,{'T':{}},v),[])
  j.update(identity='same',decision='DUPLICATE',duplicate_of='T');self.assertEqual(s.validate(j,c,{'T':{}},v),[])
 def test_invalid_id_or_reference_or_xy_only_fails(self):
  c,j,v=self.fixture();j.update(identity='same',decision='DUPLICATE',duplicate_of='invented');self.assertTrue(s.validate(j,c,{'T':{}},v))
  c,j,v=self.fixture();j['candidate_id']='invented';self.assertTrue(s.validate(j,c,{},v))
  c,j,v=self.fixture();j['evidence']=j['evidence'][:1];self.assertTrue(s.validate(j,c,{},v))
 def test_projected_slice_reference_outside_depth_cannot_merge(self):
  c,j,v=self.fixture();j.update(identity='same',decision='DUPLICATE',duplicate_of='T');v['xz']['landmarks']['T']['in_depth']=False;self.assertIn('missing_same_body_pair_evidence',s.validate(j,c,{'T':{}},v))
 def test_malformed_output_stays_review(self):
  c,j,v=self.fixture();self.assertTrue(s.validate([],c,{},v));self.assertEqual(s.review_fallback(c,'bad JSON')['decision'],'REVIEW')
 def test_catalogue_has_no_distance_suppression_and_retains_review(self):
  cs=[dict(candidate_id='C001',xyz_um=[1,1,1],zyx=[1,1,1]),dict(candidate_id='C002',xyz_um=[2,1,1],zyx=[1,1,2]),dict(candidate_id='C003',xyz_um=[3,1,1],zyx=[1,1,3])];bundle=dict(target=dict(candidate_id='T',xyz_um=[0,1,1],zyx=[1,1,0]),candidates=cs)
  results={c['candidate_id']:dict(errors=[],decision=dict(decision='ACCEPT')) for c in cs};results['C003']['decision']=dict(decision='REVIEW');a=s.assemble(bundle,results)
  self.assertEqual(len(a['accepted']),3);self.assertEqual(a['review'],['C003']);self.assertFalse(a['downstream_executed'])
 def test_nms_control_strict_radius(self):
  self.assertEqual(len(s.nms_reference([[0,0,0],[0,0,6]],[2,1],6)),2)
  self.assertEqual(len(s.nms_reference([[0,0,0],[0,0,5]],[2,1],6)),1)
 def test_xyz_mapping_and_all_three_planes(self):
  c,j,v=self.fixture()
  with tempfile.TemporaryDirectory() as td:
   ims,vs=render(np.zeros((30,30,30),np.uint8),c,[dict(candidate_id='T',xyz_um=[10,10,10])],Path(td));self.assertEqual(len(ims),3)
   for p,axes in [('XY',[0,1]),('XZ',[0,2]),('YZ',[1,2])]:
    a=vs['local_'+p];self.assertEqual(a['landmarks']['C001']['xyz_um'],[12,13,14]);self.assertEqual(a['landmarks']['C001']['uv_um'],[c['xyz_um'][i] for i in axes]);uv=np.array(a['xyz_to_png_uv'])@np.r_[c['xyz_um'],1];self.assertTrue(np.all(uv>=0));self.assertTrue(np.all(uv<a['image_size_wh']))
 def test_gcut_callback_returns_before_nms_or_segmentation(self):
  from utils.case_worker import load_module
  cfg=SimpleNamespace(BASE_DIR=ROOT,PATHS={'gcut':str(ROOT/'vendor/gcut')},GCUT={})
  with tempfile.TemporaryDirectory() as td,patch.dict(sys.modules,{'pipeline_config':cfg}):
   p=Path(td);a=np.zeros((20,20,20),np.uint8);a[5:15,5:15,5:15]=100;tifffile.imwrite(p/'raw.tif',a);tifffile.imwrite(p/'mask.tif',np.ones_like(a));m=load_module('gcut');callback=Mock(return_value={'only':'seed review'})
   with patch.object(m,'apply_nms',side_effect=AssertionError('NMS executed')),patch.object(m,'parse_swc',side_effect=AssertionError('G-Cut executed')):
    result=m.process_image_to_gcut(p/'raw.tif',p/'mask.tif',None,p,[10,10,10],55.,5.,seed_review=callback)
   self.assertEqual(result,{'only':'seed review'});callback.assert_called_once()
 def test_default_pruning_restored_without_branchwalk(self):
  spec=importlib.util.spec_from_file_location('restored_prune',ROOT/'11_prune_swc.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);paths={k:root/k for k in ['root','resampled','mask','raw','final','soma_json','soma_mask']};paths['root']=root;worker=Mock(return_value={'status':'success'});job=dict(id='95918',config={},metadata={})
   with patch.object(m,'setup_config',return_value=paths),patch.object(m,'load_module',return_value=SimpleNamespace(worker_task=worker)),patch.object(m,'check_swc'),patch('tools.run_soma_branchwalk.run',side_effect=AssertionError('Branchwalk executed')):m.run(job)
   worker.assert_called_once()
if __name__=='__main__':unittest.main()
