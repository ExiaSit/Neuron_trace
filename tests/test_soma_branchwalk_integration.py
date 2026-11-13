"""Offline traversal/export checks; not Gemma biological accuracy."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tifffile
from tools import run_soma_branchwalk as runner
from algorithms.tracing import soma_branchwalk as walk
class IntegrationTests(unittest.TestCase):
 def test_export_component_and_skip_detached_junctions(self):
  tree=[[1,1,10,10,10,2,-1],[2,3,11,10,10,1,1],[3,3,12,10,10,1,2],[4,3,13,11,10,1,3],[5,3,13,9,10,1,3],[6,3,14,12,10,1,4],[7,3,14,10,10,1,4]]
  def fake_call(prompt,meta,images,folder,stem,args):
   outlets=[]
   for b in meta['outlets']:
    cut=meta['junction_id']=='J3' and b['branch_id']=='B4'
    outlets.append(dict(branch_id=b['branch_id'],attachment_validity='contradicted' if cut else 'supported',target_ownership='unknown',decision='CUT' if cut else 'KEEP',cut_edge=b['edge'] if cut else None,evidence=[dict(view_id='initial_local_'+p+'_raw',landmark_id='J',observation='synthetic fixture, not biological evidence') for p in ['XY','XZ']],reason='synthetic'))
   return dict(request=str(folder/(stem+'_request.json')),seconds=0,image_count=len(images),judgment=dict(junction_id=meta['junction_id'],upstream_status='supported',requested_views=[],outlets=outlets))
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);swc=base/'input.swc';raw=base/'raw.tif';out=base/'run';np.savetxt(swc,tree);tifffile.imwrite(raw,np.ones((25,25,25),np.uint8));before=swc.read_bytes()
   with patch.object(runner,'runtime_identity',return_value={'test':True}),patch.object(runner,'model_call',side_effect=fake_call):
    result=runner.run('999999',swc,raw,dict(soma_x=10,soma_y=10,soma_z=10,xy_resolution=1000,z_resolution=1000),out)
   self.assertEqual(result['visited_junctions'],2);self.assertEqual(result['skipped_cut_descendant_junctions'],1)
   self.assertEqual(result['cut_edges'],[dict(parent_id=3,child_id=4)]);self.assertEqual(result['removed_nodes'],3)
   self.assertEqual(swc.read_bytes(),before);self.assertTrue(json.loads((out/'validation.json').read_text())['all_reachable_junctions_reviewed'])
   component=np.loadtxt(out/'components/B4.swc',ndmin=2).tolist();walk.tree_index(component);self.assertEqual(component[0][6],-1)
   self.assertEqual(json.loads((out/'components/B4_original_parents.json').read_text())['original_parents']['4'],3)
   self.assertEqual(np.loadtxt(out/'swc/999999.swc').tolist(),tree[:3]+[tree[4]])
   self.assertTrue((out/'junctions/J3/interactive.html').exists())
 def test_explicit_branchwalk_entry_uses_unpruned_input(self):
  spec=importlib.util.spec_from_file_location('branchwalk_entry',runner.ROOT/'11_prune_swc.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  paths={k:Path('/test')/k for k in ['root','resampled','raw','final']};job=dict(id='95918',config={},metadata={'test':1},fingerprint='123456789abc')
  with patch.object(module,'setup_config',return_value=paths),patch.object(module,'load_module',side_effect=AssertionError('Legacy pruning called')),patch.object(module,'check_swc'),patch.object(module,'save_json'),patch.object(runner,'run',return_value={'policy':walk.VERSION}) as run,patch('shutil.copy2'):
   module.run_branchwalk_experiment(job)
  self.assertEqual(run.call_args.args[:4],('95918',paths['resampled'],paths['raw'],job['metadata']))
 def test_shared_landmarks_identical_across_views(self):
  v={p:dict(raw=True,scope='local',plane=p,xyz_bounds=[[0,0,0],[5,5,5]],landmarks={'J':dict(xyz_um=[1,2,3])}) for p in ['XY','XZ','YZ']}
  landmarks,views=runner.compact_inventory(v);self.assertEqual(landmarks,{'J':[1,2,3]});self.assertEqual(views['YZ']['landmark_ids'],['J'])
  v['YZ']['landmarks']['J']['xyz_um']=[3,2,1]
  with self.assertRaises(ValueError):runner.compact_inventory(v)
class FailureTests(unittest.TestCase):
 def fixture(self):
  tree=[[1,1,10.,10.,10.,2.,-1],[2,3,11.,10.,10.,1.,1]]
  case=walk.plan_visits(tree,[10,10,10])[0];case['case_id']='J1_g1';return case
 def test_transport_error_cannot_export_cut(self):
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);request=out/'cases/J1_g1/round0_request.json'
   fake=dict(error='connection refused',request=str(request),seconds=0)
   with patch.object(runner,'model_call',return_value=fake):
    with self.assertRaises(RuntimeError):runner.review(self.fixture(),np.zeros((24,24,24),np.uint8),dict(xyz_um=[10,10,10]),out,SimpleNamespace(),[])
   self.assertFalse((out/'swc').exists());self.assertTrue((request.parent/'transport_failure.json').exists())
 def test_truncated_response_becomes_review(self):
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);folder=out/'cases/J1_g1';folder.mkdir(parents=True);(folder/'round0_response.json').write_text('{"choices":[{"finish_reason":"length"}]}')
   fake=dict(error='Truncated model output: length',request=str(folder/'round0_request.json'),seconds=0)
   with patch.object(runner,'model_call',return_value=fake):result=runner.review(self.fixture(),np.zeros((24,24,24),np.uint8),dict(xyz_um=[10,10,10]),out,SimpleNamespace(),[])
   self.assertEqual(result['judgment']['outlets'][0]['decision'],'REVIEW');self.assertIsNone(result['judgment']['outlets'][0]['cut_edge'])

if __name__=='__main__':unittest.main()
