import copy,unittest
import numpy as np
from algorithms.tracing import ownership_audit as a
from visualization.ownership_audit import projection,clipped_segments
from algorithms.tracing.ownership_consistency import audit_conflict_groups
class AuditTests(unittest.TestCase):
 def setUp(self):
  self.tree=[[1,1,0.,0.,0.,1.,-1],[2,3,20.,0.,0.,1.,1],[3,3,21.,0.,0.,1.,2],[4,3,22.,1.,0.,1.,3],[5,3,22.,-1.,0.,1.,3]]
  self.c=dict(candidate_id='A',anchor_id=2,segment_ids=[2],allowed_edges=[dict(parent_id=1,child_id=2,xyz=[20.,0.,0.])],competing_soma_id=None,protected=False)
  self.views={'v':dict(raw=True,scope='local',plane='XZ',xyz_bounds=[[18,-3,-3],[24,3,3]])}
  self.j=dict(candidate_id='A',image_support='supported',attachment_validity='contradicted',target_ownership='unknown',other_soma_id=None,cut_edge=dict(parent_id=1,child_id=2),recommendation='PROPOSE_CUT',evidence=[dict(view_id='v',xyz_bounds=[[20,0,0],[21,0,0]],observation='two separately visible fibers')],missing_evidence=[],requested_views=[])
 def test_unknown_soma_attachment_can_be_reviewed(self):
  self.assertEqual(a.validate_judgment(self.j,self.c,self.views,{}),[])
 def test_invalid_ids_and_bounds(self):
  for key,val in [('candidate_id','bad'),('cut_edge',dict(parent_id=0,child_id=99))]:
   j=copy.deepcopy(self.j);j[key]=val;self.assertTrue(a.validate_judgment(j,self.c,self.views,{}))
  j=copy.deepcopy(self.j);j['evidence'][0]['xyz_bounds'][1][0]=999;self.assertTrue(a.validate_judgment(j,self.c,self.views,{}))
  j=copy.deepcopy(self.j);j['evidence'][0]['view_id']='invented';self.assertTrue(a.validate_judgment(j,self.c,self.views,{}))
 def test_projection_hint_no_execution(self):
  c=dict(self.c,sources=['swc_xy_crossing']);self.assertFalse(a.proposed(c));self.assertEqual(a.execution_plan(self.tree,[c])['removed_ids'],[])
 def test_normal_fork_all_outlets_retained(self):
  c=dict(self.c,judgment=dict(self.j,recommendation='KEEP',attachment_validity='supported',cut_edge=None));kept,parts=a.apply_plan(self.tree,a.execution_plan(self.tree,[c]));self.assertEqual(kept,self.tree);self.assertEqual(parts,[])
 def test_mixed_subtree_held_and_joint_scheduled(self):
  c=dict(self.c,judgment=self.j,verification=self.j);other=dict(self.c,candidate_id='B',anchor_id=4,segment_ids=[3,4],judgment=dict(self.j,target_ownership='target',recommendation='KEEP'))
  plan=a.execution_plan(self.tree,[c,other]);self.assertEqual(plan['removed_ids'],[]);self.assertIn('downstream_target_hypothesis',plan['actions'][0]['hold_reasons']);self.assertEqual(len(audit_conflict_groups(self.tree,[c,other])),1)
 def test_components_parent_mapping_and_accounting(self):
  tree=self.tree[:3];c=dict(self.c,judgment=self.j,verification=self.j);plan=a.execution_plan(tree,[c]);kept,parts=a.apply_plan(tree,plan)
  self.assertEqual(len(kept),1);self.assertEqual(parts[0]['tree'][0][6],-1);self.assertEqual(parts[0]['original_parents']['2'],1)
  plan['input_tree_sha256']='bad'
  with self.assertRaises(ValueError):a.apply_plan(tree,plan)
 def test_unreviewed_fork_not_deleted(self):
  c=dict(self.c,judgment=self.j,verification=self.j);plan=a.execution_plan(self.tree,[c]);self.assertEqual(plan['removed_ids'],[]);self.assertIn('unreviewed_branchpoints',plan['actions'][0]['hold_reasons'])
 def test_duplicate_cut_resolved(self):
  tree=self.tree[:3];j=dict(self.j,target_ownership='other');c=dict(self.c,judgment=j,verification=j);d=dict(c,candidate_id='B');plan=a.execution_plan(tree,[c,d]);self.assertEqual(sum(x['status']=='execute' for x in plan['actions']),1)
 def test_zyx_and_slab_clipping(self):
  raw=np.zeros((7,9,11));raw[4,3,8]=100;lo=np.array([0,0,0]);hi=np.array([10,8,6])
  self.assertEqual(projection(raw,lo,hi,0)[3,8],100);self.assertEqual(projection(raw,lo,hi,1)[4,8],100);self.assertEqual(projection(raw,lo,hi,2)[4,3],100)
  seg=clipped_segments([[[0,0,0],[10,0,10]]],np.array([0,0,4]),np.array([10,0,4]),(0,2))[0];self.assertTrue(np.allclose(seg[:,1],[3.5,4.5]))
  self.assertEqual(clipped_segments([[[0,0,0],[10,0,0]]],np.array([0,0,4]),np.array([10,0,4]),(0,2)),[])
 def test_incomplete_output_never_proposed(self):
  self.assertTrue(a.validate_judgment({},self.c,self.views,{}));self.assertFalse(a.proposed(dict(self.c,judgment=a.review_result('API failed'))))
 def test_soma_metadata_not_first_row(self):
  with self.assertRaises(ValueError):a.target_from_metadata(dict(soma_x=100,soma_y=100,soma_z=100,xy_resolution=1000,z_resolution=1000),np.zeros((10,10,10)),self.tree)
if __name__=='__main__':unittest.main()

class ApiSafetyTests(unittest.TestCase):
 def test_api_failure_and_truncated_json_preserved_without_decision(self):
  import tempfile,json
  from pathlib import Path
  from types import SimpleNamespace
  from unittest.mock import patch
  from tools.run_ownership_audit import model_call
  import urllib.error
  with tempfile.TemporaryDirectory() as td:
   args=SimpleNamespace(endpoint='http://localhost:18484',fingerprint='test')
   with patch('urllib.request.build_opener') as opener:
    opener.return_value.open.side_effect=urllib.error.URLError('offline')
    r=model_call('prompt',{},[],Path(td),'failed',args)
    self.assertIn('error',r);self.assertNotIn('judgment',r)
   for name,response in [('truncated',dict(choices=[dict(finish_reason='length',message=dict(content='{"candidate_id":'))])),('badjson',dict(choices=[dict(finish_reason='stop',message=dict(content='{"candidate_id":'))]))]:
    with patch('urllib.request.build_opener') as opener:
     import io
     opener.return_value.open.return_value.__enter__.return_value=io.StringIO(json.dumps(response))
     r=model_call('prompt',{},[],Path(td),name,args)
     self.assertIn('error',r);self.assertNotIn('judgment',r)
 def test_prompt_and_image_changes_invalidate_cache(self):
  import tempfile
  from pathlib import Path
  from types import SimpleNamespace
  from unittest.mock import patch
  from tools.run_ownership_audit import model_call
  with tempfile.TemporaryDirectory() as td:
   with patch('urllib.request.build_opener') as opener:
    opener.return_value.open.side_effect=OSError('offline');args=SimpleNamespace(endpoint='http://localhost',fingerprint='a')
    model_call('first',{},[],Path(td),'x',args)
    with self.assertRaises(ValueError):model_call('changed',{},[],Path(td),'x',args)

class CandidateTests(unittest.TestCase):
 def test_no_competing_soma_still_generates_fork_candidates(self):
  tree=[[1,1,2,10,5,2,-1],[2,3,20,10,5,1,1],[3,3,24,10,5,1,2],[4,3,30,15,5,1,3],[5,3,30,5,5,1,3]]
  somas=[dict(soma_id=0,center_xyz=[2,10,5],radius_um=2,is_target=True)]
  graph=dict(xyz=np.array([n[2:5] for n in tree]),distances=np.zeros((1,5)),predecessors=np.full((1,5),-9999))
  cases,stats=a.generate_candidates(tree,np.zeros((15,25,40),np.uint8),somas,graph)
  self.assertTrue(any('bifurcation' in c['sources'] for c in cases));self.assertTrue(all(c['competing_soma_id'] is None for c in cases));self.assertEqual(a.execution_plan(tree,cases)['removed_ids'],[])
 def test_known_metadata_target_survives_merged_blob_detection(self):
  from algorithms.tracing.soma_ownership_context import detect_somas
  raw=np.zeros((30,50,80),np.uint8);raw[8:23,15:35,10:65]=255
  # Long connected soma-like core has a remote maximum; metadata still identifies the target.
  target=np.array([60,25,15]);somas=detect_somas(raw,target,allow_known_target=True)
  self.assertEqual(somas[0]['center_xyz'],target.tolist());self.assertTrue(somas[0]['is_target'])

class ResumeAndCompletionTests(unittest.TestCase):
 def test_model_created_timestamp_not_part_of_identity(self):
  from tools.run_ownership_audit import stable_model_metadata
  first={'data':[{'id':'model','created':1,'meta':{'n_ctx':16384}}]};second=copy.deepcopy(first);second['data'][0]['created']=2
  self.assertEqual(stable_model_metadata(first),stable_model_metadata(second));self.assertIn('created',first['data'][0])
  second['data'][0]['meta']['n_ctx']=8192;self.assertNotEqual(stable_model_metadata(first),stable_model_metadata(second))
 def test_incomplete_candidates_or_joint_review_cannot_export(self):
  from tools.run_ownership_audit import finalize
  for report in [dict(cases=[{}]),dict(cases=[dict(judgment={})])]:
   with self.assertRaises(ValueError):finalize(None,None,None,report,None)
 def test_non_string_view_id_fails_closed(self):
  base=AuditTests();base.setUp();j=copy.deepcopy(base.j);j['evidence'][0]['view_id']=['bad']
  self.assertTrue(a.validate_judgment(j,base.c,base.views,{}))
 def test_only_xy_reference_is_incomplete_for_cut_not_semantic_proof(self):
  base=AuditTests();base.setUp();views=copy.deepcopy(base.views);views['v']['plane']='XY'
  self.assertIn('incomplete_local_depth_evidence_not_accuracy_test',a.validate_judgment(base.j,base.c,views,{}))

class ActualPayloadTests(unittest.TestCase):
 def test_saved_image_inventory_matches_transmitted_bytes_and_changes_invalidate(self):
  import tempfile,json,io,base64,hashlib
  from pathlib import Path
  from types import SimpleNamespace
  from unittest.mock import patch
  from tools.run_ownership_audit import model_call
  with tempfile.TemporaryDirectory() as td:
   folder=Path(td);images=[folder/'one.png',folder/'two.png']
   for i,p in enumerate(images):p.write_bytes(bytes([i,1,2,3]))
   with patch('urllib.request.build_opener') as opener:
    opener.return_value.open.return_value.__enter__.return_value=io.StringIO(json.dumps(dict(choices=[dict(finish_reason='stop',message=dict(content='{}'))])))
    args=SimpleNamespace(endpoint='http://localhost',fingerprint='fixed');model_call('p',{},images,folder,'r',args)
    request=opener.return_value.open.call_args.args[0];body=json.loads(request.data);parts=[x for x in body['messages'][0]['content'] if x['type']=='image_url'];saved=json.loads((folder/'r_request.json').read_text())
    self.assertEqual(len(parts),saved['image_count'])
    for p,item in zip(parts,saved['images']):self.assertEqual(hashlib.sha256(base64.b64decode(p['image_url']['url'].split(',')[1])).hexdigest(),item['sha256'])
    images[0].write_bytes(b'changed')
    with self.assertRaises(ValueError):model_call('p',{},images,folder,'r',args)
