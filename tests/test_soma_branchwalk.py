import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from algorithms.tracing import soma_branchwalk as w
class WalkTests(unittest.TestCase):
 def setUp(self):
  self.tree=[[1,1,10.,10.,10.,2.,-1],[2,3,11.,10.,10.,1.,1],[3,3,12.,10.,10.,1.,2],[4,3,13.,11.,10.,1.,3],[5,3,13.,9.,10.,1.,3],[6,3,14.,12.,10.,1.,4],[7,3,14.,10.,10.,1.,4],[8,3,15.,13.,10.,1.,6],[9,3,15.,11.,10.,1.,6],[10,3,16.,14.,10.,1.,8],[11,3,16.,12.,10.,1.,8]]
 def test_root_first_and_every_fork(self):
  visits=w.plan_visits(self.tree,[10,10,10]);self.assertEqual([v['node_id'] for v in visits],[1,3,4,6,8]);self.assertEqual(visits[-1]['upstream_ids'],[1,2,3,4,6,8])
 def test_three_branch_segments_expand_all_children_not_three_outlets(self):
  n,c,r=w.tree_index(self.tree);segments=w.future_segments(n,c,3,4,3)
  self.assertEqual({s['level'] for s in segments},{1,2,3});self.assertEqual({s['end_id'] for s in segments},{4,6,7,8,9});self.assertNotIn(10,{i for s in segments for i in s['node_ids']})
 def test_no_short_terminal_or_geometry_rule(self):
  visits=w.plan_visits(self.tree,[10,10,10]);kept,parts,plan=w.execute(self.tree,visits);self.assertEqual(kept,self.tree);self.assertEqual(parts,[]);self.assertEqual(plan['removed_ids'],[])
 def fixture(self):
  c=w.plan_visits(self.tree,[10,10,10])[1];j=w.review_decisions(c,'uncertain');j['upstream_status']='supported'
  for d in j['outlets']:
   d.update(attachment_validity='supported',decision='KEEP')
  j['outlets'][0].update(attachment_validity='contradicted',decision='CUT',cut_edge=dict(parent_id=3,child_id=4),evidence=[dict(view_id='xy',landmark_id='J',observation='raw crossing'),dict(view_id='xz',landmark_id='J',observation='raw switch')])
  views={k:dict(raw=True,scope='local',plane=p,xyz_bounds=[[0,0,0],[20,20,20]],landmarks={'J':{}}) for k,p in [('xy','XY'),('xz','XZ')]};return c,j,views
 def test_unknown_ownership_can_cut_exact_outlet_with_local_evidence(self):
  c,j,v=self.fixture();self.assertEqual(w.validate_response(j,c,v,{}),[])
 def test_only_xy_or_angle_does_not_authorize_cut(self):
  c,j,v=self.fixture();j['outlets'][0]['evidence']=j['outlets'][0]['evidence'][:1];self.assertIn('cut_missing_local_xy_and_depth_evidence',w.validate_response(j,c,v,{}))
 def test_invalid_edge_branch_and_landmark_fail_closed(self):
  for change in ['edge','branch','landmark']:
   c,j,v=self.fixture()
   if change=='edge':j['outlets'][0]['cut_edge']['child_id']=6
   if change=='branch':j['outlets'][0]['branch_id']='invented'
   if change=='landmark':j['outlets'][0]['evidence'][0]['landmark_id']='invented'
   self.assertTrue(w.validate_response(j,c,v,{}))
 def test_incoming_switch_cannot_be_compensated_by_cutting_exits(self):
  c,j,v=self.fixture();j['upstream_status']='suspected_switch';self.assertTrue(w.validate_response(j,c,v,{}))
 def test_missing_exit_and_malformed_json_are_reviewable(self):
  c,j,v=self.fixture();self.assertTrue(w.validate_response(None,c,v,{}));j['outlets'].pop();self.assertTrue(w.validate_response(j,c,v,{}))
 def test_detached_component_parent_repaired_and_siblings_preserved(self):
  c,j,v=self.fixture();c['groups']=[dict(judgment=j,errors=[],calls=[dict(request='x')])];kept,parts,plan=w.execute(self.tree,[c]);self.assertEqual({int(n[0]) for n in kept},{1,2,3,5});self.assertEqual(parts[0]['tree'][0][6],-1);self.assertEqual(parts[0]['original_parents']['4'],3);self.assertEqual(plan['actions'][0]['affected_forks'],[4,6,8])
 def test_normal_bifurcation_keeps_both_outlets(self):
  c,j,v=self.fixture()
  for d in j['outlets']:d.update(decision='KEEP',attachment_validity='supported',cut_edge=None)
  self.assertEqual(w.validate_response(j,c,v,{}),[]);c['groups']=[dict(judgment=j,errors=[],calls=[])];self.assertEqual(w.execute(self.tree,[c])[0],self.tree)
 def test_metadata_root_mismatch_rejected(self):
  with self.assertRaises(ValueError):w.plan_visits(self.tree,[99,99,99])
 def test_three_view_landmark_and_png_mapping(self):
  from visualization.soma_branchwalk import render
  c=w.plan_visits(self.tree,[10,10,10])[1];raw=np.zeros((30,30,30),np.uint8);raw[10,10,12]=255
  with tempfile.TemporaryDirectory() as td:
   images,views=render(raw,c,dict(xyz_um=[10,10,10]),Path(td));self.assertEqual(len(images),7)
   for name,axes in [('XY',[0,1]),('XZ',[0,2]),('YZ',[1,2])]:
    a=views['initial_local_'+name+'_raw'];b=views['initial_local_'+name+'_hyp'];self.assertEqual(a['xyz_bounds'],b['xyz_bounds']);self.assertEqual(a['window'],b['window']);self.assertEqual(a['landmarks']['J']['xyz_um'],[12.,10.,10.]);self.assertEqual(a['landmarks']['J']['uv_um'],[c['center_xyz'][i] for i in axes]);uv=np.array(a['xyz_to_png_uv'])@np.r_[c['center_xyz'],1];self.assertTrue(np.all(uv>=0));self.assertTrue(np.all(uv<a['image_size_wh']))
if __name__=='__main__':unittest.main()
