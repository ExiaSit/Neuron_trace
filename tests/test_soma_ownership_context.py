import copy
import unittest
import numpy as np
from utils import bootstrap
from algorithms.tracing import soma_ownership_context as sc
from visualization.soma_ownership_context import clip_segment,unfolded
from tools.run_soma_ownership_experiment import verify_gate


class OwnershipTests(unittest.TestCase):
    def fixture(self):
        tree=[[1,1,0,0,0,3,-1],[2,3,16,0,0,1,1],[3,3,20,0,0,1,2],[4,3,25,0,0,1,3],[5,3,16,8,0,1,2]]
        case=dict(case_id='1',competing_soma_id=2,candidate_cut_ids=[3],cut_options=[dict(node_id=3,parent_id=2,xyz=[20,0,0],localization=dict(supported=True))],other_route_evidence=dict(support_fraction=.9,longest_weak_um=1),intermediate_soma_encounters=[])
        j=dict(ownership='other',soma_id=2,cut_node_id=3,candidate_is_soma=True,other_soma_connection_supported=True,upstream_switch_localized=True,target_origin_rejected_by_image=True,
               evidence=[dict(view='XY',observation='Raw independent soma-connected shaft'),dict(view='YZ',observation='Raw depth change onto the distinct shaft')])
        case.update(judgment=j,gate=sc.ownership_gate(case,j))
        return tree,dict(input_sha256=sc.tree_digest(tree),cases=[case])

    def test_only_selected_subtree_removed(self):
        tree,r=self.fixture();kept,audit=sc.apply_ownership_plan(tree,r)
        self.assertEqual([n[0] for n in kept],[1,2,5]);self.assertEqual(audit['removed_ids'],[3,4])

    def test_distance_and_cost_do_not_authorize_cut(self):
        _,r=self.fixture();c=r['cases'][0]
        j=dict(ownership='other',soma_id=2,cut_node_id=3,confidence=1,nearest_soma=2,graph_cost=1)
        self.assertFalse(sc.ownership_gate(c,j)['accepted'])
        for k in ['candidate_is_soma','other_soma_connection_supported','upstream_switch_localized','target_origin_rejected_by_image']:
            self.assertFalse(sc.ownership_gate(c,dict(c['judgment'],**{k:False}))['accepted'])

    def test_weak_or_third_soma_route_rejected(self):
        _,r=self.fixture();c=r['cases'][0]
        for q in [dict(support_fraction=.7,longest_weak_um=1),dict(support_fraction=.95,longest_weak_um=6)]:
            cc=dict(c,other_route_evidence=q);self.assertFalse(sc.ownership_gate(cc,cc['judgment'])['accepted'])
        cc=dict(c,intermediate_soma_encounters=[0]);self.assertFalse(sc.ownership_gate(cc,cc['judgment'])['accepted'])
        somas=[dict(soma_id=0,center_xyz=[0,0,0],radius_um=2),dict(soma_id=1,center_xyz=[10,0,0],radius_um=2)]
        self.assertEqual(sc.route_soma_encounters([[10,0,0],[0,0,0]],somas,1),[0])

    def test_wrong_identity_unlisted_cut_and_single_projection_rejected(self):
        _,r=self.fixture();c=r['cases'][0]
        for kw in [dict(ownership='target'),dict(soma_id=1),dict(cut_node_id=5),dict(cut_node_id=3.0),dict(evidence=c['judgment']['evidence'][:1])]:
            self.assertFalse(sc.ownership_gate(c,dict(c['judgment'],**kw))['accepted'])

    def test_changed_input_and_root_rejected(self):
        t,r=self.fixture();t[2][2]+=1
        with self.assertRaises(ValueError):sc.apply_ownership_plan(t,r)
        t,r=self.fixture();c=r['cases'][0];c['candidate_cut_ids']=[1];c['judgment']['cut_node_id']=1
        c['cut_options']=[dict(node_id=1,parent_id=-1,xyz=[0,0,0],localization=dict(supported=True))]
        with self.assertRaisesRegex(ValueError,'Soma/root protected'):sc.apply_ownership_plan(t,r)

    def test_supported_owner_cannot_cut_unlocalized_upstream_edge(self):
        _,r=self.fixture();c=r['cases'][0]
        c['cut_options'][0]['localization']['supported']=False
        self.assertIn('cut_lacks_measured_upstream_transition',sc.ownership_gate(c,c['judgment'])['reasons'])

    def test_cut_parent_cannot_change_in_saved_plan(self):
        t,r=self.fixture();r['cases'][0]['cut_options'][0]['parent_id']=1
        with self.assertRaises(ValueError):sc.apply_ownership_plan(t,r)

    def test_no_competing_soma_returns_no_candidates(self):
        t,_=self.fixture();raw=np.zeros((30,30,30),np.uint8)
        somas=[dict(soma_id=0,center_xyz=[0,0,0],radius_um=3)]
        graph=dict(xyz=np.array([[1,0,0],[2,0,0]]),distances=np.array([[0,1]]))
        cases,stats=sc.ownership_candidates(t,raw,somas,graph)
        self.assertEqual(cases,[]);self.assertEqual(stats['reason'],'no_competing_soma_detected')

    def test_clip_keeps_only_inside_portion(self):
        a,b=clip_segment([0,0,0],[10,10,10],1,4,6)
        np.testing.assert_allclose(a,[4,4,4]);np.testing.assert_allclose(b,[6,6,6])
        self.assertIsNone(clip_segment([0,0,0],[5,0,5],1,4,6))
        self.assertIsNotNone(clip_segment([0,5,0],[5,5,5],1,4,6))

    def test_unfolded_reveals_gap_in_known_fiber(self):
        raw=np.zeros((20,20,40),np.uint8);raw[10,10,3:35]=200;raw[10,10,18:22]=0
        a,arc=unfolded(raw,[[3,10,10],[34,10,10]])
        self.assertGreater(a[8,10],150);self.assertLess(a[8,34],10);self.assertAlmostEqual(arc[-1],31.)

    def test_local_recheck_cannot_pass_without_raw_multiview_evidence(self):
        self.assertFalse(verify_gate(dict(confirmed=True,cut_node_id=3),3))
        v=dict(confirmed=True,cut_node_id=3,same_neuron_bend_or_branch_excluded=True,raw_multiview_switch_supported=True,observations=[dict(view=p,observation='Distinct raw neurite changes course across depth') for p in ['XY','XZ']])
        self.assertTrue(verify_gate(v,3));self.assertFalse(verify_gate(v,4))

if __name__=='__main__':unittest.main()
