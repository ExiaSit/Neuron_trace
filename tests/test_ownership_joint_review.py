"""Focused safety tests for revisable, segment-scoped conflict judgments."""
import copy
import unittest
from algorithms.tracing import ownership_joint_review as joint
from algorithms.tracing import ownership_audit as audit

class JointReviewTests(unittest.TestCase):
    def setUp(self):
        self.edge=dict(parent_id=1,child_id=2)
        self.cut=dict(candidate_id='up',recommendation='PROPOSE_CUT',target_ownership='other',cut_edge=self.edge,evidence=[dict(view_id='local')])
        self.up=dict(candidate_id='up',anchor_id=2,segment_ids=[2],judgment=copy.deepcopy(self.cut),verification=copy.deepcopy(self.cut),unresolved_conflict=True,allowed_edges=[dict(self.edge,xyz=[20,0,0])])
        self.down=dict(candidate_id='down',anchor_id=3,segment_ids=[3],judgment=dict(recommendation='KEEP',target_ownership='target',cut_edge=None),reviews={})
        self.group=dict(candidate_id='up',related_ids=['down'],resolved=False,result=dict(judgment=copy.deepcopy(self.cut),errors=[]))
    def test_target_hypothesis_can_change_without_descendant_propagation(self):
        downstream=copy.deepcopy(self.down);other=dict(candidate_id='other',judgment=dict(target_ownership='target'))
        result=dict(judgment=dict(recommendation='KEEP',target_ownership='other',cut_edge=None),errors=[])
        update=joint.scoped_update(downstream,result)
        self.assertEqual(update['before']['target_ownership'],'target');self.assertEqual(downstream['judgment']['target_ownership'],'other')
        self.assertEqual(other['judgment']['target_ownership'],'target');self.assertEqual(update['segment_ids'],[3])
    def test_changed_segment_does_not_reuse_old_cut_verification(self):
        c=copy.deepcopy(self.up);joint.scoped_update(c,dict(judgment=dict(self.cut,cut_edge=dict(parent_id=2,child_id=3)),errors=[]))
        self.assertNotIn('verification',c);self.assertFalse(audit.proposed(c))
    def test_unknown_revision_keeps_conflict_unresolved(self):
        joint.scoped_update(self.down,dict(judgment=dict(recommendation='REVIEW',target_ownership='unknown'),errors=[]))
        records=joint.settle_conflicts([], [self.up,self.down], [self.group])
        self.assertTrue(self.up['unresolved_conflict']);self.assertEqual(records[0]['status'],'unresolved_REVIEW')
    def test_both_sides_revised_can_release_conflict_but_not_assign_tree(self):
        joint.scoped_update(self.down,dict(judgment=dict(recommendation='KEEP',target_ownership='other'),errors=[]))
        result=joint.settle_conflicts([], [self.up,self.down], [self.group])
        self.assertFalse(self.up['unresolved_conflict']);self.assertEqual(result[0]['status'],'segment_hypotheses_revised')
    def test_one_remaining_target_blocks_release(self):
        target=copy.deepcopy(self.down);target['candidate_id']='extra';self.group['related_ids'].append('extra')
        joint.scoped_update(self.down,dict(judgment=dict(recommendation='KEEP',target_ownership='other'),errors=[]))
        joint.settle_conflicts([], [self.up,self.down,target], [self.group]);self.assertTrue(self.up['unresolved_conflict'])
    def test_invalid_segment_output_cannot_clear_conflict(self):
        joint.scoped_update(self.down,dict(judgment=dict(recommendation='KEEP',target_ownership='other'),errors=['bad view']))
        joint.settle_conflicts([], [self.up,self.down], [self.group]);self.assertTrue(self.up['unresolved_conflict'])
    def test_review_task_budget_deduplicates_and_preserves_deferred(self):
        third=dict(self.down,candidate_id='third');group=dict(self.group,related_ids=['down','third'])
        tasks,deferred=joint.segment_review_tasks(dict(cases=[self.up,self.down,third],joint_reviews=[group,group]),maximum=1)
        self.assertEqual(len(tasks),1);self.assertEqual(tasks[0]['context_ids'],['up']);self.assertTrue(any(x['candidate_id']=='third' for x in deferred))
    def test_cited_upstream_field_must_contain_actual_cut(self):
        views=dict(local=dict(raw=True,scope='local',plane='XZ',xyz_bounds=[[0,-1,-1],[10,1,1]]))
        self.assertTrue(joint.stricter_citation_errors(self.cut,self.up,views))
        views['local']['xyz_bounds'][1][0]=21;self.assertEqual(joint.stricter_citation_errors(self.cut,self.up,views),[])
    def test_float_and_bool_cut_ids_are_not_accepted(self):
        for value in [2.0,True,'2']:
            j=copy.deepcopy(self.cut);j['cut_edge']['child_id']=value;self.assertTrue(joint.stricter_citation_errors(j,self.up,{}))

if __name__=='__main__':unittest.main()

class JointStageTests(unittest.TestCase):
    def fixture(self):
        from types import SimpleNamespace
        import numpy as np
        c=dict(candidate_id='A',center_xyz=[20,0,0],allowed_edges=[dict(parent_id=1,child_id=2,xyz=[21,0,0])],segment_ids=[2],competing_soma_id=None,stations=[],upstream_ids=[1],judgment=dict(recommendation='PROPOSE_CUT',cut_edge=dict(parent_id=1,child_id=2)))
        j=dict(candidate_id='A',image_support='supported',attachment_validity='supported',target_ownership='unknown',other_soma_id=None,cut_edge=None,recommendation='KEEP',evidence=[dict(view_id='v',region_id='view_bounds',observation='raw')],missing_evidence=[],requested_views=[])
        return c,j,np.zeros((30,30,30),np.uint8),SimpleNamespace(target_info={},endpoint='unused',fingerprint='test')
    def test_verification_centers_on_exact_edge_without_old_reasoning(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from tools.refine_ownership_conflicts import review_segment
        c,j,raw,args=self.fixture()
        with TemporaryDirectory() as td, patch('tools.refine_ownership_conflicts.runner.render_evidence') as render, patch('tools.refine_ownership_conflicts.runner.model_call') as call:
            image=str(Path(td)/'segment_local_raw.png');render.return_value=([image],dict(v=dict(raw=True,scope='local',plane='XZ',xyz_bounds=[[0,0,0],[29,29,29]],image=image)))
            call.return_value=dict(judgment=j)
            review_segment(c,[],raw,[],[],Path(td),args,verification=True)
            self.assertEqual(render.call_args.kwargs['center'],[21,0,0]);meta=call.call_args.args[1]
            self.assertEqual(meta['proposed_edge_to_check'],c['judgment']['cut_edge']);self.assertNotIn('judgment',meta);self.assertNotIn('before',meta)
    def test_exhausted_supplement_budget_returns_review_without_extra_call(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from tools.refine_ownership_conflicts import review_segment
        c,j,raw,args=self.fixture();j.update(recommendation='REQUEST_MORE',requested_views=['wider'])
        with TemporaryDirectory() as td, patch('tools.refine_ownership_conflicts.runner.render_evidence') as render, patch('tools.refine_ownership_conflicts.runner.model_call') as call:
            image=str(Path(td)/'segment_local_raw.png');render.return_value=([image],dict(v=dict(raw=True,scope='local',plane='XZ',xyz_bounds=[[0,0,0],[29,29,29]],image=image)))
            call.return_value=dict(judgment=j);result=review_segment(c,[],raw,[],[],Path(td),args,previous_supplements=2)
            self.assertEqual(call.call_count,1);self.assertEqual(result['judgment']['recommendation'],'REVIEW');self.assertEqual(result['supplements'],0)
    def test_previous_supplement_key_not_offered_again(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from tools.refine_ownership_conflicts import review_segment
        c,j,raw,args=self.fixture()
        with TemporaryDirectory() as td, patch('tools.refine_ownership_conflicts.runner.render_evidence') as render, patch('tools.refine_ownership_conflicts.runner.model_call') as call:
            image=str(Path(td)/'segment_local_raw.png');render.return_value=([image],dict(v=dict(raw=True,scope='local',plane='XZ',xyz_bounds=[[0,0,0],[29,29,29]],image=image)))
            call.return_value=dict(judgment=j);review_segment(c,[],raw,[],[],Path(td),args,previous_requests=['serial_more'])
            self.assertNotIn('serial_more',call.call_args.args[1]['request_options'])
