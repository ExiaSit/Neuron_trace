import copy
import unittest
from unittest.mock import patch
import numpy as np
from utils import bootstrap
from algorithms.tracing import model_crossing as mc
from tests.test_connection_diagnostics import image_of_paths


class ModelCrossingTests(unittest.TestCase):
    def fixture(self):
        tree=[[1,1,1,1,1,1,-1],[2,3,20,1,1,1,1],[3,3,20,20,1,1,2],
              [4,3,20,30,1,1,3],[5,3,35,1,1,1,2]]
        judgment=dict(decision='switch_to_other',crossing_node_id=2,connection_in_3d='connected',independent_shaft=True,
                      supporting_views=[dict(plane='XY',evidence='Independent shaft visible through crossing'),dict(plane='XZ',evidence='Raw depth slice resolves entry into shaft')])
        site=dict(site_id='001',node_id=2,cut_node_id=3,protected=False,direction=dict(direction_abnormal=True),
                  raw_support=dict(incoming=1.,outgoing=1.,opposite=1.),judgment=judgment)
        site['gate']=mc.cut_gate(site,judgment)
        return tree,dict(input_sha256=mc.tree_digest(tree),sites=[site])

    def test_applies_only_validated_downstream_edge_and_preserves_sibling(self):
        tree,report=self.fixture();kept,audit=mc.apply_saved_plan(tree,report)
        self.assertEqual({n[0] for n in kept},{1,2,5});self.assertEqual(audit['removed_ids'],[3,4])

    def test_changed_input_and_fabricated_edge_rejected(self):
        tree,report=self.fixture();changed=copy.deepcopy(tree);changed[2][2]+=1
        with self.assertRaises(ValueError):mc.apply_saved_plan(changed,report)
        report['sites'][0]['cut_node_id']=4
        with self.assertRaises(ValueError):mc.apply_saved_plan(tree,report)

    def test_model_score_cannot_override_missing_3d_or_turn(self):
        _,report=self.fixture();s=report['sites'][0];s['judgment']['confidence']=1.
        for key,value in [('connection_in_3d','separated'),('crossing_node_id',999),('supporting_views',[dict(plane='XY',evidence='Only one projection overlaps')])]:
            j=dict(s['judgment'],**{key:value});self.assertFalse(mc.cut_gate(s,j)['accepted'])
        s['direction']['direction_abnormal']=False;self.assertFalse(mc.cut_gate(s,s['judgment'])['accepted'])

    def test_no_cut_when_independent_shaft_raw_evidence_missing(self):
        _,r=self.fixture();s=r['sites'][0];s['raw_support']['opposite']=0.
        self.assertFalse(mc.cut_gate(s,s['judgment'])['accepted'])

    def test_truncated_or_invalid_model_json_not_actionable(self):
        for finish,text in [('length','{"decision":"switch_to_other"}'),('stop','not JSON')]:
            with self.assertRaises(ValueError):mc.parse_judgment(dict(choices=[dict(finish_reason=finish,message=dict(content=text))]))

    def test_untraced_raw_crossing_is_detected_but_straight_path_not_abnormal(self):
        tree=[[i+1,1 if i==0 else 3,x,30,20,1,-1 if i==0 else i] for i,x in enumerate(range(4,61))]
        raw=image_of_paths([[[4,30,20],[60,30,20]],[[35,5,20],[35,55,20]]])
        with patch.object(mc,'candidate_sites',return_value=[]):report,_=mc.scan_encounters(tree,raw)
        self.assertTrue(any(abs(s['center_xyz'][0]-35)<8 for s in report['sites']))
        self.assertFalse(any(s['direction']['direction_abnormal'] for s in report['sites']))

    def test_depth_separated_swc_crossing_is_kept_without_model(self):
        tree=[[1,1,4,4,4,1,-1],[2,3,10,30,20,1,1],[3,3,60,30,20,1,2],
              [4,3,35,5,42,1,1],[5,3,35,55,42,1,4]]
        raw=np.zeros((64,64,80),np.uint8);report,_=mc.scan_encounters(tree,raw)
        self.assertTrue(report['separated_crossings'])
        self.assertFalse(report['sites'])

if __name__=='__main__':unittest.main()
