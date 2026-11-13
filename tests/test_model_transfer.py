import copy
import unittest
from utils import bootstrap
from algorithms.tracing import model_transfer as mt
from tests import test_model_crossing


class ModelTransferTests(unittest.TestCase):
    def fixture(self):
        tree,report=test_model_crossing.ModelCrossingTests().fixture();s=report['sites'][0];s['root_distance_um']=20.
        v=dict(crossing_node_id=2,trace_transfers_to_distinct_3d_shaft=True,projection_overlap_only=False,
               raw_evidence_sufficient=True,same_neuron_bend_or_bifurcation_excluded=True,
               raw_observations=[dict(plane='XY',observation='Two raw shafts remain distinct across the crossing'),dict(plane='YZ',observation='The outgoing trace follows the distinct higher depth shaft')])
        s['verification']=v;s['transfer_gate']=mt.transfer_gate(s,v)
        return tree,report

    def test_actual_trace_jump_can_cut_even_when_fibers_are_separated(self):
        tree,r=self.fixture();s=r['sites'][0];s['judgment']['connection_in_3d']='separated'
        s['raw_support']['opposite']=0.
        self.assertTrue(mt.transfer_gate(s,s['verification'])['accepted'])
        kept,audit=mt.apply_verified_transfers(tree,r)
        self.assertEqual({n[0] for n in kept},{1,2,5});self.assertEqual(audit['removed_ids'],[3,4])

    def test_projection_overlap_or_unresolved_normal_branch_kept(self):
        _,r=self.fixture();s=r['sites'][0]
        for field,value in [('projection_overlap_only',True),('same_neuron_bend_or_bifurcation_excluded',False),('raw_evidence_sufficient',False),('trace_transfers_to_distinct_3d_shaft',False)]:
            v=dict(s['verification'],**{field:value});self.assertFalse(mt.transfer_gate(s,v)['accepted'])

    def test_second_opinion_cannot_override_wrong_node_or_geometry(self):
        _,r=self.fixture();s=r['sites'][0];v=dict(s['verification'],crossing_node_id=99)
        self.assertFalse(mt.transfer_gate(s,v)['accepted'])
        s['direction']['direction_abnormal']=False
        self.assertFalse(mt.transfer_gate(s,s['verification'])['accepted'])

    def test_changed_input_and_cut_edge_rejected(self):
        tree,r=self.fixture();changed=copy.deepcopy(tree);changed[2][2]+=1
        with self.assertRaises(ValueError):mt.apply_verified_transfers(changed,r)
        r['sites'][0]['cut_node_id']=4
        with self.assertRaises(ValueError):mt.apply_verified_transfers(tree,r)

if __name__=='__main__':unittest.main()
