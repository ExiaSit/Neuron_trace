import unittest
from tests import test_soma_ownership_context
from algorithms.tracing.ownership_consistency import reconcile_target_conflicts
from algorithms.tracing.soma_ownership_context import apply_ownership_plan

class ConsistencyTests(unittest.TestCase):
    def fixture(self):
        t,r=test_soma_ownership_context.OwnershipTests().fixture()
        r['cases'].append(dict(case_id='target_child',supported_run_ids=[4],judgment=dict(ownership='target',soma_id=0),gate=dict(accepted=False),status='kept_after_review'))
        return t,r

    def test_explicit_target_descendant_holds_ancestor_cut(self):
        t,r=self.fixture();fixed=reconcile_target_conflicts(t,r)
        self.assertTrue(r['cases'][0]['gate']['accepted'])
        self.assertFalse(fixed['cases'][0]['gate']['accepted'])
        self.assertEqual(fixed['cases'][0]['consistency_conflicts'],['target_child'])
        kept,audit=apply_ownership_plan(t,fixed);self.assertEqual(kept,t);self.assertEqual(audit['removed_ids'],[])

    def test_target_on_sibling_does_not_block_cut(self):
        t,r=self.fixture();r['cases'][1]['supported_run_ids']=[5]
        fixed=reconcile_target_conflicts(t,r);self.assertTrue(fixed['cases'][0]['gate']['accepted'])
        _,audit=apply_ownership_plan(t,fixed);self.assertEqual(audit['removed_ids'],[3,4])

    def test_uncertain_descendant_is_not_confirmed_target(self):
        t,r=self.fixture();r['cases'][1]['judgment']['ownership']='uncertain'
        self.assertTrue(reconcile_target_conflicts(t,r)['cases'][0]['gate']['accepted'])

    def test_changed_input_rejected(self):
        t,r=self.fixture();t[2][2]+=1
        with self.assertRaises(ValueError):reconcile_target_conflicts(t,r)

if __name__=='__main__':unittest.main()
