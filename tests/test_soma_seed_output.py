import unittest
from algorithms.tracing.soma_seed_output import normalize
from algorithms.tracing.gemma_soma_seeds import validate
class OutputTests(unittest.TestCase):
 def test_only_irrelevant_field_changes_and_input_is_immutable(self):
  j=dict(decision='REJECT',is_soma='no',identity='distinct',duplicate_of=None,evidence=['unchanged'])
  n,changes=normalize(j);self.assertEqual(j['identity'],'distinct');self.assertEqual(n['identity'],'not_applicable');self.assertEqual(n['evidence'],j['evidence']);self.assertEqual(n['decision'],'REJECT');self.assertEqual(len(changes),1)
 def test_uncertain_positive_or_duplicate_is_not_reinterpreted(self):
  for j in [None,[],dict(decision='REVIEW',is_soma='no',identity='distinct'),dict(decision='REJECT',is_soma='uncertain',identity='distinct'),dict(decision='ACCEPT',is_soma='yes',identity='distinct'),dict(decision='DUPLICATE',is_soma='yes',identity='same',duplicate_of='T')]:
   self.assertEqual(normalize(j),(j,[]))
 def test_invalid_candidate_and_views_still_fail(self):
  j=dict(candidate_id='invented',decision='REJECT',is_soma='no',identity='distinct',duplicate_of=None,evidence=[],reason='no body');n,_=normalize(j)
  errors=validate(n,dict(candidate_id='C001'),{},{});self.assertIn('invalid_candidate_id',errors);self.assertIn('missing_local_multiview_evidence',errors)
 def test_non_soma_duplicate_contradiction_is_not_repaired(self):
  j=dict(candidate_id='C001',decision='DUPLICATE',is_soma='no',identity='same',duplicate_of='T',evidence=[],reason='Contradictory output')
  n,changes=normalize(j);self.assertEqual(n,j);self.assertEqual(changes,[])
  self.assertIn('invalid_duplicate',validate(n,dict(candidate_id='C001'),{'T':{}},{}))
 def test_depth_only_duplicate_is_held_for_xy_evidence(self):
  views={p:dict(raw=True,plane=p,landmarks={'C001':dict(in_depth=True),'T':dict(in_depth=True)}) for p in ['XZ','YZ']}
  j=dict(candidate_id='C001',decision='DUPLICATE',is_soma='yes',identity='same',duplicate_of='T',evidence=[dict(view_id=p,observation='Test depth view') for p in views],reason='No XY observation')
  n,changes=normalize(j);self.assertEqual(changes,[])
  self.assertIn('missing_same_body_pair_evidence',validate(n,dict(candidate_id='C001'),{'T':{}},views))
if __name__=='__main__':unittest.main()
