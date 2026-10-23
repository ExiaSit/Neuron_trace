import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import sys,unittest
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

from algorithms.tracing.regression_guard import choose,coverage
class GuardTests(unittest.TestCase):
 def data(self):
  raw=np.zeros((28,54,105),np.float32);nodes=[]
  for i,x in enumerate(range(10,90)):
   raw[12,20,x]=180;nodes.append([i+1,3,x,20,12,1,-1 if i==0 else i])
  for i,y in enumerate(range(21,34)):
   raw[12,y,50]=180;nodes.append([81+i,3,50,y,12,1,41 if i==0 else 80+i])
  raw=ndi.gaussian_filter(raw,.6);raw=raw/raw.max()*180+3
  return np.asarray(nodes,float),raw.astype('uint8')
 def test_restore_small_supported_branch_without_cycle(self):
  a,raw=self.data();out,audit=choose(a,a[:80],raw)
  self.assertEqual(audit['decision'],'candidate_with_supported_baseline_branches');self.assertGreater(audit['restored_nodes'],0)
  self.assertGreaterEqual(coverage(a,out,raw),.98)
  seen=set()
  for n in out:
   self.assertNotIn(int(n[0]),seen)
   if int(n[6])>=0:self.assertIn(int(n[6]),seen)
   seen.add(int(n[0]))
 def test_large_loss_retains_baseline(self):
  a,raw=self.data();out,audit=choose(a,a[:20],raw);self.assertEqual(audit['decision'],'baseline');np.testing.assert_array_equal(out,a)
 def test_no_change_leaves_candidate_unchanged(self):
  a,raw=self.data();out,audit=choose(a,a,raw);self.assertEqual(audit['decision'],'candidate');np.testing.assert_array_equal(out,a)
class ForeignSomaTests(unittest.TestCase):
 def test_sparse_edge_cannot_cross_neighbor_unnoticed(self):
  from algorithms.tracing.regression_guard import foreign_contacts
  labels=np.zeros((20,20,60),int);labels[8:13,8:13,25:30]=4
  tree=np.array([[1,1,10,10,10,1,-1],[2,3,50,10,10,1,1]],float)
  self.assertEqual(foreign_contacts(tree,labels,[0,0,0]),{4})
 def test_high_baseline_coverage_does_not_excuse_new_neighbor(self):
  baseline,raw=GuardTests().data();candidate=np.vstack([baseline,[200,3,85,40,12,1,76]])
  labels=np.zeros(raw.shape,int);labels[10:15,35:44,82:90]=2
  out,audit=choose(baseline,candidate,raw,foreign_labels=labels,foreign_origin_zyx=[0,0,0])
  self.assertEqual(audit['reason'],'new_independent_soma_contact');np.testing.assert_array_equal(out,baseline)
 def test_existing_contact_is_not_counted_as_new(self):
  baseline,raw=GuardTests().data();labels=np.zeros(raw.shape,int);labels[10:15,18:23,80:90]=2
  out,audit=choose(baseline,baseline,raw,foreign_labels=labels,foreign_origin_zyx=[0,0,0]);self.assertEqual(audit['decision'],'candidate')

if __name__=='__main__':unittest.main()
