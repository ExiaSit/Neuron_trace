import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import unittest,sys
from pathlib import Path
import numpy as np

from algorithms.tracing.soma_signal_groups import filter_seeds,target_core_region
from algorithms.tracing.signal_support import SignalEvidence,prune_unsupported
class SomaCoreTests(unittest.TestCase):
 def test_offcenter_points_in_single_sphere_merge(self):
  z,y,x=np.indices((60,60,60));raw=np.where((z-30)**2+(y-30)**2+(x-30)**2<12**2,240,3).astype('uint8')
  a=[23,30,30];b=[37,30,30];kept,audit=filter_seeds(raw,[a,b],a)
  self.assertEqual(kept,[a]);self.assertTrue(audit['merged'][0]['evidence']['shared_core'])
 def test_close_distinct_bulbs_are_not_single_core(self):
  z,y,x=np.indices((50,50,70));body=((z-25)**2+(y-25)**2+(x-23)**2<8**2)|((z-25)**2+(y-25)**2+(x-43)**2<8**2);tube=((z-25)**2+(y-25)**2<2**2)&(x>=23)&(x<=43)
  raw=np.where(body|tube,240,3).astype('uint8');self.assertEqual(len(filter_seeds(raw,[[25,25,23],[25,25,43]],[25,25,23])[0]),2)
 def test_target_core_follows_measured_body_not_metadata_sphere(self):
  z,y,x=np.indices((60,60,70));raw=np.where((z-30)**2+(y-30)**2+(x-30)**2<12**2,240,3).astype('uint8')
  sl,m,a=target_core_region(raw,[30,30,40]);full=np.zeros(raw.shape,bool);full[sl]=m
  self.assertTrue(full[30,30,25]);self.assertFalse(full[30,30,48]);self.assertGreater(a['radius'],8)
 def test_forbidden_neighbor_stays_unprotected(self):
  z,y,x=np.indices((60,60,70));raw=np.where((z-30)**2+(y-30)**2+(x-30)**2<12**2,240,3).astype('uint8');foreign=np.zeros(raw.shape,bool);foreign[:,:,33:]=True
  sl,m,_=target_core_region(raw,[30,30,30],foreign);self.assertFalse(np.any(m&foreign[sl]))
 def test_empty_target_does_not_snap_to_near_body(self):
  raw=np.zeros((50,50,50),'uint8');raw[20:35,20:35,20:35]=240
  _,m,_=target_core_region(raw,[15,25,25]);self.assertFalse(m.any())
 def test_soma_plateau_is_protected_but_distant_gap_is_not(self):
  z,y,x=np.indices((60,60,90));body=(z-30)**2+(y-30)**2+(x-30)**2<12**2;tube=((z-30)**2+(y-30)**2<2**2)&(x>=30)&(x<=80);raw=np.where(body|tube,240,3).astype('uint8');raw[:,:,60:68]=3
  sl,m,_=target_core_region(raw,[30,30,39]);full=np.zeros(raw.shape,bool);full[sl]=m
  tree=[(i+1,3,float(xx),30.,30.,1.,i if i else -1) for i,xx in enumerate(range(20,81))]
  _,removed,reports=prune_unsupported(tree,SignalEvidence(raw),[39,30,30],protect_radius=1,protected_mask=full)
  self.assertTrue(reports);self.assertGreater(reports[0]['xyz'][0],55)
if __name__=='__main__':unittest.main()
