import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import unittest,sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

from algorithms.tracing.soma_signal_groups import filter_seeds
from algorithms.tracing.signal_support import SignalEvidence
from algorithms.mask_repair.root_geodesic import connect

class SomaSignalTests(unittest.TestCase):
 def volume(self,radius=3,gap=False,z_offset=0):
  z,y,x=np.indices((40,48,100));a=np.array([20,24,20]);b=np.array([20+z_offset,24,75]);body=((z-20)**2+(y-24)**2+(x-20)**2<49)|((z-b[0])**2+(y-24)**2+(x-75)**2<49)
  tube=((z-20)**2+(y-24)**2<=radius**2)&(x>=20)&(x<=75)
  if gap:tube&=(x<43)|(x>49)
  raw=np.where(body|tube,200,3).astype('uint8');return raw,a,b,body
 def test_thick_strong_connection_merges(self):
  raw,a,b,_=self.volume(radius=7);kept,audit=filter_seeds(raw,[a.tolist(),b.tolist()],a)
  self.assertEqual(len(kept),1);self.assertEqual(len(audit['merged']),1)
 def test_two_bright_bulbs_with_bright_neurite_stay_separate(self):
  raw,a,b,_=self.volume(radius=3);kept,audit=filter_seeds(raw,[a.tolist(),b.tolist()],a)
  self.assertEqual(len(kept),2);self.assertTrue(any(x['reason']=='independent_bulbs_with_narrow_neck' for x in audit['rejected']))
 def test_dark_gap_not_merged(self):
  raw,a,b,_=self.volume(gap=True);self.assertEqual(len(filter_seeds(raw,[a.tolist(),b.tolist()],a)[0]),2)
 def test_thin_contact_not_merged(self):
  raw,a,b,_=self.volume(radius=0);self.assertEqual(len(filter_seeds(raw,[a.tolist(),b.tolist()],a)[0]),2)
 def test_projection_overlap_not_merged(self):
  raw,a,b,_=self.volume(z_offset=12);self.assertEqual(len(filter_seeds(raw,[a.tolist(),b.tolist()],a)[0]),2)
 def test_diffuse_halo_not_merged(self):
  raw,a,b,_=self.volume();raw[:]=160;self.assertEqual(len(filter_seeds(raw,[a.tolist(),b.tolist()],a)[0]),2)
 def test_target_order_and_idempotence(self):
  raw,a,b,_=self.volume(radius=7);kept,_=filter_seeds(raw,[b.tolist(),a.tolist()],a);self.assertEqual(kept,[a.tolist()]);self.assertEqual(filter_seeds(raw,kept,a)[0],kept)
 def test_root_link_protects_other_soma(self):
  raw,a,b,bodies=self.volume();mask=bodies.copy();mask[20,24,48:76]=True
  own=bodies.copy();own[:,:,50:]=False;other=bodies&~own
  result,audit=connect(raw,mask,own,other)
  self.assertTrue(np.array_equal(result,mask));self.assertFalse(audit['accepted'])
class SharedContextTests(unittest.TestCase):
 def test_only_confirmed_neighbor_component_cleared(self):
  from algorithms.tracing.soma_signal_groups import exclude_confirmed_neighbors
  mask=np.zeros((20,20,40),bool);mask[5:10,5:10,5:10]=True;mask[5:10,5:10,25:30]=True
  context=dict(mask=mask,origin_xyz=np.array([100,200,300]),component_count=2)
  result=exclude_confirmed_neighbors(context,[[307,207,107]])
  self.assertFalse(result['mask'][7,7,7]);self.assertTrue(result['mask'][7,7,27]);self.assertEqual(result['component_count'],1)
  self.assertTrue(context['mask'][7,7,7])
 def test_confirmed_soma_plateau_is_not_a_dark_gap(self):
  from algorithms.tracing.signal_support import prune_unsupported
  raw=np.full((24,24,60),180,dtype='uint8');points=np.array([[12,12,x] for x in range(5,50)])
  tree=[(i+1,3,*p[::-1],1.,-1 if i==0 else i) for i,p in enumerate(points)]
  mask=np.zeros(raw.shape,bool);mask[10:15,10:15,4:51]=True
  self.assertFalse(prune_unsupported(tree,SignalEvidence(raw),points[0,::-1],protect_radius=1,protected_mask=mask)[1])
 def test_real_dark_gap_outside_confirmed_area_still_cut(self):
  from algorithms.tracing.signal_support import prune_unsupported
  raw=np.full((24,24,60),3,dtype='uint8');raw[11:14,11:14,5:50]=150;raw[:,:,28:40]=3
  points=np.array([[12,12,x] for x in range(5,50)]);tree=[(i+1,3,*p[::-1],1.,-1 if i==0 else i) for i,p in enumerate(points)]
  mask=np.zeros(raw.shape,bool);mask[:,:,0:15]=True
  self.assertTrue(prune_unsupported(tree,SignalEvidence(raw),points[0,::-1],protect_radius=1,protected_mask=mask)[1])

if __name__=='__main__':unittest.main()
