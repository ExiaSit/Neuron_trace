import unittest
from unittest.mock import patch
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.draw import line_nd
from algorithms.mask_repair.hidden_soma_caps import cap_geometry,recover_caps,bridge_clear,clipped_target
from algorithms.mask_repair import signal_reconnect as evidence
from algorithms.mask_repair.soma_bridge import solid_bridge

class HiddenSomaCapsTests(unittest.TestCase):
    def points(self):
        return np.array([[20,30,x] for x in range(40,61)])
    def geometry(self,points=None,q=(20,30,10),known=()):
        surf=np.array([q]);return cap_geometry(self.points() if points is None else points,cKDTree(surf),surf,known)
    def test_one_sided_cap_is_nominated(self):
        self.assertIsNotNone(self.geometry())
    def test_passing_fiber_side_is_not_nominated(self):
        self.assertIsNone(self.geometry(q=(20,10,50)))
    def test_existing_endpoint_not_replaced(self):
        self.assertIsNone(self.geometry(known=[(np.array([20,30,40]),np.array([0,0,-1]))]))
    def test_search_bound_70_inclusive(self):
        self.assertIsNotNone(self.geometry(q=(20,30,-30)))
        self.assertIsNone(self.geometry(q=(20,30,-31)))
    def test_short_unreliable_cap_not_nominated(self):
        self.assertIsNone(self.geometry(points=self.points()[:7]))
    def test_two_sided_bent_fiber_not_nominated(self):
        p=np.array([[20,30+i,40+abs(i)//3] for i in range(-12,13)])
        self.assertIsNone(self.geometry(points=p))
    def fixture(self):
        shape=(41,61,100);z,y,x=np.indices(shape)
        soma=(z-20)**2+(y-30)**2+(x-10)**2<=16
        mask=soma.copy();mask[20,30,40:81]=1
        lab,_=ndi.label(mask,np.ones((3,3,3)))
        target=int(lab[20,30,10]);source=int(lab[20,30,40]);surf=np.argwhere(soma&~ndi.binary_erosion(soma))
        records=[dict(component=target,soma_id=1,surface=surf,tree=cKDTree(surf))]
        raw=(5+180*np.exp(-((z-20)**2+(y-30)**2)/2)).astype('float32')
        return raw,mask,lab,records,{source:[]}
    def run_fixture(self,raw,mask,lab,records,tips):
        c=ndi.gaussian_filter(raw,(0,.6,.6))-ndi.gaussian_filter(raw,(0,4,4))
        with patch.object(evidence,'RAW_CONTRAST',c):return recover_caps(raw,c,mask,lab,records,tips)
    def test_loop_hidden_branch_cap_is_recovered(self):
        from algorithms.fragment_geometry import robust
        from skimage.morphology import skeletonize
        mask=np.zeros((41,81,100),bool)
        ring=np.array([[20,30,40],[20,29,41],[20,28,42],[20,29,43],
                       [20,30,44],[20,31,43],[20,32,42],[20,31,41]])
        mask[tuple(ring.T)]=True;mask[20,30,44:56]=True
        for dest in [[20,5,85],[20,55,85],[35,30,85]]:
            mask[line_nd((20,30,55),dest,endpoint=True)]=True
        tips,_=robust.tips_for(mask,np.zeros(3))
        self.assertTrue(all(np.linalg.norm(p-[20,30,40])>4 for p,u in tips))
        cap=self.geometry(points=np.argwhere(skeletonize(mask)),known=tips)
        self.assertEqual(cap['p'].tolist(),[20,30,40])
    def test_existing_soma_bridge_is_kept_and_used_as_target(self):
        raw,mask,lab,recs,tips=self.fixture()
        mask[20,30,14:31]=True
        added,accepted,audit=self.run_fixture(raw,mask,lab,recs,tips)
        self.assertEqual(len(accepted),1)
        self.assertEqual(accepted[0]['path_zyx'][-1],[20,30,30])
        self.assertTrue((mask|added)[mask].all())
    def test_ambiguous_soma_candidates_are_withheld(self):
        raw,mask,lab,recs,tips=self.fixture()
        recs.append(dict(recs[0],soma_id=2))
        added,accepted,audit=self.run_fixture(raw,mask,lab,recs,tips)
        self.assertFalse(added.any())
        self.assertEqual(audit['rejected'][0]['reason'],'hidden_cap_ambiguous_somas')
    def test_raw_tube_connects_and_originals_survive(self):
        raw,mask,lab,recs,tips=self.fixture();added,accepted,audit=self.run_fixture(raw,mask,lab,recs,tips)
        self.assertEqual(len(accepted),1);self.assertTrue((mask|added)[mask].all())
        self.assertEqual(ndi.label(mask|added,np.ones((3,3,3)))[1],1)
    def test_dark_gap_does_not_connect(self):
        raw,mask,lab,recs,tips=self.fixture();raw[:,:,22:29]=5
        added,accepted,audit=self.run_fixture(raw,mask,lab,recs,tips)
        self.assertFalse(added.any());self.assertEqual(accepted,[])
    def test_broad_sheet_does_not_connect(self):
        raw,mask,lab,recs,tips=self.fixture();raw[:]=100
        added,accepted,audit=self.run_fixture(raw,mask,lab,recs,tips)
        self.assertFalse(added.any())
    def test_source_already_assigned_to_soma_is_not_moved(self):
        raw,mask,lab,recs,tips=self.fixture();surf=np.argwhere(lab==next(iter(tips)))
        recs.append(dict(component=next(iter(tips)),soma_id=2,surface=surf,tree=cKDTree(surf)))
        added,accepted,audit=self.run_fixture(raw,mask,lab,recs,tips)
        self.assertFalse(added.any())
    def test_third_component_and_prior_supplement_bridge_block(self):
        lab=np.zeros((11,11,30),int);lab[5,5,2]=1;lab[5,5,25]=2
        coords=solid_bridge([[5,5,x] for x in range(2,26)],lab.shape);added=np.zeros(lab.shape,bool)
        self.assertTrue(bridge_clear(coords,lab,1,2,added))
        lab[5,6,12]=3;self.assertFalse(bridge_clear(coords,lab,1,2,added))
        lab[5,6,12]=0;added[5,5,12]=1;self.assertFalse(bridge_clear(coords,lab,1,2,added))
    def test_clip_to_assigned_branch_not_full_soma_route(self):
        lab=np.zeros((11,11,30),int);lab[5,5,2:15]=1;lab[5,5,25]=2
        p=[[5,5,x] for x in range(25,1,-1)]
        self.assertEqual(clipped_target(p,lab,1).tolist(),[5,5,14])

if __name__=='__main__':unittest.main()
