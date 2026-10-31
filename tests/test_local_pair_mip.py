import unittest
import numpy as np
from scipy import ndimage as ndi
from algorithms.mask_repair.local_pair_mip import pair_evidence, local_segment, configuration, endpoint_pairs
from algorithms.mask_repair import signal_reconnect as evidence


class LocalPairMipTests(unittest.TestCase):
    def tube(self):
        raw=np.full((41,41,65),5,np.float32)
        z,y,x=np.indices(raw.shape)
        raw += 180*np.exp(-((z-20)**2+(y-20)**2)/2)*((x>=4)&(x<=60))
        p=np.array([20,20,22]);q=np.array([20,20,35])
        a=np.array([[20,20,x] for x in range(22,9,-1)])
        b=np.array([[20,20,x] for x in range(35,48)])
        return raw,p,q,a,b

    def test_local_three_views_and_real_3d_path(self):
        raw,p,q,a,b=self.tube();corridor,report=pair_evidence(raw,p,q,a,b)
        self.assertTrue(report['passed'],report)
        self.assertEqual(len(report['views']),3)
        self.assertTrue(any(v['degenerate'] for v in report['views']))
        labels=np.zeros(raw.shape,np.int32);labels[20,20,4:23]=1;labels[20,20,35:61]=2
        contrast=ndi.gaussian_filter(raw,(0,.6,.6))-ndi.gaussian_filter(raw,(0,4,4))
        original=evidence.RAW_CONTRAST;evidence.RAW_CONTRAST=contrast
        try:result,reason=evidence.path_evidence(raw,contrast,labels,p,q,1,2,'test',corridor)
        finally:evidence.RAW_CONTRAST=original
        self.assertIsNone(reason);self.assertIsNotNone(result)

    def test_dark_gap_is_not_connected_by_bright_endpoints(self):
        raw,p,q,a,b=self.tube();raw[:,:,25:33]=5
        _,report=pair_evidence(raw,p,q,a,b)
        self.assertFalse(report['passed'])

    def test_one_projection_overlap_cannot_connect_different_depths(self):
        raw,p,q,a,b=self.tube();raw[:]=5
        raw[10,20,4:30]=200;raw[30,20,29:61]=200
        p=np.array([10,20,22]);q=np.array([30,20,35]);a=a.copy();a[:,0]=10;b=b.copy();b[:,0]=30
        _,report=pair_evidence(raw,p,q,a,b)
        self.assertFalse(report['passed'])

    def test_long_component_tail_outside_local_segment_has_no_effect(self):
        raw,p,q,a,b=self.tube()
        segment=local_segment(np.array([[20,20,x] for x in range(4,23)]),p,12)[0]
        self.assertLessEqual(np.linalg.norm(np.diff(segment,axis=0),axis=1).sum(),12)
        c1,r1=pair_evidence(raw,p,q,segment,b)
        altered=raw.copy();altered[:,:,:3]=255
        c2,r2=pair_evidence(altered,p,q,segment,b)
        self.assertEqual(r1,r2);np.testing.assert_array_equal(c1[1],c2[1])
        self.assertGreater(r1['bbox_zyx'][2][0],0)

    def test_default_endpoint_search_includes_70_but_excludes_71(self):
        for gap,expected in [(51,1),(70,1),(71,0),(100,0),(101,0),(200,0),(201,0)]:
            p=[10,10,20];q=[10,10,20+gap]
            records=[dict(component=i,endpoint_directions=[dict(point_zyx=tip)])
                     for i,tip in [(1,p),(2,q)]]
            skeletons={1:np.array([[10,10,x] for x in range(8,21)]),
                       2:np.array([[10,10,x] for x in range(20+gap,33+gap)])}
            self.assertEqual(len(list(endpoint_pairs(records,skeletons,configuration({})))),expected)
            self.assertEqual(len(list(endpoint_pairs(records,skeletons,configuration(dict(max_gap=200))))),int(gap<=200))
            self.assertEqual(len(list(endpoint_pairs(records,skeletons,configuration(dict(max_gap=50))))),0)
        with self.assertRaises(ValueError):configuration(dict(max_gap=201))

    def test_input_not_mutated_and_crop_clips_to_boundary(self):
        raw,p,q,a,b=self.tube();before=raw.copy()
        _,report=pair_evidence(raw,p,q,a,b,dict(margin=25))
        np.testing.assert_array_equal(raw,before)
        for (lo,hi),size in zip(report['bbox_zyx'],raw.shape):self.assertTrue(0<=lo<hi<=size)
        for cfg in [dict(enabled=1),dict(max_gap=-1),dict(typo=1)]:
            with self.assertRaises(ValueError):configuration(cfg)

if __name__=='__main__':unittest.main()
