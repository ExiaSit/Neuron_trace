import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import unittest,sys,tempfile,json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import tifffile

from algorithms.tracing.signal_support import SignalEvidence,prune_unsupported,prune_crossing_switches
from algorithms.mask_repair.guarded_continuation import extend,target_body,connect_roots


def tube(points,shape=(40,64,96),intensity=130):
    v=np.zeros(shape,np.float32)
    for p in points:v[tuple(np.rint(p).astype(int))]=intensity
    v=ndi.gaussian_filter(v,.6);v=v/max(float(v.max()),1)*intensity
    return np.minimum(255,v+3).astype('uint8')

def swc(points):
    return [(i+1,1 if i==0 else 3,*p[::-1],1.,-1 if i==0 else i) for i,p in enumerate(points)]

class SignalSupportTests(unittest.TestCase):
    def test_projection_overlap_is_not_3d_support(self):
        p=np.array([[20,30,x] for x in range(10,80)])
        e=SignalEvidence(tube(p))
        self.assertTrue(e.path(p)['supported'])
        self.assertFalse(e.path(p+[8,0,0])['supported'])
    def test_halo_is_not_transverse_ridge(self):
        zz,yy,xx=np.indices((40,64,96));raw=(3+120*np.exp(-((zz-20)**2+(yy-30)**2)/300)).astype('uint8')
        self.assertFalse(SignalEvidence(raw).path([[20,30,x] for x in range(10,80)])['supported'])
    def test_curved_brightening_branch_survives(self):
        a=np.linspace(-1.2,1.2,70);p=np.column_stack([np.full(70,20),30+15*np.sin(a),40+15*np.cos(a)])
        raw=tube(p);raw[:,:,48:]=np.minimum(raw[:,:,48:].astype(float)*1.7,255)
        final,removed,_=prune_unsupported(swc(p),SignalEvidence(raw),p[0,::-1],protect_radius=3)
        self.assertFalse(removed);self.assertEqual(len(final),len(p))
    def test_sustained_dark_crossing_cut(self):
        p=np.array([[20,30,x] for x in range(10,80)])
        raw=tube(p);raw[:,:,40:49]=3
        final,removed,report=prune_unsupported(swc(p),SignalEvidence(raw),p[0,::-1],protect_radius=3)
        self.assertTrue(removed);self.assertLess(len(final),40);self.assertGreater(report[0]['gap_length'],4.5)
    def test_brief_dip_is_not_cut(self):
        p=np.array([[20,30,x] for x in range(10,80)])
        raw=tube(p);raw[:,:,40]=3
        self.assertFalse(prune_unsupported(swc(p),SignalEvidence(raw),p[0,::-1],protect_radius=3)[1])
    def test_continuation_preserves_source_and_stops_at_dark(self):
        p=np.array([[20,30,x] for x in range(10,75)]);raw=tube(p)
        mask=np.zeros(raw.shape,bool);mask[20,30,10:36]=True
        result,report=extend(raw,mask)
        self.assertTrue(np.all(result[mask]));self.assertGreater(np.count_nonzero(result[:,:,36:70]),10)
        self.assertFalse(result[:,:,80:].any())
    def test_root_link_never_joins_two_known_somas(self):
        zz,yy,xx=np.indices((32,48,80));body=(zz-16)**2+(yy-24)**2+(xx-12)**2<=16
        other=(zz-16)**2+(yy-24)**2+(xx-60)**2<=16
        points=np.array([[16,24,x] for x in range(12,61)]);raw=tube(points,shape=body.shape)
        raw[body|other]=180;mask=body|other;mask[16,24,33:61]=True
        result,report=connect_roots(raw,mask,body,other)
        self.assertFalse(report['accepted']);self.assertTrue(np.array_equal(result,mask))
    def test_empty_metadata_does_not_snap_to_bright_neighbor(self):
        raw=tube(np.array([[20,30,x] for x in range(45,75)]))
        _,body,_,report=target_body(raw,[10,30,20])
        self.assertFalse(body.any());self.assertEqual(report['status'],'no_signal_at_metadata')
    def test_neighbor_soma_kept_separate(self):
        p=np.array([[20,30,x] for x in range(10,75)]);raw=tube(p)
        mask=np.zeros(raw.shape,bool);mask[20,30,10:30]=True;mask[18:23,28:33,60:66]=True
        result,_=extend(raw,mask)
        labs,_=ndi.label(result,np.ones((3,3,3)))
        self.assertNotEqual(labs[20,30,20],labs[20,30,62])

class SparseGcutTests(unittest.TestCase):
    def test_component_peaks_equal_full_volume_search(self):
        from algorithms.tracing.gcut_adapter import component_maxima
        rng=np.random.default_rng(42);values=rng.random((17,23,31))
        labels,count=ndi.label(values>.91,np.ones((3,3,3)))
        expected=[ndi.maximum_position(values,labels=labels,index=i) for i in range(1,count+1)]
        self.assertEqual(component_maxima(values,labels,count),expected)
    def test_long_single_edge_retained_by_physical_length_rule(self):
        sys.path.insert(0,str(_bootstrap.ROOT/'vendor/gcut'))
        from algorithms.tracing.gcut_adapter import make_segmentor
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'input.swc').write_text('1 1 5 10 10 3 -1\n2 3 35 10 10 1 1\n')
            (root/'soma.txt').write_text('1\n')
            segmentor=make_segmentor(str(root/'input.swc'),str(root/'soma.txt'))
            # A 30-um edge survives regardless of its one-node sampling.
            self.assertEqual(set(segmentor.tree_nodes.tree),{1,2})

    def test_distant_soma_child_not_collapsed_into_root(self):
        sys.path.insert(0,str(_bootstrap.ROOT/'vendor/gcut'))
        from algorithms.tracing.gcut_adapter import make_segmentor
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'input.swc').write_text('1 1 5 10 10 3 -1\n2 1 35 10 10 3 1\n3 3 50 10 10 1 2\n')
            (root/'soma.txt').write_text('1\n2\n')
            segmentor=make_segmentor(str(root/'input.swc'),str(root/'soma.txt'))
            self.assertEqual(segmentor.soma_ids,{1,2})
            # Both soma nodes and the 15-um non-soma tip survive.
            self.assertEqual(set(segmentor.tree_nodes.tree),{1,2,3})

class CrossingTests(unittest.TestCase):
    def test_true_bend_without_opposite_shaft_survives(self):
        p=np.array([[20,30,x] for x in range(10,40)]+[[20,y,39] for y in range(31,56)])
        result,removed,_=prune_crossing_switches(swc(p),SignalEvidence(tube(p)),p[0,::-1],protect_radius=5)
        self.assertFalse(removed)
    def test_switch_onto_independent_shaft_is_cut(self):
        p=np.array([[20,30,x] for x in range(10,40)]+[[20,y,39] for y in range(31,56)])
        independent=np.array([[20,y,39] for y in range(10,56)])
        _,removed,report=prune_crossing_switches(swc(p),SignalEvidence(tube(np.vstack([p,independent]))),p[0,::-1],protect_radius=5)
        self.assertTrue(removed);self.assertTrue(report)
    def test_one_voxel_center_offset_is_tolerated(self):
        p=np.array([[20,30,x] for x in range(10,80)])
        _,removed,_=prune_unsupported(swc(p+[1,0,0]),SignalEvidence(tube(p)),p[0,::-1],protect_radius=3)
        self.assertFalse(removed)

if __name__=='__main__':unittest.main()
