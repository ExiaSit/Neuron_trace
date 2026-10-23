import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import unittest
import numpy as np
from scipy import ndimage as ndi
from test_signal_support import tube
from algorithms.mask_repair.continuation_evidence import ContinuationEvidence
from algorithms.tracing.signal_support import SignalEvidence
from algorithms.mask_repair.guarded_continuation import extend

class RingRegressionTests(unittest.TestCase):
    def test_thin_sheet_is_not_a_tube_despite_axial_contrast(self):
        z,y,x=np.indices((41,64,96));raw=(3+100*np.exp(-(z-20)**2/2)).astype('uint8')
        points=np.array([[20,30,x] for x in range(12,80)])
        self.assertTrue(SignalEvidence(raw).path(points)['supported'])
        self.assertFalse(ContinuationEvidence(raw).path(points)['supported'])
    def test_sheet_does_not_generate_long_arcs(self):
        z,y,x=np.indices((41,64,96));raw=(3+100*np.exp(-(z-20)**2/2)).astype('uint8')
        mask=np.zeros(raw.shape,bool);mask[20,30,20:40]=True
        new,report=extend(raw,mask)
        self.assertTrue(np.array_equal(new,mask));self.assertFalse(report['traces'])
    def test_true_curve_has_two_transverse_ridges(self):
        a=np.linspace(-1.1,1.1,100);p=np.column_stack([np.full(100,20),30+18*np.sin(a),45+18*np.cos(a)])
        self.assertTrue(ContinuationEvidence(tube(p)).path(p)['supported'])
    def test_real_shaft_on_sheet_survives(self):
        p=np.array([[20,30,x] for x in range(12,80)]);raw=tube(p).astype(float)
        z,y,x=np.indices(raw.shape);raw=np.minimum(255,raw+35*np.exp(-(z-20)**2/2)).astype('uint8')
        self.assertTrue(ContinuationEvidence(raw).path(p)['supported'])
    def test_widening_cannot_touch_parallel_fragment(self):
        p=np.array([[20,30,x] for x in range(12,80)]);raw=tube(p)
        mask=np.zeros(raw.shape,bool);mask[20,30,12:30]=True;mask[20,33,40:70]=True
        new,_=extend(raw,mask);labels,_=ndi.label(new,np.ones((3,3,3)))
        self.assertTrue(np.all(new[mask]));self.assertNotEqual(labels[20,30,20],labels[20,33,50])

if __name__=='__main__':unittest.main()
