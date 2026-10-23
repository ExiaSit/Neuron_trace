"""Conservative evidence for NEW free-growing neurites; not a pruning rule.

A thin fluorescent sheet has axial contrast but no lateral ridge. Median-ring
contrast accepts it because six of eight samples leave the sheet. Require a
ridge across every opposed transverse diameter instead, at two spatial scales.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils import bootstrap as _bootstrap

import numpy as np
from scipy import ndimage as ndi
from algorithms.tracing.signal_support import SignalEvidence

class ContinuationEvidence(SignalEvidence):
    def profile(self, points, tangents=None, radius=4., center_search=False):
        stats=super().profile(points,tangents,radius,center_search)
        p=np.asarray(points,float)
        if not len(p):return stats
        t=np.asarray(tangents if tangents is not None else np.gradient(p,axis=0) if len(p)>1 else [[0.,0.,1.]],float)
        t=t/np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-8)
        ref=np.tile([1.,0.,0.],(len(p),1));ref[np.abs(t[:,0])>.9]=[0,1,0]
        u=np.cross(t,ref);u/=np.maximum(np.linalg.norm(u,axis=1)[:,None],1e-8);v=np.cross(t,u)
        for r,absolute,relative in [(2.,1.,.025),(4.,2.,.06)]:
            ring=np.array([ndi.map_coordinates(self.smooth,(p+r*(np.cos(a)*u+np.sin(a)*v)).T,order=1,mode='nearest') for a in np.linspace(0,2*np.pi,16,endpoint=False)])
            weakest=stats['center']-np.max((ring[:8]+ring[8:])/2,axis=0)
            stats['good'] &= weakest>=np.maximum(absolute,relative*stats['center'])
            stats[f'min_diameter_contrast_{int(r)}']=weakest
        return stats
