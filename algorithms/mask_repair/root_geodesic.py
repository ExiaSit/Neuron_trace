"""Curved neurite-to-soma links with local 3-D ridge evidence."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils import bootstrap as _bootstrap

import numpy as np
from utils.gpu_ndimage import ndi
from scipy.spatial import cKDTree
from skimage.graph import route_through_array
from algorithms.tracing.signal_support import SignalEvidence
from algorithms.fragment_geometry.robust import tips_for
from .soma_bridge import solid_bridge


def connect(raw, mask, body, other):
    S=np.ones((3,3,3),bool);labels,_=ndi.label(mask,S)
    ids,counts=np.unique(labels[body],return_counts=True);own=int(ids[np.argmax(counts)])
    protected=set(np.unique(labels[other]))-{0,own}
    surface=np.argwhere(body&~ndi.binary_erosion(body,S));tree=cKDTree(surface)
    evidence=SignalEvidence(raw);accepted=[];rejected=[];proposals=[]
    for cid,box in enumerate(ndi.find_objects(labels),1):
        if box is None or cid==own or cid in protected:continue
        local=labels[box]==cid
        if local.sum()<12:continue
        origin=np.array([s.start for s in box]);lo0=origin;hi0=np.array([s.stop for s in box])
        if np.linalg.norm(np.maximum(0,np.maximum(surface.min(0)-hi0,lo0-surface.max(0))))>85:continue
        tips,_=tips_for(local,origin)
        short_surface=set()
        # A thick cut surface may skeletonize into a junction rather than an
        # endpoint. Nominate its nearest outward-facing surface using local PCA.
        points=np.argwhere(local)+origin
        ds,js=tree.query(points);k=int(np.argmin(ds));point=points[k]
        if 2<=ds[k]<=45:
            cloud=points[np.linalg.norm(points-point,axis=1)<=18]
            if len(cloud)>=12:
                eig,vec=np.linalg.eigh(np.cov(cloud,rowvar=False));u=vec[:,-1]
                if u@(point-cloud.mean(0))<0:u=-u
                towards=(surface[js[k]]-point)/ds[k]
                if eig[-1]>2*max(eig[-2],1e-8) and u@towards>.6 and np.mean((cloud-point)@towards>2)<.15:
                    tips.append((point.astype(float),u))
                elif np.mean((cloud-point)@towards>2)<.10 and ds[k]<=35:
                    # Flat/fanned cut ends have no stable PCA axis. Allow only
                    # a tiny, bright missing gap after routing and third-cell checks.
                    tips.append((point.astype(float),towards));short_surface.add(tuple(point))
        for p,u in tips:
            gap,j=tree.query(p)
            if not 2<=gap<=85:continue
            q=surface[j];lo=np.maximum(0,np.minimum(p,q).astype(int)-12);hi=np.minimum(raw.shape,np.maximum(p,q).astype(int)+13)
            sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));v=evidence.smooth[sl];lab=labels[sl]
            cost=1+(30/(1+v))**2
            cost[((lab!=0)&(lab!=cid)&(lab!=own))|other[sl]]=np.inf
            try:path,total=route_through_array(cost,tuple(p.astype(int)-lo),tuple(q-lo),fully_connected=True)
            except ValueError:continue
            if not np.isfinite(total):continue
            path=np.array(path)+lo;length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
            if length>1.8*gap:continue
            tangent=path[min(7,len(path)-1)]-path[0];norm=np.linalg.norm(tangent)
            if norm<2 or tangent@u/norm<.65:continue
            gapmask=labels[tuple(path.T)]==0
            if not gapmask.any():continue
            if tuple(p) in short_surface:
                values=evidence.smooth[tuple(path[gapmask].T)]
                if gapmask.sum()>8 or values.min()<45:continue
            # Soma itself is broad; verify the actual missing segment, with two
            # voxels of tolerance at its boundary, not the soma's flat center.
            stats4=evidence.profile(path,radius=4);stats6=evidence.profile(path,radius=6)
            strong=stats4['center']>=max(45.,.25*float(np.percentile(raw[body],80)))
            broad=np.logical_or.reduce([evidence.profile(path,radius=r)['good'] for r in [8.,12.,18.]])
            good=(stats4['good']|stats6['good']|(strong&broad))[gapmask]
            run=longest=0
            for ok in good:
                run=0 if ok else run+1;longest=max(longest,run)
            fraction=float(good.mean())
            if fraction<.85 or longest>2:
                rejected.append(dict(source=cid,gap=float(gap),reason='insufficient_3d_ridge',fraction=fraction));continue
            pts=solid_bridge(path,mask.shape)
            touch=ndi.binary_dilation(np.isin(lab,[cid,own]),S)
            # A new route must not touch a third pre-existing component.
            pl=pts-lo;valid=np.all(pl>=0,axis=1)&np.all(pl<np.array(lab.shape),axis=1);tube=np.zeros(lab.shape,bool);tube[tuple(pl[valid].T)]=True
            if set(np.unique(lab[ndi.binary_dilation(tube,S)]))-{0,cid,own}:continue
            proposals.append((length,cid,pts,dict(source=cid,gap=float(gap),length=length,ridge_fraction=fraction,path_zyx=path.tolist())))
    final=mask.copy();used=set()
    for _,cid,pts,record in sorted(proposals,key=lambda x:x[0]):
        if cid in used:continue
        final[tuple(pts.T)]=True;accepted.append(record);used.add(cid)
    return final,dict(accepted=accepted,rejected=rejected,added_voxels=int(np.count_nonzero(final&~mask)))


def supplement(raw,mask,xyz,json_path=None,soma_path=None):
    from .guarded_continuation import target_body
    from algorithms.tracing.soma_signal_groups import auxiliary_groups
    from pathlib import Path
    sl,b,other,rec=target_body(raw,xyz,json_path,soma_path)
    if not b.any():return mask,dict(status='no_target_body',accepted=[],added_voxels=0)
    body=np.zeros_like(mask);body[sl]=b;neighbors=np.zeros_like(mask);neighbors[sl]=other;groups={}
    if json_path and soma_path and Path(json_path).exists() and Path(soma_path).exists():
        labs,origin,same,groups=auxiliary_groups(raw,np.asarray(xyz)[::-1],json_path,soma_path)
        lower=np.maximum(0,origin);upper=np.minimum(raw.shape,origin+np.array(labs.shape))
        if np.all(upper>lower):
            dst=tuple(slice(int(a),int(b)) for a,b in zip(lower,upper));src=tuple(slice(int(a),int(b)) for a,b in zip(lower-origin,upper-origin))
            local=labs[src];ids,ct=np.unique(local[body[dst]],return_counts=True);good=ids>0;own=int(ids[good][np.argmax(ct[good])]) if good.any() else 0
            volumes=np.bincount(labs.ravel());keep=np.flatnonzero(volumes>=40);keep=[int(x) for x in keep if x and x!=own and x not in same]
            neighbors[dst]=np.isin(local,keep)
    result,audit=connect(raw,mask|body,body,neighbors)
    audit['soma_groups']=groups
    return result,audit
