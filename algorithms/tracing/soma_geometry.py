"""Read-only soma measurements used by G-Cut. Does not modify or save input masks.

Functions extracted unchanged from 0.84_test; no mask repair entry points.
"""
from pathlib import Path
import json
import numpy as np
import tifffile
from utils.gpu_ndimage import ndi
S=np.ones((3,3,3),bool)
REJECTIONS=[]

def soma_volumes(raw):
    """Conservative 3D compact cores; ambiguous broad bright regions are withheld."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    global REJECTIONS
    REJECTIONS=[]
    smooth=ndi.gaussian_filter(raw.astype('float32'),1)
    background=ndi.gaussian_filter(smooth,6)
    contrast=smooth-background
    # Remove narrow shafts before finding soma cores. No MIP-derived growth radius.
    bright=(smooth>=100)&(contrast>=12)
    distance=ndi.distance_transform_edt(bright)
    core=distance>=2.5
    labs,n=ndi.label(core,np.ones((3,3,3)))
    volume=np.zeros(raw.shape,bool);records=[]
    for cid,box in enumerate(ndi.find_objects(labs),1):
        origin=np.maximum(0,np.array([s.start for s in box])-8)
        upper=np.minimum(raw.shape,np.array([s.stop for s in box])+8)
        sl=tuple(slice(int(a),int(b)) for a,b in zip(origin,upper))
        local_core=labs[sl]==cid;dt=distance[sl]
        if local_core.sum()<8:continue
        # Broad plateaus/halos lack a trustworthy soma boundary. Preserve old mask.
        peaks=peak_local_max(np.where(local_core,dt,0),min_distance=6,threshold_abs=2.5,exclude_border=False)
        if not len(peaks):continue
        markers=np.zeros(local_core.shape,np.int32)
        for j,point in enumerate(peaks,1):markers[tuple(point)]=j
        parts=watershed(-dt,markers,mask=local_core,watershed_line=True)
        for j,point in enumerate(peaks,1):
            seed=parts==j;pts=np.argwhere(seed);radius=float(dt[tuple(point)])
            center=point+origin
            reason=None
            if len(pts)<8:reason='tiny_core'
            elif radius>10:reason='broad_uncertain_core'
            if reason:
                REJECTIONS.append(dict(center_xyz=center[::-1].tolist(),reason=reason,core_radius=radius));continue
            eig=np.linalg.eigvalsh(np.cov(pts.astype(float),rowvar=False))
            aspect=float(np.sqrt(eig[-1]/max(eig[0],1e-6)))
            if aspect>4:
                REJECTIONS.append(dict(center_xyz=center[::-1].tolist(),reason='elongated_core',core_radius=radius));continue
            # Grow only a short distance from measured core and require half-height.
            patch=raw[sl];sig=smooth[sl];bg=background[sl]
            peak=float(np.percentile(sig[seed],95));base=float(np.median(bg[seed]))
            threshold=base+max(15,.5*(peak-base))
            near=ndi.distance_transform_edt(~seed)<=3.5
            allowed=near&(sig>=threshold)&(patch>=threshold)&(contrast[sl]>=12)
            # Competing cores cannot be crossed, even if their halos overlap.
            all_near=ndi.distance_transform_edt(~local_core)
            own=ndi.distance_transform_edt(~seed)
            allowed&=(own<=all_near+1e-6)
            grown=ndi.binary_propagation(seed&allowed,structure=np.ones((3,3,3)),mask=allowed)
            # One-voxel guard against other accepted bodies (26-neighbor separation).
            grown&=~ndi.binary_dilation(volume[sl],structure=np.ones((3,3,3)))
            if grown.sum()<25:continue
            volume[sl]|=grown
            records.append(dict(center_xyz=center[::-1].tolist(),core_radius=radius,seed_voxels=int(seed.sum()),seed_axis_ratio=aspect,threshold=threshold,background=base,soma_voxels=int(grown.sum()),bbox_zyx=[[int(a),int(b)] for a,b in zip(origin,upper)],competing_cores=len(peaks)))
    return volume,records

def target_body(raw, xyz, json_path=None, soma_path=None):
    """Use the known soma component, or a small signal-supported seed region.

    Do not snap an empty metadata position to an unrelated foreground fragment.
    """
    p=np.rint(np.asarray(xyz)[::-1]).astype(int);lo=np.maximum(0,p-24);hi=np.minimum(raw.shape,p+25)
    sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));patch=raw[sl].astype(np.float32)
    sig=ndi.gaussian_filter(patch,.65);bg=ndi.gaussian_filter(sig,5);center=tuple(p-lo)
    body=np.zeros(patch.shape,bool);other=np.zeros_like(body)
    if json_path and soma_path and Path(json_path).exists() and Path(soma_path).exists():
        bounds=json.loads(Path(json_path).read_text())['bounds_1um_zxy'];origin=np.array([bounds[0],bounds[4],bounds[2]],int)
        aux=tifffile.imread(soma_path)>0;labs,_=ndi.label(aux,S);q=p-origin;cid=0
        if np.all(q>=0) and np.all(q<aux.shape):cid=int(labs[tuple(q)])
        if not cid:
            a=np.maximum(0,q-4);b=np.minimum(aux.shape,q+5)
            if np.all(b>a):
                local=labs[tuple(slice(int(x),int(y)) for x,y in zip(a,b))];pts=np.argwhere(local>0)
                if len(pts):
                    d=np.linalg.norm(pts+a-q,axis=1);j=int(np.argmin(d))
                    if d[j]<=4:cid=int(local[tuple(pts[j])])
        a=np.maximum(lo,origin);b=np.minimum(hi,origin+np.array(aux.shape))
        if np.all(b>a):
            dst=tuple(slice(int(x),int(y)) for x,y in zip(a-lo,b-lo));src=tuple(slice(int(x),int(y)) for x,y in zip(a-origin,b-origin));v=labs[src]
            body[dst]=(v==cid)&(v>0);other[dst]=(v>0)&(v!=cid)
    # Soma brightness varies by cell; use its local contrast, not a global 100 gate.
    core=(sig-bg>=max(5.,float(sig[center]-bg[center])*.18))&(sig>=max(12.,float(sig[center])*.22))
    if not body.any():
        if sig[center]<12 or sig[center]-bg[center]<4:return sl,body,other,dict(status='no_signal_at_metadata',seed_raw=int(raw[tuple(p)]))
        seed=np.zeros_like(core);seed[center]=True
        radius=np.linalg.norm(np.indices(core.shape).transpose(1,2,3,0)-(p-lo),axis=-1)
        body=ndi.binary_propagation(seed,S,mask=core&(radius<=7))
    else:
        # Segmentation is a spatial prior, not permission to add dark voxels.
        allowed=body&(sig>=max(8.,np.percentile(sig[body],80)*.14))
        body=ndi.binary_fill_holes(allowed)
    if not body.any():return sl,body,other,dict(status='empty_supported_soma')
    # Only a short measured shell; long neurites use the directional trace below.
    allowed=ndi.binary_dilation(body,S,iterations=3)&core&~ndi.binary_dilation(other,S)
    body=ndi.binary_propagation(body&~other,S,mask=body|allowed)
    return sl,body,other,dict(status='supported_soma',seed_raw=int(raw[tuple(p)]),voxels=int(body.sum()))
