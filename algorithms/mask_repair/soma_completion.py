"""Add raw-intensity-supported soma volumes to the v5 bridge result.

MIP blobs nominate regions; independent 3D bright components seed bounded growth.
No unmeasured Z-column filling. Thresholds calibrated for this 8-bit image only.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils import bootstrap as _bootstrap

from pathlib import Path
import os
RAW_PATH=os.environ.get('RECONNECT_RAW','')
MASK_PATH=os.environ.get('RECONNECT_MASK','')
WORK=Path(os.environ.get('RECONNECT_WORK','.'))

import json
import numpy as np
import tifffile
from utils.gpu_ndimage import ndi
from skimage.measure import marching_cubes
O=WORK/'c'

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

def main():
    O.mkdir(parents=True,exist_ok=True)
    raw=tifffile.imread(RAW_PATH)
    previous=tifffile.imread(WORK/'b/connected-mask.tif')>0
    soma,records=soma_volumes(raw)
    # Soma growth may fill a body but must not silently join existing fragments.
    previous_labels,_=ndi.label(previous,np.ones((3,3,3)))
    body_labels,_=ndi.label(soma,np.ones((3,3,3)))
    guarded=np.zeros_like(soma)
    for bid,box in enumerate(ndi.find_objects(body_labels),1):
        lo=np.maximum(0,np.array([s.start for s in box])-1);hi=np.minimum(raw.shape,np.array([s.stop for s in box])+1)
        sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));body=body_labels[sl]==bid;lab=previous_labels[sl]
        ids,counts=np.unique(lab[body],return_counts=True);valid=ids>0
        anchor=int(ids[valid][np.argmax(counts[valid])]) if valid.any() else 0
        other=(lab>0)&(lab!=anchor)
        body&=~ndi.binary_dilation(other,structure=np.ones((3,3,3)))
        guarded[sl]|=body
    soma=guarded;added=soma&~previous;fixed=previous|soma
    for rec in records:
        sl=tuple(slice(a,b) for a,b in rec['bbox_zyx']);local=soma[sl];rec['previous_coverage_fraction']=float(np.sum(local&previous[sl])/max(local.sum(),1));rec['new_voxels']=int(added[sl].sum())
    assert np.all(fixed[previous])
    n0=ndi.label(previous,np.ones((3,3,3)))[1];n1=ndi.label(fixed,np.ones((3,3,3)))[1]
    for name,data in [('connected-mask',fixed),('soma-only',soma),('soma-added-only',added)]:tifffile.imwrite(O/(name+'.tif'),data.astype('uint8')*255,compression='zlib')
    report=dict(somas=records,rejected_candidates=REJECTIONS,new_soma_voxels=int(added.sum()),components_before=n0,components_after=n1,previous_mask_preserved=True,limits='3D compact-core detection, half-height bounded growth, multi-core separation and original-component guards. Broad cores above 10 voxels are withheld, including potentially real large somas. uint8 and voxel-space thresholds; no biological ground truth.')
    (O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
