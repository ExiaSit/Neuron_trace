"""Soma references without adding or dilating any reconstruction foreground."""
import json
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi


def existing_bodies(reference, mask):
    labels, _ = ndi.label(np.asarray(reference, bool)&np.asarray(mask, bool),np.ones((3,3,3)))
    bodies=[]
    for i,box in enumerate(ndi.find_objects(labels),1):
        origin=np.array([s.start for s in box]);pts=np.argwhere(labels[box]==i)+origin
        bodies.append(dict(center_xyz=pts.mean(0)[::-1].tolist(),points_zyx=pts,
                           top_z=int(pts[:,0].min()),source='existing_foreground',anchor_index=i-1))
    return bodies


def segmented_target_reference(mask, xyz, json_path, soma_path):
    """Exact seed-containing segmented soma, intersected with existing mask.

    No nearest-soma fallback, hole filling, dilation or intensity-based growth.
    Keep a small surrounding ROI for the existing directional neurite tracing.
    """
    p=np.rint(np.asarray(xyz)[::-1]).astype(int)
    if np.any(p<0) or np.any(p>=mask.shape):raise ValueError('Target outside image')
    lo=np.maximum(0,p-24);hi=np.minimum(mask.shape,p+25)
    sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
    body=np.zeros(tuple(hi-lo),bool);other=np.zeros_like(body)
    if json_path and soma_path and Path(json_path).is_file() and Path(soma_path).is_file():
        bounds=json.loads(Path(json_path).read_text())['bounds_1um_zxy']
        origin=np.array([bounds[0],bounds[4],bounds[2]],int)
        labels,_=ndi.label(tifffile.imread(soma_path)>0,np.ones((3,3,3)))
        q=p-origin;own=int(labels[tuple(q)]) if np.all(q>=0) and np.all(q<labels.shape) else 0
        a=np.maximum(lo,origin);b=np.minimum(hi,origin+labels.shape)
        if np.all(b>a):
            dst=tuple(slice(int(x),int(y)) for x,y in zip(a-lo,b-lo))
            src=tuple(slice(int(x),int(y)) for x,y in zip(a-origin,b-origin))
            lab=labels[src];body[dst]=(lab==own)&(lab>0);other[dst]=(lab>0)&(lab!=own)
    body &= mask[sl]
    report=dict(status='existing_segmented_soma' if body.any() else 'no_existing_segmented_soma',
                soma_completion_enabled=False,added_voxels=0,voxels=int(body.sum()))
    return sl,body,other,report
