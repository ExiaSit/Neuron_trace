"""Supplement endpoint connections with cropped XY/XZ/YZ MIP agreement.

Only nearby geodesic endpoint segments define the crop, never full component
bounding boxes. Projected paths nominate a corridor; raw 3-D path evidence and
topology guards must still accept it before a voxel is added to the mask.
"""
import heapq
import json
import os
import numpy as np
from utils.gpu_ndimage import ndi
from scipy.spatial import cKDTree
from skimage.graph import route_through_array

DEFAULTS = dict(enabled=True, max_gap=70, segment_length=12, margin=6,
                corridor_radius=3, min_signal_fraction=.85, max_dark_run=2)


def configuration(value=None):
    if value is None:
        value=json.loads(os.environ.get('RECONNECT_LOCAL_PAIR_MIP_CONFIG','{}'))
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):
        raise ValueError('Invalid local_pair_mip configuration')
    cfg=dict(DEFAULTS,**value)
    if type(cfg['enabled']) is not bool:raise ValueError('enabled must be boolean')
    for key in ('max_gap','segment_length','margin','corridor_radius'):
        limit=200 if key=='max_gap' else 100
        if type(cfg[key]) is not int or not 1<=cfg[key]<=limit:raise ValueError(f'{key} must be an integer in 1..{limit}')
    if not 0<cfg['min_signal_fraction']<=1:raise ValueError('Invalid min_signal_fraction')
    if type(cfg['max_dark_run']) is not int or cfg['max_dark_run']<0:raise ValueError('Invalid max_dark_run')
    return cfg


def segment_directions(segment):
    """Use reliable inward prefixes; a bend farther in must not erase a valid tip."""
    segment=np.asarray(segment,float)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(segment,axis=0),axis=1))]
    directions=[]
    for scale in [arc[-1],8.,6.]:
        indices=np.flatnonzero(arc<=scale+1e-9)
        j=int(indices[-1]);vector=segment[0]-segment[j];length=np.linalg.norm(vector)
        if arc[j]<4 or length<3 or length/arc[j]<.65:continue
        direction=vector/length
        if not any(np.allclose(direction,d) for d in directions):directions.append(direction)
    return directions


def local_segment(points, endpoint, length=12):
    """Follow the component skeleton inward for at most length, including bends."""
    points=np.asarray(points,dtype=float);endpoint=np.asarray(endpoint,float)
    if len(points)<2:return None
    local=points[np.linalg.norm(points-endpoint,axis=1)<=length+1]
    if len(local)<2:return None
    tree=cKDTree(local);distance,start=tree.query(endpoint)
    if distance>2:return None
    start=int(start);dist={start:0.};parent={};queue=[(0.,start)]
    while queue:
        d,i=heapq.heappop(queue)
        if d!=dist[i]:continue
        for j in tree.query_ball_point(local[i],1.75):
            if j==i:continue
            value=d+float(np.linalg.norm(local[j]-local[i]))
            if value<=length and value<dist.get(j,np.inf):
                dist[j]=value;parent[j]=i;heapq.heappush(queue,(value,j))
    end=max(dist,key=dist.get)
    if dist[end]<4:return None
    path=[end]
    while path[-1]!=start:path.append(parent[path[-1]])
    segment=local[path[::-1]]
    directions=segment_directions(segment)
    if not directions:return None
    return segment,directions[0]


def endpoint_pairs(records,skeletons,cfg):
    """Use each physical endpoint once, with direction from its cropped segment."""
    endpoints=[];seen=set()
    for record in records:
        cid=record['component']
        for tip in record['endpoint_directions']:
            p=np.rint(tip['point_zyx']).astype(int);key=(cid,*p.tolist())
            if key in seen:continue
            seen.add(key);local=local_segment(skeletons.get(cid,[]),p,cfg['segment_length'])
            if local is not None:endpoints.append((cid,p,local[0],segment_directions(local[0])))
    if len(endpoints)<2:return
    pairs=cKDTree([e[1] for e in endpoints]).query_pairs(cfg['max_gap'])
    for i,j in sorted(pairs):
        a,p,segment_a,u=endpoints[i];b,q,segment_b,v=endpoints[j]
        gap=float(np.linalg.norm(q-p))
        if a==b or gap<2:continue
        d=(q-p)/gap
        if max(t@d for t in u)<.5 or max(t@(-d) for t in v)<.5:continue
        yield a,b,p,q,segment_a,segment_b


def pair_evidence(raw,p,q,segment_a,segment_b,cfg=None):
    cfg=configuration(cfg)
    p=np.asarray(p,int);q=np.asarray(q,int)
    points=np.vstack([p,q,segment_a,segment_b])
    lo=np.maximum(0,np.floor(points.min(0)).astype(int)-cfg['margin'])
    hi=np.minimum(raw.shape,np.ceil(points.max(0)).astype(int)+cfg['margin']+1)
    patch=raw[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))].astype(np.float32)
    start=p-lo;end=q-lo;corridor=np.ones(patch.shape,bool);views=[]
    gap_vector=(q-p)/max(float(np.linalg.norm(q-p)),1e-9)
    legacy_directions=[]
    for segment,toward in [(segment_a,gap_vector),(segment_b,-gap_vector)]:
        segment=np.asarray(segment,float);vector=segment[0]-segment[-1]
        arc=float(np.linalg.norm(np.diff(segment,axis=0),axis=1).sum());length=float(np.linalg.norm(vector))
        legacy_directions.append(arc>=4 and length>=3 and length/max(arc,1e-9)>=.65
                                 and float(vector@toward)/max(length,1e-9)>=.5)

    report=dict(bbox_zyx=[[int(a),int(b)] for a,b in zip(lo,hi)],
                endpoints_zyx=[p.tolist(),q.tolist()],
                local_segments_zyx=[np.asarray(segment_a).tolist(),np.asarray(segment_b).tolist()],views=views)
    for axis,plane in enumerate(('XY','XZ','YZ')):
        mip=patch.max(axis=axis);smooth=ndi.gaussian_filter(mip,.6)
        contrast=smooth-ndi.gaussian_filter(mip,4)
        residual=mip-smooth
        noise=max(1.,float(1.4826*np.median(np.abs(residual-np.median(residual)))))
        axes=[i for i in range(3) if i!=axis];s=start[axes];t=end[axes];gap=float(np.linalg.norm(t-s))
        line=np.ones(mip.shape,bool)
        samples=np.rint(np.linspace(s,t,max(2,int(gap*3)+2))).astype(int);line[tuple(samples.T)]=False
        allowed=ndi.distance_transform_edt(line)<=5
        cost=1+12/(1+np.maximum(contrast,0)/noise);cost[~allowed]=np.inf
        try:path,total=route_through_array(cost,tuple(s),tuple(t),fully_connected=True)
        except ValueError:path=[s.tolist()];total=np.inf
        path=np.asarray(path,int)
        lengths=np.linalg.norm(np.diff(path,axis=0),axis=1)
        # Endpoints and projected existing fragments must not dominate gap support.
        away=(np.linalg.norm(path-s,axis=1)>1)&(np.linalg.norm(path-t,axis=1)>1)
        tested=path[away] if away.any() else path
        values=contrast[tuple(tested.T)];good=(values>2)&(values/noise>2)
        run=dark=0
        for valid in good:
            run=0 if valid else run+1;dark=max(run,dark)
        fraction=float(good.mean())
        geometry_ok=bool(np.isfinite(total) and lengths.sum()<=1.6*max(gap,1)
                         and dark<=cfg['max_dark_run'])
        required=int(np.ceil(cfg['min_signal_fraction']*len(good)))
        missing=max(0,required-int(good.sum()))
        ok=bool(geometry_ok and missing==0)
        borderline=bool(geometry_ok and missing==1 and
                        fraction>=max(.8,cfg['min_signal_fraction']-.05))
        views.append(dict(plane=plane,projection_axis=axis,projected_gap=gap,
                          degenerate=gap<2,signal_fraction=fraction,max_dark_run=dark,
                          noise_sigma=noise,length=float(lengths.sum()),passed=ok,
                          tested_pixels=len(good),supported_pixels=int(good.sum()),
                          borderline=borderline,missing_support_pixels=missing,
                          path_rc=path.tolist()))
        route=np.ones(mip.shape,bool);route[tuple(path.T)]=False
        near=ndi.distance_transform_edt(route)<=cfg['corridor_radius']
        corridor &= np.expand_dims(near,axis)
    strict=all(v['passed'] for v in views)
    tolerance=(sum(v['passed'] for v in views)==2 and
               all(v['passed'] or v['borderline'] for v in views))
    report['requires_multiscale_direction']=not all(legacy_directions)
    report['requires_strong_3d']=bool(tolerance and not strict)
    report['passed']=bool(strict or tolerance)
    report['reason']='three_view_agreement' if strict else 'one_pixel_projection_tolerance' if tolerance else 'local_mip_view_failed'
    report['corridor_voxels']=int(corridor.sum())
    return (lo,corridor),report
