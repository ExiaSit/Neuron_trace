"""Suppress duplicate soma seeds only with a measured, thick 3-D signal bridge.

MIP overlap and binary-mask connectivity alone are not evidence of ownership.
The thresholds below are conservative heuristics, not biological identity proof.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import route_through_array
from algorithms.tracing.signal_support import SignalEvidence


def bridge(evidence, a, b, same_target_region=False):
    a=np.rint(a).astype(int);b=np.rint(b).astype(int)
    gap=float(np.linalg.norm(a-b))
    if gap<1:return None, 'duplicate_position'
    if gap>110:return None, 'distance_limit'
    lo=np.maximum(0,np.minimum(a,b)-16);hi=np.minimum(evidence.raw.shape,np.maximum(a,b)+17)
    sl=tuple(slice(int(x),int(y)) for x,y in zip(lo,hi));v=evidence.smooth[sl]
    peak=max(float(v[tuple(a-lo)]),float(v[tuple(b-lo)]))
    threshold=max(35.,.18*peak)
    cost=1+(100/(1+v))**3
    cost[v<threshold]=np.inf
    try:path,total=route_through_array(cost,tuple(a-lo),tuple(b-lo),fully_connected=True)
    except ValueError:return None,'no_continuous_strong_signal'
    if not np.isfinite(total):return None,'no_continuous_strong_signal'
    p=np.asarray(path);length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    if length>1.8*gap:return None,'excessive_detour'
    values=v[tuple(p.T)];q10=float(np.percentile(values,10));minimum=float(values.min())
    if minimum<max(45.,.18*peak) or q10<max(65.,.25*peak):return None,'intensity_valley'
    width=ndi.distance_transform_edt(v>max(18.,.18*min(float(v[tuple(a-lo)]),float(v[tuple(b-lo)]))))
    inside=(np.linalg.norm(p-(a-lo),axis=1)>8)&(np.linalg.norm(p-(b-lo),axis=1)>8)
    samples=width[tuple(p[inside].T)] if inside.any() else width[tuple(p.T)]
    w10=float(np.percentile(samples,10))
    if w10<1.4:return None,'thin_contact_between_bodies'
    # A bright neurite joining two bulbous maxima is not one soma. Measure
    # the saddle thickness relative to both endpoint bodies at a higher raw
    # intensity level. Only a broad continuous body can suppress a soma seed.
    core_distance=ndi.distance_transform_edt(v>max(35.,.30*peak))
    endpoint_radii=[];endpoint_peaks=[]
    for endpoint in [a-lo,b-lo]:
        lower=np.maximum(0,endpoint-6);upper=np.minimum(v.shape,endpoint+7)
        box=tuple(slice(int(x),int(y)) for x,y in zip(lower,upper));local=core_distance[box]
        endpoint_radii.append(float(local.max()))
        endpoint_peaks.append(np.array(np.unravel_index(local.argmax(),local.shape))+lower)
    inner=(np.linalg.norm(p-(a-lo),axis=1)>endpoint_radii[0])&(np.linalg.norm(p-(b-lo),axis=1)>endpoint_radii[1])
    widths=core_distance[tuple(p.T)]
    neck=float(np.percentile(widths[inner],10)) if inner.any() else float(widths.min())
    neck_ratio=neck/max(min(endpoint_radii),1e-6)
    # Off-center metadata and duplicate maxima can lie on the edge of one
    # body. Test the interior peaks before interpreting an endpoint taper as
    # a saddle between two independent bulbs. A narrow inter-body neck still
    # fails this interior chord test.
    pa,pb=endpoint_peaks;peak_gap=float(np.linalg.norm(pa-pb))
    chord=np.rint(np.linspace(pa,pb,max(2,int(np.ceil(peak_gap*2))+1))).astype(int)
    chord_floor=float(core_distance[tuple(chord.T)].min())
    shared_core=bool(peak_gap<=min(endpoint_radii) and chord_floor>=.85*min(endpoint_radii) and min(endpoint_radii)>=2.)
    if neck_ratio<.70 and not shared_core and not same_target_region:return None,'independent_bulbs_with_narrow_neck'
    # Wide soma plateaus use a larger transverse ring, retaining a dark rim.
    good=np.logical_or.reduce([evidence.profile(p+lo,radius=r)['good'] for r in [4.,8.,12.,18.]])
    fraction=float(good.mean())
    if fraction<.8:return None,'diffuse_halo_or_plateau'
    return dict(a_zyx=a.tolist(),b_zyx=b.tolist(),length=length,gap=gap,
                minimum=minimum,q10=q10,width_q10=w10,ridge_fraction=fraction,
                endpoint_radii=endpoint_radii,neck_radius=neck,neck_ratio=neck_ratio,
                shared_core=shared_core,interior_peak_gap=peak_gap,interior_chord_floor=chord_floor,
                same_target_region=bool(same_target_region),
                merge_rule=('same_target_soma_region' if neck_ratio<.70 and not shared_core and same_target_region
                            else 'strong_3d_bridge'),
                path_zyx=(p+lo).tolist()),None


def filter_seeds(raw, coords, target, auxiliary_labels=None, auxiliary_origin=None,
                 ownership_mask_path=None, scores=None):
    """Keep target as representative; no long transitive soma chains."""
    if target is None or len(coords)<2:return coords,dict(merged=[],rejected=[])
    ev=SignalEvidence(raw);target=np.asarray(target)
    from algorithms.tracing.soma_ownership import TargetSomaOwnership
    ownership=TargetSomaOwnership(ev,target,auxiliary_labels,auxiliary_origin)
    if ownership_mask_path is not None:ownership.save(ownership_mask_path,raw.shape)
    if scores is not None and len(scores) != len(coords):
        raise ValueError("Candidate coordinates and scores differ in length")
    # The metadata target is protected; other representatives should be the
    # strongest interior core, not whichever edge happens to face the target.
    order=sorted(range(len(coords)),key=lambda i:(
        not np.allclose(coords[i],target),
        -float(scores[i]) if scores is not None else 0.,
        np.linalg.norm(np.asarray(coords[i])-target)))
    core_radii = {}
    if scores is not None:
        for coord, score in zip(coords, scores):
            if np.isfinite(score):
                key = tuple(coord)
                core_radii[key] = max(core_radii.get(key, 0.), float(score))
    kept=[];merged=[];rejected=[]
    for i in order:
        c=coords[i];representative=None
        for parent in kept:
            d=float(np.linalg.norm(np.asarray(c)-np.asarray(parent)))
            # Large somas can have peripheral maxima >20 um from their core.
            # Radius only expands the search; the 3D bridge/neck tests still
            # decide whether this is one body, without transitive merging.
            pair_limit=max(20.,2*(core_radii.get(tuple(parent),0.)
                                  +core_radii.get(tuple(c),0.)))
            if np.linalg.norm(np.asarray(parent)-target)>1 and d>pair_limit:continue
            if d>110:continue
            same_region=ownership.contains_pair(parent,c)
            record,reason=bridge(ev,parent,c,same_target_region=same_region)
            if record is not None:
                representative=parent;merged.append(dict(candidate_zyx=list(c),representative_zyx=list(parent),evidence=record));break
            rejected.append(dict(a_zyx=list(parent),b_zyx=list(c),reason=reason,
                                 same_target_region=same_region))
        if representative is None:kept.append(c)
    return kept,dict(merged=merged,rejected=rejected,
                     target_soma_ownership=ownership.audit,
                     policy='target protected; core-score representatives; radius-scaled search + strong 3D bridge; no transitive distant merges')


def auxiliary_groups(raw, target, json_path, mask_path):
    """Classify auxiliary components using exactly the same signal bridge rule."""
    import json
    from pathlib import Path
    import tifffile
    bounds=json.loads(Path(json_path).read_text())['bounds_1um_zxy'];origin=np.array([bounds[0],bounds[4],bounds[2]],int)
    aux=tifffile.imread(mask_path)>0;labs,_=ndi.label(aux,np.ones((3,3,3)))
    sizes=np.bincount(labs.ravel());target=np.asarray(target);q=np.rint(target-origin).astype(int)
    own=int(labs[tuple(q)]) if np.all(q>=0) and np.all(q<aux.shape) else 0
    ev=SignalEvidence(raw);same=[];records=[]
    for cid,box in enumerate(ndi.find_objects(labs),1):
        if box is None or cid==own or sizes[cid]<40:continue
        d=ndi.distance_transform_edt(labs[box]==cid);p=np.array(np.unravel_index(d.argmax(),d.shape))+[s.start for s in box]+origin
        if np.any(p<0) or np.any(p>=raw.shape) or np.linalg.norm(p-target)>110:continue
        rec,reason=bridge(ev,target,p)
        if rec is not None:same.append(cid);records.append(dict(component=cid,center_zyx=p.tolist(),evidence=rec))
    return labs,origin,same,dict(same_auxiliary_components=records)


def exclude_confirmed_neighbors(context, centers):
    """Remove only verified auxiliary components from the neighbor blocker.

    Do not enlarge the soma cleanup region: the connecting neurite stays a branch.
    """
    if not context or not centers:return context
    mask=context['mask'];labels,_=ndi.label(mask,np.ones((3,3,3)));remove=set();origin=np.asarray(context['origin_xyz'])[::-1]
    for point in centers:
        p=np.rint(np.asarray(point)-origin).astype(int)
        if np.any(p<0) or np.any(p>=mask.shape):continue
        label=int(labels[tuple(p)])
        if label:remove.add(label)
    result=dict(context);result['mask']=mask&~np.isin(labels,list(remove));result['component_count']=max(0,context['component_count']-len(remove))
    return result


def confirmed_mask(shape,labels,origin,same,audit):
    mask=np.zeros(shape,bool);lower=np.maximum(0,origin);upper=np.minimum(shape,origin+np.array(labels.shape))
    if np.all(upper>lower):
        dst=tuple(slice(int(a),int(b)) for a,b in zip(lower,upper));src=tuple(slice(int(a),int(b)) for a,b in zip(lower-origin,upper-origin))
        mask[dst]=np.isin(labels[src],same)
    for record in audit['same_auxiliary_components']:
        path=np.asarray(record['evidence']['path_zyx'],int)
        lo=np.maximum(0,path.min(0)-4);hi=np.minimum(shape,path.max(0)+5);sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));tube=np.zeros(hi-lo,bool);tube[tuple((path-lo).T)]=True
        mask[sl]|=ndi.distance_transform_edt(~tube)<=4
    return mask


def inside_confirmed(points_xyz,mask,fraction=.8):
    if mask is None:return False
    p=np.rint(np.asarray(points_xyz)[:,::-1]).astype(int);valid=np.all(p>=0,axis=1)&np.all(p<mask.shape,axis=1)
    return bool(valid.all() and mask[tuple(p.T)].mean()>=fraction)


def target_core_region(raw, target, forbidden=None):
    """Measure a local thick raw-signal body, including off-center metadata.

    Protect this measured volume from thin-neurite contrast tests, without
    growing along distant signal paths or changing the soma tracing origin.
    """
    p=np.rint(target).astype(int);lo=np.maximum(0,p-40);hi=np.minimum(raw.shape,p+41)
    sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));v=ndi.gaussian_filter(raw[sl].astype(np.float32),.65);q=p-lo
    if np.any(q<0) or np.any(q>=v.shape):return sl,np.zeros(v.shape,bool),dict(reason='outside_image')
    seed_value=float(v[tuple(q)]);threshold=max(35.,.30*seed_value)
    d=ndi.distance_transform_edt(v>threshold)
    a=np.maximum(0,q-12);b=np.minimum(v.shape,q+13);box=tuple(slice(int(x),int(y)) for x,y in zip(a,b));peak=np.array(np.unravel_index(d[box].argmax(),d[box].shape))+a;radius=float(d[tuple(peak)])
    empty=np.zeros(v.shape,bool)
    if seed_value<45 or radius<3:return sl,empty,dict(reason='weak_or_thin_target')
    # The supplied target must reach this core through uninterrupted bright
    # signal locally; proximity alone is not permission to snap to a neighbor.
    connected,_=ndi.label(v>threshold,np.ones((3,3,3)));cid=int(connected[tuple(q)])
    if not cid or connected[tuple(peak)]!=cid:return sl,empty,dict(reason='disconnected_core')
    axes=np.ogrid[tuple(slice(0,n) for n in v.shape)];distance=sum((axis-center)**2 for axis,center in zip(axes,peak))
    allowed=(connected==cid)&(d>=max(2.,.25*radius))&(distance<=(2*radius)**2)
    if forbidden is not None:allowed &= ~forbidden[sl]
    seed=empty.copy();seed[tuple(peak)]=allowed[tuple(peak)]
    region=ndi.binary_propagation(seed,structure=np.ones((3,3,3)),mask=allowed)
    return sl,region,dict(reason='measured_target_core',peak_zyx=(peak+lo).tolist(),radius=radius,threshold=threshold,voxels=int(region.sum()))
