"""Recover loop-hidden neurite caps after established soma bridges.

Only a one-sided cap facing a compact segmented soma is nominated. Both the
whole route to that soma and its prefix to the existing soma component need
strong raw 3-D support. Established bridges are immutable; the prefix may join
one already assigned to this soma, but cannot touch a third component.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from algorithms.mask_repair.local_pair_mip import local_segment
from algorithms.mask_repair.soma_bridge import solid_bridge
from algorithms.mask_repair import signal_reconnect as evidence


def cap_geometry(points, surface_tree, surface, known_tips=(), max_gap=70):
    points=np.asarray(points)
    if len(points)<8:return None
    distances,indices=surface_tree.query(points)
    i=int(np.argmin(distances));gap=float(distances[i])
    if not 1<=gap<=max_gap:return None
    p=points[i];q=surface[int(indices[i])]
    if any(np.linalg.norm(p-t[0])<=4 for t in known_tips):return None
    segment=local_segment(points,p,12)
    if segment is None:return None
    segment,u=segment
    extent=float(np.linalg.norm(segment[-1]-p))
    if extent<8:return None
    direction=(q-p)/gap
    angle=float(np.degrees(np.arccos(np.clip(u@direction,-1,1))))
    if angle>30:return None
    cloud=points[np.linalg.norm(points-p,axis=1)<=12]
    projection=(cloud-p)@u
    # A passing fiber has skeleton support on both sides of an interior point.
    if len(cloud)<8 or np.mean(projection>2)>.05 or np.mean(projection< -4)<.35:return None
    return dict(p=p,q=q,gap=gap,angle=angle,segment=segment,
                forward_fraction=float(np.mean(projection>2)),extent=extent)


def strong_raw(result):
    return bool(result and result['raw_signal_fraction']>=.95
                and result['signal_fraction']>=.95
                and result['transverse_ridge_fraction']>=.8
                and result['reinforced_fraction']==0
                and result['max_dark_run']<=1)


def clipped_target(path, labels, target):
    """First 26-neighbor contact with this soma's already connected component."""
    path=np.asarray(path,int)
    lo=np.maximum(0,path.min(0)-2);hi=np.minimum(labels.shape,path.max(0)+3)
    sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
    points=np.argwhere(labels[sl]==target)+lo
    if not len(points):return None
    distances,indices=cKDTree(points).query(path)
    hit=np.flatnonzero(distances<=np.sqrt(3)+1e-8)
    return points[indices[hit[0]]] if len(hit) else None


def bridge_clear(coords, labels, source, target, added):
    lo=np.maximum(0,coords.min(0)-1);hi=np.minimum(labels.shape,coords.max(0)+2)
    sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
    bridge=np.zeros(hi-lo,bool);bridge[tuple((coords-lo).T)]=True
    halo=ndi.binary_dilation(bridge,np.ones((3,3,3)))
    return (set(np.unique(labels[sl][halo]))-{0}=={source,target}
            and not np.any(added[sl]&halo))


def recover_caps(raw, contrast, connected, original_labels, records, all_tips):
    labels,n=ndi.label(connected,np.ones((3,3,3)))
    soma_ids={int(labels[tuple(rec['surface'][0])]) for rec in records}
    boxes=ndi.find_objects(original_labels);rejected=[];best={};nominated=0
    for a,tips in all_tips.items():
        box=boxes[a-1];local=original_labels[box]==a
        if local.sum()<=32:continue
        origin=np.array([s.start for s in box]);anchor=np.argwhere(local)[0]+origin
        source=int(labels[tuple(anchor)])
        if source in soma_ids:continue
        # Cheap distance bound avoids skeletonization far from every soma.
        lo=origin;hi=np.array([s.stop-1 for s in box])
        if all(np.min(np.linalg.norm(rec['surface']-np.clip(rec['surface'],lo,hi),axis=1))>70 for rec in records):continue
        points=np.argwhere(skeletonize(local))+origin
        for rec in records:
            cap=cap_geometry(points,rec['tree'],rec['surface'],tips)
            if cap is None:continue
            nominated+=1;p,q=cap['p'],cap['q'];target=int(labels[tuple(q)])
            row=dict(component=a,soma_id=rec['soma_id'],point_zyx=p.tolist())
            full,why=evidence.path_evidence(raw,contrast,labels,p,q,source,target,'hidden_cap_full_route')
            if not strong_raw(full):
                rejected.append(dict(row,reason=why or 'hidden_cap_weak_full_raw'));continue
            end=clipped_target(full['path_zyx'],labels,target)
            if end is None:
                rejected.append(dict(row,reason='hidden_cap_no_target_contact'));continue
            short,why=evidence.path_evidence(raw,contrast,labels,p,end,source,target,'hidden_cap_to_segmented_soma')
            if not strong_raw(short):
                rejected.append(dict(row,reason=why or 'hidden_cap_weak_prefix_raw'));continue
            short.update(soma_id=rec['soma_id'],source_angle=cap['angle'],small_undirected=False,
                         compact_soma_gap=cap['gap'],compact_soma_endpoint_zyx=q.tolist(),
                         full_route_evidence=full,cap_extent=cap['extent'],cap_forward_fraction=cap['forward_fraction'],
                         original_source=a,original_target=rec['component'])
            best.setdefault(source,[]).append(short)
    proposals=[]
    for source,options in best.items():
        options.sort(key=lambda x:x['full_route_evidence']['cost'])
        # Compare complete soma routes, not shortened distances to their branches.
        if not evidence.unambiguous([x['full_route_evidence']['cost'] for x in options]):
            rejected.append(dict(component=options[0]['original_source'],reason='hidden_cap_ambiguous_somas',
                                 candidates=[dict(soma_id=x['soma_id'],cost=x['full_route_evidence']['cost']) for x in options]));continue
        proposals.append(options[0])
    added=np.zeros(connected.shape,bool);accepted=[]
    for result in sorted(proposals,key=lambda x:x['full_route_evidence']['cost']):
        source,target=result['source'],result['target'];coords=solid_bridge(result['path_zyx'],connected.shape)
        if not bridge_clear(coords,labels,source,target,added):
            rejected.append(dict(component=result['original_source'],soma_id=result['soma_id'],reason='hidden_cap_third_fragment_or_bridge_collision'));continue
        fresh=~connected[tuple(coords.T)];added[tuple(coords[fresh].T)]=True
        result.update(source=result.pop('original_source'),target=result.pop('original_target'),added_voxels=int(fresh.sum()))
        result['components']=sorted([result['source'],result['target']]);accepted.append(result)
    assert ndi.label(connected|added,np.ones((3,3,3)))[1]==n-len(accepted)
    return added,accepted,dict(nominated=nominated,accepted=len(accepted),rejected=rejected)
