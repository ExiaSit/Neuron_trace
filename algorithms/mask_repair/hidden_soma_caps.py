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


def multiscale_facing_cap(points, p, direction, gap):
    """A folded thick cap needs agreement beyond an arbitrary skeleton arm.

    Limit this fallback to a 32-voxel gap. At both 12 and 24 voxels, the
    skeleton must recede from the gap and its centroid must face the soma.
    A passing fiber, a lateral attachment or a small isolated loop fails.
    Raw 3-D evidence and component collision checks remain mandatory downstream.
    """
    if gap > 32:
        return None
    rows=[]
    for radius, minimum in ((12,12),(24,24)):
        cloud=points[np.linalg.norm(points-p,axis=1)<=radius]
        if len(cloud)<minimum:return None
        projection=(cloud-p)@direction
        inward=p-cloud.mean(0);length=float(np.linalg.norm(inward))
        angle=float(np.degrees(np.arccos(np.clip(inward@direction/max(length,1e-9),-1,1))))
        if (length<radius/3 or angle>30 or np.mean(projection>2)>.05
                or np.mean(projection< -radius/3)<.35
                or -float(projection.min())<radius*2/3):return None
        rows.append(dict(radius=radius,points=len(cloud),centroid_angle=angle,
                         centroid_distance=length,forward_fraction=float(np.mean(projection>2)),
                         backward_fraction=float(np.mean(projection< -radius/3))))
    return rows


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
    cloud=points[np.linalg.norm(points-p,axis=1)<=12]
    projection=(cloud-p)@u
    # A passing fiber has skeleton support on both sides of an interior point.
    strict=(angle<=30 and len(cloud)>=8 and np.mean(projection>2)<=.05
            and np.mean(projection< -4)>=.35)
    rule='inward_segment_cap';multiscale=None
    if not strict:
        # A folded/looped cap can send the longest inward skeleton walk along Z
        # instead of along the visible neurite. For a short gap only, also test
        # the hemisphere facing the measured soma surface. Both skeleton arms
        # must remain behind the cap; a passing fiber or sideways attachment
        # must not become a candidate merely because it is close to a soma.
        facing=(cloud-p)@direction
        if (gap<=18 and angle<=60 and len(cloud)>=12 and
                np.mean(facing>2)<=.05 and np.mean(facing< -4)>=.35):
            rule='short_gap_facing_cap'
        else:
            multiscale=multiscale_facing_cap(points,p,direction,gap)
            if multiscale is None:return None
            rule='multiscale_facing_cap'
    return dict(p=p,q=q,gap=gap,angle=angle,segment=segment,
                forward_fraction=float(np.mean(projection>2)),extent=extent,
                geometry_rule=rule,
                multiscale_cap_evidence=multiscale,
                gap_facing_forward_fraction=float(np.mean(((cloud-p)@direction)>2)),
                gap_facing_backward_fraction=float(np.mean(((cloud-p)@direction)< -4)))


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
    soma_roots={rec['soma_id']:int(labels[tuple(rec['surface'][0])]) for rec in records}
    soma_ids=set(soma_roots.values())
    boxes=ndi.find_objects(original_labels);rejected=[];best={};nominated=0
    for a,tips in all_tips.items():
        box=boxes[a-1];local=original_labels[box]==a
        if local.sum()<=32:continue
        origin=np.array([s.start for s in box]);anchor=np.argwhere(local)[0]+origin
        source=int(labels[tuple(anchor)])
        # Cheap distance bound avoids skeletonization far from every soma.
        lo=origin;hi=np.array([s.stop-1 for s in box])
        if all(np.min(np.linalg.norm(rec['surface']-np.clip(rec['surface'],lo,hi),axis=1))>70 for rec in records):continue
        points=np.argwhere(skeletonize(local))+origin
        for rec in records:
            if soma_roots[rec['soma_id']]==source:continue
            cap=cap_geometry(points,rec['tree'],rec['surface'],tips)
            if cap is None:continue
            p,q=cap['p'],cap['q'];target=int(labels[tuple(q)])
            remote_somas=[]
            if source in soma_ids:
                # A segmentation crossing can put a genuine broken cap in a
                # component containing distant somas. Never infer ownership
                # from that component alone. Admit only the short, two-scale
                # cap, far from every soma already attached to its component.
                if cap['geometry_rule']!='multiscale_facing_cap':continue
                remote_somas=[dict(soma_id=r['soma_id'],distance=float(r['tree'].query(p)[0]))
                              for r in records if soma_roots[r['soma_id']]==source]
                if any(r['distance']<=max(70.,3*cap['gap']) for r in remote_somas):continue
            nominated+=1
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
            if cap['geometry_rule'] in ('short_gap_facing_cap','multiscale_facing_cap') and any(
                    e['transverse_ridge_fraction']<.95 for e in (full,short)):
                rejected.append(dict(row,reason='short_facing_cap_weak_transverse_ridge'));continue
            short.update(soma_id=rec['soma_id'],source_angle=cap['angle'],small_undirected=False,
                         compact_soma_gap=cap['gap'],compact_soma_endpoint_zyx=q.tolist(),
                         full_route_evidence=full,cap_extent=cap['extent'],cap_forward_fraction=cap['forward_fraction'],
                         cap_geometry_rule=cap['geometry_rule'],
                         multiscale_cap_evidence=cap['multiscale_cap_evidence'],
                         remote_component_somas=remote_somas,
                         cap_gap_facing_forward_fraction=cap['gap_facing_forward_fraction'],
                         cap_gap_facing_backward_fraction=cap['gap_facing_backward_fraction'],
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
    added=np.zeros(connected.shape,bool);accepted=[];parent=list(range(n+1))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for result in sorted(proposals,key=lambda x:x['full_route_evidence']['cost']):
        source,target=result['source'],result['target'];coords=solid_bridge(result['path_zyx'],connected.shape)
        if root(source)==root(target):
            rejected.append(dict(component=result['original_source'],soma_id=result['soma_id'],reason='hidden_cap_cycle'));continue
        if not bridge_clear(coords,labels,source,target,added):
            rejected.append(dict(component=result['original_source'],soma_id=result['soma_id'],reason='hidden_cap_third_fragment_or_bridge_collision'));continue
        fresh=~connected[tuple(coords.T)];added[tuple(coords[fresh].T)]=True
        parent[root(source)]=root(target)
        result.update(source=result.pop('original_source'),target=result.pop('original_target'),added_voxels=int(fresh.sum()))
        result['components']=sorted([result['source'],result['target']]);accepted.append(result)
    assert ndi.label(connected|added,np.ones((3,3,3)))[1]==n-len(accepted)
    return added,accepted,dict(nominated=nominated,accepted=len(accepted),rejected=rejected)
