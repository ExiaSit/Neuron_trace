"""Soma-top limits with conservative mask-geodesic ownership.

Original foreground is immutable. Ownership is a connectivity heuristic,
not biological ground truth. The single-target ceiling remains for comparison.
"""
import numpy as np
from scipy import ndimage

RULE = 'completed-target-soma-top-v1'


def completed_bodies(shape_report, archive, target_bodies, shape):
    """Load exact retained f regions and consolidate the same target seen by g.

    Distinct f cores are never merged here. A g target may join one f body
    only with both center proximity and >=50% overlap of the smaller region.
    """
    bodies=[]
    for i,record in enumerate(shape_report.get('completed_somas', [])):
        points=np.asarray(archive[f'body_{i}'],dtype=int)
        bodies.append(dict(record,points_zyx=points,sources=['f']))
    for target in target_bodies:
        points=np.asarray(target['points_zyx'],int)
        flat=np.ravel_multi_index(points.T,shape)
        matches=[]
        for i,body in enumerate(bodies):
            if np.linalg.norm(np.asarray(body['center_xyz'])-target['center_xyz'])>8:continue
            other=np.ravel_multi_index(body['points_zyx'].T,shape)
            overlap=len(np.intersect1d(flat,other))/max(1,min(len(flat),len(other)))
            if overlap>=.5:matches.append(i)
        if len(matches)==1:
            body=bodies[matches[0]]
            body['points_zyx']=np.unique(np.vstack([body['points_zyx'],points]),axis=0)
            body['top_z']=int(body['points_zyx'][:,0].min())
            body['sources'].append('g_target')
        else:
            bodies.append(dict(target,points_zyx=points,sources=['g_target']))
    return bodies


def foreground_graph(mask):
    """26-neighbor graph restricted to actual foreground, with voxel distances."""
    from itertools import product
    from scipy.sparse import coo_matrix
    points=np.argwhere(mask)
    lookup=np.full(mask.shape,-1,dtype=np.int32)
    lookup[tuple(points.T)]=np.arange(len(points),dtype=np.int32)
    rows=[];cols=[];weights=[]
    for delta in product((-1,0,1),repeat=3):
        if delta==(0,0,0) or next(x for x in delta if x)!=1:continue
        q=points+delta
        valid=np.all((q>=0)&(q<np.asarray(mask.shape)),axis=1)
        a=np.flatnonzero(valid);b=lookup[tuple(q[valid].T)]
        good=b>=0;a=a[good];b=b[good]
        rows.extend([a,b]);cols.extend([b,a])
        weights.extend([np.full(len(a),np.linalg.norm(delta))]*2)
    graph=coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),
                     shape=(len(points),len(points))).tocsr()
    return points,lookup,graph


def restrict_per_soma(original,proposed,bodies,ambiguity_margin=.15,min_margin=2.):
    """Use each completed body's top after competing along mask paths.

    Original voxels survive ambiguous/no ownership. New voxels do not. After
    clipping, a retained addition must still reach its assigned body; iterate
    because discarding another body's additions can remove a shared bridge.
    """
    from scipy.sparse.csgraph import dijkstra
    original=np.asarray(original,bool);proposed=np.asarray(proposed,bool)
    if original.ndim!=3 or original.shape!=proposed.shape or not proposed[original].all():
        raise ValueError('Matching 3-D masks preserving original foreground are required')
    if ambiguity_margin<0 or min_margin<0:raise ValueError('Ownership margins must be nonnegative')
    points,lookup,graph=foreground_graph(proposed)
    best=np.full(len(points),np.inf);second=best.copy();winner=np.zeros(len(points),np.uint16)
    records=[];seeds_by_body=[]
    for i,body in enumerate(bodies,1):
        pts=np.asarray(body['points_zyx'],int)
        if pts.ndim!=2 or pts.shape[1]!=3 or not len(pts):raise ValueError('Empty/invalid completed soma')
        if np.any(pts<0) or np.any(pts>=np.asarray(proposed.shape)):raise ValueError('Soma outside image')
        seeds=np.unique(lookup[tuple(pts.T)]);seeds=seeds[seeds>=0]
        if len(seeds)!=len(np.unique(pts,axis=0)):raise ValueError('Completed soma not contained in proposal')
        top=int(body['top_z'])
        if top!=int(pts[:,0].min()):raise ValueError('Soma top does not match its completed region')
        distance=dijkstra(graph,directed=True,indices=seeds,min_only=True)
        better=distance<best
        second=np.where(better,best,np.minimum(second,distance))
        winner[better]=i;best=np.minimum(best,distance)
        seeds_by_body.append(pts)
        records.append(dict(id=i,center_xyz=body['center_xyz'],top_z=top,
                            sources=body.get('sources',[body.get('source','unknown')]),seed_voxels=len(seeds)))
    reachable=np.isfinite(best)
    competing=reachable&np.isfinite(second)
    confident=reachable.copy()
    confident[competing]=(second[competing]-best[competing])>=np.maximum(min_margin,ambiguity_margin*best[competing])
    owner=np.where(confident,winner,0).astype(np.uint16)
    fresh=~original[tuple(points.T)]
    top_lookup=np.array([proposed.shape[0]]+[r['top_z'] for r in records])
    above=confident&(points[:,0]<top_lookup[owner])
    keep=fresh&confident&~above
    result=original.copy();result[tuple(points[keep].T)]=True
    detached=0;iterations=0
    while True:
        labels,_=ndimage.label(result,np.ones((3,3,3)))
        remove=np.zeros(len(points),bool)
        for i,seeds in enumerate(seeds_by_body,1):
            roots=np.unique(labels[tuple(seeds.T)]);roots=roots[roots>0]
            chosen=np.flatnonzero(keep&(owner==i))
            if len(chosen):
                remove[chosen]=~np.isin(labels[tuple(points[chosen].T)],roots)
        count=int(remove.sum())
        if not count:break
        result[tuple(points[remove].T)]=False;keep[remove]=False
        detached+=count;iterations+=1
    for record in records:
        own=owner==record['id']
        record.update(assigned_additions=int(np.count_nonzero(fresh&own)),
                      blocked_above_soma=int(np.count_nonzero(fresh&own&above)),
                      final_added_voxels=int(np.count_nonzero(keep&own)))
    owners=np.zeros(proposed.shape,np.uint16);owners[tuple(points.T)]=owner
    assert result[original].all()
    assert not np.any(keep&((owner==0)|(points[:,0]<top_lookup[owner])))
    report=dict(rule='completed-per-soma-top-geodesic-v1',mode='per_soma_top',status='applied',
                somas=records,soma_count=len(records),original_preserved=True,same_z_growth=True,
                proposed_added_voxels=int(fresh.sum()),final_added_voxels=int(keep.sum()),
                blocked_above_soma=int(np.count_nonzero(fresh&above)),
                ambiguous_additions_withheld=int(np.count_nonzero(fresh&reachable&~confident)),
                unassigned_additions_withheld=int(np.count_nonzero(fresh&~reachable)),
                detached_additions_removed=detached,connectivity_cleanup_iterations=iterations,
                ownership=dict(method='26-neighbor shortest paths within proposed foreground',
                               relative_distance_margin=ambiguity_margin,absolute_distance_margin_voxels=min_margin,
                               graph_nodes=len(points),graph_directed_edges=graph.nnz,
                               limitation='Mask geodesic competition is not validated biological ownership; unknown/ambiguous additions are withheld.'))
    return result,report,owners


def configuration(value='per_soma_top'):
    if value not in ('per_soma_top', 'below_soma_top', 'unrestricted'):
        raise ValueError('mask_growth_direction must be per_soma_top, below_soma_top or unrestricted')
    return value


def select_top(xyz, shape_report, continuation_report, max_center_distance=8.):
    """Match a completed soma region, never the whole neurite component.

    f supplies validated 3-D soma-shape regions; g supplies the target body
    anchored by metadata/auxiliary soma segmentation. Their union is part of
    the completed mask, so use its smallest measured Z.
    """
    xyz = np.asarray(xyz, float)
    sources = []
    candidates = []
    for anchor in shape_report.get('anchors', []):
        top = anchor.get('completed_soma_top_z')
        bounds = anchor.get('completed_soma_bbox_zyx')
        if top is None or bounds is None:
            continue
        center = np.asarray(anchor['center_xyz'], float)
        distance = float(np.linalg.norm(center-xyz))
        inside = all(lo <= value < hi for value, (lo,hi) in zip(xyz[::-1],bounds))
        if inside and distance <= max_center_distance:
            candidates.append((distance, anchor))
    if candidates:
        distance, anchor = min(candidates, key=lambda row: row[0])
        sources.append(dict(stage='f', top_z=int(anchor['completed_soma_top_z']),
                            center_xyz=anchor['center_xyz'], center_distance=distance))
    body = continuation_report.get('soma', {})
    if body.get('status') == 'supported_soma' and body.get('completed_soma_top_z') is not None:
        sources.append(dict(stage='g', top_z=int(body['completed_soma_top_z']),
                            center_xyz=xyz.tolist()))
    return (min(item['top_z'] for item in sources) if sources else None), sources


def restrict(original, proposed, top_z, mode='below_soma_top'):
    """Clip only additions above the soma top and discard newly detached pieces."""
    configuration(mode)
    original = np.asarray(original, bool)
    proposed = np.asarray(proposed, bool)
    if original.ndim != 3 or proposed.shape != original.shape:
        raise ValueError('Expected matching ZYX masks')
    if not proposed[original].all():
        raise ValueError('Repair proposal already lost original foreground')
    additions = proposed & ~original
    report = dict(rule=RULE, mode=mode, proposed_added_voxels=int(additions.sum()),
                  soma_top_z=top_z, same_z_growth=True, original_preserved=True)
    if mode == 'unrestricted':
        return proposed.copy(), dict(report, blocked_above_soma=0, detached_additions_removed=0,
                                     final_added_voxels=int(additions.sum()), status='disabled')
    if top_z is None:
        return original.copy(), dict(report, blocked_above_soma=0, detached_additions_removed=0,
                                     unlocalized_soma_additions_withheld=int(additions.sum()),
                                     final_added_voxels=0, status='no_supported_target_soma')
    if isinstance(top_z, (bool, np.bool_)) or int(top_z) != top_z or not 0 <= top_z < original.shape[0]:
        raise ValueError('Soma top Z must be a valid integer slice')
    top_z = int(top_z)
    blocked = int(additions[:top_z].sum())
    additions[:top_z] = False
    # Remove only additions that lose their original-mask attachment because
    # of this clipping. A preexisting independent soma proposal below the
    # ceiling must not be deleted as an unrelated side effect of this rule.
    initially_reached = ndimage.binary_propagation(original, structure=np.ones((3,3,3)),
                                                   mask=proposed)
    reached = ndimage.binary_propagation(original, structure=np.ones((3,3,3)),
                                         mask=original | additions)
    detached_mask = additions & initially_reached & ~reached
    detached = int(detached_mask.sum())
    result = original | (additions & ~detached_mask)
    assert np.array_equal(result[:top_z], original[:top_z])
    assert result[original].all()
    report.update(soma_top_z=top_z, blocked_above_soma=blocked,
                  detached_additions_removed=detached,
                  final_added_voxels=int(np.count_nonzero(result & ~original)), status='applied')
    return result, report
