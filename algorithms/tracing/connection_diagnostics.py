"""Local XYZ boxes and three cropped MIPs, persisted before connection cuts.

Coordinates are 1-um image XYZ. Projection overlap alone never creates or
removes an edge. Gap cuts need local anchors, loss of transverse ridge, two
thin views and raw 3-D evidence. Crossing-switch cuts are separately labelled
ownership heuristics, supported by continuation probes in raw 3-D/thin views.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from algorithms.tracing.signal_support import resample_line, tree_paths, SignalEvidence

POLICY = 'initial-short-leaves-thin-local-connection-neighbor-soma-v2'
DEFAULTS = dict(arm_length_um=12., margin_um=6., sample_step_um=.5,
                crossing_distance_um=1.5, crossing_merge_um=6.,
                min_crossing_angle_degrees=30., min_gap_um=3.,
                min_resolving_views=2, diagnostic_slab_um=3.,
                local_anchor_window_um=5., max_gap_anchor_ratio=.6,
                branch_scan_window_um=24., branch_scan_step_um=12.,
                switch_min_angle_degrees=60., switch_probe_start_um=3.,
                switch_probe_end_um=11.)
PLANES = [('XY', 0, (0, 1)), ('XZ', 1, (0, 2)), ('YZ', 2, (1, 2))]


def tree_digest(tree):
    a = np.asarray(sorted(tree, key=lambda n: int(n[0])), dtype='<f8')
    return hashlib.sha256(a.tobytes()).hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def _prefix(points, length):
    points = np.asarray(points, float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    end = min(length, float(arc[-1]))
    keep = points[arc < end]
    last = np.array([np.interp(end, arc, points[:, j]) for j in range(3)])
    return np.vstack([keep, last]) if len(keep) else points[:1]


def _window(points, at, length):
    points = np.asarray(points, float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    lo, hi = max(0., at-length), min(float(arc[-1]), at+length)
    samples = np.r_[lo, arc[(arc > lo) & (arc < hi)], hi]
    return np.column_stack([np.interp(samples, arc, points[:, j]) for j in range(3)])


def _projected_intersection(samples, owner, a, b, axes):
    """Include actual shallow-angle intersections, not only angular near misses."""
    def edges(i):
        return [(samples[i, list(axes)], samples[j, list(axes)])
                for j in (i-1, i+1) if 0 <= j < len(samples) and owner[j] == owner[i]]
    def cross(u, v):
        return float(u[0]*v[1]-u[1]*v[0])
    for p, q in edges(a):
        for r, s in edges(b):
            u, v = q-p, s-r
            denominator = cross(u, v)
            if abs(denominator) < 1e-8:
                continue  # Collinear overlap is not an isolated crossing.
            t, w = cross(r-p, v)/denominator, cross(r-p, u)/denominator
            if 0 <= t <= 1 and 0 <= w <= 1:
                return True
    return False


def candidate_sites(tree, cfg=None):
    cfg = dict(DEFAULTS, **(cfg or {}))
    nodes, children, paths = tree_paths(tree)
    sites = []
    by_start = {}
    for path in paths:
        by_start.setdefault(path[0], []).append(path)
    # Every topological branchpoint, including the soma; protected later.
    for nid in sorted(nodes):
        if len(children[nid]) < 2:
            continue
        arms = []
        for path in by_start.get(nid, []):
            xyz = _prefix([nodes[i][2:5] for i in path], cfg['arm_length_um'])
            arms.append(dict(role='outgoing', node_ids=path, xyz=xyz.tolist()))
        incoming = [nodes[nid][2:5]]; cur = nid; length = 0.
        while int(nodes[cur][6]) in nodes and length < cfg['arm_length_um']:
            cur = int(nodes[cur][6]); q = nodes[cur][2:5]
            length += np.linalg.norm(np.asarray(q)-incoming[-1]); incoming.append(q)
        sites.append(dict(kind='branchpoint', center_xyz=list(nodes[nid][2:5]),
                          node_id=nid, arms=arms,
                          context_xyz=_prefix(incoming, cfg['arm_length_um']).tolist()))
    # Sample all branches, including long edges and self-crossings.
    samples=[]; owner=[]; arcs=[]; tangents=[]; path_xyz=[]
    for i,path in enumerate(paths):
        xyz=np.asarray([nodes[n][2:5] for n in path],float);path_xyz.append(xyz)
        p=resample_line(xyz,.75)
        if len(p)<2:continue
        samples.extend(p);owner.extend([i]*len(p))
        arcs.extend(np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))])
        t=np.gradient(p,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-8);tangents.extend(t)
    if not samples:return sites
    samples=np.asarray(samples);owner=np.asarray(owner);arcs=np.asarray(arcs);tangents=np.asarray(tangents)
    proposals=[]
    for plane,_,axes in PLANES:
        pairs=cKDTree(samples[:,axes]).query_pairs(cfg['crossing_distance_um'])
        for a,b in sorted(pairs):
            i,j=int(owner[a]),int(owner[b])
            if i==j and abs(arcs[a]-arcs[b])<4:continue
            # Do not report ordinary adjacent points around a shared branchpoint as crossings.
            shared=set(paths[i]) & set(paths[j]) if i!=j else set()
            if any(np.linalg.norm(samples[a]-nodes[n][2:5])<cfg['arm_length_um'] and
                   np.linalg.norm(samples[b]-nodes[n][2:5])<cfg['arm_length_um'] for n in shared):continue
            u=tangents[a,list(axes)];v=tangents[b,list(axes)]
            norm=float(np.linalg.norm(u)*np.linalg.norm(v))
            if norm<.05:continue
            shallow = abs(float(u@v)/norm)>np.cos(np.radians(cfg['min_crossing_angle_degrees']))
            local_self = i==j and abs(arcs[a]-arcs[b])<2*cfg['arm_length_um']
            if (shallow or local_self) and not _projected_intersection(samples,owner,a,b,axes):continue
            if i>j:i,j,a,b=j,i,b,a
            distance=float(np.linalg.norm(samples[a,list(axes)]-samples[b,list(axes)]))
            proposals.append((distance,i,j,a,b,plane))
    clusters={}
    for _,i,j,a,b,plane in sorted(proposals):
        point_a,point_b=samples[a],samples[b]
        key=(i,j);group=clusters.setdefault(key,[])
        match=next((s for s in group if np.linalg.norm(point_a-np.asarray(s['pair_xyz'][0]))<=cfg['crossing_merge_um'] and
                    np.linalg.norm(point_b-np.asarray(s['pair_xyz'][1]))<=cfg['crossing_merge_um']),None)
        if match is not None:
            if plane not in match['overlap_planes']:match['overlap_planes'].append(plane)
            continue
        arms=[dict(role='crossing_path',node_ids=paths[k],
                   xyz=_window(path_xyz[k],arcs[idx],cfg['arm_length_um']).tolist()) for k,idx in [(i,a),(j,b)]]
        site=dict(kind='crossing',center_xyz=((point_a+point_b)/2).tolist(),
                  pair_xyz=[point_a.tolist(),point_b.tolist()],
                  separation_um=float(np.linalg.norm(point_a-point_b)),
                  overlap_planes=[plane],arms=arms,context_xyz=[])
        group.append(site);sites.append(site)
    for i,site in enumerate(sites,1):site['site_id']=f'{i:03d}'
    return sites


def extended_sites(tree, soma_xyz, cfg):
    """Inspect long branch interiors and potential switches onto crossing shafts."""
    nodes,children,paths=tree_paths(tree);sites=[]
    for path in paths:
        xyz=np.asarray([nodes[i][2:5] for i in path],float)
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        if arc[-1]<=cfg['branch_scan_window_um']:continue
        for at in np.arange(cfg['branch_scan_window_um']/2,arc[-1],cfg['branch_scan_step_um']):
            points=_window(xyz,at,cfg['branch_scan_window_um']/2)
            center=[np.interp(at,arc,xyz[:,j]) for j in range(3)]
            sites.append(dict(kind='path_window',center_xyz=center,context_xyz=[],
                arms=[dict(role='branch_interior',node_ids=path,xyz=points.tolist())]))
    for nid,n in nodes.items():
        # A real topological bifurcation cannot be rejected by this switch test.
        if int(n[6]) not in nodes or len(children[nid])!=1:continue
        center=np.asarray(n[2:5],float)
        if np.linalg.norm(center-soma_xyz)<=14:continue
        cur=nid;back=[center];length=0.
        while int(nodes[cur][6]) in nodes and length<7:
            cur=int(nodes[cur][6]);q=np.asarray(nodes[cur][2:5],float)
            length+=np.linalg.norm(q-back[-1]);back.append(q)
        if length<6:continue
        child=children[nid][0];ids=[nid,child];forward=[center,np.asarray(nodes[child][2:5],float)]
        length=np.linalg.norm(forward[1]-center);cur=child
        while len(children[cur])==1 and length<7:
            cur=children[cur][0];q=np.asarray(nodes[cur][2:5],float)
            length+=np.linalg.norm(q-forward[-1]);forward.append(q);ids.append(cur)
        if length<6:continue
        incoming=center-back[-1];incoming/=max(np.linalg.norm(incoming),1e-8)
        outgoing=forward[-1]-center;outgoing/=max(np.linalg.norm(outgoing),1e-8)
        angle=float(np.degrees(np.arccos(np.clip(incoming@outgoing,-1,1))))
        if angle<cfg['switch_min_angle_degrees']:continue
        steps=np.linspace(cfg['switch_probe_start_um'],cfg['switch_probe_end_um'],17)
        opposite=center-steps[:,None]*outgoing;ahead=center+steps[:,None]*incoming
        sites.append(dict(kind='crossing_switch',center_xyz=center.tolist(),node_id=nid,
            switch_child_id=child,angle_degrees=angle,context_xyz=np.asarray(back).tolist(),
            probe_opposite_xyz=opposite.tolist(),probe_ahead_xyz=ahead.tolist(),
            arms=[dict(role='outgoing',node_ids=ids,xyz=np.asarray(forward).tolist())]))
    return sites


def _profile(volume, coordinates):
    """Allow one-voxel centering error; estimate local background independently."""
    smooth=ndi.gaussian_filter(np.asarray(volume,np.float32),.5)
    best=ndi.maximum_filter(smooth,size=3,mode='nearest')
    values=ndi.map_coordinates(best,np.asarray(coordinates).T,order=1,mode='nearest')
    bg=float(np.percentile(smooth,20))
    low=smooth[smooth<=np.percentile(smooth,50)]
    noise=max(1.,float(1.4826*np.median(np.abs(low-np.median(low)))))
    return values,bg,noise


def _runs(mask, arc):
    starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
    ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])
    return [(int(a),int(b),float(arc[b]-arc[a])) for a,b in zip(starts,ends)]


def _diagnose_full_box(raw_crop, lo_xyz, xyz, cfg):
    p=resample_line(xyz,cfg['sample_step_um']);local=p-np.asarray(lo_xyz)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    report=dict(status='uncertain',cut_xyz=None,reason='insufficient_path_or_anchor_signal',views=[])
    if np.any(local < 0) or np.any(local > np.asarray(raw_crop.shape)[::-1]-1):
        report['reason']='path_outside_image'
        return report
    if len(p)<3 or arc[-1]<cfg['min_gap_um']+4:return report
    values,bg,noise=_profile(raw_crop,local[:,::-1])
    start=arc<=2.;end=arc>=arc[-1]-2.
    anchor=min(float(np.percentile(values[start],80)),float(np.percentile(values[end],80)))
    threshold=bg+max(3.,3*noise,.15*(anchor-bg))
    anchors_ok=anchor-bg>=max(6.,5*noise)
    weak=values<threshold
    report.update(anchor_signal=anchor,background=bg,noise=noise,threshold=threshold,
                  raw_3d_fraction=float((~weak).mean()),anchors_supported=bool(anchors_ok))
    view_weak=[]
    for plane,axis,axes in PLANES:
        v,vbg,vnoise=_profile(raw_crop.max(axis=axis),local[:,list(axes)[::-1]])
        vanchor=min(float(np.percentile(v[start],80)),float(np.percentile(v[end],80)))
        vt=vbg+max(3.,3*vnoise,.15*(vanchor-vbg));vw=v<vt
        projected_length=float(np.linalg.norm(np.diff(p[:,axes],axis=0),axis=1).sum())
        report['views'].append(dict(plane=plane,signal_fraction=float((~vw).mean()),
                                   threshold=vt,anchors_supported=bool(vanchor-vbg>=max(6.,5*vnoise)),
                                   projected_length_um=projected_length))
        view_weak.append(vw)
    gaps=[]
    for a,b,length in _runs(weak,arc):
        if length<cfg['min_gap_um'] or arc[a]<2 or arc[-1]-arc[b]<2:continue
        votes=[]
        for k,(_,_,axes) in enumerate(PLANES):
            projected_gap=float(np.linalg.norm(np.diff(p[a:b+1,list(axes)],axis=0),axis=1).sum())
            fraction=float(view_weak[k][a:b+1].mean())
            agrees=bool(projected_gap>=1.5 and fraction>=.7 and report['views'][k]['anchors_supported'])
            votes.append(dict(plane=PLANES[k][0],weak_fraction=fraction,
                              projected_gap_um=projected_gap,agrees=agrees))
        gaps.append(dict(start_xyz=p[a].tolist(),end_xyz=p[b].tolist(),length_um=length,
                         view_votes=votes,resolving_views=sum(v['agrees'] for v in votes)))
    report['gaps']=gaps
    qualifying=[g for g in gaps if g['resolving_views']>=cfg['min_resolving_views']]
    if anchors_ok and qualifying:
        gap=qualifying[0];report.update(status='disconnected',reason='two_view_gap_confirmed_in_raw_3d',
                                       cut_xyz=gap['start_xyz'],selected_gap=gap)
    elif anchors_ok and not gaps and all(v['signal_fraction']>=.85 for v in report['views']) and (~weak).mean()>=.85:
        report.update(status='supported',reason='three_views_and_raw_3d_support')
    elif anchors_ok:report['reason']='views_or_3d_disagree'
    return report


def _thin_profiles(crop, local, thickness):
    """Follow path depth in each view; distant structures cannot fill its gap."""
    result=[]
    for plane,axis,axes in PLANES:
        depth_axis=2-axis
        centers=np.rint(local[:,depth_axis]).astype(int)
        values=np.empty(len(local),float)
        half=int(thickness)//2
        bounds=[]
        for center in np.unique(centers):
            lo=max(0,int(center)-half);hi=min(crop.shape[axis],int(center)+half+1)
            section=[slice(None)]*3;section[axis]=slice(lo,hi)
            mip=crop[tuple(section)].max(axis=axis)
            select=centers==center
            values[select],_,_=_profile(mip,local[select][:,list(axes)[::-1]])
            bounds.append([int(center),lo,hi])
        result.append(dict(plane=plane,values=values,depth_bounds=bounds))
    return result


def diagnose_path(raw_crop, lo_xyz, xyz, cfg):
    """Require a raw 3-D gap and local anchors in two depth-limited views.

    Full-box results remain audit context, but cannot veto a gap hidden by
    unrelated foreground. Weak entire paths and unanchored endings stay uncertain.
    """
    cfg=dict(DEFAULTS,**cfg)
    previous=_diagnose_full_box(raw_crop,lo_xyz,xyz,cfg)
    if previous['reason']=='path_outside_image':return previous
    p=resample_line(xyz,cfg['sample_step_um']);local=p-np.asarray(lo_xyz)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if len(p)<3 or arc[-1]<cfg['min_gap_um']+4:return previous
    values,bg,noise=_profile(raw_crop,local[:,::-1])
    reference=float(np.percentile(values,80))
    threshold=bg+max(3.,3*noise,.25*(reference-bg))
    thin=_thin_profiles(raw_crop,local,cfg['diagnostic_slab_um'])
    ridge=SignalEvidence(raw_crop).profile(local[:,::-1],center_search=True)
    report=dict(previous)
    report.pop('selected_gap',None)
    report.update(status='uncertain',cut_xyz=None,reason='no_locally_anchored_gap',
                  full_box_status=previous['status'],full_box_reason=previous['reason'],
                  full_box_gaps=previous.get('gaps',[]),gaps=[],threshold=threshold,
                  raw_3d_fraction=float((values>=threshold).mean()),
                  projection_mode='path_following_thin_slab',slab_um=cfg['diagnostic_slab_um'])
    for a,b,length in _runs(values<threshold,arc):
        if length<cfg['min_gap_um']:continue
        left=(arc>=arc[a]-cfg['local_anchor_window_um'])&(arc<arc[a])
        right=(arc<=arc[b]+cfg['local_anchor_window_um'])&(arc>arc[b])
        if left.sum()<3 or right.sum()<3:continue
        anchor=min(float(np.percentile(values[left],80)),float(np.percentile(values[right],80)))
        gap_signal=float(np.percentile(values[a:b+1],80))
        ratio=max(0.,gap_signal-bg)/max(1.,anchor-bg)
        anchors_ok=anchor-bg>=max(6.,4*noise)
        votes=[]
        for k,(_,_,axes) in enumerate(PLANES):
            v=thin[k]['values']
            va=min(float(np.percentile(v[left],80)),float(np.percentile(v[right],80)))
            vg=float(np.percentile(v[a:b+1],80))
            vr=max(0.,vg-bg)/max(1.,va-bg)
            projected_gap=float(np.linalg.norm(np.diff(p[a:b+1,list(axes)],axis=0),axis=1).sum())
            votes.append(dict(plane=PLANES[k][0],anchor_signal=va,gap_signal=vg,
                gap_anchor_ratio=vr,projected_gap_um=projected_gap,
                agrees=bool(projected_gap>=1.5 and va-bg>=max(6.,4*noise)
                            and vr<=cfg['max_gap_anchor_ratio'])))
        gap=dict(start_xyz=p[a].tolist(),end_xyz=p[b].tolist(),length_um=length,
                 anchor_signal=anchor,gap_signal=gap_signal,gap_anchor_ratio=ratio,
                 anchors_supported=bool(anchors_ok),view_votes=votes,
                 resolving_views=sum(v['agrees'] for v in votes))
        absent=ridge['ridge'][a:b+1]<np.maximum(1.2,1.25*ridge['noise'][a:b+1])
        gap['absent_ridge_fraction']=float(absent.mean())
        gap['absent_ridge_run_um']=max((r[2] for r in _runs(absent,arc[a:b+1])),default=0.)
        report['gaps'].append(gap)
    qualifying=[g for g in report['gaps'] if g['anchors_supported']
                and g['gap_anchor_ratio']<=cfg['max_gap_anchor_ratio']
                and g['absent_ridge_fraction']>=.5 and g['absent_ridge_run_um']>=2.
                and g['resolving_views']>=cfg['min_resolving_views']]
    if qualifying:
        gap=qualifying[0]
        report.update(status='disconnected',reason='local_anchors_thin_views_and_raw_3d_gap',
                      cut_xyz=gap['start_xyz'],selected_gap=gap)
    elif previous['status']=='supported':
        report.update(status='supported',reason=previous['reason'])
    # Keep exact sampling and depth bounds so every new decision is reviewable.
    report['profile']=dict(xyz=p.tolist(),arc_um=arc.tolist(),raw_3d=values.tolist(),
        ridge=ridge['ridge'].tolist(),ridge_noise=ridge['noise'].tolist(),
        thin_views=[dict(plane=v['plane'],values=v['values'].tolist(),depth_bounds=v['depth_bounds']) for v in thin])
    return report


def _thin_ridge_support(crop, local, cfg):
    """Transverse contrast in thin MIPs, with each projection's own geometry."""
    views=[]
    for plane,axis,axes in PLANES:
        p=local[:,list(axes)[::-1]]
        length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        tangent=np.gradient(p,axis=0)
        tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-8)
        normal=np.column_stack([-tangent[:,1],tangent[:,0]])
        centers=np.rint(local[:,2-axis]).astype(int);good=np.zeros(len(p),bool)
        for depth in np.unique(centers):
            half=int(cfg['diagnostic_slab_um'])//2
            section=[slice(None)]*3
            section[axis]=slice(max(0,int(depth)-half),min(crop.shape[axis],int(depth)+half+1))
            mip=ndi.gaussian_filter(crop[tuple(section)].max(axis=axis).astype(float),.5)
            select=centers==depth;points=p[select];perp=normal[select]
            center=np.max([ndi.map_coordinates(mip,(points+d*perp).T,order=1,mode='nearest') for d in [-1,0,1]],axis=0)
            sides=np.asarray([ndi.map_coordinates(mip,(points+d*perp).T,order=1,mode='nearest') for d in [-4,-3,3,4]])
            bg=np.median(sides,axis=0);noise=np.maximum(1.,.25*1.4826*np.median(np.abs(sides-bg),axis=0))
            good[select]=(center-bg>=np.maximum.reduce([np.full_like(center,3.),2.5*noise,.1*center]))&(center>=10)
        views.append(dict(plane=plane,projected_length_um=length,support_fraction=float(good.mean())))
    return views


def diagnose_switch(crop, lo, site, cfg):
    """A heuristic ownership test, not proof that touching fibers are disconnected."""
    evidence=SignalEvidence(crop);reports={}
    paths=dict(incoming=list(reversed(site['context_xyz'])),outgoing=site['arms'][0]['xyz'],
               opposite=site['probe_opposite_xyz'],ahead=site['probe_ahead_xyz'])
    for name,xyz in paths.items():
        local=resample_line(xyz,.5)-lo
        if np.any(local<4) or np.any(local>np.array(crop.shape)[::-1]-5):
            return dict(status='uncertain',cut_xyz=None,reason='switch_probe_outside_image',views=[])
        centered=evidence.profile(local[:,::-1])
        # Resampled SWCs can miss the voxel ridge by one voxel. Apply the same
        # tolerance as gap protection only to traced paths, never to probes.
        r=evidence.profile(local[:,::-1],center_search=True) if name in ('incoming','outgoing') else centered
        reports[name]=dict(raw_3d_support=float(r['good'].mean()),median_ridge=float(np.median(r['ridge'])),
                           centered_raw_3d_support=float(centered['good'].mean()),
                           centering_tolerance_um=1. if name in ('incoming','outgoing') else 0.,
                           views=_thin_ridge_support(crop,local,cfg),xyz=(local+lo).tolist())
    votes=[]
    for a,b in zip(reports['opposite']['views'],reports['ahead']['views']):
        votes.append(dict(plane=a['plane'],agrees=bool(min(a['projected_length_um'],b['projected_length_um'])>=3
                     and a['support_fraction']>=.65 and b['support_fraction']<=.4)))
    supported=all(reports[k]['raw_3d_support']>=.8 for k in ['incoming','outgoing','opposite'])
    likely=supported and reports['ahead']['raw_3d_support']<=.25 and reports['opposite']['median_ridge']>=4 \
        and sum(v['agrees'] for v in votes)>=cfg['min_resolving_views']
    return dict(status='suspected_switch' if likely else 'uncertain',
        cut_xyz=site['center_xyz'] if likely else None,
        reason='likely_switch_to_independent_shaft' if likely else 'insufficient_switch_evidence',
        evidence_type='ownership_heuristic_not_proven_gap',probes=reports,views=votes,
        resolving_views=sum(v['agrees'] for v in votes),angle_degrees=site['angle_degrees'])


def _cut_node(nodes, arm, cut_xyz):
    """Choose the downstream endpoint of the existing edge containing the gap."""
    ids=arm['node_ids'];p=np.asarray([nodes[n][2:5] for n in ids],float)
    a=p[:-1];v=p[1:]-a;vv=(v*v).sum(1)
    t=np.clip(((np.asarray(cut_xyz)-a)*v).sum(1)/np.maximum(vv,1e-12),0,1)
    dist=np.linalg.norm(a+t[:,None]*v-cut_xyz,axis=1)
    return int(ids[int(np.argmin(dist))+1])


def _protected_subtree(cut,nodes,children):
    stack=[cut]
    while stack:
        n=stack.pop()
        if int(nodes[n][1])==1 or int(nodes[n][6]) not in nodes:return True
        stack.extend(children[n])
    return False


def diagnose_connections(tree, raw, output_dir, soma_xyz, cfg=None):
    """Save all pre-cut evidence. This function never changes the input tree."""
    cfg=dict(DEFAULTS,**(cfg or {}));out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    nodes,children,_=tree_paths(tree);sites=candidate_sites(tree,cfg)+extended_sites(tree,np.asarray(soma_xyz),cfg)
    for i,site in enumerate(sites,1):site['site_id']=f'{i:03d}'
    report=dict(policy=POLICY,mode='diagnose_before_cut',input_sha256=tree_digest(tree),
                parameters=cfg,sites=sites,proposed_cut_ids=[],
                interpretation='Image support is not proof of biological connectivity; uncertain sites are kept.')
    for site in sites:
        points=np.vstack([np.asarray(a['xyz']) for a in site['arms']]+([np.asarray(site['context_xyz'])] if site['context_xyz'] else []))
        if site['kind']=='crossing_switch':
            points=np.vstack([points,site['probe_opposite_xyz'],site['probe_ahead_xyz']])
        lo=np.maximum(0,np.floor(points.min(0)-cfg['margin_um'])).astype(int)
        hi=np.minimum(np.array(raw.shape)[::-1],np.ceil(points.max(0)+cfg['margin_um']+1)).astype(int)
        site['bbox_xyz']=[[int(a),int(b)] for a,b in zip(lo,hi)]
        site['bbox_zyx']=site['bbox_xyz'][::-1]
        if np.any(hi<=lo):
            site['status']='uncertain';site['reason']='outside_image';continue
        crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
        site['proposed_cut_ids']=[]
        for arm in site['arms']:
            arm['diagnosis']=d=(diagnose_switch(crop,lo,site,cfg) if site['kind']=='crossing_switch'
                                else diagnose_path(crop,lo,np.asarray(arm['xyz']),cfg))
            if d['status'] in ('disconnected','suspected_switch'):
                cut=site['switch_child_id'] if site['kind']=='crossing_switch' else _cut_node(nodes,arm,d['cut_xyz'])
                # Preserve the root and soma interior; this is a connection diagnostic.
                protected=_protected_subtree(cut,nodes,children) or np.linalg.norm(np.asarray(d['cut_xyz'])-soma_xyz)<=12.
                d['cut_node_id']=cut;d['protected']=bool(protected)
                if not protected:site['proposed_cut_ids'].append(cut)
        if site['proposed_cut_ids']:site['status']='cut_proposed'
        elif all(a['diagnosis']['status']=='supported' for a in site['arms']):
            site['status']='crossing_paths_supported' if site['kind']=='crossing' else 'supported'
        else:site['status']='uncertain'
        if site['kind']=='crossing':
            # Never delete a branch merely because two independent paths overlap in 2-D.
            site['connection_interpretation']='projection_overlap_not_a_swc_connection'
            if site['separation_um']>3.:
                site['connection_interpretation']='spatially_separated_crossing'
        report['proposed_cut_ids'].extend(site['proposed_cut_ids'])
        from visualization.connection_diagnostics import render_site
        site['image']=f"site_{site['site_id']}.png"
        render_site(crop,lo,site,out/site['image'])
    report['proposed_cut_ids']=sorted(set(report['proposed_cut_ids']))
    from collections import Counter
    report['counts']=dict(Counter(s['status'] for s in sites))
    save_json(out/'diagnosis.json',report)
    from visualization.connection_diagnostics import write_gallery
    write_gallery(report,out)
    return report


def apply_connection_cuts(tree, report):
    """Apply a previously saved diagnostic decision to exactly its input tree."""
    if tree_digest(tree)!=report['input_sha256']:raise ValueError('Diagnostic input tree has changed')
    nodes,children,_=tree_paths(tree);removed=set();events=[]
    for cut in report['proposed_cut_ids']:
        if cut not in nodes:raise ValueError('Unknown diagnostic cut node')
        if cut in removed:continue
        if _protected_subtree(cut,nodes,children):raise ValueError('Cut would remove root/soma')
        deleted=set();stack=[cut]
        while stack:
            n=stack.pop()
            if n in deleted:continue
            deleted.add(n);stack.extend(children[n])
        removed.update(deleted);events.append(dict(cut_node_id=cut,removed_ids=sorted(deleted),
            evidence_sites=[s['site_id'] for s in report['sites'] if cut in s.get('proposed_cut_ids',[])],
            evidence_types=sorted({a['diagnosis'].get('evidence_type','raw_3d_gap')
                for s in report['sites'] if cut in s.get('proposed_cut_ids',[])
                for a in s['arms'] if a['diagnosis'].get('cut_node_id')==cut})))
    return [n for n in tree if int(n[0]) not in removed],dict(
        policy=POLICY,input_sha256=report['input_sha256'],events=events,removed_ids=sorted(removed))


def diagnose_then_cut(tree,raw,output_dir,soma_xyz):
    report=diagnose_connections(tree,raw,output_dir,soma_xyz)
    # Re-read the persisted plan: all diagnostic images/JSON exist before edits.
    saved=json.loads((Path(output_dir)/'diagnosis.json').read_text())
    kept,audit=apply_connection_cuts(tree,saved)
    audit['diagnosis_path']=str(Path(output_dir)/'diagnosis.json')
    audit['counts']=report['counts'];audit['site_count']=len(report['sites'])
    save_json(Path(output_dir)/'cuts.json',audit)
    return kept,audit
