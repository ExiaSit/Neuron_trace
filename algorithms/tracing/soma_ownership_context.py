"""Whole-field soma routes and full upstream evidence for ownership review.

Shortest routes generate hypotheses only; they never establish neuron identity.
All image coordinates are the pipeline's resampled 1-um XYZ coordinates.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import remove_small_objects,skeletonize
from algorithms.tracing.signal_support import tree_paths,resample_line,SignalEvidence
from algorithms.tracing.connection_diagnostics import tree_digest
from algorithms.tracing.model_crossing import direction_record,DEFAULTS

POLICY='gemma-global-soma-upstream-ownership-v3'

def detect_somas(raw,target,allow_known_target=False):
    smooth=ndi.gaussian_filter(raw.astype(np.float32),.7)
    edt=ndi.distance_transform_edt(smooth>=90)
    labels,count=ndi.label(edt>=2,structure=ndi.generate_binary_structure(3,2));sizes=np.bincount(labels.ravel());items=[]
    for label in range(1,count+1):
        if sizes[label]<100:continue
        zyx=np.asarray(ndi.maximum_position(edt,labels,label));radius=float(edt[tuple(zyx)])
        if radius<4:continue
        items.append(dict(center_xyz=zyx[::-1].tolist(),radius_um=radius,core_voxels=int(sizes[label]),status='image_derived_candidate_not_ground_truth'))
    items.sort(key=lambda s:np.linalg.norm(np.asarray(s['center_xyz'])-target))
    if not items or np.linalg.norm(np.asarray(items[0]['center_xyz'])-target)>10:
        if not allow_known_target:raise ValueError('Target soma not found in raw image')
        # A bright connected component can merge several soma cores. Its remote
        # EDT maximum must not replace the known metadata target location.
        xyz=np.asarray(target);lo=np.maximum(0,np.floor(xyz-5)).astype(int);hi=np.minimum(raw.shape[::-1],np.ceil(xyz+6)).astype(int)
        local=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
        if not local.size or float(np.percentile(local,90))<float(np.percentile(raw,50))+10:raise ValueError('Metadata target lacks local image support')
        voxel=np.rint(xyz).astype(int);radius=max(2.,min(10.,float(edt[tuple(voxel[::-1])])) )
        items.insert(0,dict(center_xyz=xyz.tolist(),radius_um=radius,core_voxels=None,status='known_metadata_target_local_signal_checked_geometry_estimated'))
    for i,s in enumerate(items):s['soma_id']=i;s['is_target']=i==0
    return items

def graph_from_image(raw,somas):
    smooth=ndi.gaussian_filter(raw.astype(np.float32),.6)
    contrast=smooth-ndi.gaussian_filter(smooth,4.)
    mask=remove_small_objects(((contrast>3.)&(smooth>12))|(smooth>90),min_size=30,connectivity=3)
    sk=skeletonize(mask,method='lee').astype(bool)
    # Remove spherical soma cores; these are not complete soma segmentations.
    coords=np.argwhere(sk);xyz=coords[:,::-1];outside=np.ones(len(coords),bool)
    for s in somas:outside &= np.linalg.norm(xyz-s['center_xyz'],axis=1)>s['radius_um']
    coords=coords[outside];xyz=coords[:,::-1]
    if not len(coords):raise ValueError('Empty image skeleton')
    lookup=np.full(raw.shape,-1,np.int32);lookup[tuple(coords.T)]=np.arange(len(coords),dtype=np.int32)
    rows=[];cols=[];weights=[];quality=np.maximum(contrast[tuple(coords.T)],0.)
    for dz in [-1,0,1]:
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                offset=(dz,dy,dx)
                if offset<=(0,0,0):continue
                q=coords+offset;valid=np.all((q>=0)&(q<raw.shape),axis=1);a=np.flatnonzero(valid);b=lookup[tuple(q[valid].T)]
                good=b>=0;a=a[good];b=b[good]
                cost=np.linalg.norm(offset)*(1+8/(1+np.minimum(quality[a],quality[b])))
                rows.extend([a,b]);cols.extend([b,a]);weights.extend([cost,cost])
    graph=coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(len(coords),len(coords))).tocsr()
    distances=[];predecessors=[];seed_counts=[]
    for soma in somas:
        d=np.linalg.norm(xyz-soma['center_xyz'],axis=1)
        seeds=np.flatnonzero((d<=soma['radius_um']+3.)&(smooth[tuple(coords.T)]>=20))
        seed_counts.append(len(seeds))
        if not len(seeds):
            distances.append(np.full(len(coords),np.inf));predecessors.append(np.full(len(coords),-9999,np.int32));continue
        dist,pred,_=dijkstra(graph,directed=False,indices=seeds,min_only=True,return_predecessors=True)
        distances.append(dist);predecessors.append(pred)
    return dict(xyz=xyz.astype(float),distances=np.asarray(distances),predecessors=np.asarray(predecessors),
                seed_counts=seed_counts,graph_vertices=len(coords),graph_edges=graph.nnz//2)

def route_to_soma(graph,soma_id,vertex,somas):
    if not np.isfinite(graph['distances'][soma_id,vertex]):return []
    ids=[];seen=set();i=int(vertex)
    while i>=0:
        if i in seen:raise ValueError('Cycle in shortest route')
        seen.add(i);ids.append(i);i=int(graph['predecessors'][soma_id,i])
    points=graph['xyz'][ids[::-1]]
    return np.vstack([somas[soma_id]['center_xyz'],points]).tolist()

def ancestors(nodes,nid):
    result=[];seen=set()
    while nid in nodes:
        if nid in seen:raise ValueError('Cyclic input SWC')
        seen.add(nid);result.append(nid);nid=int(nodes[nid][6])
    return result[::-1]

def path_evidence(evidence,xyz,exclude_start_um=0.):
    p=resample_line(xyz,.5)
    if len(p)<2:return dict(support_fraction=0.,longest_weak_um=None,length_um=0.)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    mask=arc>=exclude_start_um;p=p[mask];arc=arc[mask]
    if len(p)<2:return dict(support_fraction=0.,longest_weak_um=None,length_um=0.)
    r=evidence.profile(p[:,::-1],center_search=True);run=longest=0.
    for j,good in enumerate(r['good']):
        run=0 if good else run+(arc[j]-arc[j-1] if j else 0.);longest=max(run,longest)
    return dict(support_fraction=float(r['good'].mean()),longest_weak_um=float(longest),length_um=float(arc[-1]-arc[0]),
                median_ridge=float(np.median(r['ridge'])))

def ownership_candidates(tree,raw,somas,graph,crossing_sites=()):
    nodes,children,paths=tree_paths(tree);target=np.asarray(somas[0]['center_xyz']);evidence=SignalEvidence(raw)
    ids=np.asarray(list(nodes));points=np.array([nodes[i][2:5] for i in ids]);snap,vertex=cKDTree(graph['xyz']).query(points)
    costs=graph['distances'][:,vertex].T
    if len(somas)<2:
        return [],dict(nodes=len(nodes),candidate_count=0,input_sha256=tree_digest(tree),reason='no_competing_soma_detected')
    foreign=1+np.argmin(costs[:,1:],axis=1);foreign_cost=costs[np.arange(len(ids)),foreign]
    margin=np.full(len(ids),-np.inf);finite=np.isfinite(foreign_cost)
    margin[finite]=costs[finite,0]-foreign_cost[finite]
    valid=(snap<=2.5)&finite&(costs[:,0]>1.25*foreign_cost)&(margin>15)
    assigned=np.where(valid,foreign,-1);index={int(n):i for i,n in enumerate(ids)}
    anchors={}
    for s in crossing_sites:
        if not s.get('protected'):anchors.setdefault(s['cut_node_id'],[]).append(s['site_id'])
    candidates=[];covered=set()
    for path in paths:
        xyz=np.array([nodes[n][2:5] for n in path]);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        labels=assigned[[index[n] for n in path]];start=0
        while start<len(path):
            stop=start+1
            while stop<len(path) and labels[stop]==labels[start]:stop+=1
            owner=int(labels[start]);span=float(arc[stop-1]-arc[start])
            if owner<1 or span<12: start=stop;continue
            representative=(start+stop-1)//2;nid=path[representative];up=ancestors(nodes,nid)
            # Propose only existing upstream edges. Include recent crossing and fork options.
            options=[];distance=0.;last=nodes[nid][2:5]
            for n in reversed(up[1:]):
                distance+=np.linalg.norm(np.asarray(nodes[n][2:5])-last);last=nodes[n][2:5]
                if distance>60:break
                parent=int(nodes[n][6])
                if np.linalg.norm(np.asarray(nodes[n][2:5])-target)<=14:continue
                if n in anchors or (len(children.get(parent,[]))>1 and distance<=35):options.append(n)
            start_node=path[max(1,start)]
            if np.linalg.norm(np.asarray(nodes[start_node][2:5])-target)>14:options.append(start_node)
            options=list(dict.fromkeys(options))[:4]
            # Also offer the first sustained ownership transition anywhere along
            # the complete upstream route, even when it is >60 um away.
            run=0.;transition=None
            for j,n in enumerate(up[1:],1):
                if assigned[index[n]]==owner:
                    if transition is None:transition=n
                    run+=np.linalg.norm(np.asarray(nodes[n][2:5])-nodes[up[j-1]][2:5])
                    if run>=8:
                        k=up.index(transition)
                        nearby=[m for m in up[max(1,k-20):k+1] if m in anchors or len(children.get(int(nodes[m][6]),[]))>1]
                        options.extend(nearby[-2:]+[transition]);break
                else:run=0.;transition=None
            options=[n for n in dict.fromkeys(options) if np.linalg.norm(np.asarray(nodes[n][2:5])-target)>14]
            if not options:start=stop;continue
            key=(owner,options[0])
            if key in covered:start=stop;continue
            covered.add(key);v=int(vertex[index[nid]])
            other=route_to_soma(graph,owner,v,somas);target_route=route_to_soma(graph,0,v,somas)
            up_xyz=[nodes[n][2:5] for n in up]
            quality=path_evidence(evidence,other,somas[owner]['radius_um']+3.)
            # Image-supported routes are kept even if weak, but cannot authorize a cut.
            detail=dict(case_id=f'{len(candidates)+1:03d}',representative_id=nid,center_xyz=list(nodes[nid][2:5]),
                        competing_soma_id=owner,supported_run_ids=path[start:stop],supported_run_um=span,
                        candidate_cut_ids=options,upstream_ids=up,upstream_xyz=up_xyz,
                        downstream_xyz=xyz[start:stop].tolist(),other_route_xyz=other,target_route_xyz=target_route,
                        other_route_evidence=quality,upstream_evidence=path_evidence(evidence,up_xyz,14.),
                        target_route_evidence=path_evidence(evidence,target_route,somas[0]['radius_um']+3.) if target_route else None,
                        target_graph_cost=float(costs[index[nid],0]) if np.isfinite(costs[index[nid],0]) else None,
                        other_graph_cost=float(foreign_cost[index[nid]]),snap_distance_um=float(snap[index[nid]]),
                        intermediate_soma_encounters=route_soma_encounters(other,somas,owner),
                        cut_options=[dict(node_id=n,parent_id=int(nodes[n][6]),xyz=list(nodes[n][2:5]),crossing_sites=anchors.get(n,[])) for n in options])
            up_arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(np.asarray(up_xyz),axis=0),axis=1))]
            for option in detail['cut_options']:
                k=up.index(option['node_id']);delta=up_arc-up_arc[k]
                after=[j for j in range(k,len(up)) if assigned[index[up[j]]]==owner]
                distance=float(delta[after[0]]) if after else None
                before=[j for j in range(k) if -12<=delta[j]<=-4]
                # A localization guard, not a neuron ownership classifier.
                target_before=[]
                for j in before:
                    ix=index[up[j]];tc=costs[ix,0];oc=costs[ix,owner]
                    target_before.append(bool(np.isfinite(tc) and tc+5<oc))
                window=[p for p,d in zip(up_xyz,delta) if -8<=d<=8]
                local=path_evidence(evidence,window)
                direction=direction_record(nodes,up,k,DEFAULTS)
                transition=bool(target_before and np.mean(target_before)>=.5 and distance is not None and distance<=12.)
                option['localization']=dict(distance_to_foreign_supported_run_um=distance,
                    target_to_other_cost_transition=transition,local_signal=local,direction=direction,
                    supported=bool(distance is not None and distance<=12. and
                        (transition or direction['direction_abnormal'] or (local['longest_weak_um'] or 0)>=2.)))
            candidates.append(detail);start=stop
    diagnostics=dict(nodes=len(nodes),nodes_near_image_skeleton=int((snap<=2.5).sum()),nodes_preferring_other_route=int(valid.sum()),
                     unknown_or_unsupported_nodes=int((snap>2.5).sum()),graph_vertices=graph['graph_vertices'],graph_edges=graph['graph_edges'],
                     seed_counts=graph['seed_counts'],candidate_count=len(candidates),input_sha256=tree_digest(tree))
    return candidates,diagnostics

def route_soma_encounters(route,somas,origin_id):
    if not route:return []
    p=np.asarray(route)
    return [s['soma_id'] for s in somas if s['soma_id']!=origin_id and
            np.min(np.linalg.norm(p-s['center_xyz'],axis=1))<=s['radius_um']+2.]

def ownership_gate(case,judgment):
    reasons=[];quality=case['other_route_evidence']
    if case.get('intermediate_soma_encounters'):reasons.append('competing_route_passes_another_soma')
    if judgment.get('ownership')!='other':reasons.append('other_ownership_not_confirmed')
    if judgment.get('soma_id')!=case['competing_soma_id']:reasons.append('wrong_soma')
    if type(judgment.get('cut_node_id')) is not int or judgment.get('cut_node_id') not in case['candidate_cut_ids']:reasons.append('cut_not_localized_to_existing_upstream_edge')
    option=next((o for o in case.get('cut_options',[]) if o['node_id']==judgment.get('cut_node_id')),None)
    if option is None or not option.get('localization',{}).get('supported'):reasons.append('cut_lacks_measured_upstream_transition')
    for key in ['candidate_is_soma','other_soma_connection_supported','upstream_switch_localized','target_origin_rejected_by_image']:
        if judgment.get(key) is not True:reasons.append(key+'_not_established')
    views=judgment.get('evidence',[])
    if not isinstance(views,list):views=[]
    planes={v.get('view') for v in views if isinstance(v,dict) and isinstance(v.get('observation'),str) and len(v['observation'])>=15}
    if not planes&{'XZ','YZ','path_unfolded'} or len(planes&{'XY','XZ','YZ','path_unfolded'})<2:reasons.append('insufficient_global_evidence')
    if quality['support_fraction']<.8 or quality['longest_weak_um'] is None or quality['longest_weak_um']>3:reasons.append('competing_soma_route_not_continuously_supported')
    return dict(accepted=not reasons,reasons=reasons,nearest_soma_or_shortest_route_alone_never_cuts=True)

def apply_ownership_plan(tree,report):
    if tree_digest(tree)!=report['input_sha256']:raise ValueError('Changed input')
    nodes,children,_=tree_paths(tree);events=[];removed=set()
    candidates=[c for c in report['cases'] if c.get('gate',{}).get('accepted')]
    candidates.sort(key=lambda c:len(ancestors(nodes,c['judgment']['cut_node_id'])))
    for c in candidates:
        if not ownership_gate(c,c['judgment'])['accepted']:raise ValueError('Invalid persisted decision')
        cut=c['judgment']['cut_node_id']
        if cut in removed:continue
        if cut not in nodes:raise ValueError('Unknown cut node')
        option=next(o for o in c['cut_options'] if o['node_id']==cut)
        if option['parent_id']!=int(nodes[cut][6]) or not np.allclose(option['xyz'],nodes[cut][2:5]):raise ValueError('Cut edge changed')
        descendants=set();stack=[cut]
        while stack:
            n=stack.pop()
            if n in descendants:raise ValueError('Cycle')
            if int(nodes[n][1])==1 or int(nodes[n][6]) not in nodes:raise ValueError('Soma/root protected')
            descendants.add(n);stack.extend(children[n])
        removed.update(descendants);events.append(dict(case_id=c['case_id'],cut_node_id=cut,competing_soma_id=c['competing_soma_id'],removed_ids=sorted(descendants)))
    return [n for n in tree if int(n[0]) not in removed],dict(events=events,removed_ids=sorted(removed))
