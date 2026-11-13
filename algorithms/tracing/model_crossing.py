"""XY encounter scan and evidence-gated local VLM ownership review.

This optional pass operates on an existing pruned SWC in 1-um XYZ coordinates.
It does not infer biological identity from proximity or a model score alone.
"""
import json
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, remove_small_objects
from algorithms.tracing.signal_support import tree_paths, resample_line, SignalEvidence
from algorithms.tracing.connection_diagnostics import candidate_sites, tree_digest

POLICY='gemma-xy-encounter-direction-v1'
DEFAULTS=dict(raw_ridge_threshold=6., xy_tolerance_um=3., merge_distance_um=8.,
              refine_radius_um=6., tangent_um=8., context_um=24.,
              minimum_turn_degrees=45., soma_protection_um=14., separated_z_um=4.)

def angle(a,b):
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return None if norm<1e-7 else float(np.degrees(np.arccos(np.clip(np.dot(a,b)/norm,-1,1))))

def walk_up(nodes,nid,length):
    ids=[nid]; arc=0.
    while int(nodes[ids[-1]][6]) in nodes and arc<length:
        nxt=int(nodes[ids[-1]][6]);arc+=np.linalg.norm(np.asarray(nodes[nxt][2:5])-nodes[ids[-1]][2:5]);ids.append(nxt)
    return ids

def point_at(xyz,distance):
    xyz=np.asarray(xyz,float)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    return np.array([np.interp(min(distance,arc[-1]),arc,xyz[:,j]) for j in range(3)]),float(arc[-1])

def direction_record(nodes,path,index,cfg):
    nid=path[index];center=np.asarray(nodes[nid][2:5]);up=walk_up(nodes,nid,40.)
    back=np.array([nodes[i][2:5] for i in up]);forward=np.array([nodes[i][2:5] for i in path[index:]])
    rear,lb=point_at(back,8.);ahead,lf=point_at(forward,8.)
    turn=angle(center-rear,ahead-center) if min(lb,lf)>=6 else None
    baseline=[]
    for dist in [8.,16.,24.]:
        a,_=point_at(back,dist);b,_=point_at(back,dist+8);c,_=point_at(back,max(0,dist-8))
        if lb>=dist+8:
            value=angle(a-b,c-a)
            if value is not None:baseline.append(value)
    median=float(np.median(baseline)) if baseline else 0.
    mad=float(np.median(np.abs(np.asarray(baseline)-median))) if baseline else 0.
    threshold=max(cfg['minimum_turn_degrees'],median+3*1.4826*mad+20.)
    origin=np.array([nodes[i][2:5] for i in path[:index+1]])
    initial,_=point_at(origin,10.)
    rear16,_=point_at(back,16.);ahead16,_=point_at(forward,16.)
    return dict(local_turn_degrees=turn,turn_16um_degrees=angle(center-rear16,ahead16-center),
                initial_heading_change_degrees=angle(initial-origin[0],ahead-center),
                upstream_turns_degrees=baseline,adaptive_threshold_degrees=threshold,
                direction_abnormal=bool(turn is not None and turn>=threshold),
                incoming_unit=((center-rear)/max(np.linalg.norm(center-rear),1e-8)).tolist(),
                outgoing_unit=((ahead-center)/max(np.linalg.norm(ahead-center),1e-8)).tolist())

def raw_junctions(raw,cfg):
    mip=raw.max(0).astype(float);smooth=ndi.gaussian_filter(mip,.6)
    mask=remove_small_objects((smooth-ndi.gaussian_filter(mip,5)>cfg['raw_ridge_threshold'])&(smooth>18),min_size=12)
    skeleton=skeletonize(mask)
    degree=ndi.convolve(skeleton.astype(int),np.ones((3,3),int))-skeleton
    labels,n=ndi.label(skeleton&(degree>=3))
    centers=np.array(ndi.center_of_mass(skeleton,labels,range(1,n+1))).reshape(-1,2)[:,::-1]
    return centers,skeleton

def scan_encounters(tree,raw,cfg=None):
    cfg=dict(DEFAULTS,**(cfg or {}));nodes,children,paths=tree_paths(tree)
    roots=[i for i in nodes if int(nodes[i][6]) not in nodes]
    if len(roots)!=1:raise ValueError('Expected one connected rooted tree')
    root=roots[0];soma=np.asarray(nodes[root][2:5]);distance={root:0.};stack=[root]
    while stack:
        p=stack.pop()
        for c in children[p]:distance[c]=distance[p]+np.linalg.norm(np.asarray(nodes[c][2:5])-nodes[p][2:5]);stack.append(c)
    if len(distance)!=len(nodes):raise ValueError('Disconnected or cyclic SWC')
    raw_centers,skeleton=raw_junctions(raw,cfg);proposals=[];separated=[]
    xyz_paths=[np.array([nodes[i][2:5] for i in p]) for p in paths]
    path_lookup={tuple(p):k for k,p in enumerate(paths)}
    for k,xyz in enumerate(xyz_paths):
        if len(xyz)<3:continue
        d,indices=cKDTree(xyz[:,:2]).query(raw_centers)
        for j in np.flatnonzero(d<=cfg['xy_tolerance_um']):
            proposals.append((k,int(indices[j]),dict(kind='raw_xy_junction',xy=raw_centers[j].tolist(),junction_id=int(j))))
    for site in candidate_sites(tree):
        if site['kind']!='crossing' or 'XY' not in site['overlap_planes']:continue
        if abs(site['pair_xyz'][0][2]-site['pair_xyz'][1][2])>cfg['separated_z_um']:
            separated.append(dict(pair_xyz=site['pair_xyz'],separation_um=site['separation_um'],status='separated_in_depth_keep'));continue
        for arm,point in zip(site['arms'],site['pair_xyz']):
            k=path_lookup[tuple(arm['node_ids'])];idx=int(np.argmin(np.linalg.norm(xyz_paths[k]-point,axis=1)))
            proposals.append((k,idx,dict(kind='swc_xy_crossing',pair_xyz=site['pair_xyz'])))
    grouped={}
    for k,idx,source in proposals:
        path=paths[k];xyz=xyz_paths[k];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        local=np.flatnonzero(np.abs(arc-arc[idx])<=cfg['refine_radius_um'])
        valid=[int(j) for j in local if j<len(path)-1]
        if not valid:continue
        records={j:direction_record(nodes,path,j,cfg) for j in valid}
        best=max(valid,key=lambda j:records[j]['local_turn_degrees'] if records[j]['local_turn_degrees'] is not None else -1)
        nid=path[best];cut=path[best+1]
        match=next((s for s in grouped.get(k,[]) if abs(s['path_arc_um']-float(arc[best]))<=cfg['merge_distance_um']),None)
        if match is not None:
            match['sources'].append(source);continue
        center=xyz[best];incoming=walk_up(nodes,nid,cfg['context_um']);incoming=list(reversed(incoming))
        end=best+1
        while end<len(path)-1 and arc[end]-arc[best]<cfg['context_um']:end+=1
        outgoing=path[best:end+1]
        context=np.vstack([[nodes[i][2:5] for i in incoming],[nodes[i][2:5] for i in outgoing]])
        for src in [source]:
            if 'pair_xyz' in src:context=np.vstack([context,src['pair_xyz']])
        lo=np.maximum(0,np.floor(context.min(0)-9)).astype(int);hi=np.minimum(raw.shape[::-1],np.ceil(context.max(0)+10)).astype(int)
        protected=bool(np.linalg.norm(center-soma)<=cfg['soma_protection_um'] or int(nodes[nid][1])==1)
        site=dict(path_id=k,node_id=nid,cut_node_id=cut,center_xyz=center.tolist(),path_arc_um=float(arc[best]),
                  root_distance_um=float(distance[nid]),sources=[source],direction=records[best],protected=protected,
                  incoming_ids=incoming,outgoing_ids=outgoing,bbox_xyz=[lo.tolist(),hi.tolist()],
                  ancestors=walk_up(nodes,nid,1e9),status='protected_soma' if protected else 'pending_model')
        grouped.setdefault(k,[]).append(site)
    sites=sorted([s for group in grouped.values() for s in group],key=lambda s:(s['root_distance_um'],s['cut_node_id']))
    for i,s in enumerate(sites,1):s['site_id']=f'{i:03d}'
    return dict(policy=POLICY,input_sha256=tree_digest(tree),parameters=cfg,sites=sites,
                separated_crossings=separated,raw_xy_junction_count=len(raw_centers),target_soma_xyz=soma.tolist()),skeleton

def raw_support(raw,site,nodes):
    lo,hi=np.asarray(site['bbox_xyz']);crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
    evidence=SignalEvidence(crop);out={}
    center=np.asarray(site['center_xyz']);d=site['direction']
    for name,ids in [('incoming',site['incoming_ids']),('outgoing',site['outgoing_ids'])]:
        points=np.array([nodes[i][2:5] for i in ids]);p=resample_line(points,.5)-lo
        # Only assess the nearest 8 um on each side of the encounter.
        p=p[np.linalg.norm(p+lo-center,axis=1)<=8.]
        out[name]=float(evidence.profile(p[:,::-1],center_search=True)['good'].mean()) if len(p)>1 else 0.
    for name,vec in [('opposite',-np.array(d['outgoing_unit'])),('ahead',np.array(d['incoming_unit']))]:
        p=center+np.arange(3.,10.,.5)[:,None]*vec-lo
        valid=np.all((p>=0)&(p<np.array(crop.shape[::-1])-1),axis=1)
        out[name]=float(evidence.profile(p[:,::-1])['good'].mean()) if valid.all() else None
    return out

def parse_judgment(response):
    choice=response['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('Truncated or incomplete model response')
    text=choice['message'].get('content','').strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
    obj=json.loads(text)
    if obj.get('decision') not in ['switch_to_other','same_neuron','projection_overlap','real_bifurcation','uncertain']:raise ValueError('Invalid decision')
    return obj

def cut_gate(site,judgment):
    reasons=[]
    if site['protected']:reasons.append('soma_protected')
    if not site['direction']['direction_abnormal']:reasons.append('no_abnormal_direction_change')
    if judgment.get('decision')!='switch_to_other':reasons.append('model_did_not_identify_switch')
    if judgment.get('crossing_node_id')!=site['node_id']:reasons.append('wrong_crossing_node')
    if judgment.get('connection_in_3d')!='connected':reasons.append('3d_entry_not_established')
    if judgment.get('independent_shaft') is not True:reasons.append('independent_shaft_not_established')
    views={v.get('plane') for v in judgment.get('supporting_views',[]) if isinstance(v,dict) and isinstance(v.get('evidence'),str) and len(v['evidence'].strip())>=10}
    views&={'XY','XZ','YZ'}
    if len(views)<2 or not views&{'XZ','YZ'}:reasons.append('insufficient_multiview_evidence')
    if min(site['raw_support']['incoming'],site['raw_support']['outgoing'])<.55:reasons.append('weak_raw_3d_path_support')
    # A reverse continuation of the new direction supports an independent shaft.
    if site['raw_support'].get('opposite') is None or site['raw_support']['opposite']<.5:reasons.append('independent_shaft_not_supported_in_raw_3d')
    return dict(accepted=not reasons,reasons=reasons,model_confidence_used=False)

def apply_saved_plan(tree,report):
    if tree_digest(tree)!=report['input_sha256']:raise ValueError('Input SWC changed after diagnosis')
    nodes,children,_=tree_paths(tree);removed=set();events=[]
    for site in report['sites']:
        if not site.get('gate',{}).get('accepted'):continue
        if not cut_gate(site,site['judgment'])['accepted']:raise ValueError('Saved decision fails gate')
        cut=site['cut_node_id']
        if cut not in nodes or int(nodes[cut][6])!=site['node_id']:raise ValueError('Invalid cut edge')
        if cut in removed:continue
        descendants=set();stack=[cut]
        while stack:
            i=stack.pop()
            if i in descendants:raise ValueError('Cycle')
            if int(nodes[i][1])==1 or int(nodes[i][6]) not in nodes:raise ValueError('Cannot delete soma/root')
            descendants.add(i);stack.extend(children[i])
        removed.update(descendants);events.append(dict(site_id=site['site_id'],cut_node_id=cut,removed_ids=sorted(descendants)))
    kept=[n for n in tree if int(n[0]) not in removed]
    return kept,dict(policy=POLICY,events=events,removed_ids=sorted(removed),input_sha256=report['input_sha256'])
