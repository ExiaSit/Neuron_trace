"""V4 evidence inventory, candidate union and conservative disconnection audit.

Geometry only schedules model review. Schema validity is not image truth.
"""
from collections import Counter
from copy import deepcopy
import hashlib,json
import numpy as np
from scipy.spatial import cKDTree
from algorithms.tracing.signal_support import tree_paths
from algorithms.tracing import soma_ownership_context as sc
from algorithms.tracing import model_crossing as mc
from algorithms.tracing.connection_diagnostics import candidate_sites,tree_digest

VERSION='raw-attachment-audit-v4.3'

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def target_from_metadata(meta,raw,tree):
    xyz=np.array([meta['soma_x']*meta['xy_resolution']/1000,meta['soma_y']*meta['xy_resolution']/1000,meta['soma_z']*meta['z_resolution']/1000])
    if not np.isfinite(xyz).all() or not np.all((xyz>=0)&(xyz<np.array(raw.shape[::-1]))):raise ValueError('Metadata soma outside image')
    nodes,_,_=tree_paths(tree);roots=[n for n in nodes.values() if int(n[6])==-1]
    if len(roots)!=1 or np.linalg.norm(xyz-roots[0][2:5])>15:raise ValueError('Metadata soma and SWC root mismatch')
    somas=sc.detect_somas(raw,xyz,allow_known_target=True)
    return somas,dict(known_target_xyz=xyz.tolist(),source='per-cell native metadata converted nm/1000 to image um',detected_center_xyz=somas[0]['center_xyz'],root_id=int(roots[0][0]),root_distance_um=float(np.linalg.norm(xyz-roots[0][2:5])),manually_confirmed_paths=[])

def descendants(children,start):
    seen=set();stack=[start]
    while stack:
        n=stack.pop()
        if n in seen:raise ValueError('Cyclic/duplicate child')
        seen.add(n);stack.extend(children[n])
    return seen

def generate_candidates(tree,raw,somas,graph,history=None,merge_um=8.):
    nodes,children,paths=tree_paths(tree);sites=candidate_sites(tree)
    raw_centers,_=mc.raw_junctions(raw,mc.DEFAULTS)
    raw_sites=[]
    for path in paths:
        if len(path)<2:continue
        xyz=np.array([nodes[n][2:5] for n in path]);dist,ix=cKDTree(xyz[:,:2]).query(raw_centers)
        for j in np.flatnonzero(dist<=mc.DEFAULTS['xy_tolerance_um']):
            n=path[max(1,int(ix[j]))];raw_sites.append(dict(cut_node_id=n,site_id='raw_'+str(j),sources=[dict(kind='raw_xy_junction')]))
    scan=dict(sites=raw_sites,separated_crossings=[s for s in sites if 'XY' in s.get('overlap_planes',[]) and s.get('separation_um',0)>4])
    global_cases,global_stats=sc.ownership_candidates(tree,raw,somas,graph,scan['sites'])
    proposals=[];source_counts=Counter()
    def add(n,source,extra=None):
        if n not in nodes or int(nodes[n][6]) not in nodes:return
        source_counts[source]+=1;proposals.append((n,source,extra or {}))
    for c in global_cases:
        for o in c['cut_options']:add(o['node_id'],'soma_competition',dict(global_case=c['case_id'],competitor=c['competing_soma_id'],representative=c['representative_id']))
    for s in scan['sites']:
        for src in s['sources']:add(s['cut_node_id'],src['kind'])
    # Include depth-separated XY overlaps too: they can hide a tracing jump.
    for s in sites:
        if s['kind']=='branchpoint':
            for child in children[s['node_id']]:add(child,'bifurcation')
        elif 'XY' in s.get('overlap_planes',[]):
            for arm,xyz in zip(s['arms'],s['pair_xyz']):
                n=min(arm['node_ids'],key=lambda n:np.linalg.norm(np.array(nodes[n][2:5])-xyz));add(n,'swc_xy_crossing')
    evidence=mc.SignalEvidence(raw)
    for path in paths:
        xyz=np.array([nodes[n][2:5] for n in path]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        # Long interiors are sampled throughout, not only within 60 um of a soma.
        for at in np.arange(12,arc[-1]-3,24):
            j=int(np.argmin(abs(arc-at)));n=path[j]
            direction=mc.direction_record(nodes,path,j,mc.DEFAULTS)
            support=evidence.profile(xyz[max(0,j-3):j+4,::-1],center_search=True)
            if direction['direction_abnormal']:add(n,'interior_direction')
            elif not np.all(support['good']):add(n,'interior_weak_signal')
    for h in history or []:add(h['node_id'],h['source'],h)
    # Same unbranched segment + 3D proximity. Never merge distinct sibling edges.
    segments={}
    for k,path in enumerate(paths):
        for n in path[1:]:segments[n]=k
    groups=[]
    for n,source,extra in sorted(proposals,key=lambda p:(not p[1].startswith('historical'),p[0],p[1])):
        xyz=np.array(nodes[n][2:5]);g=next((g for g in groups if segments.get(n)==g['segment'] and np.linalg.norm(xyz-np.array(nodes[g['anchor']][2:5]))<=merge_um),None)
        if g is None:g=dict(anchor=n,segment=segments.get(n),node_ids=[],sources=[],provenance=[]);groups.append(g)
        if n not in g['node_ids']:g['node_ids'].append(n)
        if source not in g['sources']:g['sources'].append(source)
        g['provenance'].append(dict(extra,node_id=n,source=source))
    cases=[];lookup=cKDTree(graph['xyz'])
    for i,g in enumerate(groups,1):
        n=g['anchor'];up=sc.ancestors(nodes,n);center=list(nodes[n][2:5]);_,vertex=lookup.query(center)
        costs=graph['distances'][:,vertex];finite=[j for j in range(1,len(somas)) if np.isfinite(costs[j])]
        competitor=min(finite,key=lambda j:costs[j]) if finite else None
        route=sc.route_to_soma(graph,competitor,int(vertex),somas) if competitor is not None else []
        segment=paths[g['segment']] if g['segment'] is not None else [n]
        # Edge refinement remains local. Farther upstream has separate scheduled cases/stations.
        options=set(g['node_ids']);idx=segment.index(n)
        options.update(segment[max(1,idx-2):idx+3]);options=sorted(options)
        options=[x for x in options if np.linalg.norm(np.array(nodes[x][2:5])-somas[0]['center_xyz'])>14 and int(nodes[x][1])!=1]
        allowed=[dict(parent_id=int(nodes[x][6]),child_id=x,xyz=list(nodes[x][2:5]),parent_xyz=list(nodes[int(nodes[x][6])][2:5])) for x in options]
        stations=[]
        for name,points in [('upstream',[nodes[x][2:5] for x in up]),('competitor',route)]:
            if len(points)<2:continue
            p=np.array(points);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
            for at in np.arange(12,arc[-1],24):
                j=int(np.argmin(abs(arc-at)));stations.append(dict(id=f'{name}_{j}',kind=name,center_xyz=p[j].tolist(),arc_um=float(arc[j])))
        for s in somas:
            if s['soma_id'] in [0,competitor]:stations.append(dict(id=f'soma_{s["soma_id"]}',kind='soma',center_xyz=s['center_xyz']))
        downstream=[nodes[x][2:5] for x in segment[idx:]]
        cases.append(dict(candidate_id=f'{i:04d}',anchor_id=n,center_xyz=center,sources=g['sources'],provenance=g['provenance'],allowed_edges=allowed,
            upstream_ids=up,upstream_xyz=[nodes[x][2:5] for x in up],segment_ids=segment[max(0,idx-2):min(len(segment),idx+12)],downstream_xyz=downstream,
            competing_soma_id=competitor,other_route_xyz=route,stations=stations,
            protected=bool(int(nodes[n][1])==1 or np.linalg.norm(np.array(center)-somas[0]['center_xyz'])<=14)))
    return cases,dict(by_source_before_dedup=dict(source_counts),before_dedup=len(proposals),after_dedup=len(cases),merge_um=merge_um,global_statistics=global_stats,depth_separated_pairs=len(scan['separated_crossings']))

def validate_judgment(j,c,views,options):
    errors=[]
    required=['candidate_id','image_support','attachment_validity','target_ownership','other_soma_id','cut_edge','recommendation','evidence','missing_evidence','requested_views']
    if not isinstance(j,dict):return ['not_object']
    if any(k not in j for k in required):return ['missing_fields']
    if j['candidate_id']!=c['candidate_id']:errors.append('candidate_id')
    for key,values in [('image_support',['supported','unsupported','uncertain']),('attachment_validity',['supported','contradicted','uncertain']),('target_ownership',['target','other','unknown']),('recommendation',['KEEP','PROPOSE_CUT','REQUEST_MORE','REVIEW'])]:
        if j[key] not in values:errors.append(key)
    edge=j['cut_edge'];allowed=[{k:o[k] for k in ['parent_id','child_id']} for o in c['allowed_edges']]
    if edge is not None and edge not in allowed:errors.append('invalid_edge')
    if j['other_soma_id'] is not None and (j['other_soma_id']!=c['competing_soma_id'] or j['target_ownership']!='other'):errors.append('unprovided_soma')
    if not isinstance(j['evidence'],list) or not isinstance(j['missing_evidence'],list) or not isinstance(j['requested_views'],list):return errors+['invalid_lists']
    for e in j['evidence']:
        if not isinstance(e,dict) or not isinstance(e.get('view_id'),str) or e.get('view_id') not in views:errors.append('invalid_view');continue
        v=views[e['view_id']]
        if not v['raw']:errors.append('overlay_not_raw_evidence')
        try:
            bounds=np.array(v['xyz_bounds']);b=bounds.copy() if e.get('region_id')=='view_bounds' else np.array(e['xyz_bounds'],float)
            if b.shape!=(2,3) or not np.isfinite(b).all() or not np.all((b[0]>=bounds[0])&(b[1]<=bounds[1])&(b[0]<=b[1])):errors.append('invalid_evidence_range')
        except (KeyError,ValueError,TypeError):errors.append('invalid_evidence_range')
        if e.get('region_id')=='view_bounds':e['xyz_bounds']=deepcopy(v['xyz_bounds'])
        if not isinstance(e.get('observation'),str) or not e['observation'].strip():errors.append('empty_observation')
    if any(not isinstance(v,str) or v not in options for v in j['requested_views']):errors.append('invalid_request_option')
    if j['recommendation']=='REQUEST_MORE' and not j['requested_views']:errors.append('empty_request')
    if j['recommendation']=='KEEP' and j['attachment_validity']!='supported':errors.append('unsupported_keep')
    if j['recommendation']=='PROPOSE_CUT':
        if edge is None or j['attachment_validity']!='contradicted' or not j['evidence']:errors.append('unlocalized_proposal')
        cited_local_depth=[views[e['view_id']] for e in j['evidence'] if isinstance(e,dict) and isinstance(e.get('view_id'),str) and e['view_id'] in views and views[e['view_id']].get('raw') and views[e['view_id']].get('scope')=='local' and views[e['view_id']].get('plane') in ['XZ','YZ']]
        if not cited_local_depth:errors.append('incomplete_local_depth_evidence_not_accuracy_test')
        if edge in allowed:
            xyz=np.array(next(o['xyz'] for o in c['allowed_edges'] if o['child_id']==edge['child_id']))
            local=[v for v in views.values() if v['raw'] and v['scope']=='local' and np.all((xyz>=v['xyz_bounds'][0])&(xyz<=v['xyz_bounds'][1]))]
            if not local:errors.append('unimaged_cut')
    return sorted(set(errors))

def review_result(reason):
    return dict(image_support='uncertain',attachment_validity='uncertain',target_ownership='unknown',other_soma_id=None,cut_edge=None,recommendation='REVIEW',evidence=[],missing_evidence=[reason],requested_views=[])

def proposed(c):
    a=c.get('judgment',{});b=c.get('verification',{})
    return a.get('recommendation')=='PROPOSE_CUT' and b.get('recommendation')=='PROPOSE_CUT' and a.get('cut_edge')==b.get('cut_edge') and not c.get('protected') and not c.get('errors') and not c.get('verification_errors')

def impact(tree,c,cases):
    nodes,children,paths=tree_paths(tree);edge=c['judgment']['cut_edge'];ids=descendants(children,edge['child_id'])
    judgments=[dict(candidate_id=t['candidate_id'],anchor_id=t['anchor_id'],target_ownership=t.get('judgment',{}).get('target_ownership','unknown'),recommendation=t.get('judgment',{}).get('recommendation','REVIEW'),segment_ids=sorted(set(t['segment_ids'])&ids)) for t in cases if set(t['segment_ids'])&ids]
    return dict(edge=edge,node_ids=sorted(ids),nodes=len(ids),branches=sum(bool(set(p[1:])&ids) for p in paths),length_um=sum(float(np.linalg.norm(np.array(nodes[n][2:5])-nodes[int(nodes[n][6])][2:5])) for n in ids if int(nodes[n][6]) in nodes),judgments=judgments,
        involves_soma=any(int(nodes[n][1])==1 or int(nodes[n][6])==-1 for n in ids),confirmed_paths=[],unreviewed_branchpoints=[n for n in ids if len(children[n])>1 and not any(n in t['segment_ids'] and t.get('judgment',{}).get('recommendation')!='REVIEW' for t in cases)])

def execution_plan(tree,cases):
    actions=[]
    for c in cases:
        if c.get('judgment',{}).get('recommendation')!='PROPOSE_CUT':continue
        a=impact(tree,c,cases);a['candidate_id']=c['candidate_id'];reasons=[]
        if not proposed(c):reasons.append('same_model_repeat_did_not_confirm_edge')
        if a['involves_soma']:reasons.append('soma_or_root')
        if c.get('unresolved_conflict'):reasons.append('unresolved_joint_conflict')
        # Mixed/unknown downstream segments cannot be silently assigned by this junction.
        other=[j for j in a['judgments'] if j['candidate_id']!=c['candidate_id']]
        if any(j['target_ownership']=='target' for j in a['judgments']):reasons.append('downstream_target_hypothesis')
        if any(j['recommendation'] in ['REVIEW','REQUEST_MORE'] or j['target_ownership']=='unknown' for j in other):reasons.append('unresolved_downstream_segments')
        if a['unreviewed_branchpoints']:reasons.append('unreviewed_branchpoints')
        a['hold_reasons']=reasons;a['status']='REVIEW' if reasons else 'eligible';actions.append(a)
    # Shallowest eligible disconnections first; fully record nested/deduplicated plans.
    nodes,children,_=tree_paths(tree);removed=set()
    for a in sorted(actions,key=lambda a:len(sc.ancestors(nodes,a['edge']['child_id']))):
        if a['status']!='eligible':continue
        if a['edge']['child_id'] in removed:a['status']='covered_by_ancestor';continue
        a['status']='execute';removed.update(a['node_ids'])
    return dict(input_tree_sha256=tree_digest(tree),actions=actions,removed_ids=sorted(removed))

def validate_tree(tree):
    nodes={int(n[0]):n for n in tree}
    if len(nodes)!=len(tree):raise ValueError('Duplicate IDs')
    roots=[n for n in nodes if int(nodes[n][6])==-1]
    if len(roots)!=1:raise ValueError('Expected one root')
    for n in nodes:
        seen=set()
        while n!=-1:
            if n in seen:raise ValueError('Cycle')
            if n not in nodes:raise ValueError('Dangling parent')
            seen.add(n);n=int(nodes[n][6])
    return True

def apply_plan(tree,plan):
    if tree_digest(tree)!=plan['input_tree_sha256']:raise ValueError('Changed input graph')
    validate_tree(tree);nodes,children,_=tree_paths(tree);removed=set();components=[]
    for a in plan['actions']:
        if a['status']!='execute':continue
        e=a['edge']
        if int(nodes[e['child_id']][6])!=e['parent_id']:raise ValueError('Edge mismatch')
        ids=descendants(children,e['child_id'])
        if ids!=set(a['node_ids']) or removed&ids or a['hold_reasons']:raise ValueError('Invalid affected set')
        part=deepcopy([n for n in tree if int(n[0]) in ids]);mapping={str(int(n[0])):int(n[6]) for n in part}
        for row in part:
            if int(row[6]) not in ids:row[6]=-1
        validate_tree(part);components.append(dict(candidate_id=a['candidate_id'],tree=part,original_parents=mapping));removed.update(ids)
    if removed!=set(plan['removed_ids']):raise ValueError('Audit removal mismatch')
    kept=[n for n in tree if int(n[0]) not in removed];validate_tree(kept)
    return kept,components
