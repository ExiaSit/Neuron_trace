"""Soma-rooted branch traversal. No length, soma-entry or geometry pruning rules."""
from collections import defaultdict,deque
import hashlib,json
import numpy as np
VERSION='gemma-soma-branchwalk-1'

def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def tree_index(tree):
    nodes={int(n[0]):n for n in tree};children={n:[] for n in nodes}
    if len(nodes)!=len(tree):raise ValueError('Duplicate node IDs')
    roots=[n for n in nodes if int(nodes[n][6])==-1]
    if len(roots)!=1:raise ValueError('Expected one soma-rooted tree')
    for n,row in nodes.items():
        if not np.isfinite(row).all():raise ValueError('Non-finite row')
        p=int(row[6])
        if p!=-1:
            if p not in nodes:raise ValueError('Dangling parent')
            children[p].append(n)
    seen=set();queue=[roots[0]]
    while queue:
        n=queue.pop()
        if n in seen:raise ValueError('Cycle')
        seen.add(n);queue.extend(children[n])
    if seen!=set(nodes):raise ValueError('Disconnected cycle')
    return nodes,{k:sorted(v) for k,v in children.items()},roots[0]

def ancestry(nodes,n):
    result=[]
    while n!=-1:result.append(n);n=int(nodes[n][6])
    return result[::-1]

def descendants(children,n):
    result=set();stack=[n]
    while stack:
        v=stack.pop();result.add(v);stack.extend(children[v])
    return result

def segment(nodes,children,parent,child):
    ids=[parent,child]
    while len(children[ids[-1]])==1:ids.append(children[ids[-1]][0])
    xyz=[nodes[n][2:5] for n in ids]
    return dict(branch_id='B'+str(child),parent_id=parent,child_id=child,end_id=ids[-1],node_ids=ids,xyz=xyz,length_um=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()))

def future_segments(nodes,children,parent,child,depth=3):
    """Three consecutive branch segments along EACH forward path, all outlets."""
    result=[];todo=[(parent,child,1)]
    while todo:
        p,c,level=todo.pop(0);s=segment(nodes,children,p,c);s['level']=level;result.append(s)
        if level<depth:todo.extend((s['end_id'],n,level+1) for n in children[s['end_id']])
    return result

def plan_visits(tree,target_xyz,lookahead=3):
    nodes,children,root=tree_index(tree)
    if np.linalg.norm(np.array(nodes[root][2:5])-target_xyz)>15:raise ValueError('Metadata soma and root disagree')
    visits=[];queue=deque([root]);order=0
    while queue:
        n=queue.popleft()
        if n==root or len(children[n])>1:
            order+=1;branches=[dict(segment(nodes,children,n,c),future=future_segments(nodes,children,n,c,lookahead)) for c in children[n]]
            visits.append(dict(junction_id=f'J{n}',node_id=n,order=order,is_soma_start=n==root,center_xyz=nodes[n][2:5],upstream_ids=ancestry(nodes,n),upstream_xyz=[nodes[k][2:5] for k in ancestry(nodes,n)],branches=branches))
        queue.extend(children[n])
    return visits

def review_decisions(case,reason):
    return dict(junction_id=case['junction_id'],upstream_status='uncertain',requested_views=[],outlets=[dict(branch_id=b['branch_id'],attachment_validity='uncertain',target_ownership='unknown',decision='REVIEW',cut_edge=None,evidence=[],reason=reason) for b in case['branches']])

def validate_response(j,case,views,options):
    if not isinstance(j,dict):return ['not_object']
    errors=[]
    if j.get('junction_id')!=case['junction_id']:errors.append('wrong_junction')
    if j.get('upstream_status') not in ['supported','suspected_switch','uncertain']:errors.append('invalid_upstream_status')
    requested=j.get('requested_views')
    if not isinstance(requested,list) or any(not isinstance(k,str) or k not in options for k in requested):errors.append('invalid_requested_views')
    outlets=j.get('outlets')
    if not isinstance(outlets,list):return errors+['missing_outlets']
    expected={b['branch_id']:b for b in case['branches']};seen=set()
    for result in outlets:
        if not isinstance(result,dict) or not isinstance(result.get('branch_id'),str) or result['branch_id'] not in expected:errors.append('unknown_branch');continue
        bid=result['branch_id'];b=expected[bid]
        if bid in seen:errors.append('duplicate_branch')
        seen.add(bid)
        if result.get('attachment_validity') not in ['supported','contradicted','uncertain']:errors.append('invalid_attachment')
        if result.get('target_ownership') not in ['target','other','unknown']:errors.append('invalid_ownership')
        if result.get('decision') not in ['KEEP','CUT','REVIEW']:errors.append('invalid_decision')
        if not isinstance(result.get('reason'),str):errors.append('missing_reason')
        edge=result.get('cut_edge');allowed=dict(parent_id=b['parent_id'],child_id=b['child_id'])
        if edge is not None and (edge!=allowed or any(type(edge.get(k)) is not int for k in allowed)):errors.append('invalid_edge')
        evidence=result.get('evidence')
        if not isinstance(evidence,list):errors.append('invalid_evidence');continue
        planes=set()
        for e in evidence:
            if not isinstance(e,dict) or not isinstance(e.get('view_id'),str) or e['view_id'] not in views:errors.append('invalid_view');continue
            v=views[e['view_id']]
            if not v['raw']:errors.append('hypothesis_is_not_raw_evidence')
            if not isinstance(e.get('observation'),str) or not e['observation'].strip():errors.append('empty_observation')
            if not isinstance(e.get('landmark_id'),str) or e['landmark_id'] not in v['landmarks']:errors.append('unmapped_landmark')
            # For a cut, cited local projections must include both actual edge endpoints.
            lo,hi=np.array(v['xyz_bounds'],float);points=np.array(b['xyz'][:2])
            if v['raw'] and v['scope']=='local' and np.all((points>=lo-.5)&(points<=hi+.5)):planes.add(v['plane'])
        if result.get('decision')=='CUT':
            if result.get('attachment_validity')!='contradicted' or edge!=allowed:errors.append('unlocalized_cut')
            if not ({'XY'}<=planes and bool(planes&{'XZ','YZ'})):errors.append('cut_missing_local_xy_and_depth_evidence')
            if j.get('upstream_status')=='suspected_switch':errors.append('incoming_route_requires_review_not_downstream_cut')
            if requested:errors.append('cut_with_outstanding_request')
        elif edge is not None:errors.append('non_cut_has_edge')
        if result.get('decision')=='KEEP' and result.get('attachment_validity')!='supported':errors.append('unsupported_keep')
    if seen!=set(expected):errors.append('outlets_not_fully_reviewed')
    return sorted(set(errors))

def execute(tree,visits):
    nodes,children,root=tree_index(tree);removed=set();actions=[];components=[]
    for visit in visits:
        for group in visit.get('groups',[]):
            for decision in group['judgment']['outlets']:
                if decision['decision']!='CUT':continue
                edge=decision['cut_edge'];child=edge['child_id'];ids=descendants(children,child)
                if child==root or int(nodes[child][6])!=edge['parent_id']:raise ValueError('Invalid cut boundary')
                if group['errors']:raise ValueError('Invalid output cannot authorize cut')
                if child in removed:continue
                if ids&removed:raise ValueError('Unexpected overlapping traversal cuts')
                part=[list(row) for row in tree if int(row[0]) in ids];original_parents={str(int(row[0])):int(row[6]) for row in part}
                for row in part:
                    if int(row[6]) not in ids:row[6]=-1
                tree_index(part);removed|=ids
                action=dict(junction_id=visit['junction_id'],branch_id=decision['branch_id'],edge=edge,node_ids=sorted(ids),nodes=len(ids),length_um=sum(float(np.linalg.norm(np.array(nodes[n][2:5])-nodes[int(nodes[n][6])][2:5])) for n in ids),
                    affected_forks=sorted(n for n in ids if len(children[n])>1),model_calls=[c['request'] for c in group['calls']],decision=decision,scope='Disconnected component; descendants have NOT all been assigned other-neuron identity')
                actions.append(action);components.append(dict(branch_id=decision['branch_id'],tree=part,original_parents=original_parents))
    kept=[row for row in tree if int(row[0]) not in removed];tree_index(kept)
    return kept,components,dict(actions=actions,removed_ids=sorted(removed),input_tree_sha256=fingerprint(tree))
