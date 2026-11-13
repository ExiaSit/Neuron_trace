"""Bounded bidirectional conflict review; model labels remain revisable hypotheses."""
from copy import deepcopy
from algorithms.tracing import ownership_audit as oa

def segment_review_tasks(report,maximum=12,per_conflict=2):
    cases={c['candidate_id']:c for c in report['cases']};tasks={};deferred=[]
    for group in report.get('joint_reviews',[]):
        related=sorted(group['related_ids'],key=lambda cid:cases[cid].get('judgment',{}).get('target_ownership')!='target')
        for cid in related[:per_conflict]:
            if cases[cid].get('protected'):continue
            if cid not in tasks and len(tasks)>=maximum:
                deferred.append(dict(candidate_id=cid,reason='global_joint_segment_budget'));continue
            task=tasks.setdefault(cid,dict(candidate_id=cid,context_ids=[],conflicts=[]))
            if group['candidate_id'] not in task['context_ids'] and len(task['context_ids'])<2:task['context_ids'].append(group['candidate_id'])
            task['conflicts'].append(group['candidate_id'])
        deferred.extend(dict(candidate_id=cid,reason='per_conflict_segment_budget') for cid in related[per_conflict:])
    return list(tasks.values()),deferred

def scoped_update(case,result):
    """Update only this sampled segment; no descendant identity propagation."""
    before=deepcopy(case['judgment']);case['pre_segment_joint_judgment']=before
    case.setdefault('reviews',{})['segment_joint']=result
    case['judgment']=deepcopy(result['judgment']);case['errors']=result['errors']
    # A changed cut needs its own fresh cut check; old verification cannot authorize it.
    case.pop('verification',None);case.pop('verification_errors',None)
    return dict(candidate_id=case['candidate_id'],segment_ids=case['segment_ids'],before=before,after=deepcopy(case['judgment']))

def settle_conflicts(tree,cases,groups):
    index={c['candidate_id']:c for c in cases};out=[]
    for group in groups:
        c=index[group['candidate_id']];result=group['result'];review=result['judgment']
        if group.get('resolved'):
            out.append(dict(candidate_id=c['candidate_id'],status='upstream_proposal_retracted',related_ids=group['related_ids']));continue
        related=[index[cid] for cid in group['related_ids']]
        # Both-sided review must positively revise every blocking segment. Unknown
        # is unresolved, not a confirmed target and not permission to remove it.
        revised=bool(related) and all(t.get('reviews',{}).get('segment_joint') and not t['reviews']['segment_joint']['errors'] and t['judgment']['target_ownership']=='other' and t['judgment']['recommendation']=='KEEP' for t in related)
        same_edge=review.get('cut_edge')==c.get('judgment',{}).get('cut_edge')==c.get('verification',{}).get('cut_edge')
        if revised and review.get('recommendation')=='PROPOSE_CUT' and not result['errors'] and same_edge:
            c['unresolved_conflict']=False
            out.append(dict(candidate_id=c['candidate_id'],status='segment_hypotheses_revised',related_ids=group['related_ids'],note='Scoped model revisions only; executor still checks the entire affected set.'))
        else:
            c['unresolved_conflict']=True
            out.append(dict(candidate_id=c['candidate_id'],status='unresolved_REVIEW',related_ids=group['related_ids'],note='Unresolved does not establish target ownership.'))
    return out

def stricter_citation_errors(j,c,views):
    """Integrity/localization completeness, explicitly not semantic image validation."""
    if not isinstance(j,dict) or j.get('recommendation')!='PROPOSE_CUT':return []
    edge=j.get('cut_edge')
    if not isinstance(edge,dict) or any(type(edge.get(k)) is not int for k in ['parent_id','child_id']):return ['cut_edge_requires_integer_ids']
    option=next((o for o in c['allowed_edges'] if o['child_id']==edge['child_id'] and o['parent_id']==edge['parent_id']),None)
    if option is None:return ['edge_not_allowed']
    endpoints=[option['xyz'],option.get('parent_xyz',option['xyz'])];localized=False
    for e in j.get('evidence',[]):
        v=views.get(e.get('view_id')) if isinstance(e,dict) and isinstance(e.get('view_id'),str) else None
        if not v or not v.get('raw') or v.get('scope')!='local' or v.get('plane') not in ['XZ','YZ']:continue
        lo,hi=v['xyz_bounds']
        if all(all(a<=x<=b for a,x,b in zip(lo,xyz,hi)) for xyz in endpoints):localized=True
    return [] if localized else ['cited_local_depth_view_does_not_contain_selected_edge']
