"""Hold subtree cuts that contradict an explicit downstream target-ownership review."""
from copy import deepcopy
from algorithms.tracing.signal_support import tree_paths
from algorithms.tracing.connection_diagnostics import tree_digest


def reconcile_target_conflicts(tree,report):
    if tree_digest(tree)!=report['input_sha256']:raise ValueError('Changed input')
    result=deepcopy(report);nodes,children,_=tree_paths(tree)
    target_cases=[c for c in result['cases'] if c.get('judgment',{}).get('ownership')=='target' and c['judgment'].get('soma_id')==0]
    held=[]
    for c in result['cases']:
        if not c.get('gate',{}).get('accepted'):continue
        cut=c['judgment']['cut_node_id'];descendants=set();stack=[cut]
        while stack:
            n=stack.pop()
            if n in descendants:raise ValueError('Cyclic tree')
            descendants.add(n);stack.extend(children[n])
        conflicts=[t['case_id'] for t in target_cases if any(n in descendants for n in t['supported_run_ids'])]
        if not conflicts:continue
        c['pre_consistency_gate']=c['gate'];c['consistency_conflicts']=conflicts
        c['gate']=dict(accepted=False,reasons=['unresolved_downstream_target_ownership_conflict'])
        c['status']='held_for_target_ownership_conflict'
        held.append(dict(case_id=c['case_id'],cut_node_id=cut,conflicting_target_cases=conflicts,potential_removed_nodes=len(descendants)))
    result['consistency']=dict(policy='hold_ancestor_cut_when_model_labels_downstream_target',held=held,
        note='Contradictory model judgments are unresolved, not converted into a confirmed target identity. Unrelated accepted subtree cuts remain eligible.')
    return result


def audit_conflict_groups(tree,cases):
    """V4: schedule bounded joint review, without promoting model labels to truth."""
    from algorithms.tracing.ownership_audit import proposed,descendants
    nodes,children,_=tree_paths(tree);groups=[];seen=set()
    for c in cases:
        if not proposed(c):continue
        cut=c['judgment']['cut_edge']['child_id'];ids=descendants(children,cut)
        related=[t['candidate_id'] for t in cases if t['candidate_id']!=c['candidate_id'] and set(t['segment_ids'])&ids and t.get('judgment',{}).get('target_ownership')=='target']
        key=(cut,tuple(related))
        if related and key not in seen:
            seen.add(key);groups.append(dict(candidate_id=c['candidate_id'],cut_edge=c['judgment']['cut_edge'],related_ids=related,potential_nodes=len(ids)))
    return groups
