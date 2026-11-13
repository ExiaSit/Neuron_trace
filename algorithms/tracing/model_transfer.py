"""Second-pass distinction between projection overlap and actual trace transfer."""
from algorithms.tracing.model_crossing import tree_digest,tree_paths

def eligible(site):
    return (not site['protected'] and site['direction']['direction_abnormal']
            and site.get('judgment',{}).get('decision')=='switch_to_other'
            and min(site['raw_support']['incoming'],site['raw_support']['outgoing'])>=.55)

def transfer_gate(site,verification):
    reasons=[]
    if not eligible(site):reasons.append('not_a_supported_abnormal_switch_candidate')
    if verification.get('crossing_node_id')!=site['node_id']:reasons.append('wrong_crossing_node')
    if verification.get('trace_transfers_to_distinct_3d_shaft') is not True:reasons.append('trace_transfer_not_confirmed')
    if verification.get('projection_overlap_only') is not False:reasons.append('projection_overlap_not_excluded')
    if verification.get('raw_evidence_sufficient') is not True:reasons.append('raw_evidence_insufficient')
    observations=verification.get('raw_observations',[])
    if not isinstance(observations,list):observations=[]
    planes={v.get('plane') for v in observations if isinstance(v,dict) and isinstance(v.get('observation'),str) and len(v['observation'].strip())>=15}&{'XY','XZ','YZ'}
    if len(planes)<2 or not planes&{'XZ','YZ'}:reasons.append('insufficient_raw_multiview_evidence')
    if verification.get('same_neuron_bend_or_bifurcation_excluded') is not True:reasons.append('normal_bend_or_branch_not_excluded')
    return dict(accepted=not reasons,reasons=reasons,confidence_used=False,
                physical_contact_required=False,straight_reverse_probe_used_as_veto=False)

def apply_verified_transfers(tree,report):
    if tree_digest(tree)!=report['input_sha256']:raise ValueError('Input SWC changed')
    nodes,children,_=tree_paths(tree);removed=set();events=[]
    for site in sorted(report['sites'],key=lambda s:s['root_distance_um']):
        if not site.get('transfer_gate',{}).get('accepted'):continue
        if not transfer_gate(site,site['verification'])['accepted']:raise ValueError('Invalid persisted verification')
        cut=site['cut_node_id']
        if cut not in nodes or int(nodes[cut][6])!=site['node_id']:raise ValueError('Invalid cut edge')
        if cut in removed:continue
        descendants=set();stack=[cut]
        while stack:
            i=stack.pop()
            if i in descendants:raise ValueError('Cycle')
            if int(nodes[i][1])==1 or int(nodes[i][6]) not in nodes:raise ValueError('Protected soma/root')
            descendants.add(i);stack.extend(children[i])
        removed.update(descendants);events.append(dict(site_id=site['site_id'],node_id=site['node_id'],cut_node_id=cut,removed_ids=sorted(descendants)))
    return [n for n in tree if int(n[0]) not in removed],dict(events=events,removed_ids=sorted(removed),input_sha256=report['input_sha256'])
