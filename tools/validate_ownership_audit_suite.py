#!/usr/bin/env python3
"""Check actual exported graphs, component provenance and shared model budgets."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from algorithms.tracing import ownership_audit as oa
from algorithms.tracing.connection_diagnostics import save_json

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(run):
    report=read(run/'diagnosis.json');plan=read(run/'execution_plan.json');summary=read(run/'summary.json')
    original=np.loadtxt(report['input']['swc'],ndmin=2).tolist();kept=np.loadtxt(run/'swc'/f'{summary["cell"]}.swc',ndmin=2).tolist();nodes={int(n[0]):n for n in original};kept_ids={int(n[0]) for n in kept};removed=set(nodes)-kept_ids
    oa.validate_tree(kept);assert all(n==nodes[int(n[0])] for n in kept);assert removed==set(plan['removed_ids']);assert sha(report['input']['swc'])==report['input']['swc_sha256'];assert sha(report['input']['raw'])==report['input']['raw_sha256']
    expected,_=oa.apply_plan(original,plan);assert expected==kept;seen=set();component_checks=[]
    for action in plan['actions']:
        if action['status']!='execute':continue
        cid=action['candidate_id'];part=np.loadtxt(run/'components'/f'{cid}.swc',ndmin=2).tolist();ids={int(n[0]) for n in part};audit=read(run/'components'/f'{cid}_original_parents.json');oa.validate_tree(part)
        assert not (seen&ids);assert ids==set(action['node_ids']);assert not action['hold_reasons']
        for row in part:
            n=int(row[0]);assert row[:6]==nodes[n][:6];assert audit['original_parents'][str(n)]==int(nodes[n][6]);assert int(row[6])==(int(nodes[n][6]) if int(nodes[n][6]) in ids else -1)
        seen|=ids;component_checks.append(dict(candidate_id=cid,nodes=len(ids),root_parent_repaired=True,original_parent_mapping_preserved=True))
        for evidence in action['model_evidence']:
            assert all(Path(evidence[k]).exists() for k in ['request','prompt','response','call_record'])
    assert seen==removed;assert len(list((run/'components').glob('*.swc')))==len(component_checks)
    budgets={c['candidate_id']:sum(p['supplements'] for p in c.get('reviews',{}).values())+sum(g['result']['supplements'] for g in report.get('joint_reviews',[]) if g['candidate_id']==c['candidate_id']) for c in report['cases']}
    assert all(n<=2 for n in budgets.values());assert len(read(run/'model_request_refs.json'))==summary['calls']
    requests=[read(p) for p in read(run/'model_request_refs.json')];assert all(r['image_count']==len(r['images']) for r in requests)
    for req in requests:
        assert all(sha(p['path'])==p['sha256'] for p in req['images'])
    result=dict(run=str(run),cell=summary['cell'],input_nodes=len(original),output_nodes=len(kept),removed_nodes=len(removed),component_checks=component_checks,
        valid_root_parents_acyclic=True,retained_rows_exactly_unchanged=True,removal_set_equals_plan=True,original_input_hashes_unchanged=True,
        max_supplement_rounds=max(budgets.values(),default=0),supplement_budgets=budgets,all_actual_model_image_hashes_match=True,unique_calls=summary['calls'],biological_accuracy_evaluated=False)
    save_json(run/'final_artifact_validation.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();config=read(args.config);base=ROOT.parent/'test/0.84/runs'
    results=[validate(base/item['final_run']) for item in config['experiments']];save_json(Path(config['log_directory'])/'final_suite_validation.json',results);print(json.dumps([{k:v for k,v in r.items() if k!='supplement_budgets'} for r in results],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
