#!/usr/bin/env python3
"""Recheck downstream segments of joint cases, then reapply from pre-Gemma input.

Parent evidence remains immutable and is hash checked before reuse. This phase
never treats a prior target judgment as a permanent truth or assigns descendants.
"""
from pathlib import Path
import argparse,hashlib,html,json,os,shutil,sys
from collections import Counter
from copy import deepcopy
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from tools import run_ownership_audit as runner
from algorithms.tracing import ownership_audit as oa
from algorithms.tracing import ownership_joint_review as joint
from algorithms.tracing.connection_diagnostics import save_json
from visualization.swc_comparison import render_swc_comparison
BASE=ROOT.parent/'test/0.84'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return runner.sha(p)

def check_parent(parent):
    report=read(parent/'diagnosis.json')
    if not (parent/'validation.json').exists() or any('judgment' not in c for c in report['cases']):raise ValueError('Finish parent experiment first')
    for key in ['swc','raw']:
        if sha(report['input'][key])!=report['input'][key+'_sha256']:raise ValueError('Changed '+key)
    _,metadata_path=runner.cell_metadata(BASE/'data',report['input']['cell'])
    if sha(metadata_path)!=report['input']['metadata_sha256']:raise ValueError('Changed target metadata')
    for path,digest in report['input']['code'].items():
        if sha(ROOT/path)!=digest or sha(parent/'source'/path)!=digest:raise ValueError('Changed parent implementation: '+path)
    for request in parent.glob('cases/*/*/round*_request.json'):
        r=read(request)
        saved_text=request.with_name(request.name.replace('_request.json','_prompt.txt'))
        if saved_text.read_text()!=r['final_text']:raise ValueError('Changed expanded prompt')
        identity={key:r[key] for key in ['final_text','images','parameters','run_fingerprint']}
        if oa.fingerprint(identity)!=r['fingerprint']:raise ValueError('Changed request identity')
        if r['image_count']!=len(r['images']):raise ValueError('Image inventory mismatch')
        for image in r['images']:
            if sha(image['path'])!=image['sha256']:raise ValueError('Changed image; do not reuse judgments: '+image['path'])
    return report

def review_segment(c,contexts,raw,tree,somas,out,args,previous_supplements=0,verification=False,previous_requests=()):
    folder=out/'cases'/c['candidate_id']/('segment_verify' if verification else 'segment_joint');folder.mkdir(parents=True,exist_ok=True)
    edge=c['judgment'].get('cut_edge') if verification else None
    center=next((e['xyz'] for e in c['allowed_edges'] if edge and e['child_id']==edge['child_id'] and e['parent_id']==edge['parent_id']),c['center_xyz'])
    images,views=runner.render_evidence(raw,tree,c,somas,folder,prefix='segment',global_context=True,center=center)
    images=[p for p in images if '_hypothesis.png' not in p];views={k:v for k,v in views.items() if v['image'] in images}
    for i,peer in enumerate(contexts[:2]):
        ims,vs=runner.render_evidence(raw,tree,peer,somas,folder,prefix=f'peer{i}',global_context=True)
        selected=[p for p in ims if p.endswith('_local_raw.png') or p.endswith('_global_raw.png')]
        images+=selected;views.update({k:v for k,v in vs.items() if v['image'] in selected})
    opts={k:v for k,v in runner.request_options(c).items() if k not in previous_requests};used=[];calls=[];supplements=0;budget=max(0,2-previous_supplements)
    # Full path/station records stay in diagnosis. A compact menu avoids wasting
    # context on repeated IDs and full-precision arc-length decimals.
    for round_i in range(budget+1):
        metadata=dict(candidate_id=c['candidate_id'],mode='joint_segment_verification' if verification else 'joint_segment',known_target=args.target_info,
            competing_soma=[s for s in somas if s['soma_id']==c['competing_soma_id']],allowed_edges=c['allowed_edges'],sampled_segment_ids=c['segment_ids'],proposed_edge_to_check=edge,
            segment_scope='Judge ONLY the named candidate segment. In joint_segment mode KEEP means this segment attachment is supported, not that the peer upstream attachment is supported. Do not assign descendants or copy a peer identity.',
            peer_context=[dict(candidate_id=p['candidate_id'],center_xyz=p['center_xyz'],role='upstream junction' if p['anchor_id'] in c['upstream_ids'] else 'related segment') for p in contexts],
            views={k:{a:b for a,b in v.items() if a not in ['image','window']} for k,v in views.items()},
            request_options={k:dict(center_xyz=v['center_xyz'],kind=v['kind']) for k,v in opts.items() if k not in used},supplement_rounds_remaining=budget-round_i,
            trust='No old model conclusions are supplied; all SWC and graph routes are hypotheses, no confirmed paths.')
        result=runner.model_call((ROOT/'prompts/attachment_review_v4.txt').read_text(),metadata,images,folder,f'round{round_i}',args);calls.append(result)
        j=result.get('judgment');errors=[result['error']] if result.get('error') else oa.validate_judgment(j,c,views,{k:v for k,v in opts.items() if k not in used})
        if not errors:errors=joint.stricter_citation_errors(j,c,views)
        if errors:j=dict(candidate_id=c['candidate_id'],**oa.review_result('Joint segment output incomplete: '+';'.join(errors)));break
        if j['recommendation']!='REQUEST_MORE':break
        if round_i==budget:j=dict(j,recommendation='REVIEW',missing_evidence=j['missing_evidence']+['supplement_budget_exhausted']);break
        key=j['requested_views'][0];used.append(key);opt=opts[key]
        if key=='unfolded':
            with runner.RENDER_LOCK:im=runner.render_unfolded(raw,c,folder)
            xyz=np.array(c['upstream_xyz']+c['other_route_xyz']);ims=[im];vs={'unfolded_raw':dict(view_id='unfolded_raw',raw=True,scope='route_sampling',plane='unfolded',xyz_bounds=[xyz.min(0).tolist(),xyz.max(0).tolist()],image=im)}
        else:
            ims,vs=runner.render_evidence(raw,tree,c,somas,folder,prefix=f'new{round_i}_{key}',center=opt['center_xyz'],radius=30 if key=='wider' else 18,global_context=False,slice_offset=3 if key=='serial_more' else 0)
            ims=[p for p in ims if '_hypothesis.png' not in p]
        # Retain the main junction and both peer local fields; spend the remaining
        # slots on newly observed slices/fields. All discarded view IDs are removed.
        base=[p for p in images if p.endswith('_local_raw.png') or p.endswith('segment_global_raw.png')]
        images=list(dict.fromkeys(base+ims))[-7:];views.update(vs);views={k:v for k,v in views.items() if v['image'] in images};supplements+=1
    r=dict(judgment=j,errors=errors,calls=calls,supplements=supplements,requested=used,images=images,views=views);save_json(folder/'result.json',r);return r

def all_calls(report):
    by_request={}
    for c in report['cases']:
        for r in c.get('reviews',{}).values():
            for call in r['calls']:by_request[call['request']]=call
    for group in report.get('joint_reviews',[]):
        for call in group['result']['calls']:by_request[call['request']]=call
    return list(by_request.values())

def export(out,parent,report,tree,raw,args):
    # Every effect is recalculated from the original pre-model tree, never from
    # the parent's already-pruned SWC.
    plan=oa.execution_plan(tree,report['cases'])
    cases_by_id={c['candidate_id']:c for c in report['cases']}
    for action in plan['actions']:
        c=cases_by_id[action['candidate_id']]
        calls=[call for phase in c.get('reviews',{}).values() for call in phase['calls']]+[call for group in report.get('joint_reviews',[]) if group['candidate_id']==c['candidate_id'] for call in group['result']['calls']]
        action['model_evidence']=[dict(request=call['request'],prompt=call['request'].replace('_request.json','_prompt.txt'),response=call['request'].replace('_request.json','_response.json'),call_record=call['request'].replace('_request.json','_call.json')) for call in calls]
        action['scope_note']='Disconnect this component only; no claim that all affected structure belongs to another neuron.'
    save_json(out/'execution_plan.json',plan)
    save_json(out/'diagnosis.json',report)
    if sha(report['input']['swc'])!=report['input']['swc_sha256']:raise ValueError('Input changed before execution')
    kept,components=oa.apply_plan(tree,read(out/'execution_plan.json'));(out/'swc').mkdir(exist_ok=True)
    target=out/'swc'/f'{args.cell}.swc';fmt=['%d','%d','%.17g','%.17g','%.17g','%.17g','%d'];np.savetxt(target,kept,fmt=fmt)
    for component in components:
        directory=out/'components';directory.mkdir(exist_ok=True);cid=component['candidate_id'];np.savetxt(directory/f'{cid}.swc',component.pop('tree'),fmt=fmt);save_json(directory/f'{cid}_original_parents.json',component)
    loaded=np.loadtxt(target,ndmin=2).tolist();oa.validate_tree(loaded);original={int(n[0]):n for n in tree};assert all(n==original[int(n[0])] for n in loaded)
    versions=[(k,Path(v['path'])) for k,v in read(parent/'summary.json')['versions'].items() if k!='raw-first v4'];versions.append(('raw-first v4 + segment joint review',target))
    render_swc_comparison(raw,versions,out/'comparison.png',title=f'{args.cell}: original input, historical outputs and scoped joint review')
    calls=all_calls(report);execute=[a for a in plan['actions'] if a['status']=='execute'];old=read(parent/'summary.json')
    summary=dict(old,reviewed=sum(bool(c.get('reviews')) for c in report['cases']),versions={k:runner.metrics(p) for k,p in versions},recommendations=dict(Counter(c['judgment']['recommendation'] for c in report['cases'] if c.get('reviews'))),
        calls=len(calls),new_segment_calls=sum(len(c.get('reviews',{}).get('segment_joint',{}).get('calls',[]))+len(c.get('reviews',{}).get('segment_verify',{}).get('calls',[])) for c in report['cases']),
        model_seconds=sum(c['seconds'] for c in calls),call_errors=sum('error' in c for c in calls),executed_edges=[a['edge'] for a in execute],removed_nodes=len(plan['removed_ids']),removed_length_um=sum(a['length_um'] for a in execute),max_action_nodes=max([a['nodes'] for a in execute] or [0]),
        conflicts=dict(parent=old['conflicts'],current_model_conflict_groups=sum(g.get('origin')!='historical_untrusted_conflict' for g in report.get('joint_reviews',[])),historical_attention_groups=sum(g.get('origin')=='historical_untrusted_conflict' for g in report.get('joint_reviews',[])),segment_rechecks=len(report['segment_updates']),final=dict(Counter(x['status'] for x in report['joint_settlement'])),deferred_segments=report['deferred_segments'],new_after_segment_review=report.get('new_conflicts_after_segments',[])),
        supplements=old['supplements']+sum(r['supplements'] for c in report['cases'] for k,r in c.get('reviews',{}).items() if k in ['segment_joint','segment_verify']),
        supplement_reasons=old['supplement_reasons']+[dict(candidate_id=c['candidate_id'],phase=k,requests=r['requested']) for c in report['cases'] for k,r in c.get('reviews',{}).items() if k in ['segment_joint','segment_verify'] and r['requested']],
        execution_status_counts=dict(Counter(a['status'] for a in plan['actions'])),program_holds=dict(Counter(reason for a in plan['actions'] for reason in a['hold_reasons'])),
        localization_integrity_holds=report.get('integrity_holds',[]),
        parent_run=str(parent),accuracy=None)
    save_json(out/'summary.json',summary);save_json(out/'model_request_refs.json',[c['request'] for c in calls]);save_json(out/'validation.json',dict(valid_root_parents_acyclic=True,retained_rows_unchanged=True,original_input_unchanged=sha(report['input']['swc'])==report['input']['swc_sha256'],components=len(components),removed_ids_match_audit=True,biological_accuracy_evaluated=False))
    blocks=[]
    for c in report['cases']:
        if not c.get('reviews'):continue
        # Human viewers are rendered locally from the same input in delivery pass.
        folder=out/'cases'/c['candidate_id'];folder.mkdir(parents=True,exist_ok=True)
        image_paths=list(dict.fromkeys(p for r in c['reviews'].values() for p in r['images']))
        pictures=''.join(f'<a href="{os.path.relpath(p,out)}"><img loading="lazy" src="{os.path.relpath(p,out)}"></a>' for p in image_paths)
        payload=dict(judgment=c['judgment'],before_segment_review=c.get('pre_segment_joint_judgment'),repeat=c.get('verification'),format_or_localization_errors=c.get('errors',[]),repeat_errors=c.get('verification_errors',[]),execution=[a for a in plan['actions'] if a['candidate_id']==c['candidate_id']])
        blocks.append(f'<section id="case-{c["candidate_id"]}"><h2>{c["candidate_id"]} / node {c["anchor_id"]}</h2><p><a href="cases/{c["candidate_id"]}/interactive.html">原始局部交互图</a></p><pre>{html.escape(json.dumps(payload,ensure_ascii=False,indent=2))}</pre><div class="grid">{pictures}</div></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Scoped joint review</title><style>body{font:16px system-ui;margin:30px}pre{white-space:pre-wrap}img{max-width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}section{margin-top:30px}</style><h1>双方局部段联合复核</h1><p>所有剪断从模型介入前的同一输入重新计算。旧 target 可被复核，但新结论不传播到整棵子树。没有人工真值，准确率未评估。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><img src="comparison.png">'+''.join(blocks))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent-run',type=Path,required=True);ap.add_argument('--run-name',required=True);ap.add_argument('--endpoint',default='http://127.0.0.1:18484');ap.add_argument('--max-segment-reviews',type=int,default=12);args=ap.parse_args()
    if Path(args.run_name).name!=args.run_name:ap.error('Invalid run name')
    parent=args.parent_run.resolve();parent_report=check_parent(parent);out=BASE/'runs'/args.run_name;out.mkdir(parents=True,exist_ok=True)
    args.cell=parent_report['input']['cell'];args.target_info=parent_report['target'];runtime=runner.runtime_identity(args.endpoint)
    if runtime!=read(parent/'runtime.json'):raise ValueError('Model/runtime changed; do not combine different services in joint refinement')
    sources=['tools/refine_ownership_conflicts.py','algorithms/tracing/ownership_joint_review.py','tools/run_ownership_audit.py','algorithms/tracing/ownership_audit.py','visualization/ownership_audit.py','prompts/ownership_audit_v4.txt','prompts/attachment_review_v4.txt']
    history=parent_report['input'].get('history')
    history_files={str(p.resolve()):sha(p) for p in Path(history).glob('*.json')} if history else {}
    identity=dict(parent=str(parent),parent_diagnosis_sha256=sha(parent/'diagnosis.json'),parent_input=parent_report['input'],history_reference_sha256=history_files,code={p:sha(ROOT/p) for p in sources},runtime=runtime,maximum=args.max_segment_reviews)
    if (out/'stage_identity.json').exists() and read(out/'stage_identity.json')!=identity:raise ValueError('Stage identity changed')
    save_json(out/'stage_identity.json',identity);save_json(out/'runtime.json',runtime);args.fingerprint=oa.fingerprint(identity)
    for p in sources:
        dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    report=read(out/'diagnosis.json') if (out/'diagnosis.json').exists() else deepcopy(parent_report)
    tree=np.loadtxt(report['input']['swc'],ndmin=2).tolist();raw=tifffile.imread(report['input']['raw']);somas=report['somas'];index={c['candidate_id']:c for c in report['cases']}
    # Retroactive localization/integrity checks apply before any final-stage cut.
    if 'integrity_holds' not in report:
        holds=[]
        for c in report['cases']:
            for key,phase in [('judgment','ownership'),('verification','verification')]:
                if key not in c or phase not in c.get('reviews',{}):continue
                extra=joint.stricter_citation_errors(c[key],c,c['reviews'][phase]['views'])
                if extra:
                    holds.append(dict(candidate_id=c['candidate_id'],phase=phase,reasons=extra));c['errors' if key=='judgment' else 'verification_errors']=extra
        report['integrity_holds']=holds
    if 'segment_tasks' not in report:
        tasks,deferred=joint.segment_review_tasks(report,args.max_segment_reviews);report['segment_tasks']=tasks;report['deferred_segments']=deferred
        save_json(out/'diagnosis.json',report)
    tasks=report['segment_tasks'];deferred=report['deferred_segments'];report.setdefault('segment_updates',[]);done={x['candidate_id'] for x in report['segment_updates']}
    for task in tasks:
        if task['candidate_id'] in done:continue
        c=index[task['candidate_id']];contexts=[index[i] for i in task['context_ids']]
        spent=sum(r['supplements'] for r in c.get('reviews',{}).values())+sum(g['result']['supplements'] for g in report.get('joint_reviews',[]) if g['candidate_id']==c['candidate_id'])
        previous_requests=[k for r in c.get('reviews',{}).values() for k in r['requested']]+[k for g in report.get('joint_reviews',[]) if g['candidate_id']==c['candidate_id'] for k in g['result']['requested']]
        result=review_segment(c,contexts,raw,tree,somas,out,args,spent,previous_requests=previous_requests);update=joint.scoped_update(c,result);update['context_ids']=task['context_ids']
        if c['judgment']['recommendation']=='PROPOSE_CUT':
            verification=review_segment(c,contexts,raw,tree,somas,out,args,spent+result['supplements'],verification=True,previous_requests=previous_requests+result['requested']);c['reviews']['segment_verify']=verification;c['verification']=verification['judgment'];c['verification_errors']=verification['errors']
        report['segment_updates'].append(update);save_json(out/'diagnosis.json',report);print(json.dumps(update,ensure_ascii=False),flush=True)
    report['joint_settlement']=joint.settle_conflicts(tree,report['cases'],report.get('joint_reviews',[]))
    reviewed_pairs={(g['candidate_id'],t) for g in report.get('joint_reviews',[]) for t in g['related_ids']}
    remaining=runner.audit_conflict_groups(tree,report['cases'])
    report['new_conflicts_after_segments']=[g for g in remaining if any((g['candidate_id'],t) not in reviewed_pairs for t in g['related_ids'])]
    for g in report['new_conflicts_after_segments']:index[g['candidate_id']]['unresolved_conflict']=True
    save_json(out/'joint_reviews.json',dict(upstream=report.get('joint_reviews',[]),segment_updates=report['segment_updates'],settlement=report['joint_settlement'],deferred=deferred,new_unresolved=report['new_conflicts_after_segments']));save_json(out/'diagnosis.json',report)
    export(out,parent,report,tree,raw,args)
if __name__=='__main__':main()
