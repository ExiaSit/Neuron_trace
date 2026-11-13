#!/usr/bin/env python3
"""Reproducible raw-first Gemma attachment audit; consumes existing pre-model SWC."""
from pathlib import Path
import argparse,base64,hashlib,html,json,os,shutil,sys,time,urllib.request,urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Lock
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from algorithms.tracing import ownership_audit as oa
from algorithms.tracing import soma_ownership_context as sc
from algorithms.tracing.connection_diagnostics import save_json,tree_digest
from algorithms.tracing.ownership_consistency import audit_conflict_groups
from visualization.ownership_audit import render_evidence as _render_evidence,render_unfolded
RENDER_LOCK=Lock()
def render_evidence(*args,**kwargs):
    with RENDER_LOCK:return _render_evidence(*args,**kwargs)
from visualization.projection_viewer import write_site_viewer
from visualization.swc_comparison import render_swc_comparison
from tools.run_connection_experiment import metrics,cell_metadata
BASE=ROOT.parent/'test/0.84'
PARAMS=dict(model='gemma4-31b-local',temperature=0,seed=42,max_tokens=2048,cache_prompt=False)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def api(path,endpoint):
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(endpoint+path,timeout=15) as r:return json.load(r)

def stable_model_metadata(models):
    from copy import deepcopy
    result=deepcopy(models)
    for model in result.get('data',[]):model.pop('created',None)
    return result

def runtime_identity(endpoint):
    models=stable_model_metadata(api('/v1/models',endpoint));cmd=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            exe=(p/'exe').resolve()
            if exe.name=='llama-server':
                argv=(p/'cmdline').read_bytes().decode().split('\0')
                if '18484' in argv:cmd=argv;break
        except (OSError,PermissionError):continue
    if not cmd:raise ValueError('Cannot verify local llama-server launch')
    files={}
    for key in ['--model','--mmproj']:
        file=cmd[cmd.index(key)+1];st=Path(file).stat();files[key]=dict(path=file,size=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(file))
    source=json.loads((BASE/'models/gemma-4-31B-it-qat-q4_0/source.json').read_text())
    expected={x['rfilename']:x.get('lfs',{}).get('sha256') for x in source['siblings']}
    for file in files.values():
        if file['sha256']!=expected.get(Path(file['path']).name):raise ValueError('Model file differs from pinned official manifest')
    return dict(models=models,cmdline=cmd,files=files,binary_sha256=sha(cmd[0]),model_source=source)

def request_options(c):
    opts={'serial_more':dict(kind='serial',center_xyz=c['center_xyz']),'wider':dict(kind='wider',center_xyz=c['center_xyz'])}
    opts['unfolded']=dict(kind='unfolded',center_xyz=c['center_xyz'])
    opts.update({s['id']:s for s in c['stations']});return opts

def model_call(prompt,metadata,images,folder,stem,args):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    final=prompt+'\nCASE AND VIEW INVENTORY:\n'+json.dumps(metadata,ensure_ascii=False,separators=(',',':'))
    image_items=[dict(path=str(p),sha256=sha(p)) for p in images]
    identity=dict(final_text=final,images=image_items,parameters=PARAMS,run_fingerprint=args.fingerprint)
    digest=oa.fingerprint(identity);saved=folder/(stem+'_call.json')
    if saved.exists():
        cache=json.loads(saved.read_text())
        if cache['fingerprint']!=digest:raise ValueError('Changed prompt/image/runtime: use a new run')
        return cache
    (folder/(stem+'_prompt.txt')).write_text(final)
    save_json(folder/(stem+'_request.json'),dict(**identity,endpoint=args.endpoint,image_count=len(images),fingerprint=digest))
    content=[dict(type='text',text=final)]
    for p in images:
        content.append(dict(type='text',text='Image file: '+Path(p).name))
        content.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode())))
    body=dict(PARAMS,messages=[dict(role='user',content=content)])
    start=time.monotonic();result=dict(fingerprint=digest,request=str(folder/(stem+'_request.json')),image_count=len(images))
    try:
        req=urllib.request.Request(args.endpoint+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req,timeout=360) as r:response=json.load(r)
        save_json(folder/(stem+'_response.json'),response);result['usage']=response.get('usage');result['timings']=response.get('timings')
        choice=response['choices'][0]
        if choice.get('finish_reason')!='stop':raise ValueError('Truncated model output: '+str(choice.get('finish_reason')))
        text=choice['message'].get('content','').strip()
        if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
        result['judgment']=json.loads(text)
    except (ValueError,KeyError,IndexError,TypeError,AttributeError,OSError,urllib.error.URLError) as e:result['error']=str(e)
    result['seconds']=time.monotonic()-start;save_json(saved,result);return result

def review(c,raw,tree,somas,args,out,mode='ownership',edge=None,joint=None):
    folder=out/'cases'/c['candidate_id']/mode;folder.mkdir(parents=True,exist_ok=True)
    prompt=(ROOT/'prompts'/('ownership_audit_v4.txt' if mode=='ownership' else 'attachment_review_v4.txt')).read_text()
    view_center=next((o['xyz'] for o in c['allowed_edges'] if edge and o['child_id']==edge['child_id']),c['center_xyz'])
    images,views=render_evidence(raw,tree,c,somas,folder,prefix=mode,center=view_center)
    # Stage-one raw-only observation prevents a full colored route from supplying the answer.
    saved_images=list(images)
    if mode in ['ownership','joint']:images=[p for p in images if '_hypothesis.png' not in p]
    else:images=sorted(images,key=lambda p:'_hypothesis.png' in p)
    views={k:v for k,v in views.items() if v['image'] in images}
    opts=request_options(c);opts['hypotheses']=dict(kind='hypotheses',center_xyz=c['center_xyz']);calls=[];used=[];more_count=0
    # Joint case is raw contextual evidence, not previous prose or a claimed truth label.
    if joint:
        for k,t in enumerate(joint[:2]):
            ims,vs=render_evidence(raw,tree,t,somas,folder,prefix=f'joint{k}',global_context=True)
            selected=[p for p in ims if p.endswith('_local_raw.png') or p.endswith('_global_raw.png')]
            images.extend(selected);views.update({key:v for key,v in vs.items() if v['image'] in selected})
            for station in t['stations']:
                if station['kind']=='upstream':opts['related_'+t['candidate_id']+'_'+station['id']]=station
    budget=max(0,2-sum(r['supplements'] for r in c.get('reviews',{}).values()))
    for round_i in range(budget+1):
        inventory={key:{k:v for k,v in value.items() if k not in ['image','window']} for key,value in views.items()}
        metadata=dict(candidate_id=c['candidate_id'],known_target=args.target_info,competing_soma=[s for s in somas if s['soma_id']==c['competing_soma_id']],
            allowed_edges=c['allowed_edges'],sampled_segment_ids=c['segment_ids'],upstream_node_count=len(c['upstream_ids']),
            trust=dict(swc='hypothesis',competitor_route='image_skeleton_hypothesis',other_somas='unconfirmed',manual_paths=[]),
            locator_points=dict(A='upstream sample approximately 8 um before J; unconfirmed',B='outgoing sample approximately 8 um after J; unconfirmed'),views=inventory,request_options={k:v for k,v in opts.items() if k not in used},supplement_rounds_remaining=budget-round_i,
            mode=mode,proposed_edge_to_check=edge,joint_branches=[dict(candidate_id=t['candidate_id'],center_xyz=t['center_xyz'],relationship='downstream hypothesis to inspect, no trusted ownership label') for t in joint or []])
        call=model_call(prompt,metadata,images,folder,f'round{round_i}',args);calls.append(call)
        j=call.get('judgment');errors=[call['error']] if call.get('error') else oa.validate_judgment(j,c,views,{k:v for k,v in opts.items() if k not in used})
        if errors:
            j=dict(candidate_id=c['candidate_id'],**oa.review_result('Invalid/incomplete model output: '+';'.join(errors)));break
        if j['recommendation']!='REQUEST_MORE':break
        if round_i>=budget:
            j=dict(j,recommendation='REVIEW',missing_evidence=j['missing_evidence']+['supplement_budget_exhausted']);break
        new_ids=j['requested_views'][:1] # one focused new observation per round, deterministic budget
        if not new_ids or new_ids[0] in used:
            j=dict(j,recommendation='REVIEW',missing_evidence=j['missing_evidence']+['no_new_view_requested']);break
        key=new_ids[0];used.append(key);opt=opts[key];center=opt['center_xyz'];radius=30 if key=='wider' else 18
        if key=='hypotheses':
            upstream=[s for s in c['stations'] if s['kind']=='upstream' and np.linalg.norm(np.array(s['center_xyz'])-c['center_xyz'])>4]
            if upstream:center=upstream[-1]['center_xyz']
        if key=='unfolded':
            with RENDER_LOCK:im=render_unfolded(raw,c,folder)
            points=np.array(c['upstream_xyz']+c['other_route_xyz']);ims=[im]
            vs={'unfolded_raw':dict(view_id='unfolded_raw',raw=True,scope='route_sampling',plane='unfolded',xyz_bounds=[points.min(0).tolist(),points.max(0).tolist()],image=im,window=[0,180])}
        else:
            offset=3 if key=='serial_more' or (key=='hypotheses' and np.linalg.norm(np.array(center)-c['center_xyz'])<1) else 0
            ims,vs=render_evidence(raw,tree,c,somas,folder,prefix=f'extra{round_i}_{key}',center=center,radius=radius,global_context=key=='hypotheses',slice_offset=offset)
            if key!='hypotheses':ims=[p for p in ims if '_hypothesis.png' not in p]
        # Keep initial raw, annotation and global; replace serial sheet with new local evidence.
        images=[p for p in images if '_slices.png' not in p and 'extra' not in Path(p).name]+ims
        # Cap context at seven images, retain only actually sent view IDs.
        images=images[:7];views.update(vs);views={k:v for k,v in views.items() if v['image'] in images};more_count+=1
    result=dict(judgment=j,errors=errors,calls=calls,supplements=more_count,requested=used,views=views,images=images)
    save_json(folder/'result.json',result);return result

def historical_cases(path):
    if not path:return []
    report=json.loads((path/'diagnosis.json').read_text());cuts=json.loads((path/'cuts.json').read_text());cons=json.loads((path/'consistency.json').read_text());items=[]
    for e in cuts['events']:items.append(dict(node_id=e['cut_node_id'],source='historical_cut',case_id=e['case_id']))
    for n in sorted({e['cut_node_id'] for e in cons['held']}):items.append(dict(node_id=n,source='historical_conflict'))
    for c in report['cases']:
        if c.get('judgment',{}).get('ownership')=='target':items.append(dict(node_id=c['representative_id'],source='historical_target_hypothesis',case_id=c['case_id']))
    # Caller-supplied history is 97117; use prior v2 attention point only if it exists there.
    v2=path.parent/'20261111_97117_gemma_transfer_v2/diagnosis.json'
    if v2.exists():items.append(dict(node_id=1886,source='historical_large_bend'))
    return items

def finalize(out,tree,raw,report,args):
    if any('judgment' not in c for c in report['cases']) or 'conflict_statistics' not in report:
        raise ValueError('Complete candidate and joint review before any pruning export')
    plan=oa.execution_plan(tree,report['cases']);save_json(out/'execution_plan.json',plan)
    # Read saved diagnostics/plan back before any export. Never mutate the original input.
    if sha(args.swc)!=report['input']['swc_sha256']:raise ValueError('SWC changed before execution')
    kept,components=oa.apply_plan(tree,json.loads((out/'execution_plan.json').read_text()))
    swc=out/'swc'/f'{args.cell}.swc';swc.parent.mkdir(exist_ok=True)
    np.savetxt(swc,kept,fmt=['%d','%d','%.17g','%.17g','%.17g','%.17g','%d'])
    for component in components:
        folder=out/'components';folder.mkdir(exist_ok=True);cid=component['candidate_id']
        np.savetxt(folder/f'{cid}.swc',component.pop('tree'),fmt=['%d','%d','%.17g','%.17g','%.17g','%.17g','%d']);save_json(folder/f'{cid}_original_parents.json',component)
    # Export equality/accounting checks include roundtrip serialization.
    loaded=np.loadtxt(swc,ndmin=2).tolist();oa.validate_tree(loaded)
    original={int(n[0]):n for n in tree};assert all(n==original[int(n[0])] for n in loaded)
    assert sha(args.swc)==report['input']['swc_sha256']
    versions=[('v0.84 pre-Gemma',Path(args.swc))]
    if args.history:
        versions.extend([('local v2',BASE/'runs/20261111_97117_gemma_transfer_v2/swc'/f'{args.cell}.swc'),('global v3 checked',Path(args.history)/'swc'/f'{args.cell}.swc')])
    versions.extend([(v,BASE/'data/references'/v/f'{args.cell}.swc') for v in ['v0.81','v0.83b']]);versions=[(k,p) for k,p in versions if p.exists()];versions.append(('raw-first v4',swc))
    render_swc_comparison(raw,versions,out/'comparison.png',title=f'{args.cell}: historical algorithms and new experiment; no manual truth')
    calls=[call for c in report['cases'] for phase in c.get('reviews',{}).values() for call in phase['calls']]
    for joint in report.get('joint_reviews',[]):calls+=joint['result']['calls']
    execute=[a for a in plan['actions'] if a['status']=='execute']
    summary=dict(cell=args.cell,candidates=report['candidate_statistics'],reviewed=sum('judgment' in c and not c.get('protected') for c in report['cases']),
        recommendations=dict(Counter(c['judgment']['recommendation'] for c in report['cases'] if c.get('reviews'))),
        unsent=dict(Counter('protected' if c['protected'] else 'outside_representative_scope' for c in report['cases'] if not c.get('reviews'))),
        raw_first_round_recommendations=dict(Counter(c['reviews']['ownership']['calls'][0].get('judgment',{}).get('recommendation','API_ERROR') for c in report['cases'] if c.get('reviews'))),
        supplements=sum(r['supplements'] for c in report['cases'] for r in c.get('reviews',{}).values())+sum(j['result']['supplements'] for j in report.get('joint_reviews',[])),
        supplement_reasons=[dict(candidate_id=c['candidate_id'],phase=k,requests=r['requested']) for c in report['cases'] for k,r in c.get('reviews',{}).items() if r['requested']]+[dict(candidate_id=j['candidate_id'],phase='joint',requests=j['result']['requested']) for j in report.get('joint_reviews',[]) if j['result']['requested']],
        conflicts=report.get('conflict_statistics',{}),calls=len(calls),model_seconds=sum(c['seconds'] for c in calls),
        call_errors=sum('error' in c for c in calls),executed_edges=[a['edge'] for a in execute],removed_nodes=len(plan['removed_ids']),removed_length_um=sum(a['length_um'] for a in execute),max_action_nodes=max([a['nodes'] for a in execute] or [0]),
        versions={k:metrics(p) for k,p in versions},accuracy=None,ground_truth='unavailable; deletion size and structural validity are not accuracy',scope='complete' if not args.representative else f'representative subset {args.representative}')
    save_json(out/'summary.json',summary);save_json(out/'validation.json',dict(valid_root_parents_acyclic=True,retained_rows_unchanged=True,input_unchanged=True,removal_audit_matches=True,components=len(components),biological_accuracy_evaluated=False))
    render_report(out,raw,tree,report,summary,plan,args)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

def render_report(out,raw,tree,report,summary,plan,args):
    blocks=[];nodes,_,paths=sc.tree_paths(tree);actions={a['candidate_id']:a for a in plan['actions']}
    for c in report['cases']:
        if not c.get('reviews'):continue
        folder=out/'cases'/c['candidate_id'];p=np.array(c['center_xyz']);lo=np.maximum(0,np.floor(p-24)).astype(int);hi=np.minimum(raw.shape[::-1],np.ceil(p+25)).astype(int)
        crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
        arms=[dict(xyz=[nodes[n][2:5] for n in path],diagnosis=dict(status='hypothesis')) for path in paths if any(np.all((np.array(nodes[n][2:5])>=lo)&(np.array(nodes[n][2:5])<hi)) for n in path)]
        site=dict(site_id=c['candidate_id'],kind='ownership_audit',status=c['judgment']['recommendation'],center_xyz=p.tolist(),image='ownership/ownership_local_raw.png',bbox_xyz=np.array([lo,hi]).T.tolist(),arms=arms,context_xyz=[],pair_xyz=[])
        write_site_viewer(crop,lo,site,folder/'interactive.html')
        images=list(dict.fromkeys(p for r in c['reviews'].values() for p in r['images']))
        imgs=''.join(f'<a href="{Path(p).relative_to(out)}"><img loading="lazy" src="{Path(p).relative_to(out)}"></a>' for p in images)
        actual=actions.get(c['candidate_id']);body=dict(candidate=c['candidate_id'],sources=c['sources'],model=c.get('judgment'),repeat_review=c.get('verification'),execution=actual,unresolved_conflict=c.get('unresolved_conflict',False))
        blocks.append(f'<section id="case-{c["candidate_id"]}"><h2>{c["candidate_id"]} · edge to {c["anchor_id"]} · {c["judgment"]["recommendation"]}</h2><a href="cases/{c["candidate_id"]}/interactive.html">交互原图：切片/薄层/窗口/骨架</a><pre>{html.escape(json.dumps(body,ensure_ascii=False,indent=2))}</pre><div class="grid">{imgs}</div></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Gemma raw-first attachment audit</title><style>body{font:16px system-ui;margin:30px;background:#f5f6f7}section{background:white;padding:20px;margin:20px 0}pre{white-space:pre-wrap}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{max-width:100%}</style>'+f'<h1>{args.cell} · 原图优先连接审核 v4</h1><p>Gemma 判断、程序执行、未解决病例分别记录。无人工真值，准确率未评估。同一模型两轮检查不是独立模型验证。</p><p><a href="diagnosis.json">诊断</a> · <a href="execution_plan.json">执行及影响范围</a> · <a href="summary.json">统计</a> · <a href="joint_reviews.json">联合复核</a> · <a href="swc/{args.cell}.swc">实验 SWC</a></p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><img src="comparison.png">'+''.join(blocks))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cell',required=True);ap.add_argument('--swc',required=True);ap.add_argument('--run-name',required=True);ap.add_argument('--history',type=Path);ap.add_argument('--endpoint',default='http://127.0.0.1:18484');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--limit',type=int);ap.add_argument('--representative',type=int,default=0);ap.add_argument('--finalize-only',action='store_true');ap.add_argument('--workers',type=int,default=1);args=ap.parse_args()
    if Path(args.run_name).name!=args.run_name:ap.error('run-name must be a directory name')
    out=BASE/'runs'/args.run_name;out.mkdir(parents=True,exist_ok=True);args.swc=str(Path(args.swc).resolve())
    rawpath=BASE/'data/inputs/raw'/f'{args.cell}.tif';raw=tifffile.imread(rawpath);tree=np.loadtxt(args.swc,ndmin=2).tolist();oa.validate_tree(tree)
    meta,mp=cell_metadata(BASE/'data',args.cell);somas,target=oa.target_from_metadata(meta,raw,tree);args.target_info=target
    files=['tools/run_soma_ownership_pipeline.py','tools/report_ownership_audit.py','tools/run_ownership_audit.py','algorithms/tracing/ownership_audit.py','algorithms/tracing/ownership_consistency.py','algorithms/tracing/soma_ownership_context.py','algorithms/tracing/model_crossing.py','algorithms/tracing/connection_diagnostics.py','algorithms/tracing/signal_support.py','visualization/ownership_audit.py','visualization/soma_ownership_context.py','visualization/projection_viewer.py','visualization/assets/projection_viewer.html','prompts/ownership_audit_v4.txt','prompts/attachment_review_v4.txt','algorithms/tracing/prune.py','algorithms/tracing/terminal_pruning.py']
    identity=dict(version=oa.VERSION,cell=args.cell,swc=args.swc,swc_sha256=sha(args.swc),raw=str(rawpath),raw_sha256=sha(rawpath),metadata_sha256=sha(mp),code={p:sha(ROOT/p) for p in files},parameters=PARAMS,views=dict(radius_um=18,wider_um=30,window=[0,180],slice_step_um=1,image_min_tokens=1120,image_max_tokens=1120,dpi=200),representative=args.representative,workers=args.workers,history=str(args.history) if args.history else None)
    ip=out/'input.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Input/config/source changed: use a new run directory')
    save_json(ip,identity)
    for p in files:
        dst=out/'source'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
    dp=out/'diagnosis.json'
    if dp.exists():report=json.loads(dp.read_text())
    else:
        graph=sc.graph_from_image(raw,somas);np.savez_compressed(out/'graph_routes.npz',**graph)
        cases,stats=oa.generate_candidates(tree,raw,somas,graph,historical_cases(args.history));report=dict(input=identity,target=target,somas=somas,cases=cases,candidate_statistics=stats)
        # Explicit representative sampling covers each source and includes ordinary forks.
        if args.representative:
            selected=[];seen=set()
            for c in sorted(cases,key=lambda c:('soma_competition' in c['sources'],c['anchor_id'])):
                if c['protected']:continue
                if any(s not in seen for s in c['sources']):selected.append(c['candidate_id']);seen.update(c['sources'])
            selected=list(dict.fromkeys(selected+[c['candidate_id'] for c in cases if not c['protected']]))[:args.representative];report['selected_ids']=selected
        save_json(dp,report)
    print(json.dumps(report['candidate_statistics']),flush=True)
    if args.prepare_only:return
    runtime=runtime_identity(args.endpoint);rp=out/'runtime.json'
    if rp.exists() and json.loads(rp.read_text())!=runtime:raise ValueError('Model/runtime changed; new run required')
    save_json(rp,runtime);args.fingerprint=oa.fingerprint(dict(input=identity,runtime=runtime));reviewed=0
    if not args.finalize_only:
        def process(c):
            c['reviews']={};r=review(c,raw,tree,somas,args,out);c['reviews']['ownership']=r;c['judgment']=r['judgment'];c['errors']=r['errors']
            if c['judgment']['recommendation']=='PROPOSE_CUT':
                r=review(c,raw,tree,somas,args,out,mode='verification',edge=c['judgment']['cut_edge']);c['reviews']['verification']=r;c['verification']=r['judgment'];c['verification_errors']=r['errors']
            return c
        pending=[]
        for c in report['cases']:
            if c.get('judgment'):continue
            if c['protected'] or ('selected_ids' in report and c['candidate_id'] not in report['selected_ids']):
                c['judgment']=dict(candidate_id=c['candidate_id'],**oa.review_result('protected_target_soma' if c['protected'] else 'not_selected_representative_scope'));continue
            pending.append(c)
        if args.limit:pending=pending[:args.limit]
        # Worker returns a private case copy; only the main thread persists the shared report.
        from copy import deepcopy
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(process,deepcopy(c)) for c in pending]
            for future in as_completed(futures):
                result=future.result();c=next(c for c in report['cases'] if c['candidate_id']==result['candidate_id']);c.update(result)
                save_json(dp,report);reviewed+=1
                print(json.dumps(dict(case=c['candidate_id'],node=c['anchor_id'],sources=c['sources'],judgment=c['judgment']['recommendation'],owner=c['judgment']['target_ownership'],errors=c['errors']),ensure_ascii=False),flush=True)
        if args.limit:return
        groups=audit_conflict_groups(tree,report['cases'])
        if args.history:
            old=json.loads((args.history/'diagnosis.json').read_text());held=json.loads((args.history/'consistency.json').read_text())['held']
            seen={g['cut_edge']['child_id'] for g in groups}
            for h in held:
                n=h['cut_node_id']
                if n in seen:continue
                c=next((c for c in report['cases'] if any(o['child_id']==n for o in c['allowed_edges'])),None)
                if c is None:continue
                targets=[x['representative_id'] for x in old['cases'] if x['case_id'] in h['conflicting_target_cases']]
                related=[t['candidate_id'] for t in report['cases'] if any(n in t['segment_ids'] for n in targets)]
                edge=next({k:o[k] for k in ['parent_id','child_id']} for o in c['allowed_edges'] if o['child_id']==n)
                groups.append(dict(candidate_id=c['candidate_id'],cut_edge=edge,related_ids=related,potential_nodes=h['potential_removed_nodes'],origin='historical_untrusted_conflict'));seen.add(n)
        # One bounded joint case per candidate. Adjacent historical/current cut
        # options share its local field; preserve every attention edge in audit.
        grouped={}
        for group in groups:
            key=group['candidate_id']
            if key not in grouped:grouped[key]=dict(group,attention_edges=[group['cut_edge']])
            else:
                grouped[key]['related_ids']=list(dict.fromkeys(grouped[key]['related_ids']+group['related_ids']))
                grouped[key]['attention_edges'].append(group['cut_edge'])
        groups=list(grouped.values())
        report['joint_reviews']=report.get('joint_reviews',[]);done={j['candidate_id'] for j in report['joint_reviews']}
        for group in groups:
            c=next(c for c in report['cases'] if c['candidate_id']==group['candidate_id'])
            if c['candidate_id'] in done:continue
            related=[t for t in report['cases'] if t['candidate_id'] in group['related_ids']]
            before=dict(judgment=c['judgment'],verification=c.get('verification'))
            r=review(c,raw,tree,somas,args,out,mode='joint',edge=group['cut_edge'],joint=related)
            resolved=r['judgment']['recommendation']=='KEEP' and not r['errors']
            if resolved:c['pre_joint_judgment']=c['judgment'];c['judgment']=r['judgment']
            else:c['unresolved_conflict']=True
            report['joint_reviews'].append(dict(**group,before=before,result=r,resolved=resolved,resolution='upstream_proposal_retracted_not_target_truth' if resolved else 'unresolved_preserve_attachment'))
            done.add(c['candidate_id']);save_json(dp,report)
        report['conflict_statistics']=dict(new=len(groups),resolved=sum(j['resolved'] for j in report['joint_reviews']),unresolved=sum(not j['resolved'] for j in report['joint_reviews']))
        save_json(out/'joint_reviews.json',report['joint_reviews']);save_json(dp,report)
    finalize(out,tree,raw,report,args)

if __name__=='__main__':main()
