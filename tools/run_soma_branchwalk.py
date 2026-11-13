#!/usr/bin/env python3
"""Gemma-only pruning: visit every reachable soma-rooted junction, look ahead 3 branches."""
from pathlib import Path
import argparse,base64,html,json,shutil,sys,time
from collections import Counter
from types import SimpleNamespace
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from algorithms.tracing import soma_branchwalk as walk
from algorithms.tracing.connection_diagnostics import save_json
from visualization.soma_branchwalk import render
from tools.run_ownership_audit import model_call,runtime_identity,sha,PARAMS
from tools.run_connection_experiment import metrics,cell_metadata
from visualization.swc_comparison import render_swc_comparison
BASE=ROOT.parent/'test/0.84'

def read(p):return json.loads(Path(p).read_text())

def compact_inventory(views):
    """Keep one shared XYZ table; never repeat it for each projection."""
    landmarks={};inventory={}
    for key,v in views.items():
        for label,point in v['landmarks'].items():
            xyz=point['xyz_um']
            if label in landmarks and landmarks[label]!=xyz:raise ValueError('Inconsistent landmark XYZ')
            landmarks[label]=xyz
        inventory[key]=dict(raw=v['raw'],scope=v['scope'],plane=v['plane'],xyz_bounds=v['xyz_bounds'],landmark_ids=list(v['landmarks']))
    return landmarks,inventory

def options(case):
    opts={'more_slices':dict(center_xyz=case['center_xyz'],offset=3)}
    for b in case['branches']:
        for seg in b['future']:opts['forward_'+seg['branch_id']]=dict(center_xyz=seg['xyz'][len(seg['xyz'])//2],offset=0)
    for i in range(0,len(case['upstream_xyz']),24):opts['upstream_'+str(case['upstream_ids'][i])]=dict(center_xyz=case['upstream_xyz'][i],offset=0)
    return opts

def review(case,raw,target,out,args,history):
    folder=out/'cases'/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
    images,views=render(raw,case,target,folder);initial=list(images);opts=options(case);used=[];calls=[]
    prompt=(ROOT/'prompts/soma_branchwalk_v1.txt').read_text()
    for round_i in range(3):
        branches=[dict(branch_id=b['branch_id'],edge=dict(parent_id=b['parent_id'],child_id=b['child_id']),edge_xyz=b['xyz'][:2],future=[dict(branch_id=s['branch_id'],level=s['level'],start_xyz=s['xyz'][0],end_xyz=s['xyz'][-1],length_um=round(s['length_um'],2),end_node=s['end_id']) for s in b['future']]) for b in case['branches']]
        landmarks,inventory=compact_inventory(views)
        metadata=dict(junction_id=case['junction_id'],case_id=case['case_id'],junction_xyz=case['center_xyz'],known_target_soma=target,lookahead_branch_segments=3,
            outlet_groups_note='Every immediate outlet is checked; this is one group of up to three.',outlets=branches,upstream_nodes=len(case['upstream_ids']),upstream_history=history,
            path_trust='All SWC paths and all prior model decisions remain hypotheses.',landmarks_xyz_um=landmarks,projection_mapping=dict(XY=['X','Y','depth Z'],XZ=['X','Z','depth Y'],YZ=['Y','Z','depth X']),views=inventory,request_options={k:v for k,v in opts.items() if k not in used},supplement_rounds_remaining=2-round_i)
        call=model_call(prompt,metadata,images,folder,f'round{round_i}',args);calls.append(call)
        if call.get('error') and not Path(call['request'].replace('_request.json','_response.json')).exists():
            save_json(folder/'transport_failure.json',call);raise RuntimeError('Model service failed; traversal is incomplete, no final pruning export: '+call['error'])
        j=call.get('judgment');errors=[call['error']] if call.get('error') else walk.validate_response(j,case,views,{k:v for k,v in opts.items() if k not in used})
        if errors:j=walk.review_decisions(case,'Invalid model output: '+'; '.join(errors));break
        if not j['requested_views']:break
        if round_i==2:
            j=walk.review_decisions(case,'Supplement budget exhausted; connection retained');break
        key=j['requested_views'][0];used.append(key);opt=opts[key]
        ims,vs=render(raw,case,target,folder,prefix='extra'+str(round_i),focus=opt['center_xyz'],supplement=True,offset=opt['offset'])
        keep=[p for p in initial if p.endswith(('local_raw.png','local_hypothesis.png','forward3_raw.png','upstream_raw.png','upstream_hypothesis.png'))]
        images=keep+ims;views.update(vs);views={k:v for k,v in views.items() if v['image'] in images}
    result=dict(case_id=case['case_id'],judgment=j,errors=errors,calls=calls,requested=used,views=views,images=images)
    save_json(folder/'result.json',result);return result

def write_viewer(raw,visit,target,folder,judgments):
    points=visit['upstream_xyz']+[target['xyz_um']]+[p for b in visit['branches'] for s in b['future'] for p in s['xyz']];xyz=np.array(points)
    lo=np.maximum(0,np.floor(xyz.min(0)-8)).astype(int);hi=np.minimum(raw.shape[::-1],np.ceil(xyz.max(0)+9)).astype(int);crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
    tracks=[dict(id='upstream',xyz=visit['upstream_xyz'],color='#48dfff')]
    for i,b in enumerate(visit['branches']):
        for s in b['future']:tracks.append(dict(id=s['branch_id'],xyz=s['xyz'],color=['#ffb347','#62df80','#ef80f5'][i%3]))
    payload=dict(origin_xyz=lo.tolist(),shape_zyx=list(crop.shape),voxels_base64=base64.b64encode(crop.tobytes()).decode(),junction=visit['center_xyz'],target=target['xyz_um'],tracks=tracks,judgments=judgments,junction_id=visit['junction_id'])
    template=(ROOT/'visualization/assets/soma_branchwalk_viewer.html').read_text();(folder/'interactive.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')))

def export(raw,tree,report,out):
    # Persist the full diagnosis and action plan before writing any changed SWC.
    save_json(out/'diagnosis.json',report);kept,components,plan=walk.execute(tree,report['visits']);save_json(out/'execution_plan.json',plan)
    if sha(report['input']['swc'])!=report['input']['swc_sha256'] or sha(report['input']['raw'])!=report['input']['raw_sha256']:raise ValueError('Input changed')
    kept2,components2,plan2=walk.execute(tree,read(out/'diagnosis.json')['visits']);assert plan2==read(out/'execution_plan.json') and kept2==kept
    swc=out/'swc'/f'{report["input"]["cell"]}.swc';swc.parent.mkdir(exist_ok=True);fmt=['%d','%d','%.17g','%.17g','%.17g','%.17g','%d'];np.savetxt(swc,kept,fmt=fmt)
    for part in components:
        folder=out/'components';folder.mkdir(exist_ok=True);np.savetxt(folder/(part['branch_id']+'.swc'),part.pop('tree'),fmt=fmt);save_json(folder/(part['branch_id']+'_original_parents.json'),part)
    original={int(n[0]):n for n in tree};loaded=np.loadtxt(swc,ndmin=2).tolist();walk.tree_index(loaded);assert all(row==original[int(row[0])] for row in loaded)
    assert set(plan['removed_ids'])==set(original)-{int(n[0]) for n in loaded}
    calls=[call for v in report['visits'] for g in v.get('groups',[]) for call in g['calls']];decisions=[d for v in report['visits'] for g in v.get('groups',[]) for d in g['judgment']['outlets']]
    versions=[('Unpruned input',Path(report['input']['swc']))]+[(v,BASE/'data/references'/v/f'{report["input"]["cell"]}.swc') for v in ['v0.81','v0.83b'] if (BASE/'data/references'/v/f'{report["input"]["cell"]}.swc').exists()]+[('Gemma soma branch walk',swc)]
    render_swc_comparison(raw,versions,out/'comparison.png',title=f'{report["input"]["cell"]}: no legacy pruning; no manual ground truth')
    summary=dict(cell=report['input']['cell'],policy=walk.VERSION,legacy_pruning_steps=[],input_nodes=len(tree),output_nodes=len(kept),input_junctions=len(report['visits']),visited_junctions=sum(v.get('status')=='reviewed' for v in report['visits']),skipped_cut_descendant_junctions=sum(v.get('status')=='detached_before_visit' for v in report['visits']),
        outgoing_decisions=dict(Counter(d['decision'] for d in decisions)),model_calls=len(calls),model_seconds=sum(c['seconds'] for c in calls),supplements=sum(len(g['requested']) for v in report['visits'] for g in v.get('groups',[])),cut_edges=[a['edge'] for a in plan['actions']],removed_nodes=len(plan['removed_ids']),removed_length_um=sum(a['length_um'] for a in plan['actions']),versions={k:metrics(p) for k,p in versions},accuracy=None)
    save_json(out/'summary.json',summary);save_json(out/'validation.json',dict(single_valid_root=True,acyclic_no_dangling_parents=True,retained_rows_unchanged=True,removed_set_matches_audit=True,original_inputs_unchanged=True,legacy_steps_executed=[],all_reachable_junctions_reviewed=all(v.get('status') in ['reviewed','detached_before_visit'] for v in report['visits']),biological_accuracy_evaluated=False))
    rows=[]
    for v in report['visits']:
        if v['status']!='reviewed':continue
        folder=out/'junctions'/v['junction_id'];folder.mkdir(parents=True,exist_ok=True);ds=[d for g in v['groups'] for d in g['judgment']['outlets']];write_viewer(raw,v,report['target'],folder,ds)
        evidence=[]
        for g in v['groups']:
            for call in g['calls']:
                request=Path(call['request']);stem=request.name.replace('_request.json','')
                evidence.append(' · '.join(f'<a href="{os_rel(request.parent/(stem+suffix),out)}">{g["case_id"]} {stem} {label}</a>' for suffix,label in [('_prompt.txt','实际提示词'),('_request.json','图片清单'),('_response.json','原始响应')] if (request.parent/(stem+suffix)).exists()))
        images=list(dict.fromkeys(p for g in v['groups'] for p in g['images']));images_html=''.join(f'<a href="{os_rel(p,out)}"><img loading="lazy" src="{os_rel(p,out)}"></a>' for p in images)
        rows.append(f'<details id="{v["junction_id"]}"><summary>{v["junction_id"]} / 遍历序号 {v["order"]} / '+html.escape(', '.join(d['branch_id']+': '+d['decision'] for d in ds))+'</summary><p><a href="junctions/'+v['junction_id']+'/interactive.html">三视图联动：上游、向前三段与逐层切片</a></p><pre>'+html.escape(json.dumps(ds,ensure_ascii=False,indent=2))+'</pre><p>'+'<br>'.join(evidence)+'</p><div class="grid">'+images_html+'</div></details>')
    prompt=(ROOT/'prompts/soma_branchwalk_v1.txt').read_text();command=f'/home/pzy/anaconda3/envs/hb_seg/bin/python {ROOT}/tools/run_soma_branchwalk.py --cell {summary["cell"]} --run-name <新的目录名>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Soma branch walk</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:30px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f3f5;padding:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{max-width:100%}summary{cursor:pointer;font-weight:bold}details{margin:15px 0;padding:12px;border:1px solid #ccc}</style><h1>'+summary['cell']+'：从目标 soma 出发，每个分叉检查向前三段 branch</h1><p>输入为剪枝前 SWC，旧剪枝步骤全部跳过。沿每条路径连续展开三段 branch，所有直接出口都审核；KEEP/REVIEW 继续遍历，CUT 保存组件并停止访问其下游。MIP 统一 XYZ 坐标；投影重叠不表示三维连接。模型此前走过的路径也只是待核实假设。</p><p><a href="diagnosis.json">完整诊断</a> · <a href="execution_plan.json">执行及组件影响审计</a> · <a href="validation.json">结构验证</a> · <a href="swc/'+summary['cell']+'.swc">结果 SWC</a></p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><img src="comparison.png"><h2>逐分叉原图与判断</h2>'+''.join(rows)+'<h2>提示词全文</h2><pre>'+html.escape(prompt)+'</pre><h2>复现</h2><pre>'+html.escape(command)+'</pre><p>没有人工归属真值，准确率未评估；删除数量与结构检查不代表生物学正确性。</p>')
    return summary

def os_rel(p,out):
    import os
    return os.path.relpath(p,out)

def run(cell,swc,rawpath,metadata,out,endpoint='http://127.0.0.1:18484',prepare_only=False,limit=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);tree=np.loadtxt(swc,ndmin=2).tolist();raw=tifffile.imread(rawpath);nodes,children,root=walk.tree_index(tree)
    if raw.ndim!=3 or raw.dtype!=np.uint8:raise ValueError('Expected resampled uint8 ZYX volume')
    target_xyz=[metadata['soma_x']*metadata['xy_resolution']/1000,metadata['soma_y']*metadata['xy_resolution']/1000,metadata['soma_z']*metadata['z_resolution']/1000]
    if not np.all((np.array(target_xyz)>=0)&(np.array(target_xyz)<raw.shape[::-1])):raise ValueError('Soma metadata outside image')
    if not np.all((np.array(tree)[:,2:5]>=0)&(np.array(tree)[:,2:5]<raw.shape[::-1])):raise ValueError('SWC coordinates outside image')
    target=dict(xyz_um=target_xyz,source='known per-cell metadata (native nm -> resampled image um)',root_id=root,root_xyz=nodes[root][2:5],root_distance_um=float(np.linalg.norm(np.array(target_xyz)-nodes[root][2:5])),voxel_size_um_xyz=[1,1,1])
    files=['tools/run_soma_branchwalk.py','algorithms/tracing/soma_branchwalk.py','visualization/soma_branchwalk.py','visualization/assets/soma_branchwalk_viewer.html','prompts/soma_branchwalk_v1.txt','tools/run_ownership_audit.py','visualization/ownership_audit.py','visualization/soma_ownership_context.py','algorithms/tracing/connection_diagnostics.py']
    files+=['11_prune_swc.py','12_guard_regression.py','tools/run_connection_experiment.py','tools/run_soma_ownership_pipeline.py','visualization/swc_comparison.py']
    identity=dict(version=walk.VERSION,cell=str(cell),swc=str(Path(swc).resolve()),raw=str(Path(rawpath).resolve()),swc_sha256=sha(swc),raw_sha256=sha(rawpath),metadata=metadata,code={p:sha(ROOT/p) for p in files},lookahead=3,group_size=3,legacy_pruning_steps=[],model_parameters=PARAMS)
    if (out/'input.json').exists() and read(out/'input.json')!=identity:raise ValueError('Input/prompt/view/code changed: use a new run name')
    save_json(out/'input.json',identity)
    for p in files:
        dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    if (out/'diagnosis.json').exists():report=read(out/'diagnosis.json')
    else:report=dict(input=identity,target=target,visits=walk.plan_visits(tree,target_xyz),traversal='Breadth first from soma; complete each junction before advancing; each path looks ahead 3 branch segments');save_json(out/'diagnosis.json',report)
    if prepare_only:return dict(junctions=len(report['visits']),input_nodes=len(tree),target=target)
    runtime=runtime_identity(endpoint)
    if (out/'runtime.json').exists() and read(out/'runtime.json')!=runtime:raise ValueError('Changed model/runtime: use a new run')
    save_json(out/'runtime.json',runtime);args=SimpleNamespace(endpoint=endpoint,fingerprint=walk.fingerprint(dict(input=identity,runtime=runtime)))
    completed=0;removed=set();history={}
    for visit in report['visits']:
        if visit.get('status')=='reviewed':
            for group in visit['groups']:
                for d in group['judgment']['outlets']:
                    history[d['branch_id']]=d['decision']
                    if d['decision']=='CUT':removed|=walk.descendants(children,d['cut_edge']['child_id'])
            continue
        if visit['node_id'] in removed:visit['status']='detached_before_visit';continue
        visit.setdefault('groups',[])
        for start in range(0,len(visit['branches']),3):
            case=dict(visit,branches=visit['branches'][start:start+3],case_id=visit['junction_id']+'_g'+str(start//3+1));existing=next((g for g in visit['groups'] if g['case_id']==case['case_id']),None)
            if existing:continue
            passed=[dict(branch_id='B'+str(n),decision=history['B'+str(n)],trust='model hypothesis only') for n in visit['upstream_ids'] if 'B'+str(n) in history]
            result=review(case,raw,target,out,args,passed);visit['groups'].append(result);save_json(out/'diagnosis.json',report)
        for group in visit['groups']:
            for d in group['judgment']['outlets']:
                history[d['branch_id']]=d['decision']
                if d['decision']=='CUT':removed|=walk.descendants(children,d['cut_edge']['child_id'])
        visit['status']='reviewed';completed+=1;save_json(out/'diagnosis.json',report)
        print(json.dumps(dict(junction=visit['junction_id'],order=visit['order'],decisions={d['branch_id']:d['decision'] for g in visit['groups'] for d in g['judgment']['outlets']},removed_so_far=len(removed)),ensure_ascii=False),flush=True)
        if limit and completed>=limit:return dict(incomplete=True,reviewed_this_call=completed)
    save_json(out/'diagnosis.json',report);result=export(raw,tree,report,out);print(json.dumps(result,ensure_ascii=False),flush=True);return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cell',required=True);ap.add_argument('--run-name',required=True);ap.add_argument('--swc',type=Path);ap.add_argument('--raw',type=Path);ap.add_argument('--endpoint',default='http://127.0.0.1:18484');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--limit',type=int);args=ap.parse_args()
    if not args.cell.isdigit() or Path(args.run_name).name!=args.run_name:ap.error('Invalid cell/run name')
    meta,_=cell_metadata(BASE/'data',args.cell);swc=args.swc or BASE/'data/source_v083b/gcut_output/selected_resampled_step1'/f'{args.cell}.swc';raw=args.raw or BASE/'data/inputs/raw'/f'{args.cell}.tif'
    print(run(args.cell,swc,raw,meta,BASE/'runs'/args.run_name,args.endpoint,args.prepare_only,args.limit))
if __name__=='__main__':main()
