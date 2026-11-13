#!/usr/bin/env python3
"""Test Gemma soma classification and deduplication, stopping before G-Cut."""
from pathlib import Path
import argparse,html,json,shutil,sys,types
from collections import Counter
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from utils.case_worker import load_module
from algorithms.tracing import gemma_soma_seeds as seeds
from algorithms.tracing.soma_seed_output import normalize
from algorithms.tracing.connection_diagnostics import save_json
from visualization.gemma_soma_seeds import render,overview
from tools.run_ownership_audit import model_call,runtime_identity,sha,PARAMS
from tools.run_connection_experiment import cell_metadata
BASE=ROOT.parent/'test/0.84'

def read(p):return json.loads(Path(p).read_text())

def review_all(raw,bundle,out,args,identity,prompt):
    save_json(out/'candidates.json',bundle)
    if args.prepare_only:return dict(prepared=True,candidates=len(bundle['candidates']))
    runtime=runtime_identity(args.endpoint)
    if (out/'runtime.json').exists() and read(out/'runtime.json')!=runtime:raise ValueError('Runtime changed; use a new run')
    save_json(out/'runtime.json',runtime);args.fingerprint=seeds.fingerprint(dict(input=identity,runtime=runtime))
    results={};retained=[bundle['target']];completed=0
    for candidate in bundle['candidates']:
        cid=candidate['candidate_id'];folder=out/'cases'/cid;folder.mkdir(parents=True,exist_ok=True)
        refs=[p for p in retained if np.linalg.norm(np.array(p['xyz_um'])-candidate['xyz_um'])<=48]
        # No candidate is suppressed by distance, score, target masks or NMS.
        images,views=render(raw,candidate,refs,folder);save_json(folder/'views.json',views)
        inventory={k:{x:v[x] for x in ['raw','plane','xyz_bounds','landmarks']} for k,v in views.items()}
        metadata=dict(candidate=dict(candidate_id=cid,xyz_um=candidate['xyz_um']),known_target=bundle['target'],previously_retained=[dict(candidate_id=p['candidate_id'],xyz_um=p['xyz_um'],trust='known_metadata' if p['candidate_id']=='T' else 'model_hypothesis') for p in retained],allowed_duplicate_ids=[p['candidate_id'] for p in refs],views=inventory,voxel_size_um_xyz=[1,1,1],array_axes='ZYX',coordinate_axes='XYZ um',comparison_selection_radius_um=48,comparison_radius_is_not_a_suppression_rule=True)
        call=model_call(prompt,metadata,images,folder,'review',args)
        if call.get('error') and not (folder/'review_response.json').exists():
            save_json(out/'INCOMPLETE.json',dict(candidate_id=cid,error=call['error'],downstream_executed=False));raise RuntimeError('Model request failed; no final soma catalogue: '+call['error'])
        judgment,normalization=normalize(call.get('judgment'))
        errors=[call['error']] if call.get('error') else seeds.validate(judgment,candidate,{p['candidate_id']:p for p in refs},views)
        decision=seeds.review_fallback(candidate,'Invalid output: '+'; '.join(errors)) if errors else judgment
        result=dict(candidate=candidate,decision=decision,errors=errors,call=call,normalization=normalization,reference_ids=[p['candidate_id'] for p in refs],images=images)
        save_json(folder/'result.json',result);results[cid]=result
        if decision['decision']=='ACCEPT':retained.append(candidate)
        save_json(out/'decisions.json',results)
        print(json.dumps(dict(candidate=cid,decision=decision['decision'],duplicate_of=decision['duplicate_of'],completed=len(results),total=len(bundle['candidates'])),ensure_ascii=False),flush=True)
        completed+=1
        if args.limit and completed>=args.limit:return dict(incomplete=True,completed=completed)
    # Complete diagnosis is durable before forming the proposed retained catalogue.
    save_json(out/'diagnosis.json',dict(input=identity,bundle=bundle,results=results,downstream_executed=False))
    if any(sha(identity[k])!=identity[k+'_sha256'] for k in ['raw','mask']):raise ValueError('Source inputs changed')
    catalogue=seeds.assemble(bundle,results);save_json(out/'soma_catalogue.json',catalogue)
    save_json(out/'proposed_soma_context.json',dict(target_zyx=bundle['target']['zyx'],centers_zyx=[p['zyx'] for p in catalogue['accepted']],radii=[p.get('gsdt_score') for p in catalogue['accepted']],status='experimental_proposal_not_applied_to_gcut',unresolved_candidates=catalogue['review']))
    stats=dict(cell=args.cell,policy=seeds.VERSION,intensity_percentile=bundle['intensity_percentile'],intensity_threshold=bundle['intensity_threshold'],baseline_threshold=bundle['baseline_threshold'],relaxed_threshold=bundle['relaxed_threshold'],baseline_gsdt_peaks=len(bundle['baseline_gsdt']),relaxed_gsdt_peaks=len(bundle['relaxed_gsdt']),union_candidates=len(bundle['candidates']),reviewed=len(results),baseline_nms_including_target=len(bundle['baseline_nms_zyx']),relaxed_nms_including_target=len(bundle['relaxed_nms_zyx']),decisions=dict(Counter(r['decision']['decision'] for r in results.values())),raw_model_decisions=dict(Counter((r['call'].get('judgment') if isinstance(r['call'].get('judgment'),dict) else {}).get('decision','invalid') for r in results.values())),retained_including_known_target=len(catalogue['accepted']),model_calls=len(results),model_seconds=sum(r['call']['seconds'] for r in results.values()),invalid_outputs=[k for k,r in results.items() if r['errors']],gcut_executed=False,pruning_executed=False,biological_accuracy=None)
    stats['field_normalizations']=sum(bool(r['normalization']) for r in results.values())
    historical=BASE/'data/source_v083b/logs'/args.cell/'soma-context.json'
    if historical.exists():stats['historical_full_gcut_seed_count']=len(read(historical)['centers_zyx']);stats['historical_full_seed_source']=str(historical)
    save_json(out/'summary.json',stats);overview(raw,bundle,results,out/'comparison.png')
    rows=[]
    for cid,r in results.items():
        links=''.join(f'<a href="cases/{cid}/{Path(p).name}"><img loading="lazy" src="cases/{cid}/{Path(p).name}"></a>' for p in r['images'])
        rows.append(f'<details id="{cid}"><summary>{cid} · '+r['decision']['decision']+((' → '+str(r['decision']['duplicate_of'])) if r['decision']['duplicate_of'] else '')+' · XYZ '+str(r['candidate']['xyz_um'])+'</summary><pre>'+html.escape(json.dumps(r['decision'],ensure_ascii=False,indent=2))+'</pre><p><a href="cases/'+cid+'/review_prompt.txt">实际完整提示词</a> · <a href="cases/'+cid+'/review_request.json">模型输入清单</a> · <a href="cases/'+cid+'/review_response.json">模型原始响应</a> · <a href="cases/'+cid+'/views.json">坐标映射</a></p>'+links+'</details>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+args.cell+' Gemma 胞体点测试</title><style>body{font:16px/1.6 system-ui;max-width:1350px;margin:30px auto;padding:20px}img{max-width:100%}pre{white-space:pre-wrap}details{border:1px solid #bbb;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:bold}</style><h1>'+args.cell+'：G-Cut 前的胞体识别与判重</h1><p>按指定 mask 内百分位计算的亮度阈值为 '+str(bundle['intensity_threshold'])+'。距离阈值与百分位见下表。保留两档峰值的并集，只合并完全同坐标的提名。每个候选都由 Gemma 审核；不预先做 NMS、目标 mask 去重或同胞体自动合并。</p><p>T 是已知目标元数据点，不计为模型检出的胞体。ACCEPT=新增胞体；DUPLICATE=同一胞体；REJECT=非胞体；REVIEW=尚未判断清楚。待复核点保留在诊断里，没有偷偷作为非胞体丢弃。本轮没有运行 G-Cut 分配或任何后续剪枝。</p><p>对照中的 NMS 是同一 GSDT 提名上的消融控制；历史完整流程还含其他提名和过滤，不能把两者混为同一结果。模型与历史结果都不是人工真值，准确率未评估。</p><pre>'+html.escape(json.dumps(stats,ensure_ascii=False,indent=2))+'</pre><img src="comparison.png"><p><a href="candidates.json">所有候选</a> · <a href="soma_catalogue.json">拟保留胞体与重复映射</a> · <a href="diagnosis.json">完整诊断</a></p>'+''.join(rows)+'<h2>提示词模板</h2><pre>'+html.escape(prompt)+'</pre>')
    save_json(out/'validation.json',dict(all_candidates_reviewed=set(results)=={c['candidate_id'] for c in bundle['candidates']},duplicate_refs_are_retained=True,inputs_unchanged=True,downstream_gcut_or_pruning_executed=False,model_input_images_per_call=3,biological_accuracy_evaluated=False))
    print(json.dumps(stats,ensure_ascii=False),flush=True);return stats

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cell',default='95918');ap.add_argument('--run-name',required=True);ap.add_argument('--raw',type=Path);ap.add_argument('--mask',type=Path);ap.add_argument('--threshold',type=float,default=4.);ap.add_argument('--baseline-threshold',type=float,default=5.);ap.add_argument('--percentile',type=float,default=55.);ap.add_argument('--endpoint',default='http://127.0.0.1:18484');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--limit',type=int);args=ap.parse_args()
    if not args.cell.isdigit() or args.run_name in ['.','..'] or Path(args.run_name).name!=args.run_name:ap.error('Invalid ID/run name')
    out=BASE/'runs'/args.run_name;out.mkdir(parents=True,exist_ok=True);raw=args.raw or BASE/'data/inputs/raw'/f'{args.cell}.tif';mask=args.mask or BASE/'data/source_v083b/mask'/f'{args.cell}.tif';meta,mp=cell_metadata(BASE/'data',args.cell)
    target=np.rint([meta['soma_z']*meta['z_resolution']/1000,meta['soma_y']*meta['xy_resolution']/1000,meta['soma_x']*meta['xy_resolution']/1000]).astype(int).tolist()
    files=['tools/run_gemma_soma_seeds.py','algorithms/tracing/gemma_soma_seeds.py','algorithms/tracing/soma_seed_output.py','visualization/gemma_soma_seeds.py','prompts/gemma_soma_seeds_v1.txt','algorithms/tracing/gcut_stage.py','algorithms/tracing/gcut_adapter.py','tools/run_ownership_audit.py']
    identity=dict(version=seeds.VERSION,cell=args.cell,raw=str(raw.resolve()),mask=str(mask.resolve()),raw_sha256=sha(raw),mask_sha256=sha(mask),metadata=meta,metadata_sha256=sha(mp),baseline=args.baseline_threshold,relaxed=args.threshold,percentile=args.percentile,model_parameters=PARAMS,code={f:sha(ROOT/f) for f in files},scope='soma seed diagnosis only; no G-Cut split or pruning')
    if (out/'input.json').exists() and read(out/'input.json')!=identity:raise ValueError('Input/code/prompt changed; use a new run directory')
    save_json(out/'input.json',identity)
    for f in files:
        dest=out/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    prompt=(out/'source/prompts/gemma_soma_seeds_v1.txt').read_text()
    sys.modules['pipeline_config']=SimpleNamespace(BASE_DIR=out,PATHS={'gcut':str(ROOT/'vendor/gcut')},GCUT={'target_percentile':args.percentile,'gsdt_threshold_x':args.baseline_threshold})
    stage=load_module('gcut')
    result=stage.process_image_to_gcut(raw,mask,None,out,target,args.percentile,args.baseline_threshold,seed_review=lambda volume,bundle:review_all(volume,bundle,out,args,identity,prompt),relaxed_seed_threshold=args.threshold)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
