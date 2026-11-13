#!/usr/bin/env python3
"""Audit output-field normalization; never changes model evidence or reruns inference."""
from pathlib import Path
import argparse,html,json,shutil,sys
from collections import Counter
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from algorithms.tracing import gemma_soma_seeds as seeds
from algorithms.tracing.soma_seed_output import normalize,VERSION
from algorithms.tracing.connection_diagnostics import save_json
from tools.run_ownership_audit import sha
from visualization.gemma_soma_seeds import overview

def read(path):return json.loads(Path(path).read_text())
def finish(out,partial=False):
    out=Path(out);bundle=read(out/'candidates.json');original=read(out/'decisions.json');results={};changes=[]
    if not partial and (not (out/'summary.json').exists() or len(original)!=len(bundle['candidates'])):raise ValueError('Finish inference first')
    for cid,r in original.items():
        c=r['candidate'];j,notes=normalize(r['call'].get('judgment'));views=read(out/'cases'/cid/'views.json')
        errors=[r['call']['error']] if r['call'].get('error') else seeds.validate(j,c,{k:True for k in r['reference_ids']},views)
        decision=seeds.review_fallback(c,'Invalid output: '+'; '.join(errors)) if errors else j
        results[cid]=dict(r,decision=decision,errors=errors,normalization=notes)
        if notes:changes.append(dict(candidate_id=cid,changes=notes,before_decision=r['decision'],after_decision=decision,remaining_errors=errors,raw_response_sha256=sha(out/'cases'/cid/'review_response.json')))
    # The only adjustment concerns REJECT. Thus every subsequent model request
    # saw exactly the same retained reference set; no counterfactual model reuse.
    assert [k for k,r in original.items() if r['decision']['decision']=='ACCEPT']==[k for k,r in results.items() if r['decision']['decision']=='ACCEPT']
    count=dict(Counter(r['decision']['decision'] for r in results.values()))
    if partial:return dict(completed=len(results),total=len(bundle['candidates']),decisions=count,field_normalizations=len(changes))
    dest=out/'normalized';dest.mkdir(exist_ok=True)
    manifest=dict(version=VERSION,original_input_sha256=sha(out/'input.json'),original_diagnosis_sha256=sha(out/'diagnosis.json'),model_requests_unchanged=True,images_unchanged=True,model_is_soma_and_decision_unchanged=True,retained_reference_sequence_unchanged=True,new_model_calls=0,changes=changes,source={p:sha(ROOT/p) for p in ['algorithms/tracing/soma_seed_output.py','tools/finish_gemma_soma_seed_review.py','algorithms/tracing/gemma_soma_seeds.py','visualization/gemma_soma_seeds.py']})
    for p in manifest['source']:
        target=dest/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,target)
    save_json(dest/'normalization_audit.json',manifest);save_json(dest/'decisions.json',results)
    catalogue=seeds.assemble(bundle,results);save_json(dest/'soma_catalogue.json',catalogue)
    summary=read(out/'summary.json');summary.update(decisions=count,strict_parser_decisions=summary['decisions'],field_normalizations=len(changes),normalization_version=VERSION,invalid_outputs=[k for k,r in results.items() if r['errors']])
    sources={name:Counter() for name in ['both_thresholds','baseline_only','relaxed_only']}
    for r in results.values():
        thresholds=r['candidate']['threshold_sources']
        name='both_thresholds' if len(thresholds)==2 else ('baseline_only' if bundle['baseline_threshold'] in thresholds else 'relaxed_only')
        sources[name][r['decision']['decision']]+=1
    summary['decisions_by_threshold_source']={k:dict(v) for k,v in sources.items()}
    summary['held_for_review']=[dict(candidate_id=cid,source='output_validation' if r['errors'] else 'model_uncertainty',validation_errors=r['errors'],raw_model_decision=(r['call'].get('judgment') or {}).get('decision'),reason=r['decision']['reason']) for cid,r in results.items() if r['decision']['decision']=='REVIEW']
    summary['limitations']=['No human soma labels: accuracy and recall are not evaluated.', 'Only proposed GSDT peaks plus known target T are considered; absent proposals cannot be recovered.', 'Sequential reference acceptance is order-dependent; earlier Gemma ACCEPT decisions remain hypotheses.', 'Only retained references within 48 um are rendered for identity comparison; this selects context and does not suppress points.', 'Projected reference crosses on serial slices may lie outside the section; in_depth in views.json distinguishes them.']
    save_json(dest/'summary.json',summary)
    context=read(out/'proposed_soma_context.json');context['unresolved_candidates']=catalogue['review'];context['normalization_audit']='normalization_audit.json';save_json(dest/'proposed_soma_context.json',context)
    identity=read(out/'input.json')
    if sha(identity['raw'])!=identity['raw_sha256'] or sha(identity['mask'])!=identity['mask_sha256']:raise ValueError('Input changed')
    overview(tifffile.imread(identity['raw']),bundle,results,dest/'comparison.png')
    rows=[]
    for cid,r in results.items():
        raw=r['call'].get('judgment');shown=raw if isinstance(raw,dict) else {};decision=r['decision'];photos=''.join(f'<a href="../cases/{cid}/{Path(p).name}"><img loading="lazy" src="../cases/{cid}/{Path(p).name}"></a>' for p in r['images'])
        rows.append(f'<details id="{cid}"><summary>{cid} · 模型 {html.escape(str(shown.get("decision","INVALID")))} / 程序 {decision["decision"]}'+((' → '+decision['duplicate_of']) if decision['duplicate_of'] else '')+' · XYZ '+str(r['candidate']['xyz_um'])+'</summary><p>'+html.escape(decision['reason'])+'</p><pre>'+html.escape(json.dumps(dict(raw_model_output=raw,applied_output=decision,field_normalization=r['normalization'],remaining_errors=r['errors']),ensure_ascii=False,indent=2))+'</pre><p>'+f'<a href="../cases/{cid}/review_prompt.txt">实际完整提示词</a> · <a href="../cases/{cid}/review_request.json">请求与图片哈希</a> · <a href="../cases/{cid}/review_response.json">原始响应</a> · <a href="../cases/{cid}/views.json">XYZ 坐标映射</a></p>'+photos+'</details>')
    visual_notes=''
    if (out/'qualitative_review.json').exists():
        observations=read(out/'qualitative_review.json')
        shutil.copy2(out/'qualitative_review.json',dest/'qualitative_review.json')
        visual_notes='<h2>优先核查的局部病例</h2><p>以下为 Codex 对保存图片的抽查，不是人工标签，不改写 Gemma 结果。</p>'
        for note in observations['notes']:
            visual_notes+='<p><a href="#'+note['candidate_id']+'">'+html.escape(note['candidate_id']+'：'+note['purpose'])+'</a> — '+html.escape(note['observation'])+'</p>'
    table_rows=[]
    for cid,r in results.items():
        c=r['candidate'];d=r['decision']
        table_rows.append('<tr><td><a href="#'+cid+'">'+cid+'</a></td><td>'+str(c['xyz_um'])+'</td><td>'+str(c['threshold_sources'])+'</td><td>'+d['decision']+'</td><td>'+str(d['duplicate_of'] or '')+'</td></tr>')
    quick_table='<h2>候选索引</h2><p>点击 ID 展开该点的模型原答、三视图和连续切片；阈值来源表示该点由哪一档 GSDT 提名。</p><table style="border-collapse:collapse" cellpadding="8"><thead><tr><th>ID</th><th>XYZ (µm)</th><th>阈值来源</th><th>程序结果</th><th>同一胞体</th></tr></thead><tbody>'+''.join(table_rows)+'</tbody></table>'
    navigation='<script>function openCase(){const e=document.getElementById(location.hash.slice(1));if(e&&e.tagName==="DETAILS"){e.open=true;e.scrollIntoView();}}window.addEventListener("hashchange",openCase);openCase();</script>'
    explanation='仅把明确 REJECT 且 is_soma=no 的不适用归属字段规范为 not_applicable；没有改变模型的是否胞体、接受/拒绝判定或原图证据。原始响应及严格解析结果完整保留。待复核回答不会被改成拒绝。此修复不改变已保留胞体序列，因此后续模型输入中的参考集合也没有变化。'
    command=f'/home/pzy/anaconda3/envs/hb_seg/bin/python {ROOT}/tools/run_gemma_soma_seeds.py --cell {summary["cell"]} --run-name <新目录名> --threshold {identity["relaxed"]} --baseline-threshold {identity["baseline"]} --percentile {identity["percentile"]}'
    (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Gemma 胞体点实验</title><style>body{font:16px/1.6 system-ui;max-width:1350px;margin:30px auto;padding:20px}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f6;padding:12px}details{border:1px solid #bbb;margin:12px 0;padding:12px}summary{cursor:pointer;font-weight:bold}</style><h1>'+summary['cell']+'：Gemma 胞体识别与判重</h1><p>本轮止于胞体点审核，没有执行 G-Cut 分配或任何后续剪枝。T 是已知元数据点，不算模型新检出的胞体。GSDT 保留严格大于阈值的语义；降低阈值后，保留原/新两档峰值并集，逐点审核；新结果不经过 NMS 或目标 mask 去重。</p><p>对照 NMS 只作用于同一 GSDT 来源，用于分开观察阈值变化和模型筛选。历史完整流程还包含其他提名、过滤和合并，点数不可直接等同于这个对照。没有人工胞体标签，准确率未评估；模型只审核已提名点，未提名的胞体不会被本轮补出。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><p>'+explanation+'</p><p><a href="normalization_audit.json">逐条字段修复审计</a> · <a href="soma_catalogue.json">拟保留胞体 / 重复映射 / 拒绝 / 待复核</a> · <a href="../candidates.json">全部提名与来源</a> · <a href="../index.html">原始严格解析报告</a></p><img src="comparison.png">'+visual_notes+quick_table+'<h2>逐点原图与模型判断</h2>'+''.join(rows)+'<h2>复现命令</h2><pre>'+html.escape(command)+'</pre><h2>本次实际提示词模板</h2><pre>'+html.escape((out/'source/prompts/gemma_soma_seeds_v1.txt').read_text())+'</pre>'+navigation)
    save_json(dest/'validation.json',dict(all_candidates_reviewed=len(results)==len(bundle['candidates']),retained_reference_sequence_unchanged=True,raw_model_decisions_unchanged=True,raw_inputs_unchanged=True,gcut_or_pruning_executed=False,biological_accuracy_evaluated=False))
    (dest/'REPORT.txt').write_text(summary['cell']+' 胞体点实验\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'+explanation+'\n\n'+command+'\n\n模型判定、程序应用结果、原始图像与每条实际请求/响应见 index.html；没有人工真值，准确率未评估。\n')
    print(json.dumps(summary,ensure_ascii=False));return summary
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--partial',action='store_true');args=ap.parse_args();result=finish(args.run,args.partial)
    if args.partial:print(json.dumps(result,ensure_ascii=False))
