#!/usr/bin/env python3
"""Finalize whole-soma ownership decisions with cross-case target protection."""
from pathlib import Path
import argparse,html,json,shutil,sys
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from algorithms.tracing.ownership_consistency import reconcile_target_conflicts
from algorithms.tracing.soma_ownership_context import apply_ownership_plan
from algorithms.tracing.connection_diagnostics import save_json
from tools.run_connection_experiment import metrics
from visualization.swc_comparison import render_swc_comparison


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();out=Path(args.run).resolve()
    report=json.loads((out/'diagnosis.json').read_text());identity=json.loads((out/'input.json').read_text())
    if report.get('consistency'):raise ValueError('Already reconciled; original results preserved in model_only_before_consistency')
    if any(not c.get('status') for c in report['cases']):raise ValueError('Model review incomplete')
    tree=np.loadtxt(identity['swc'],ndmin=2).tolist();final=reconcile_target_conflicts(tree,report)
    backup=out/'model_only_before_consistency';backup.mkdir(exist_ok=False)
    for name in ['diagnosis.json','cuts.json','summary.json','index.html','comparison.png','removed_segments.swc','swc/97117.swc']:
        p=out/name
        if p.exists():
            q=backup/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    # Save all conflict decisions before reapplying the revised plan.
    save_json(out/'diagnosis.json',final);save_json(out/'consistency.json',final['consistency'])
    kept,audit=apply_ownership_plan(tree,json.loads((out/'diagnosis.json').read_text()))
    np.savetxt(out/'swc/97117.swc',kept,fmt=['%d','%d','%.5f','%.5f','%.5f','%.5f','%d'],header='n type x y z r parent; full upstream, competing soma and cross-case consistency review')
    save_json(out/'cuts.json',audit)
    removed=set(audit['removed_ids']);parts=[n for n in tree if int(n[0]) in removed]
    if parts:np.savetxt(out/'removed_segments.swc',parts,fmt=['%d','%d','%.5f','%.5f','%.5f','%.5f','%d'])
    elif (out/'removed_segments.swc').exists():(out/'removed_segments.swc').unlink()
    old=json.loads((backup/'summary.json').read_text());versions=[(k,Path(v['path'])) for k,v in old['versions'].items()]
    render_swc_comparison(tifffile.imread(identity['raw']),versions,out/'comparison.png',title='97117: full upstream, competing soma and cross-case consistency')
    summary=dict(old,versions={k:metrics(p) for k,p in versions},cuts=len(audit['events']),removed_nodes=len(removed),
        model_reviewed=sum('judgment' in c for c in final['cases']),independent_local_reviews=sum('verification' in c for c in final['cases']),
        consistency_held_proposals=len(final['consistency']['held']),status_counts={s:sum(c['status']==s for c in final['cases']) for s in set(c['status'] for c in final['cases'])},
        note='Experimental ownership hypotheses, no manual truth. Cross-case contradictions block ancestor subtree cuts. Original model-only output is archived, not the final result.')
    save_json(out/'summary.json',summary)
    blocks=[]
    for c in sorted(final['cases'],key=lambda c:(not c['gate']['accepted'],c['case_id'])):
        imgs=''.join(f'<a href="{Path(p).relative_to(out)}"><img loading="lazy" src="{Path(p).relative_to(out)}"></a>' for p in c['images']+c.get('detail_images',[]))
        blocks.append(f'<section id="case-{c["case_id"]}"><h2>{c["case_id"]} · 候选胞体 S{c["competing_soma_id"]} · {c["status"]}</h2><pre>{html.escape(json.dumps({k:c.get(k) for k in ["cut_options","judgment","gate","verification","consistency_conflicts"]},ensure_ascii=False,indent=2))}</pre><div class="grid">{imgs}</div></section>')
    rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["nodes"]}</td><td>{v["length_um"]:.2f}</td><td>{v["tips"]}</td></tr>' for k,v in summary['versions'].items())
    conflicts=''.join('<li>上游候选 <a href="#case-'+h['case_id']+'">'+h['case_id']+'</a>，节点 '+str(h['cut_node_id'])+'：与下游目标归属 '+', '.join('<a href="#case-'+t+'">'+t+'</a>' for t in h['conflicting_target_cases'])+' 矛盾，暂不执行此处整树删除。</li>' for h in final['consistency']['held'])
    (out/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>97117 完整上游与胞体归属</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px;background:#f4f6f8}}section{{background:white;padding:20px;margin:20px 0}}pre{{white-space:pre-wrap}}img{{width:100%}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}td,th{{padding:10px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>97117：完整上游、候选胞体路径与归属一致性</h1>
<p>共 {len(final['somas'])} 个图像候选胞体（含目标）、{len(final['cases'])} 个路径归属疑点。执行 {len(audit['events'])} 处剪断，删除 {len(removed)} 节点。没有人工归属真值，删除量不代表准确率。</p>
<p>判断同时查看目标胞体到分支的完整上游、候选胞体到分支的三维图像路径、三个正交视角及沿路径展开的原始信号。最短路径仅提出候选；模型提出的剪断还须通过位置约束、独立局部复核和跨位置一致性检查。</p>
<p>青色：目标胞体出发的 SWC；洋红：其他胞体的图像路径假设；绿色：目标胞体的图像路径假设；橙色：被检查段。原始图与叠加图并排，局部薄层仅叠加实际层内线段。</p>
<p><a href="swc/97117.swc">最终 SWC</a> · <a href="diagnosis.json">完整诊断</a> · <a href="cuts.json">实际剪断</a> · <a href="summary.json">指标</a> · <a href="consistency.json">归属冲突</a></p>
<h2>跨位置归属冲突</h2><p>同一模型对上游与下游可能给出相互矛盾的归属。本轮拦截 {len(final['consistency']['held'])} 个相关剪断提议；这表示仍未解决，不能视为已证明属于目标。局部未剪断也不代表最终保留，需看下方每处的最终影响。</p><ul>{conflicts}</ul>
<table><tr><th>版本</th><th>节点</th><th>长度 µm</th><th>末梢</th></tr>{rows}</table><img src="comparison.png">{''.join(blocks)}</html>''',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
