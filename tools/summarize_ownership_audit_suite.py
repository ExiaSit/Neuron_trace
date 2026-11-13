#!/usr/bin/env python3
"""Readable results and unresolved-case inventory for the completed audit suite."""
from pathlib import Path
import argparse,collections,html,json,os,sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from algorithms.tracing.connection_diagnostics import save_json

def read(p):return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();config=read(args.config);out=Path(config['log_directory']);base=ROOT.parent/'test/0.84/runs';rows=[];stats=[]
    def link(p,label):return f'<a href="{os.path.relpath(p,out)}">{html.escape(label)}</a>'
    for item in config['experiments']:
        run=base/item['final_run'];s=read(run/'summary.json');r=read(run/'diagnosis.json');plan=read(run/'execution_plan.json');v=read(run/'final_artifact_validation.json')
        cases=[c for c in r['cases'] if c.get('reviews')];supp={c['candidate_id'] for c in cases if any(x['supplements'] for x in c['reviews'].values())}|{j['candidate_id'] for j in r.get('joint_reviews',[]) if j['result']['supplements']}
        unresolved=dict(attachment_REVIEW=[c['candidate_id'] for c in cases if c['judgment']['recommendation']=='REVIEW'],held_cut_candidates=[a['candidate_id'] for a in plan['actions'] if a['status']=='REVIEW'],unknown_ownership=[c['candidate_id'] for c in cases if c['judgment']['target_ownership']=='unknown'],
            deferred_joint_segments=[dict(candidate_id=cid,related_upstream_cases=[g['candidate_id'] for g in r.get('joint_reviews',[]) if cid in g['related_ids']]) for cid in sorted({x['candidate_id'] for x in r.get('deferred_segments',[])})],
            note='Upstream proposal retraction removes a conflicting cut proposal; it does not prove target identity. REVIEW, unknown ownership, and deferred segment review are different unresolved categories.')
        save_json(run/'unresolved_inventory.json',unresolved)
        result=dict(cell=s['cell'],scope=s['scope'],candidates_before_dedup=s['candidates']['before_dedup'],candidates_after_dedup=s['candidates']['after_dedup'],reviewed=s['reviewed'],recommendations=s['recommendations'],supplement_candidates=len(supp),supplement_rounds=s['supplements'],calls=s['calls'],model_seconds=s['model_seconds'],executed_edges=s['executed_edges'],removed_nodes=s['removed_nodes'],removed_length_um=s['removed_length_um'],max_action_nodes=s['max_action_nodes'],unresolved_attachment=len(unresolved['attachment_REVIEW']),held_cuts=len(unresolved['held_cut_candidates']),unknown_ownership=len(unresolved['unknown_ownership']),deferred_unique_segments=len(unresolved['deferred_joint_segments']),accuracy=None,run=str(run));stats.append(result)
        rows.append(f'<tr><td>{s["cell"]}</td><td>{"完整候选审核" if s["scope"]=="complete" else "代表性 8 处"}</td><td>{s["reviewed"]}</td><td>{s["recommendations"].get("KEEP",0)} / {s["recommendations"].get("PROPOSE_CUT",0)} / {s["recommendations"].get("REVIEW",0)}</td><td>{len(s["executed_edges"])} / {s["removed_nodes"]}</td><td>{s["calls"]}</td><td>'+link(run/'delivery.html','证据导航')+' · '+link(run/'report.html','完整报告')+' · '+link(run/'swc'/f'{s["cell"]}.swc','最终 SWC')+' · '+link(run/'unresolved_inventory.json','未解决清单')+'</td></tr>')
    save_json(out/'final_suite_summary.json',stats);main_run=base/config['experiments'][0]['final_run'];s=read(main_run/'summary.json');test=read(out/'test_summary.json')
    versions=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["nodes"]}</td><td>{v["length_um"]:.3f}</td><td>{v["tips"]}</td></tr>' for k,v in s['versions'].items())
    examples=[('0013','4492：局部转接提议与 86 节点混合下游；上游复核撤回剪断'),('0151','2558 → 2559：中间阶段移出 3 节点，最终因引用切片不覆盖边而否决'),('0256','4583 附近：旧 target 改为 unknown，连接仍 KEEP'),('0019','6663：模型提出 other / 连接 contradicted，但未定位剪断边，仍 REVIEW'),('0147','本轮联合复核预算未覆盖的局部段；关联三个历史上游病例')]
    example_links=''.join('<li>'+link(str(main_run/'index.html')+'#case-'+cid,label)+' · '+link(main_run/'cases'/cid/'interactive.html','局部原图')+' · '+link(main_run/'cases'/cid/'global_context.html','完整来路')+'</li>' if (main_run/'cases'/cid/'global_context.html').exists() else '<li>'+link(main_run/'cases'/cid/'interactive.html',label)+'</li>' for cid,label in examples)
    source_files=['algorithms/tracing/ownership_audit.py','visualization/ownership_audit.py','algorithms/tracing/ownership_consistency.py','algorithms/tracing/ownership_joint_review.py','algorithms/tracing/soma_ownership_context.py','tools/run_ownership_audit.py','tools/refine_ownership_conflicts.py','tools/run_soma_ownership_pipeline.py']
    sources='<ul>'+''.join('<li>'+link(ROOT/f,f)+'</li>' for f in source_files)+'</ul>'
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v0.84 Gemma 实验交付</title><style>body{font:16px/1.7 system-ui;max-width:1400px;margin:35px auto;padding:20px;color:#182737}h1{font-size:28px}td,th{padding:12px;border-bottom:1px solid #cfd7e0;text-align:left}table{width:100%}a{color:#175ba6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f7;padding:16px}.result{padding:18px;background:#fff4db;border-left:5px solid #b57b16}</style>'''
    doc+='<h1>v0.84：Gemma 原图审核与双侧冲突复核</h1><div class="result"><strong>流程已实现并完成实验，尚未证明剪枝效果改善。</strong><p>97117 最终没有执行剪断，保留 5943 个节点；它仍可能含有其他神经元分支。15 条拟剪建议全部未达到执行条件。没有人工真值，准确率、误剪率和目标长度召回未评估。</p></div>'
    doc+='<h2>结果入口</h2><table><tr><th>ID</th><th>范围</th><th>送审</th><th>KEEP / PROPOSE_CUT / REVIEW</th><th>实际剪边 / 移出节点</th><th>调用</th><th>可检查结果</th></tr>'+''.join(rows)+'</table><p>KEEP 只描述当前连接；unknown 表示归属仍不清楚。候选来源可多选，按来源计数的合计不等于去重后的病例数。</p>'
    doc+='<h2>97117 的变化与限制</h2><p>867 条来源提议合并成 362 个候选，其中 16 个位于保护范围，实际送审 346 个。完整审核后对 5 个上游病例联合复核，均撤回对应剪断假设；随后复核 5 个下游段，其中一个旧 target 改为 unknown。5 个上游病例由 1 个当前模型冲突及 4 个历史关注位置组成，不能称作确认了 5 条目标路径。另有 1 个不同局部段因预算未进一步联合复核（记录关联三个上游病例）。</p>'
    doc+='<p>最终 157 KEEP、15 PROPOSE_CUT、174 REVIEW；15 条提议均被执行检查保留。总计 420 次调用、47 次补图。模型对同一接点的复核会改变结论，也会引用未穿过拟剪边的切片。定位和结构校验能够拦截这些具体问题，不能证明其他生物学判断正确。</p>'
    doc+='<p>中间阶段曾生成 1 处、3 节点的实验剪断；最终定位审计否决了它，最终结果重新从同一份模型介入前 SWC 导出。中间组件保留在 '+link(base/'20261111_97117_raw_audit_v4/components/0151.swc','中间组件 SWC')+' 和 '+link(base/'20261111_97117_raw_audit_v4/components/0151_original_parents.json','原父节点审计')+'。最终没有移出组件。</p>'
    doc+='<h2>同输入对照</h2><table><tr><th>版本</th><th>节点</th><th>长度 µm</th><th>末梢</th></tr>'+versions+'</table><p>v0.81、v0.83b、v2、v3_checked 都是历史算法结果，不是人工真值。旧 v3_checked 的 11 条冲突提议对应 4 个不同剪断位置。</p>'
    doc+='<h2>先检查这些具体位置</h2><ul>'+example_links+'</ul><p>逐病例页面保留模型原话、实际输入 PNG、连续切片、上下游上下文与程序动作；可按节点或候选编号搜索。交互窗口是人工检查工具，不是额外的模型输入。</p>'
    doc+='<h2>实际修改与提示词</h2><p>候选合并、元数据目标验证、原图优先证据、可请求补图、同模型剪点复核、上下游联合复核和组件审计已接入参数化实验入口。基础剪枝文件本轮未改；严格小于 7.75 µm 的前置规则及现有其他基础规则保留。</p>'+sources
    doc+='<p>两套模板全文：'+link(ROOT/'prompts/ownership_audit_v4.txt','归属/连接初审')+' · '+link(ROOT/'prompts/attachment_review_v4.txt','剪点/联合复核')+'。各样本完整报告也内嵌全文。每次展开文本、文件名说明、图像顺序、参数和原始响应可从证据导航逐一打开。</p>'
    doc+='<h2>复现与工程验证</h2><p>'+link(ROOT/'OWNERSHIP_AUDIT.txt','完整运行说明')+' · '+link(args.config,'三样本配置')+' · '+link(out/'final_suite_validation.json','实际 SWC/组件/输入/图像校验')+' · '+link(out/'test_summary.json',str(test['unique_tests'])+' 项针对性测试')+' · '+link(main_run/'vision_input_audit.json','真实模型视觉输入审计')+'</p><p>模型复用本地官方 QAT 四位量化 Gemma 4 31B 与既有 llama.cpp。实际视觉编码约 1089–1160 token/图；源数据仍为 1 µm 体素。未重跑 nnU-Net、APP2、G-Cut。</p>'
    doc+='<p>统计导出阶段修复过一个受保护病例缺少 reviews 字段的问题，'+link(main_run/'export_repair_identity.json','修复审计')+' 校验推理代码 AST 未变化、只改变导出函数，保留两版源码，没有重复模型调用。</p>'
    doc+='<p>106245、95918 相关文件已分别复制 64 个并校验哈希，保存在 test/0.84/data；本轮模型实验没有包含这两个样本。</p>'
    doc+='<details><summary>全部统计</summary><pre>'+html.escape(json.dumps(stats,ensure_ascii=False,indent=2))+'</pre></details></html>'
    (out/'overview.html').write_text(doc);shutil.copy2(__file__,out/Path(__file__).name);print(out/'overview.html')
if __name__=='__main__':main()
