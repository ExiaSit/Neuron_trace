#!/usr/bin/env python3
"""Read-only report of an in-progress or completed soma branch walk."""
from pathlib import Path
import argparse,html,json
from collections import Counter

def raw_outlets(group):
 judgment=group["calls"][-1].get("judgment")
 outlets=judgment.get("outlets") if isinstance(judgment,dict) else None
 return outlets if isinstance(outlets,list) else []

def report(out):
 out=Path(out);r=json.loads((out/'diagnosis.json').read_text());done=[v for v in r['visits'] if v.get('status')=='reviewed'];skipped=[v for v in r['visits'] if v.get('status')=='detached_before_visit'];calls=[];rows=[];decisions=[];groups=[]
 for f in sorted((out/'cases').glob('*/result.json')):
  g=json.loads(f.read_text());groups.append(g);calls+=g['calls'];ds=g['judgment']['outlets'];decisions+=ds
  case=f.parent.name;relative=f.parent.relative_to(out);links=' · '.join(f'<a href="{relative}/{Path(p).name}">{html.escape(Path(p).stem)}</a>' for p in g['images'])
  rows.append(f'<details><summary>{case}: '+html.escape(', '.join(d['branch_id']+' '+d['decision'] for d in ds))+'</summary><pre>'+html.escape(json.dumps(g['judgment'],ensure_ascii=False,indent=2))+'</pre><p>'+links+'</p><p><a href="'+str(relative)+'/result.json">输入图映射与实际请求、响应记录</a></p></details>')
 status=dict(reviewed_junctions=len(done),all_input_junctions=len(r['visits']),detached_before_visit=len(skipped),completed_calls=len(calls),outlet_decisions=dict(Counter(d['decision'] for d in decisions)),model_seconds=sum(c.get('seconds',0) for c in calls),final_export=(out/'summary.json').exists(),ownership_counts=dict(Counter(d['target_ownership'] for d in decisions)),upstream_status_counts=dict(Counter(g['judgment']['upstream_status'] for g in groups)),invalid_output_cases=[g['case_id'] for g in groups if g['errors']],supplement_requests=[dict(case_id=g['case_id'],views=g['requested']) for g in groups if g['requested']],model_last_outlet_decisions=dict(Counter(d.get('decision','missing') for g in groups for d in raw_outlets(g) if isinstance(d,dict))))
 title=r['input']['cell']+' Gemma 从 soma 逐分叉审核'
 (out/'progress.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:20px}pre{white-space:pre-wrap}details{border:1px solid #bbb;padding:12px;margin:12px 0}summary{cursor:pointer}</style><h1>'+title+'</h1><p>旧剪枝步骤全部关闭。从未剪枝的 SWC 开始。每条前向路径连续展开三个 branch 段；已走过的完整上游仍是假设。</p><p>此页是已落盘结果快照；未完成时不代表完整测试，也不产生最终 SWC。没有人工真值，生物学准确率未评估。</p><pre>'+html.escape(json.dumps(status,ensure_ascii=False,indent=2))+'</pre>'+('<p><a href="index.html">已完成：完整结果及三视图联动</a></p>' if status['final_export'] else '')+''.join(rows))
 (out/'progress.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
 if status['final_export']:
  s=json.loads((out/'summary.json').read_text());plan=json.loads((out/'execution_plan.json').read_text());validation=json.loads((out/'validation.json').read_text());runtime=json.loads((out/'runtime.json').read_text())
  checks=[]
  for f in (out/'cases').glob('*/*_response.json'):
   v=json.loads(f.read_text());checks.append(dict(file=str(f.relative_to(out)),finish_reason=v['choices'][0].get('finish_reason'),usage=v.get('usage',{})))
  (out/'response_validation.json').write_text(json.dumps(checks,indent=2))
  text=[r['input']['cell']+' / Gemma soma branch walk 实验报告', '输入与流程',json.dumps(r['input'],ensure_ascii=False,indent=2), '结果（没有人工标签，不评价准确率）',json.dumps(s,ensure_ascii=False,indent=2),'工程验证',json.dumps(validation,indent=2),'实际剪断及影响',json.dumps(plan['actions'],ensure_ascii=False,indent=2),'查看入口：index.html；逐个分叉的原图、实际提示词与原始响应均可点击；junctions/J*/interactive.html 为共享 XYZ 光标的三视图。','CUT 只断开模型明确定位的当前分叉出口边。上游已疑似串线、或分支中段无法定位的疑点保持 REVIEW；不会生成重连路径。下游组件单独保存，不宣称其全部属于其他神经元。','模型断言与程序执行分开记录：cases/*/round*_response.json 是模型原始回答；result.json 是格式/定位检查后的判断；execution_plan.json 是实际执行。','没有执行旧短末梢/胞体/近点/Zigzag规则；未改动输入或保留节点的数据。','交互页面未通过浏览器实测：自动浏览器安全策略禁止 file://。已运行渲染坐标与组件导出测试。','实际最终提示词：cases/*/round*_prompt.txt（模板+该分叉坐标和图片清单）；模板全文如下：',(out/'source/prompts/soma_branchwalk_v1.txt').read_text(),'模型启动参数',json.dumps(runtime['cmdline'],ensure_ascii=False), '复现：/home/pzy/anaconda3/envs/hb_seg/bin/python /data/disk/Neuron_tracing_method/v0.84/tools/run_soma_branchwalk.py --cell '+r['input']['cell']+' --run-name <新目录名>']
  (out/'REPORT.txt').write_text('\n\n'.join(text)+'\n')
 return status
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();print(json.dumps(report(args.run),ensure_ascii=False))
