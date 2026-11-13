#!/usr/bin/env python3
"""Human-readable cross-version and per-edge evidence report, no accuracy claims."""
from pathlib import Path
import argparse,html,json,sys,hashlib,re
from collections import Counter
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from algorithms.tracing.connection_diagnostics import save_json
from visualization.ownership_audit import view_panel,PLANES
import matplotlib.pyplot as plt
BASE=ROOT.parent/'test/0.84'

def read(p):return json.loads(Path(p).read_text())
def lines(tree):
    d={int(n[0]):n for n in tree};return [[d[int(n[6])][2:5],n[2:5]] for n in tree if int(n[6]) in d]

def comparisons(run,report,summary,plan):
    raw=tifffile.imread(report['input']['raw']);versions={k:np.loadtxt(v['path'],ndmin=2) for k,v in summary['versions'].items() if k not in ['v0.81','v0.83b']}
    cases=[c for c in report['cases'] if any(s.startswith('historical') for s in c['sources'])]
    execute_ids={a['candidate_id'] for a in plan['actions'] if a['status']=='execute'}
    cases+= [c for c in report['cases'] if c['candidate_id'] in execute_ids and c not in cases]
    # Ordinary forks, projection overlap, weak signals supplement history-focused examples.
    for source in ['bifurcation','swc_xy_crossing','interior_weak_signal']:
        cases+=next(([c] for c in report['cases'] if source in c['sources'] and c.get('reviews') and c not in cases),[])
    directory=run/'local_comparisons';directory.mkdir(exist_ok=True);records=[]
    for c in cases:
        center=np.array(c['center_xyz']);lo=np.maximum(0,np.floor(center-20)).astype(int);hi=np.minimum(raw.shape[::-1],np.ceil(center+21)).astype(int)-1
        fig,axs=plt.subplots(len(versions)+1,3,figsize=(13,3.8*(len(versions)+1)),layout='constrained')
        for ax,(name,axis,plane) in zip(axs[0],PLANES):view_panel(ax,raw,lo,hi,axis,plane,center,'RAW '+name)
        for row,(version,tree) in enumerate(versions.items(),1):
            routes=[(p,'cyan') for p in lines(tree)]
            for ax,(name,axis,plane) in zip(axs[row],PLANES):view_panel(ax,raw,lo,hi,axis,plane,center,version+' '+name,routes)
        fig.suptitle(f'{c["candidate_id"]} / node {c["anchor_id"]}: matched field, raw on top. Overlays are hypotheses, not labels.')
        p=directory/f'{c["candidate_id"]}.png';fig.savefig(p,dpi=130);plt.close(fig)
        records.append(dict(candidate_id=c['candidate_id'],node_id=c['anchor_id'],sources=c['sources'],model=c.get('judgment'),image=str(p.relative_to(run)),executed=c['candidate_id'] in execute_ids,interactive=f'cases/{c["candidate_id"]}/interactive.html'))
    save_json(directory/'index.json',records);return records

def vision_audit(runtime,log,run):
    text=Path(log).read_text(errors='replace');tokens=[int(x) for x in re.findall(r'image_tokens->nx = (\d+)',text)]
    requests=[read(p) for p in run.glob('cases/*/*/round*_request.json')]
    responses=[read(p) for p in run.glob('cases/*/*/round*_response.json')]
    checks=[]
    from PIL import Image
    for req in requests:
        checks.append(dict(fingerprint=req['fingerprint'],count=req['image_count'],matches_list=req['image_count']==len(req['images']),images=[dict(path=i['path'],sha_matches=hashlib.sha256(Path(i['path']).read_bytes()).hexdigest()==i['sha256'],size=list(Image.open(i['path']).size)) for i in req['images']]))
    return dict(cmdline=runtime['cmdline'],model_files=runtime['files'],context_per_slot=16384,requests=len(requests),saved_image_count=sum(r['image_count'] for r in requests),
        logged_image_chunks=len(tokens),visual_token_histogram=dict(Counter(tokens)),min_visual_tokens=min(tokens) if tokens else None,max_visual_tokens=max(tokens) if tokens else None,
        max_prompt_tokens=max([r.get('usage',{}).get('prompt_tokens',0) for r in responses] or [0]),max_total_tokens=max([r.get('usage',{}).get('total_tokens',0) for r in responses] or [0]),
        non_stop_responses=sum(r['choices'][0].get('finish_reason')!='stop' for r in responses),context_truncation_lines=len(re.findall('truncated = 1',text)),
        checks=checks,log=str(log),log_scope='shared final server log; includes representative-cell calls if already run; match aggregate across runs',
        note='Actual encoder grids may round above nominal 1120. More visual tokens do not add biological source resolution; RAW data remain 1 um voxels.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--server-log',type=Path,required=True);args=ap.parse_args();run=args.run
    report=read(run/'diagnosis.json');summary=read(run/'summary.json');plan=read(run/'execution_plan.json');joint=read(run/'joint_reviews.json')
    records=comparisons(run,report,summary,plan);vision=vision_audit(read(run/'runtime.json'),args.server_log,run);save_json(run/'vision_input_audit.json',vision)
    src=run/'source/prompts';templates={p.name:p.read_text() for p in src.glob('*.txt')}
    template_html=''.join(f'<h3>{html.escape(k)}</h3><pre>{html.escape(v)}</pre>' for k,v in templates.items())
    table=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["nodes"]}</td><td>{v["length_um"]:.2f}</td><td>{v["tips"]}</td></tr>' for k,v in summary['versions'].items())
    local=''.join(f'<article><h3>{r["candidate_id"]} / 节点 {r["node_id"]}</h3><p>{html.escape(", ".join(r["sources"]))} · 实际执行：{r["executed"]}</p><a href="{r["interactive"]}">查看连续切片/薄层</a> · <a href="index.html#case-{r["candidate_id"]}">模型输入、判断与程序动作</a><a href="{r["image"]}"><img loading="lazy" src="{r["image"]}"></a></article>' for r in records)
    actions=''
    for a in plan['actions']:
        cid=a['candidate_id'];parts=f'<a href="components/{cid}.swc">独立组件 SWC</a> · <a href="components/{cid}_original_parents.json">原连接映射</a>' if a['status']=='execute' else ''
        actions+=f'<h3>{cid}: {a["edge"]["parent_id"]} → {a["edge"]["child_id"]} / {a["status"]}</h3><p><a href="index.html#case-{cid}">原图证据与两轮请求响应</a> · {parts}</p><pre>{html.escape(json.dumps(a,ensure_ascii=False,indent=2))}</pre>'
    if not actions:actions='<p>本轮没有达到程序执行条件的拟剪断边。没有生成移除组件。</p>'
    command=f'/home/pzy/anaconda3/envs/hb_seg/bin/python {ROOT}/tools/run_ownership_audit.py --cell {summary["cell"]} --swc {report["input"]["swc"]} --run-name {run.name} --workers {report["input"]["workers"]}'
    if report['input']['history']:command+=' --history '+str(report['input']['history'])
    if report['input']['representative']:command+=' --representative '+str(report['input']['representative'])
    command+='\n# 相同目录用于严格指纹续跑；独立重跑时改为新的 run-name。'
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Gemma v4 实验交付</title><style>body{{max-width:1400px;margin:30px auto;padding:20px;font:16px/1.6 system-ui}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f6;padding:16px}}img{{max-width:100%}}td,th{{padding:8px 20px;border-bottom:1px solid #ccc}}article{{border-top:2px solid #ccc;margin-top:30px}}</style>
<h1>{summary['cell']}：Gemma 原图优先连接审核实验</h1>
<p>模型结论、程序执行与未解决区域分开记录。没有人工归属真值，准确率未评估。删除数量、JSON 合法及 SWC 拓扑检查均不代表生物学准确性提高。</p>
<p><a href="index.html">全部病例/原图</a> · <a href="diagnosis.json">完整诊断</a> · <a href="execution_plan.json">执行审计</a> · <a href="joint_reviews.json">冲突复核</a> · <a href="vision_input_audit.json">真实视觉输入审计</a> · <a href="validation.json">结构检查</a></p>
<h2>实际代码变化与限制</h2><p>新增 ownership_audit.py 候选合并、严格输出验证和可追踪组件导出；新增原图优先 renderer；新增参数化 run_ownership_audit.py；扩展 ownership_consistency.py 联合复核；soma_ownership_context.detect_somas 增加显式元数据目标选项，默认旧行为不变。基础 prune.py、terminal_pruning.py 本轮未改。</p>
<p>候选包括胞体竞争、SWC XY 交错、原图交叉、分叉、弱信号/方向异常及历史关注点。去重限定同一无分叉段与 8 µm 三维距离。全上游每 24 µm 提供局部请求点，局部视野半径 18 µm；最多两次补图，不明确则 REVIEW。该预算不能保证长路径全部核实，未检查的路径也不能成为可信锚点。</p>
<p>第一轮实际只发送原图、A/B/J 定位点和胞体位置；完整路径标注保存在匹配图片中，可补充请求或由第二轮查看。两轮都由同一个 Gemma 完成，不能称为独立模型验证。首次试运行仍引用彩线路径的结果已单独保留，没有混入本次缓存。</p>
<p>执行要求：合法候选边，模型判定连接 contradicted 并 PROPOSE_CUT，同一模型第二轮选择同一边，输入哈希一致，目标胞体受保护，无未解决的下游 target/unknown 混合判断或未审分叉。所有提议先保存完整影响范围，再统一处理祖先/后代重叠。移除组件只代表从目标树断开，不代表已确认属于另一个神经元；不自动重连。</p>
<h2>统计与公平对照</h2><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre><table><tr><th>版本</th><th>节点</th><th>长度 µm</th><th>末梢</th></tr>{table}</table><img src="comparison.png">
<h2>实际动作与影响范围</h2>{actions}<h2>联合复核</h2><pre>{html.escape(json.dumps(joint,ensure_ascii=False,indent=2))}</pre>
<h2>两套提示词全文</h2><p>以下是版本化模板全文；每次实际发送的完整文本（含病例、允许边、视图与补图菜单）保存在 cases/&lt;ID&gt;/&lt;phase&gt;/roundN_prompt.txt，请求清单与原始响应在相邻 JSON 中。</p>{template_html}
<h2>复现命令</h2><pre>{html.escape(command)}</pre><p>输入为既有模型介入前 SWC，未重跑 nnU-Net、APP2 或 G-Cut。源码、模板、输入、模型 GGUF、视觉编码文件、启动参数均保存指纹；更改后必须使用新运行目录。</p>
<h2>代表性同视野局部对照</h2>{local}</html>'''
    (run/'report.html').write_text(doc);save_json(run/'report_source.json',dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(run/'report.html')
if __name__=='__main__':main()
