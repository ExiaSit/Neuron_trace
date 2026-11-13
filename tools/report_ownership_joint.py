#!/usr/bin/env python3
"""Report a staged bilateral audit, including immutable parent model requests."""
from pathlib import Path
import argparse,html,json,sys,os,hashlib,re,shutil
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from tools.report_ownership_audit import comparisons
from algorithms.tracing.connection_diagnostics import save_json
from PIL import Image

def read(path):return json.loads(Path(path).read_text())
def pre(value):return '<pre>'+html.escape(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))+'</pre>'
def link(path,label,run):return '<a href="'+os.path.relpath(path,run)+'">'+html.escape(label)+'</a>'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--server-log',type=Path,required=True);args=ap.parse_args();run=args.run.resolve()
    report=read(run/'diagnosis.json');summary=read(run/'summary.json');plan=read(run/'execution_plan.json');runtime=read(run/'runtime.json');joint=read(run/'joint_reviews.json');parent=Path(summary['parent_run'])
    requests=[Path(p) for p in read(run/'model_request_refs.json')];responses=[];checks=[];calls=[]
    for path in requests:
        req=read(path);images=[]
        for item in req['images']:
            p=Path(item['path']);ok=hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
            if not ok:raise ValueError('Changed input image')
            with Image.open(p) as im:size=list(im.size)
            images.append(dict(path=str(p),sha_matches=ok,size=size))
        checks.append(dict(request=str(path),count=req['image_count'],matches_list=req['image_count']==len(images),images=images))
        response=path.with_name(path.name.replace('_request.json','_response.json'))
        if response.exists():responses.append(read(response))
        call=path.with_name(path.name.replace('_request.json','_call.json'))
        if call.exists():calls.append(read(call))
    log=args.server_log.read_text(errors='replace');tokens=[int(x) for x in re.findall(r'image_tokens->nx = (\d+)',log)]
    vision=dict(cmdline=runtime['cmdline'],model_files=runtime['files'],context_per_slot=16384,requests=len(requests),saved_image_count=sum(r['count'] for r in checks),checks=checks,
        logged_image_chunks=len(tokens),visual_token_histogram=dict(Counter(tokens)),min_visual_tokens=min(tokens) if tokens else None,max_visual_tokens=max(tokens) if tokens else None,
        max_prompt_tokens=max([r.get('usage',{}).get('prompt_tokens',0) for r in responses] or [0]),max_total_tokens=max([r.get('usage',{}).get('total_tokens',0) for r in responses] or [0]),
        non_stop_responses=sum(r['choices'][0].get('finish_reason')!='stop' for r in responses),call_errors=sum('error' in c for c in calls),context_truncation_lines=len(re.findall('truncated = 1',log)),
        log=str(args.server_log),log_scope='Shared server log includes interrupted pilots, main audit and representative samples. Aggregate encoder count is not a per-run image count. Actual sent payload bytes/count are unit-tested and all saved images hash-checked.',
        note='Vision token grid rounding may exceed nominal1120. Raw volume remains 1 um voxels. Context overflow or truncated output fails closed to REVIEW.')
    save_json(run/'vision_input_audit.json',vision)
    records=comparisons(run,report,summary,plan)
    versions=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["nodes"]}</td><td>{v["length_um"]:.3f}</td><td>{v["tips"]}</td></tr>' for k,v in summary['versions'].items())
    actions=[]
    for a in plan['actions']:
        cid=a['candidate_id'];links=[link(run/'index.html', '逐病例原图与判断（查找 '+cid+'）',run)]
        if a['status']=='execute':links += [link(run/'components'/f'{cid}.swc','移出组件 SWC',run),link(run/'components'/f'{cid}_original_parents.json','原父节点映射',run)]
        actions.append(f'<h3>{cid}: {a["edge"]["parent_id"]} → {a["edge"]["child_id"]} / {a["status"]}</h3>'+ ' · '.join(links)+pre(a))
    local=''.join(f'<article><h3>{r["candidate_id"]} / 节点 {r["node_id"]}</h3><p>{html.escape(", ".join(r["sources"]))}；实际执行：{r["executed"]}</p><a href="{r["interactive"]}">局部原图交互</a> · <a href="index.html#case-{r["candidate_id"]}">模型输入/输出</a><a href="{r["image"]}"><img loading="lazy" src="{r["image"]}"></a></article>' for r in records)
    templates=''.join('<h3>'+html.escape(p.name)+'</h3>'+pre(p.read_text()) for p in sorted((run/'source/prompts').glob('*.txt')))
    inp=report['input'];command=f'/home/pzy/anaconda3/envs/hb_seg/bin/python {ROOT}/tools/run_ownership_audit.py --cell {summary["cell"]} --swc {inp["swc"]} --run-name {parent.name} --workers {inp["workers"]}'
    if inp['history']:command+=' --history '+inp['history']
    if inp['representative']:command+=' --representative '+str(inp['representative'])
    command+=f'\n/home/pzy/anaconda3/envs/hb_seg/bin/python {ROOT}/tools/refine_ownership_conflicts.py --parent-run {parent} --run-name {run.name} --max-segment-reviews {read(run/"stage_identity.json")["maximum"]}'
    command+=f'\n/home/pzy/anaconda3/envs/hb_seg/bin/python {ROOT}/tools/report_ownership_joint.py --run {run} --server-log {args.server_log.resolve()}\n/home/pzy/anaconda3/envs/hb_seg/bin/python {ROOT}/tools/build_audit_delivery.py --run {run} --browser-check'
    command+='\n# 新的独立复现实验应更换两个 run-name 及对应 parent-run。原目录只支持严格指纹匹配的续跑。'
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Gemma 原图与双侧冲突复核</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f6;padding:16px}img{max-width:100%}td,th{padding:8px 20px;border-bottom:1px solid #ddd}article{border-top:2px solid #ddd;margin-top:32px}</style>'''
    doc+=f'<h1>{summary["cell"]}：原图优先连接审核与双方局部段复核</h1><p><a href="delivery.html">逐病例证据导航</a> · <a href="diagnosis.json">完整诊断</a> · <a href="execution_plan.json">执行与影响审计</a> · <a href="joint_reviews.json">冲突前后结论</a> · <a href="vision_input_audit.json">实际视觉输入审计</a> · <a href="validation.json">结构检查</a></p>'
    doc+='<p>没有人工归属标签，生物学准确率未评估。KEEP / target 仍是模型假设；REVIEW 表示尚不知道。所有动作均从首次 Gemma 介入前的同一 SWC 重新计算，没有在 v3 或本轮第一阶段输出上继续删枝。模型第二轮和联合复核均来自同一 Gemma，不是独立模型验证。</p>'
    doc+='<h2>现状差异与实际改动</h2><p>旧流程依赖胞体竞争候选、彩色路径假设和一次局部判断，冲突主要被拦截。现在合并胞体竞争、SWC XY 交错、原图交叉、分叉、路径中段弱信号/方向线索及历史关注点；按同一无分叉段与 8 µm 三维距离去重。找不到其他胞体也可审核连接，所有几何线索都不能自动删除。</p>'
    doc+='<p>已知目标来自样本元数据，验证体素坐标、图像边界、根距离和局部信号；其他胞体、SWC 上游与图像骨架竞争路径均标为假设。当前没有人工确认路径。完整路径保存在诊断中，全局原图与局部细节分开。首轮发送局部原图、连续切片与全局原图，彩色假设保存在匹配视图中，供补图或复核。</p>'
    doc+='<p>新增 ownership_audit.py、visualization/ownership_audit.py 与参数化 run_ownership_audit.py；扩展 ownership_consistency.py；soma_ownership_context.py 增加显式元数据目标支持（旧默认行为不变）。ownership_joint_review.py 与 refine_ownership_conflicts.py 在上游复核后进一步检查下游段，保存旧 target 与新判断；每冲突最多选两段，总预算 12 段。新标签不传播到整棵子树。基础 prune.py / terminal_pruning.py 本轮未改，已有未提交改动保留。</p>'
    doc+='<h2>补图、边定位与执行条件</h2><p>局部半径 18 µm，可扩大到 30 µm；全上游与竞争路径每 24 µm 提供请求站点。最多两次补图是每候选所有阶段共享预算。可请求更多切片、上游交叉、竞争路线疑点或候选胞体。仍不清楚则 REVIEW；不会机械选择更靠近根的祖先。全上游可访问不等于每处都已经核实，有界预算无法证明长路径全程归属。</p>'
    doc+='<p>执行必须同时满足：合法且准确定位的允许边；连接 contradicted + PROPOSE_CUT；同一模型复核选择相同边；引用局部深度原图覆盖该边；输入版本一致；没有目标胞体、target/unknown 混合判断、未审核分叉或未解决冲突。视图引用与格式检查只验证完整性，不证明图像判断正确。输出诊断与计划先落盘，再统一处理重复与祖先/后代剪断。每个组件保留节点 ID，独立根 parent=-1，原 parent 另存审计；不自动生成重连。</p>'
    doc+='<h2>模型输出结构</h2>'+pre(dict(candidate_id='provided ID',image_support='supported / unsupported / uncertain',attachment_validity='supported / contradicted / uncertain',target_ownership='target / other / unknown',other_soma_id='provided ID or null',cut_edge='allowed {parent_id, child_id} or null',recommendation='KEEP / PROPOSE_CUT / REQUEST_MORE / REVIEW',evidence=[dict(view_id='existing raw view',region_id='view_bounds or explicit in-view xyz_bounds',observation='具体原图观察')],missing_evidence=[],requested_views=['existing request option']))
    doc+='<h2>统计与同输入整体对照</h2>'+pre(summary)+'<table><tr><th>版本</th><th>节点</th><th>长度 µm</th><th>末梢</th></tr>'+versions+'</table><img src="comparison.png">'
    doc+='<h2>每条拟剪边及实际动作</h2>'+(''.join(actions) or '<p>没有达到执行条件的剪断边，没有移出组件。</p>')
    doc+='<h2>冲突复核前后及未解决病例</h2>'+pre(joint)
    doc+='<h2>两套最终提示词模板全文</h2><p>每个病例真正发送的展开文本（含允许边、坐标、视图清单、补图菜单和作用范围）由 delivery.html 链接到 roundN_prompt.txt；图片与 API 参数在同名 request.json，原始模型回答在 response.json。第一轮旧结论不传给第二轮。</p>'+templates
    doc+='<h2>可复现命令</h2>'+pre(command)+'<p>服务复用现有官方 QAT GGUF 与视觉编码文件，未安装其他推理框架。实际启动参数与文件 SHA256：</p>'+pre(dict(cmdline=runtime['cmdline'],model_files=runtime['files']))
    doc+='<h2>代表性同视野局部对照</h2>'+local+'</html>'
    (run/'report.html').write_text(doc);target=run/'source/tools/report_ownership_joint.py';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,target)
    save_json(run/'report_source.json',dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()));print(run/'report.html')
if __name__=='__main__':main()
