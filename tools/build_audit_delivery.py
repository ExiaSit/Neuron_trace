#!/usr/bin/env python3
"""Link all evidence and validate exported audit artifacts without changing decisions."""
from pathlib import Path
from html.parser import HTMLParser
import argparse,html,json,hashlib,sys,shutil,os
from copy import deepcopy
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from algorithms.tracing.connection_diagnostics import save_json
import numpy as np
import tifffile
from visualization.projection_viewer import write_site_viewer
from algorithms.tracing import ownership_audit as oa

def local_view(run,raw,tree,c):
    """Human-only viewer for a staged run referencing immutable parent evidence."""
    from algorithms.tracing.soma_ownership_context import tree_paths
    nodes,_,paths=tree_paths(tree);p=np.array(c['center_xyz']);lo=np.maximum(0,np.floor(p-24)).astype(int);hi=np.minimum(raw.shape[::-1],np.ceil(p+25)).astype(int)
    crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
    arms=[dict(xyz=[nodes[n][2:5] for n in path],diagnosis=dict(status='hypothesis')) for path in paths if any(np.all((np.array(nodes[n][2:5])>=lo)&(np.array(nodes[n][2:5])<hi)) for n in path)]
    folder=run/'cases'/c['candidate_id'];folder.mkdir(parents=True,exist_ok=True)
    source=c['reviews']['ownership']['images'][0]
    site=dict(site_id=c['candidate_id'],kind='joint ownership audit',status=c['judgment']['recommendation'],center_xyz=p.tolist(),image=os.path.relpath(source,folder),bbox_xyz=np.array([lo,hi]).T.tolist(),arms=arms,context_xyz=[],pair_xyz=[])
    write_site_viewer(crop,lo,site,folder/'interactive.html')

def cut_view(run,raw,tree,c,action):
    nodes={int(n[0]):n for n in tree};edge=action['edge'];parent=nodes[edge['parent_id']][2:5];child=nodes[edge['child_id']][2:5]
    p=np.array(child);lo=np.maximum(0,np.floor(p-18)).astype(int);hi=np.minimum(raw.shape[::-1],np.ceil(p+19)).astype(int)
    crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]];folder=run/'cases'/c['candidate_id']
    site=dict(site_id=c['candidate_id']+' cut edge',kind='exact parent-child edge',status=action['status'],center_xyz=child,bbox_xyz=np.array([lo,hi]).T.tolist(),image=os.path.relpath(c['reviews']['ownership']['images'][0],folder),context_xyz=c['upstream_xyz'],pair_xyz=[parent,child],arms=[dict(xyz=[parent,child],diagnosis=dict(status='disconnected'))])
    dest=folder/'cut_edge.html';write_site_viewer(crop,lo,site,dest)
    text=dest.read_text().replace('定位交错分支 ${i+1}',"定位${i===0?'父节点 P':'子节点 C'}")
    text=text.replace('<h1 id="title">',f'<p style="padding:12px">拟剪边 P={edge["parent_id"]} {parent} → C={edge["child_id"]} {child}。程序状态：{action["status"]}。红线仅标识待检查的 SWC 边，不是图像证据；这是人工查看工具，未作为额外模型输入。</p><h1 id="title">')
    dest.write_text(text)

def global_view(run,raw,c,report):
    points=c['upstream_xyz']+c['other_route_xyz']+c['downstream_xyz']+[report['target']['known_target_xyz']]
    somas={x['soma_id']:x for x in report['somas']};pair=[report['target']['known_target_xyz']]
    if c['competing_soma_id'] is not None:pair.append(somas[c['competing_soma_id']]['center_xyz']);points+=pair[1:]
    xyz=np.array(points);lo=np.maximum(0,np.floor(xyz.min(0)-8)).astype(int);hi=np.minimum(raw.shape[::-1],np.ceil(xyz.max(0)+9)).astype(int)
    crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
    folder=run/'cases'/c['candidate_id'];source=next((p for p in c['reviews']['ownership']['images'] if p.endswith('_global_raw.png')),c['reviews']['ownership']['images'][0])
    site=dict(site_id=c['candidate_id']+' 全局',kind='上游与胞体',status='路径待核实；仅供人工复核',center_xyz=c['center_xyz'],bbox_xyz=np.array([lo,hi]).T.tolist(),
              image=os.path.relpath(source,folder),context_xyz=c['upstream_xyz'],arms=[],pair_xyz=pair)
    site['arms']=[dict(xyz=p,diagnosis=dict(status='uncertain')) for p in [c['downstream_xyz'],c['other_route_xyz']] if len(p)>1]
    dest=run/'cases'/c['candidate_id']/'global_context.html';write_site_viewer(crop,lo,site,dest)
    content=dest.read_text().replace('定位交错分支 ${i+1}',"定位${i===0?'目标胞体':'候选胞体'}")
    content=content.replace('<body>','<body><p style="padding:12px">完整上游与胞体位置，仅用于人工复核。青色为 SWC 上游假设，黄色为下游/竞争路线假设，均不是归属真值；这些交互切片不是额外的模型输入。</p>')
    content=content.replace('这是原始体素的显示视图。','青色为 SWC 上游假设，黄色为下游/竞争路径假设，均不是归属真值。此交互图仅用于人工复核，并非额外模型输入。这是原始体素的显示视图。')
    dest.write_text(content)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--browser-check',action='store_true');args=ap.parse_args();run=args.run.resolve()
    report=json.loads((run/'diagnosis.json').read_text());plan=json.loads((run/'execution_plan.json').read_text());summary=json.loads((run/'summary.json').read_text());rows=[]
    raw=tifffile.imread(report['input']['raw']);tree=np.loadtxt(report['input']['swc'],ndmin=2).tolist()
    inventory=[];request_files=[Path(p) for p in json.loads((run/'model_request_refs.json').read_text())] if (run/'model_request_refs.json').exists() else list(run.glob('cases/*/*/round*_request.json'))
    for f in request_files:
        req=json.loads(f.read_text());checks=[]
        for item in req['images']:
            checks.append(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'])
        if req['image_count']!=len(req['images']) or not all(checks):raise ValueError('Changed model image evidence: '+str(f))
        # Preserve the actual order of every transmitted text part as well as
        # image references, including the filename captions added by model_call.
        content=[dict(type='text',text=req['final_text'])]
        for item in req['images']:
            content.append(dict(type='text',text='Image file: '+Path(item['path']).name))
            content.append(dict(type='image_reference',**item))
        folder=run/'sent_content'/f.parent.parent.name/f.parent.name;folder.mkdir(parents=True,exist_ok=True)
        stem=f.name.replace('_request.json','')
        save_json(folder/(stem+'.json'),dict(source_request=str(f),note='Lossless text parts and ordered image references reconstructed from the frozen model_call payload; image data are retained at their hashed paths.',role='user',content=content))
        (folder/(stem+'.txt')).write_text('\n\n'.join(p['text'] for p in content if p['type']=='text'))
        inventory.append(dict(request=os.path.relpath(f,run),image_count=req['image_count'],hashes_match=True))
    save_json(run/'evidence_integrity.json',dict(requests=inventory,all_hashes_match=True,caption_text='Image file: <basename>, as constructed in model_call',
        log_scope_note='Shared server log also includes interrupted pilot runs. Its aggregate image count is not a per-run reception count; sent payload/image-list equality is separately unit-tested.'))
    requests_times=[f.stat().st_mtime for f in request_files];responses_times=[p.stat().st_mtime for f in request_files for p in [f.with_name(f.name.replace('_request.json','_response.json'))] if p.exists()]
    save_json(run/'experiment_metrics.json',dict(by_source_after_dedup=dict(Counter(s for c in report['cases'] for s in c['sources'])),
        reviewed_by_source=dict(Counter(s for c in report['cases'] if c.get('reviews') for s in c['sources'])),
        image_count_per_request=dict(Counter(x['image_count'] for x in inventory)),
        request_wall_span_seconds=max(responses_times)-min(requests_times) if requests_times and responses_times else None,
        cumulative_request_seconds=summary['model_seconds'],accuracy=None))
    raw_proposals=[]
    for c in report['cases']:
        calls=c.get('reviews',{}).get('ownership',{}).get('calls',[])
        if not calls:continue
        j=calls[-1].get('judgment',{})
        if not isinstance(j,dict) or j.get('recommendation')!='PROPOSE_CUT':continue
        edge=j.get('cut_edge');allowed=[{k:o[k] for k in ['parent_id','child_id']} for o in c['allowed_edges']]
        record=dict(candidate_id=c['candidate_id'],raw_judgment=j,program_judgment=c['judgment'],execution=[a for a in plan['actions'] if a['candidate_id']==c['candidate_id']])
        if isinstance(edge,dict) and all(type(edge.get(k)) is int for k in ['parent_id','child_id']) and edge in allowed:
            cc=deepcopy(c);cc['judgment']=j;record['impact_using_final_available_segment_judgments']=oa.impact(tree,cc,report['cases'])
        else:record['impact_unavailable_reason']='Model did not supply a valid allowed integer edge'
        raw_proposals.append(record)
    save_json(run/'all_model_proposal_impacts.json',raw_proposals)
    attention={a['candidate_id'] for a in plan['actions']}
    attention.update(g['candidate_id'] for g in report.get('joint_reviews',[]))
    attention.update(cid for g in report.get('joint_reviews',[]) for cid in g['related_ids'])
    for c in report['cases']:
        if not c.get('reviews'):continue
        if not (run/'cases'/c['candidate_id']/'interactive.html').exists():local_view(run,raw,tree,c)
        if c['candidate_id'] in attention or any(s.startswith('historical') for s in c['sources']):global_view(run,raw,c,report)
        paths=sorted(f for f in request_files if f.parent.parent.name==c['candidate_id']);links=[]
        for req in paths:
            stem=req.name.replace('_request.json','');folder=req.parent
            links.append(f'<a href="sent_content/{c["candidate_id"]}/{folder.name}/{stem}.json">{folder.name}/{stem} 完整文本与图像顺序</a>')
            for suffix,label in [('_prompt.txt','实际提示词'),('_request.json','图片/参数清单'),('_response.json','模型原始响应')]:
                p=folder/(stem+suffix)
                if p.exists():links.append(f'<a href="{os.path.relpath(p,run)}">{folder.name}/{stem} {label}</a>')
        if (run/'cases'/c['candidate_id']/'global_context.html').exists():links.insert(0,f'<a href="cases/{c["candidate_id"]}/global_context.html">完整上游/胞体交互原图（人工复核）</a>')
        actions=[a for a in plan['actions'] if a['candidate_id']==c['candidate_id']]
        if actions:
            cut_view(run,raw,tree,c,actions[0]);links.insert(0,f'<a href="cases/{c["candidate_id"]}/cut_edge.html">精确父子边定位/连续切片</a>')
        decision=c['judgment'];model_raw=c['reviews']['ownership']['calls'][-1].get('judgment',{})
        raw_text=str(model_raw.get('recommendation','API_ERROR'))+' / '+str(model_raw.get('target_ownership','unknown')) if isinstance(model_raw,dict) else 'INVALID_OUTPUT'
        repeat=c.get('reviews',{}).get('verification',{}).get('calls',[])
        if repeat:
            v=repeat[-1].get('judgment',{});raw_text+='; repeat='+str(v.get('recommendation','API_ERROR') if isinstance(v,dict) else 'INVALID_OUTPUT')
        for phase in ['segment_joint','segment_verify']:
            stage=c.get('reviews',{}).get(phase,{}).get('calls',[])
            if stage:
                j=stage[-1].get('judgment',{});raw_text+='; '+phase+'='+str(j.get('recommendation','API_ERROR') if isinstance(j,dict) else 'INVALID_OUTPUT')
        rows.append(f'<tr data-case="{c["candidate_id"]}"><td><a href="index.html#case-{c["candidate_id"]}">{c["candidate_id"]} / {c["anchor_id"]}</a></td><td>{html.escape(raw_text)}</td><td>{decision["recommendation"]}<br>{decision["target_ownership"]}</td><td>{html.escape("; ".join(str(a["edge"]["parent_id"])+" → "+str(a["edge"]["child_id"])+" / "+a["status"]+": "+", ".join(a["hold_reasons"]) for a in actions) or "无执行计划")}</td><td><a href="cases/{c["candidate_id"]}/interactive.html">原图交互检查</a><br>'+ '<br>'.join(links)+'</td></tr>')
    stats=html.escape(json.dumps({k:summary[k] for k in ['reviewed','recommendations','conflicts','calls','model_seconds','executed_edges','removed_nodes','removed_length_um','accuracy']},ensure_ascii=False,indent=2))
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>{summary['cell']} 审核证据入口</title><style>body{{font:15px/1.5 system-ui;margin:32px}}td,th{{padding:12px;border-bottom:1px solid #ccc;vertical-align:top}}pre{{white-space:pre-wrap}}a{{color:#135bb1}}table{{width:100%}}</style><h1>{summary['cell']}：模型判断、程序动作、原图证据</h1><p><a href="report.html">完整报告与两套提示词全文</a> · <a href="index.html">逐病例图像</a> · <a href="comparison.png">整体对照</a> · <a href="local_comparisons/index.json">同视野局部对照清单</a> · <a href="execution_plan.json">动作和影响范围</a> · <a href="joint_reviews.json">联合复核</a> · <a href="all_model_proposal_impacts.json">所有模型剪断提议的影响（含被拒绝项）</a> · <a href="evidence_integrity.json">证据文件校验</a> · <a href="experiment_metrics.json">候选来源与耗时补充统计</a></p><p>未提供人工归属真值。KEEP / target 是模型假设；REVIEW 是未解决；独立保存的组件不等于已确认属于其他神经元。结构验证不代表生物学准确率。</p><p><strong>实际送审 {summary["reviewed"]}；KEEP {summary["recommendations"].get("KEEP",0)}；PROPOSE_CUT {summary["recommendations"].get("PROPOSE_CUT",0)}；REVIEW {summary["recommendations"].get("REVIEW",0)}；实际剪断 {len(summary["executed_edges"])}；移出节点 {summary["removed_nodes"]}。</strong></p><details><summary>完整统计与冲突预算记录</summary><pre>{stats}</pre></details><p><label>查找节点/候选/结论 <input id="filter" style="font:inherit;padding:8px;width:320px" placeholder="例如 2559、0013、REVIEW"></label></p><table><tr><th>候选/节点</th><th>模型原始建议（末轮）</th><th>程序审核后状态</th><th>程序动作</th><th>可追溯证据</th></tr>{''.join(rows)}</table><script>document.getElementById('filter').oninput=function(){{const q=this.value.trim().toLowerCase();for(const row of document.querySelectorAll('tr[data-case]'))row.hidden=q&&!row.textContent.toLowerCase().includes(q);}};</script></html>'''
    (run/'delivery.html').write_text(doc)
    # The shared viewer template also serves older flat reports. In per-case
    # directories its diagnostic/index links must point two levels upward.
    for pattern in ['cases/*/interactive.html','cases/*/global_context.html','cases/*/cut_edge.html']:
        for p in run.glob(pattern):
            content=p.read_text().replace('href="diagnosis.json"','href="../../diagnosis.json"').replace('href="interactive.html"','href="../../delivery.html"')
            p.write_text(content)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src'] and v and not v.startswith(('#','http:','https:','data:')):self.paths.append(v.split('#')[0])
    files=list(run.glob('*.html'))+list(run.glob('cases/*/interactive.html'))+list(run.glob('cases/*/global_context.html'))+list(run.glob('cases/*/cut_edge.html'));missing=[]
    for p in files:
        parser=Links();parser.feed(p.read_text());missing.extend(str(p.parent/q) for q in parser.paths if not (p.parent/q).exists())
    if missing:raise ValueError('Missing links: '+str(missing[:10]))
    result=dict(html_files=len(files),local_links_valid=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.browser_check:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(viewport=dict(width=1440,height=1000));errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto((run/'delivery.html').as_uri());page.screenshot(path=str(run/'delivery_preview.png'))
            examples=sorted(run.glob('cases/*/interactive.html'));globals_=sorted(run.glob('cases/*/global_context.html'));tested=[]
            for file in examples[:2]+examples[-1:]+globals_[:1]+sorted(run.glob('cases/*/cut_edge.html'))[:1]:
                page.goto(file.as_uri());page.wait_for_function("document.body.dataset.ready==='true'");assert page.locator('canvas').count()==3
                page.locator('#overlay').check();page.locator('#mode').select_option('slice');assert page.evaluate("viewerDebug.planes.every(p=>viewerDebug.bounds(p)[1]-viewerDebug.bounds(p)[0]===1)")
                page.locator('#mode').select_option('full');page.locator('#whiteN').fill('120');page.locator('#whiteN').dispatch_event('input');tested.append(str(file.relative_to(run)))
            page.screenshot(path=str(run/'interactive_preview.png'));assert not errors,errors;browser.close();result.update(browser_views=tested,page_errors=errors,three_planes_slice_overlay_window=True)
    target=run/'source/tools/build_audit_delivery.py';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),target)
    save_json(run/'delivery_validation.json',result);print(run/'delivery.html')
if __name__=='__main__':main()
