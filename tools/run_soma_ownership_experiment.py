#!/usr/bin/env python3
"""Experimental local Gemma review using full upstream and competing soma routes."""
from pathlib import Path
import argparse,base64,hashlib,html,json,shutil,sys,time,urllib.request
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from algorithms.tracing import soma_ownership_context as sc
from algorithms.tracing.connection_diagnostics import save_json
from visualization.soma_ownership_context import render_case,render_cut_detail
from visualization.swc_comparison import render_swc_comparison
from tools.run_connection_experiment import metrics

PROMPT='''Review ownership of a traced neuron branch using complete upstream context and a competing soma. Images are XY, XZ, YZ raw/overlaid full-route views, then raw signal unfolded along paths. T is the known target soma; S labels are UNVALIDATED image-derived soma candidates. Cyan is existing SWC from T; magenta and green are independent image-graph route HYPOTHESES from competing soma and T, respectively. Orange is the questioned run. C0, C1 etc mark possible upstream cut edges. Every route can accidentally change fibers at a crossing. Proximity to a soma, shortest-path cost, colored overlays or a bright unfolded ridge are NOT neuron identity evidence. Inspect clean raw panels. Determine if the candidate is a real soma, whether its neurite can be followed continuously in multiple views to the questioned run, and whether the SWC first switches from T's neurite onto this independent fiber. Compare the COMPLETE route from T, including upstream junctions, with the route back to the other soma. Consider a change in trajectory relative to the upstream course; an ordinary bend or real bifurcation is not a switch. Unfolded images show signal continuity, not biological identity. If a path is ambiguous at a crossing or traverses another soma, answer uncertain. Do not force deletion. A switch can be a tracing jump across two depth-separated fibers, not necessarily physical contact. If ownership is other, choose the provided cut edge at the FIRST supported transition, not merely near the foreign soma. The node is the CHILD to remove with its descendants. Each cut option has a measured localization.supported flag: do not choose an unsupported option. It requires proximity along the upstream path to foreign-supported signal and an actual trajectory, signal, or competitive-route transition. This geometric flag is necessary but does not establish identity. If no supplied cut has both visible raw evidence and supported localization, return uncertain. Return JSON only: ownership (target|other|uncertain), soma_id (integer or null), cut_node_id (provided integer or null), candidate_is_soma (bool), other_soma_connection_supported (bool), upstream_switch_localized (bool), target_origin_rejected_by_image (bool), evidence (array of {view: XY|XZ|YZ|path_unfolded, observation: specific raw-image evidence}), explanation (string), uncertainty (string). Explain in Chinese; do not invent observations. At least two independent views, including XZ or YZ, must support other ownership. A cost/nearest-soma classification alone must remain uncertain.'''
VERIFY='''Independently recheck a proposed trace cut. Images are full-context XY, then local XY, XZ, YZ at the proposed cut. Each local view has clean raw panels and overlaid panels; bottom row is a thin slab with CLIPPED overlays. Cyan is SWC, magenta a competing-soma graph hypothesis, green a target-soma graph hypothesis. Colored lines do not prove connectivity. Check whether the SWC actually switches from its upstream fiber onto the distinct fiber leading toward the other soma at the supplied cut. Could this instead be an ordinary bend or genuine bifurcation of the target? A projection crossing alone is insufficient. A depth-separated tracing jump can be a switch if the outgoing trace follows the other fiber. You may reject the previous proposal. Return JSON only: confirmed (bool), cut_node_id (same supplied integer), same_neuron_bend_or_branch_excluded (bool), raw_multiview_switch_supported (bool), observations (array of {view: XY|XZ|YZ, observation: specific raw-image evidence}), explanation (Chinese string). If evidence is ambiguous, confirmed=false. Do not accept solely because a previous model proposed the cut.'''


def request(prompt,metadata,images,folder,endpoint,stem):
    content=[dict(type='text',text=prompt+'\nMetadata: '+json.dumps(metadata,ensure_ascii=False))];hashes={}
    for f in images:
        b=Path(f).read_bytes();hashes[f]=hashlib.sha256(b).hexdigest()
        content.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(b).decode())))
    params=dict(model='gemma4-31b-local',temperature=0,seed=42,max_tokens=2048,cache_prompt=False)
    save_json(folder/(stem+'_request.json'),dict(prompt=prompt,metadata=metadata,image_sha256=hashes,endpoint=endpoint,parameters=params))
    body=dict(params,messages=[dict(role='user',content=content)])
    req=urllib.request.Request(endpoint+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req,timeout=360) as response:r=json.load(response)
    save_json(folder/(stem+'_response.json'),r)
    c=r['choices'][0]
    if c.get('finish_reason')!='stop':raise ValueError('Incomplete model output')
    s=c['message'].get('content','').strip()
    if s.startswith('```'):s=s.split('\n',1)[1].rsplit('```',1)[0]
    obj=json.loads(s)
    if not isinstance(obj,dict):raise ValueError('Expected JSON object')
    return obj,time.monotonic()-start


def verify_gate(v,cut):
    obs=v.get('observations',[])
    if not isinstance(obs,list):return False
    views={o.get('view') for o in obs if isinstance(o,dict) and isinstance(o.get('observation'),str) and len(o['observation'])>=15}
    return all(v.get(k) is True for k in ['confirmed','same_neuron_bend_or_branch_excluded','raw_multiview_switch_supported']) and v.get('cut_node_id')==cut and len(views&{'XY','XZ','YZ'})>=2 and bool(views&{'XZ','YZ'})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-name',default='20261111_97117_global_soma_v3_checked');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--limit',type=int);ap.add_argument('--endpoint',default='http://127.0.0.1:18484');args=ap.parse_args()
    if Path(args.run_name).name!=args.run_name:ap.error('Invalid run name')
    base=ROOT.parent/'test/0.84';out=base/'runs'/args.run_name;out.mkdir(parents=True,exist_ok=True)
    src=base/'runs/20261110_97117_current_v084_parallel/swc/97117.swc';rawpath=base/'data/inputs/raw/97117.tif'
    tree=np.loadtxt(src,ndmin=2).tolist();raw=tifffile.imread(rawpath)
    identity={k:v for k,v in [('swc',str(src)),('swc_sha256',hashlib.sha256(src.read_bytes()).hexdigest()),('raw',str(rawpath)),('raw_sha256',hashlib.sha256(rawpath.read_bytes()).hexdigest())]}
    if (out/'input.json').exists() and json.loads((out/'input.json').read_text())!=identity:raise ValueError('Input changed')
    save_json(out/'input.json',identity)
    codefiles=['algorithms/tracing/soma_ownership_context.py','visualization/soma_ownership_context.py','tools/run_soma_ownership_experiment.py','algorithms/tracing/signal_support.py']
    code={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in codefiles}
    if (out/'code.json').exists() and json.loads((out/'code.json').read_text())!=code:raise ValueError('Code changed; use a new run')
    save_json(out/'code.json',code)
    for p in codefiles:
        dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    plan=out/'diagnosis.json'
    report=json.loads(plan.read_text()) if plan.exists() else {}
    if report.get('schema')!='full-upstream-v3.2':
        somas=json.loads((out/'somas.json').read_text()) if (out/'somas.json').exists() else sc.detect_somas(raw,np.asarray(tree[0][2:5]))
        save_json(out/'somas.json',somas)
        if (out/'graph_routes.npz').exists():
            with np.load(out/'graph_routes.npz') as g:graph={k:g[k] for k in g.files}
            for k in ['graph_vertices','graph_edges']:graph[k]=int(graph[k])
            graph['seed_counts']=graph['seed_counts'].tolist()
        else:
            graph=sc.graph_from_image(raw,somas);np.savez_compressed(out/'graph_routes.npz',**graph)
        sites=json.loads((base/'runs/20261110_97117_gemma_crossing_v1/diagnosis.json').read_text())['sites']
        cases,statistics=sc.ownership_candidates(tree,raw,somas,graph,sites)
        report=dict(schema='full-upstream-v3.2',policy=sc.POLICY,input_sha256=sc.tree_digest(tree),somas=somas,cases=cases,statistics=statistics)
        save_json(plan,report)
    print(json.dumps(report['statistics']),flush=True)
    if args.prepare_only:return
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.endpoint+'/v1/models',timeout=10) as r:save_json(out/'model_runtime.json',json.load(r))
    reviewed=0
    for c in report['cases']:
        if c.get('status'):continue
        folder=out/'cases'/c['case_id'];folder.mkdir(parents=True,exist_ok=True)
        c['images']=render_case(raw,c,report['somas'],folder)
        metadata={k:c[k] for k in ['case_id','representative_id','competing_soma_id','cut_options','intermediate_soma_encounters']}
        metadata['somas']=[s for s in report['somas'] if s['soma_id'] in [0,c['competing_soma_id']]]
        metadata['cut_label_map']={'C'+str(j):o['node_id'] for j,o in enumerate(c['cut_options'])}
        try:
            c['judgment'],c['seconds']=request(PROMPT,metadata,c['images'],folder,args.endpoint,'ownership')
            c['gate']=sc.ownership_gate(c,c['judgment'])
            if c['gate']['accepted']:
                cut=c['judgment']['cut_node_id'];c['detail_images']=render_cut_detail(raw,c,cut,folder)
                c['verification'],c['verification_seconds']=request(VERIFY,dict(cut_node_id=cut,competing_soma_id=c['competing_soma_id'],somas=metadata['somas']),[c['images'][0]]+c['detail_images'],folder,args.endpoint,'verification')
                c['verification_accepted']=verify_gate(c['verification'],cut)
                if not c['verification_accepted']:c['gate']=dict(accepted=False,reasons=['local_independent_recheck_failed'])
            c['status']='cut_proposed' if c['gate']['accepted'] else 'kept_after_review'
        except (ValueError,KeyError,TypeError,IndexError) as e:
            c['status']='uncertain_invalid_output';c['error']=str(e);c['gate']=dict(accepted=False,reasons=['invalid_output'])
        save_json(folder/'diagnosis.json',c);save_json(plan,report)
        print(json.dumps(dict(case=c['case_id'],owner=c.get('judgment',{}).get('ownership'),soma=c['competing_soma_id'],cut=c.get('judgment',{}).get('cut_node_id'),status=c['status'],gate=c['gate']),ensure_ascii=False),flush=True)
        reviewed+=1
        if args.limit and reviewed>=args.limit:return
    if any(not c.get('status') for c in report['cases']):raise ValueError('Incomplete review')
    # No input mutation until every saved diagnostic and independent verification is complete.
    saved=json.loads(plan.read_text())
    for c in saved['cases']:
        if c.get('gate',{}).get('accepted') and not verify_gate(c.get('verification',{}),c['judgment']['cut_node_id']):raise ValueError('Unverified cut')
    kept,audit=sc.apply_ownership_plan(tree,saved)
    (out/'swc').mkdir(exist_ok=True);swc=out/'swc/97117.swc'
    np.savetxt(swc,kept,fmt=['%d','%d','%.5f','%.5f','%.5f','%.5f','%d'],header='n type x y z r parent')
    removed=[n for n in tree if int(n[0]) in audit['removed_ids']]
    if removed:np.savetxt(out/'removed_segments.swc',removed,fmt=['%d','%d','%.5f','%.5f','%.5f','%.5f','%d'])
    save_json(out/'cuts.json',audit)
    versions=[('v0.81',base/'data/references/v0.81/97117.swc'),('v0.83b',base/'data/references/v0.83b/97117.swc'),('v0.84 before model',src),('previous local v2',base/'runs/20261111_97117_gemma_transfer_v2/swc/97117.swc'),('global soma v3',swc)]
    render_swc_comparison(raw,versions,out/'comparison.png',title='97117: complete upstream and competing-soma review')
    result=dict(policy=sc.POLICY,versions={k:metrics(p) for k,p in versions},candidates=len(saved['cases']),soma_candidates=len(saved['somas']),cuts=len(audit['events']),removed_nodes=len(audit['removed_ids']),accuracy=None,
        note='Experimental model judgments; no manual ownership ground truth. Graph routes, soma candidates and deletion counts do not establish biological correctness.',status_counts={s:sum(c['status']==s for c in saved['cases']) for s in set(c['status'] for c in saved['cases'])})
    save_json(out/'summary.json',result)
    blocks=[]
    for c in sorted(saved['cases'],key=lambda c:not c['gate']['accepted']):
        imgs=''.join(f'<a href="{Path(p).relative_to(out)}"><img loading="lazy" src="{Path(p).relative_to(out)}"></a>' for p in c['images']+c.get('detail_images',[]))
        blocks.append(f'<section id="case-{c["case_id"]}"><h2>{c["case_id"]} · 胞体 S{c["competing_soma_id"]} · {c["status"]}</h2><pre>{html.escape(json.dumps({k:c.get(k) for k in ["cut_options","judgment","gate","verification"]},ensure_ascii=False,indent=2))}</pre><div class="grid">{imgs}</div></section>')
    rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["nodes"]}</td><td>{v["length_um"]:.2f}</td><td>{v["tips"]}</td></tr>' for k,v in result['versions'].items())
    (out/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>97117 完整上游与胞体归属 v3</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px;background:#f4f6f8}}section{{background:white;padding:20px;margin:20px 0}}pre{{white-space:pre-wrap}}img{{width:100%}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}td,th{{padding:10px}}</style>
<h1>97117：完整上游与候选胞体归属</h1><p>从模型剪枝前的 v0.84 重新判断，13 个图像候选胞体、{len(saved['cases'])} 个路径归属疑点。独立局部复核后执行 {len(audit['events'])} 处剪断，删除 {len(audit['removed_ids'])} 节点。图路径是待验证假设，删除数量不代表准确率；尚无人工归属真值。</p>
<p>青色：目标胞体出发的 SWC；洋红：候选胞体的图像路径；绿色：目标胞体的图像路径；橙色：被检查分支。C 标记可选剪断点。原始图与叠加图并排；沿路径展开显示全程信号连续性；局部薄层叠加已按层范围裁剪。</p>
<p><a href="diagnosis.json">完整诊断</a> · <a href="cuts.json">实际剪断</a> · <a href="swc/97117.swc">新 SWC</a> · <a href="summary.json">指标</a></p>
<table><tr><th>版本</th><th>节点</th><th>长度 µm</th><th>末梢</th></tr>{rows}</table><img src="comparison.png">{''.join(blocks)}</html>''',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
