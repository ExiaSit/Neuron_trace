#!/usr/bin/env python3
"""Optional local Gemma crossing pass on a frozen v0.84 result."""
from pathlib import Path
import argparse,base64,hashlib,html,json,shutil,sys,time,urllib.request
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from algorithms.tracing import model_crossing as mc
from algorithms.tracing.connection_diagnostics import save_json
from visualization.model_crossing import render_encounter
from visualization.swc_comparison import render_swc_comparison
from tools.run_connection_experiment import metrics

PROMPT='''Review a fluorescence neuron trace at an XY-MIP crossing, following the cyan incoming path from the target soma into the yellow outgoing path. Determine whether the trace switches onto an independent neurite at the red X. Images: (1) global XY with target soma; (2) XY raw/overlay and 3-um slab; (3) XZ; (4) YZ; (5) consecutive clean XY slices. Each view has clean images: the SWC overlay is a hypothesis, not proof. Thin-slab overlays may show paths outside the slab. Follow raw signal across depth. Distinguish a real bifurcation, a smooth same-neuron bend, a depth-separated projection crossing, and an abrupt turn into another shaft. Geometry measurements below are aids, not labels. A large angle, nearest soma, or a single projection is insufficient. No soma is assigned to the competing shaft unless its full path is visible. Only say switch_to_other when there is visible entry into a distinct continuing shaft in 3D and corroborating non-XY views. Otherwise answer uncertain or the appropriate alternative. Do not extrapolate identity of all downstream branches. Return a JSON object only: decision (switch_to_other|same_neuron|projection_overlap|real_bifurcation|uncertain), crossing_node_id (provided integer), connection_in_3d (connected|separated|unclear), independent_shaft (boolean), supporting_views (array of {plane: XY|XZ|YZ, evidence: specific raw-image observation}), direction_interpretation (string), alternative_explanation (string), confidence (0..1, uncalibrated). Explain in Chinese. Never fabricate image evidence.'''

def request(site,folder,endpoint):
    metadata={k:site[k] for k in ['node_id','center_xyz','direction','raw_support']}
    content=[dict(type='text',text=PROMPT+'\nGeometry: '+json.dumps(metadata))]
    image_hashes={}
    for file in site['images']:
        data=Path(file).read_bytes();image_hashes[file]=hashlib.sha256(data).hexdigest()
        content.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(data).decode())))
    body=dict(model='gemma4-31b-local',messages=[dict(role='user',content=content)],temperature=0,seed=42,max_tokens=2048,cache_prompt=False)
    save_json(folder/'request_manifest.json',dict(prompt=PROMPT,metadata=metadata,image_sha256=image_hashes,
              endpoint=endpoint,parameters={k:v for k,v in body.items() if k!='messages'}))
    req=urllib.request.Request(endpoint.rstrip('/')+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));start=time.monotonic()
    with opener.open(req,timeout=300) as r:response=json.load(r)
    save_json(folder/'response.json',response)
    return response,time.monotonic()-start

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',default='97117');ap.add_argument('--input-run',required=True)
    ap.add_argument('--run-name',required=True);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--endpoint',default='http://127.0.0.1:18484');args=ap.parse_args()
    if not args.cell.isdigit() or Path(args.run_name).name!=args.run_name:ap.error('Invalid cell or run name')
    base=ROOT.parent/'test/0.84';out=base/'runs'/args.run_name;out.mkdir(parents=True,exist_ok=args.resume)
    src=Path(args.input_run)/f'swc/{args.cell}.swc';raw_path=base/f'data/inputs/raw/{args.cell}.tif'
    tree=np.loadtxt(src,ndmin=2).tolist();raw=tifffile.imread(raw_path);nodes,_,_=mc.tree_paths(tree)
    identity=dict(swc=str(src),swc_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),raw=str(raw_path),raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
    if (out/'input.json').exists() and json.loads((out/'input.json').read_text())!=identity:raise ValueError('Resume input changed')
    save_json(out/'input.json',identity)
    code_files=['algorithms/tracing/model_crossing.py','visualization/model_crossing.py','tools/run_model_crossing_experiment.py']
    code={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code_files}
    if (out/'code.json').exists() and json.loads((out/'code.json').read_text())!=code:raise ValueError('Resume code changed; use new run')
    save_json(out/'code.json',code)
    for p in code_files:
        dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    plan=out/'diagnosis.json'
    if plan.exists():report=json.loads(plan.read_text())
    else:
        report,skeleton=mc.scan_encounters(tree,raw)
        tifffile.imwrite(out/'raw_xy_skeleton.tif',skeleton.astype(np.uint8)*255)
        for s in report['sites']:
            if not s['protected']:s['raw_support']=mc.raw_support(raw,s,nodes)
        save_json(plan,report)
    print(json.dumps(dict(sites=len(report['sites']),protected=sum(s['protected'] for s in report['sites']),
         abnormal=sum(s['direction']['direction_abnormal'] for s in report['sites']),depth_separated=len(report['separated_crossings']))),flush=True)
    if args.prepare_only:return
    for site in report['sites']:
        if site['status']!='pending_model':continue
        folder=out/'sites'/site['site_id'];folder.mkdir(parents=True,exist_ok=True)
        site['images']=render_encounter(raw,nodes,site,report['target_soma_xyz'],folder)
        save_json(folder/'geometry.json',site)
        try:
            response,elapsed=request(site,folder,args.endpoint);site['elapsed_seconds']=elapsed
            site['judgment']=mc.parse_judgment(response);site['gate']=mc.cut_gate(site,site['judgment'])
            site['status']='cut_proposed' if site['gate']['accepted'] else 'kept_after_review'
        except (ValueError,KeyError,TypeError,IndexError) as e:
            site['status']='uncertain_invalid_model_output';site['error']=str(e);site['gate']=dict(accepted=False,reasons=['invalid_model_output'])
        # Network/service errors abort; never silently declare an incomplete scan finished.
        save_json(folder/'diagnosis.json',site);save_json(plan,report)
        print(json.dumps(dict(site=site['site_id'],node=site['node_id'],status=site['status'],decision=site.get('judgment',{}).get('decision'),turn=site['direction']['local_turn_degrees'],gate=site.get('gate')),ensure_ascii=False),flush=True)
    # Persisted evidence is complete before any SWC changes.
    saved=json.loads(plan.read_text());kept,audit=mc.apply_saved_plan(tree,saved)
    (out/'swc').mkdir(exist_ok=True);target=out/f'swc/{args.cell}.swc'
    np.savetxt(target,kept,fmt=['%d','%d','%.5f','%.5f','%.5f','%.5f','%d'],header='n type x y z r parent')
    save_json(out/'cuts.json',audit)
    versions=[('v0.81',base/f'data/references/v0.81/{args.cell}.swc'),('v0.83b',base/f'data/references/v0.83b/{args.cell}.swc'),('v0.84 before model',src),('v0.84 Gemma crossing',target)]
    versions=[(label,p) for label,p in versions if p.exists()]
    render_swc_comparison(raw,versions,out/'comparison.png',title=f'{args.cell}: XY encounter / direction / Gemma review')
    result=dict(policy=mc.POLICY,versions={label:metrics(p) for label,p in versions},sites=len(saved['sites']),
                model_reviewed=sum('judgment' in s for s in saved['sites']),depth_separated=len(saved['separated_crossings']),
                cuts=len(audit['events']),removed_nodes=len(audit['removed_ids']),accuracy=None,
                note='No manual ground truth. Cut decisions are evidence-gated model hypotheses, not validated biological identity.')
    save_json(out/'summary.json',result)
    blocks=[]
    for s in saved['sites']:
        if 'images' not in s:continue
        imgs=''.join(f'<a href="{Path(p).relative_to(out)}"><img loading="lazy" src="{Path(p).relative_to(out)}"></a>' for p in s['images'])
        blocks.append(f'<section><h2>{s["site_id"]} · 节点 {s["node_id"]} · {s["status"]}</h2><pre>{html.escape(json.dumps({k:s.get(k) for k in ["direction","raw_support","judgment","gate"]},ensure_ascii=False,indent=2))}</pre><div class="grid">{imgs}</div></section>')
    rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["nodes"]}</td><td>{v["length_um"]:.2f}</td></tr>' for k,v in result['versions'].items())
    (out/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>{args.cell} Gemma 交错剪枝</title>
<style>body{{font:16px system-ui;max-width:1450px;margin:30px auto;padding:20px;background:#f4f6f8}}section{{background:white;padding:20px;margin:20px 0}}pre{{white-space:pre-wrap}}img{{width:100%}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}td{{padding:10px}}</style>
<h1>{args.cell}：沿分支检查 XY 交错，Gemma 多视角诊断</h1><p>扫描 {len(saved['sites'])} 个方向性位置，另有 {len(saved['separated_crossings'])} 对 SWC 投影交错在深度上分离，保留。模型通过本机运行。</p>
<p>先保存诊断再执行剪断：{len(audit['events'])} 处，删除 {len(audit['removed_ids'])} 个下游节点。无人工真值，删除数量不是准确率。邻近胞体或方向变化单独均不足以剪断。</p>
<p><a href="diagnosis.json">全部诊断</a> · <a href="cuts.json">实际剪断</a> · <a href="swc/{args.cell}.swc">新 SWC</a> · <a href="summary.json">指标</a></p>
<table><tr><th>版本</th><th>节点</th><th>长度 µm</th></tr>{rows}</table><img src="comparison.png">{''.join(blocks)}''',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
