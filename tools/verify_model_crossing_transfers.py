#!/usr/bin/env python3
"""Recheck direction-anomalous VLM proposals for actual SWC transfers, then cut."""
from pathlib import Path
import argparse,base64,hashlib,html,json,shutil,sys,time,urllib.request
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from algorithms.tracing.model_transfer import eligible,transfer_gate,apply_verified_transfers
from algorithms.tracing.connection_diagnostics import save_json
from tools.run_connection_experiment import metrics
from visualization.swc_comparison import render_swc_comparison
from visualization.projection_viewer import render_projection_comparison

PROMPT='''Independently inspect the clean microscopy images around the red X. Cyan and yellow lines identify the incoming and outgoing portions of a proposed SWC trace; they do NOT prove a biological connection. Ignore numerical turn angles and any prior model opinion. Decide whether the SWC itself transfers from one 3D neurite onto a distinct neurite at the marked node, rather than merely passing a projected crossing while staying on the same neurite. A transfer may jump over a gap: physical contact between the two actual neurites is NOT required. Conversely, two depth-separated neurites overlapping in XY do not justify a cut unless this specific SWC changes which one it follows. A V-shaped overlay or an abrupt angle alone is NOT evidence of distinct neurites. Inspect raw intensities, depth separation, the continuing shaft and the alternative same-neuron bend/bifurcation. If the clean images cannot establish this, do not confirm. Global context and XY/XZ/YZ raw/overlay full/slab images and consecutive XY slices follow. Overlays on thin slabs may lie outside the slab. Return ONLY JSON with: crossing_node_id (integer), trace_transfers_to_distinct_3d_shaft (boolean), projection_overlap_only (boolean), raw_evidence_sufficient (boolean), same_neuron_bend_or_bifurcation_excluded (boolean), raw_observations (array of {plane: XY|XZ|YZ, observation: concrete evidence from raw image not just overlay direction}), explanation (string), uncertainty (string). Explain in Chinese. Do not infer neuron identity from nearest soma or self-reported confidence.'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-run',required=True);ap.add_argument('--run-name',required=True)
    ap.add_argument('--endpoint',default='http://127.0.0.1:18484');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if Path(args.run_name).name!=args.run_name:ap.error('Invalid run name')
    source=Path(args.source_run);base=ROOT.parent/'test/0.84';out=base/'runs'/args.run_name
    first=json.loads((source/'diagnosis.json').read_text())
    if any(s['status']=='pending_model' for s in first['sites']):raise ValueError('Finish first-pass scan before verification')
    out.mkdir(exist_ok=args.resume,parents=True)
    identity=json.loads((source/'input.json').read_text());tree=np.loadtxt(identity['swc']).tolist();raw=tifffile.imread(identity['raw'])
    for name in ['swc','raw']:
        if hashlib.sha256(Path(identity[name]).read_bytes()).hexdigest()!=identity[name+'_sha256']:raise ValueError('Input changed')
    source_hash=hashlib.sha256((source/'diagnosis.json').read_bytes()).hexdigest()
    provenance=dict(source_run=str(source),source_diagnosis_sha256=source_hash,policy='gemma-xy-direction-independent-transfer-verification-v2')
    if (out/'provenance.json').exists() and json.loads((out/'provenance.json').read_text())!=provenance:raise ValueError('Source changed')
    save_json(out/'provenance.json',provenance);save_json(out/'input.json',identity)
    files=['algorithms/tracing/model_transfer.py','tools/verify_model_crossing_transfers.py','visualization/projection_viewer.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    if (out/'code.json').exists() and json.loads((out/'code.json').read_text())!=hashes:raise ValueError('Verification code changed')
    save_json(out/'code.json',hashes)
    for p in files:
        dst=out/'source'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
    plan=out/'diagnosis.json';report=json.loads(plan.read_text()) if plan.exists() else first
    report['policy']=provenance['policy'];opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for s in report['sites']:
        dest=out/'sites'/s['site_id'];dest.mkdir(parents=True,exist_ok=True)
        if 'images' in s:
            local=[]
            for p in s['images']:
                dst=dest/Path(p).name
                if not dst.exists():shutil.copy2(p,dst)
                local.append(str(dst))
            s['images']=local
        if 'verification' in s or 'verification_error' in s:continue
        if not eligible(s):
            s['transfer_gate']=dict(accepted=False,reasons=['not_a_supported_abnormal_switch_candidate']);continue
        extra=dest/'raw_windowed.png'
        if not extra.exists():
            lo,hi=np.asarray(s['bbox_xyz']);crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
            render_projection_comparison(crop,lo,s,extra)
        if str(extra) not in s['images']:s['images'].append(str(extra))
        content=[dict(type='text',text=PROMPT+f'\nMarked crossing_node_id: {s["node_id"]}. Center XYZ um: {s["center_xyz"]}. The last image is an additional clean XY/XZ/YZ sheet with shared contrast window, thin slabs and single slices to reveal faint signal. Windowing is display-only, not new evidence.')]
        hashes={}
        for p in s['images']:
            data=Path(p).read_bytes();hashes[p]=hashlib.sha256(data).hexdigest()
            content.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(data).decode())))
        body=dict(model='gemma4-31b-local',messages=[dict(role='user',content=content)],temperature=0,seed=42,max_tokens=2048,cache_prompt=False)
        save_json(dest/'verification_request.json',dict(prompt=content[0]['text'],image_sha256=hashes,parameters={k:v for k,v in body.items() if k!='messages'}))
        req=urllib.request.Request(args.endpoint.rstrip('/')+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        start=time.monotonic()
        with opener.open(req,timeout=300) as r:response=json.load(r)
        save_json(dest/'verification_response.json',response);s['verification_seconds']=time.monotonic()-start
        try:
            ch=response['choices'][0]
            if ch['finish_reason']!='stop':raise ValueError('Incomplete response')
            text=ch['message']['content'].strip()
            if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
            verification=json.loads(text)
            if not isinstance(verification,dict):raise ValueError('Not a JSON object')
            s['verification']=verification;s['transfer_gate']=transfer_gate(s,verification)
        except (ValueError,KeyError,TypeError,IndexError) as e:
            s['verification_error']=str(e);s['transfer_gate']=dict(accepted=False,reasons=['invalid_verification_response'])
        save_json(dest/'verification.json',s);save_json(plan,report)
        print(json.dumps(dict(site=s['site_id'],node=s['node_id'],gate=s['transfer_gate']),ensure_ascii=False),flush=True)
    for s in report['sites']:
        if s['protected']:continue
        s['status']='verified_cut_proposed' if s['transfer_gate']['accepted'] else 'kept_after_verification'
    save_json(plan,report)
    kept,audit=apply_verified_transfers(tree,json.loads(plan.read_text()));save_json(out/'cuts.json',audit)
    (out/'swc').mkdir(exist_ok=True);cell=Path(identity['swc']).stem;swc=out/f'swc/{cell}.swc'
    np.savetxt(swc,kept,fmt=['%d','%d','%.5f','%.5f','%.5f','%.5f','%d'],header='n type x y z r parent')
    versions=[('v0.81',base/f'data/references/v0.81/{cell}.swc'),('v0.83b',base/f'data/references/v0.83b/{cell}.swc'),('v0.84 before Gemma',Path(identity['swc'])),('v0.84 Gemma verified',swc)]
    versions=[(k,p) for k,p in versions if p.exists()];render_swc_comparison(raw,versions,out/'comparison.png',title=f'{cell}: independent model transfer verification')
    summary=dict(policy=report['policy'],sites=len(report['sites']),model_reviewed=sum('judgment' in s for s in report['sites']),
                 verified=sum('verification' in s for s in report['sites']),cuts=len(audit['events']),removed_nodes=len(audit['removed_ids']),
                 versions={k:metrics(p) for k,p in versions},accuracy=None,note='Experimental model-assisted cuts; no manual truth. All original inputs preserved.')
    save_json(out/'summary.json',summary)
    blocks=[]
    for s in report['sites']:
        if 'images' not in s:continue
        pics=''.join(f'<a href="sites/{s["site_id"]}/{Path(p).name}"><img loading="lazy" src="sites/{s["site_id"]}/{Path(p).name}"></a>' for p in s['images'])
        detail={k:s.get(k) for k in ['direction','judgment','verification','transfer_gate']}
        blocks.append(f'<section id="site_{s["site_id"]}"><h2>{s["site_id"]} · 节点 {s["node_id"]} · '+('拟剪断' if s['transfer_gate']['accepted'] else '保留')+f'</h2><pre>{html.escape(json.dumps(detail,ensure_ascii=False,indent=2))}</pre><div class="grid">{pics}</div></section>')
    rows=''.join(f'<tr><td>{k}</td><td>{m["nodes"]}</td><td>{m["length_um"]:.2f}</td></tr>' for k,m in summary['versions'].items())
    cuts=''.join(f'<li><a href="#site_{e["site_id"]}">位置 {e["site_id"]}，节点 {e["node_id"]}</a>：从下游节点 {e["cut_node_id"]} 剪断，删除 {len(e["removed_ids"])} 节点。</li>' for e in audit['events'])
    (out/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>{cell} Gemma 归属复核</title><style>body{{font:16px system-ui;max-width:1450px;margin:30px auto;padding:20px;background:#f4f6f8}}section{{background:white;padding:20px;margin:20px 0}}pre{{white-space:pre-wrap}}img{{width:100%}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}td{{padding:10px}}</style>
<h1>{cell}：Gemma 交错诊断与定向复核</h1><p>沿分支扫描 XY 交错，结合异常转向与三维原图。第一次模型诊断后，对异常转向的串线候选独立复核，不提供前次模型结论和角度数值。只有明确判定 SWC 转入另一根三维纤维、排除单纯投影和正常分叉/弯曲时才剪断。</p>
<p>真实纤维之间不必物理相连：跨间隙跳到另一根纤维也可能是错误串线。直线反向探针仅作参考，不否决模型在曲线纤维上的判断。</p>
<p>首轮审阅 {summary['model_reviewed']} 处，定向复核 {summary['verified']} 处，实际剪断 {summary['cuts']} 处，删除 {summary['removed_nodes']} 节点。尚无人工真值，不能把节点减少解释为准确率提高。</p>
<p><a href="summary.json">指标</a> · <a href="diagnosis.json">诊断</a> · <a href="cuts.json">剪断记录</a> · <a href="swc/{cell}.swc">新 SWC</a></p><ul>{cuts}</ul><table><tr><th>版本</th><th>节点</th><th>长度 µm</th></tr>{rows}</table><img src="comparison.png">{''.join(blocks)}''',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
