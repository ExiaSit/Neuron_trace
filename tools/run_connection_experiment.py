#!/usr/bin/env python3
"""Replay a frozen cell's pruning; write every new result under test/0.84/runs."""
from pathlib import Path
import argparse,csv,hashlib,html,json,shutil,sys,types
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import bootstrap
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from utils.case_worker import load_module
from tools.compare_mip_repair import swc_metrics
from visualization.swc_comparison import render_swc_comparison
from algorithms.tracing.connection_diagnostics import POLICY,save_json


def samples(path):
    tree=np.loadtxt(path,ndmin=2);nodes={int(n[0]):n for n in tree};points=[];weights=[]
    for n in tree:
        if int(n[6]) not in nodes:continue
        a=nodes[int(n[6])][2:5];b=n[2:5];length=float(np.linalg.norm(b-a))
        if length<=0:continue
        count=max(1,int(np.ceil(length/.5)))
        points.extend(a+(b-a)*((np.arange(count)+.5)/count)[:,None]);weights.extend([length/count]*count)
    return np.asarray(points).reshape(-1,3),np.asarray(weights)


def metrics(path):
    m=swc_metrics(path);tree=np.loadtxt(path,ndmin=2)
    _,counts=np.unique(tree[tree[:,6]>=0,6],return_counts=True)
    m.update(branchpoints=int((counts>1).sum()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),path=str(path))
    return m


def proximity(a,b):
    p,w=samples(a);q,_=samples(b)
    if not len(p) or not len(q):return None
    d=cKDTree(q).query(p)[0]
    return dict(tolerance_um=2.,covered_length_fraction=float(w[d<=2].sum()/w.sum()),
                mean_distance_um=float(np.average(d,weights=w)),metric='arc-length-weighted spatial proximity; not accuracy')


def cell_metadata(data, cid):
    path=data/'metadata'/f'{cid}.json'
    if path.exists():return json.loads(path.read_text()),path
    if cid=='60003':return json.loads((data/'metadata.json').read_text()),data/'metadata.json'
    raise FileNotFoundError(f'Missing per-cell metadata: {path}')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cell',default='60003');parser.add_argument('--run-name')
    parser.add_argument('--compare-run',help='Previous v0.84 run; defaults to latest-result.txt')
    parser.add_argument('--render-workers',type=int,default=1,help='Parallel figure exporters; pruning stays sequential')
    args=parser.parse_args();cid=args.cell
    if not cid.isdigit():parser.error('cell must be numeric')
    if args.render_workers<1:parser.error('render-workers must be positive')
    experiment=ROOT.parent/'test/0.84';data=experiment/'data'
    meta,meta_path=cell_metadata(data,cid)
    provenance_path=data/f'copy-manifest-{cid}.json'
    if not provenance_path.exists() and cid=='60003':provenance_path=data/'copy-manifest.json'
    provenance=json.loads(provenance_path.read_text())
    pointer=experiment/f'latest-result-{cid}.txt'
    if not pointer.exists():pointer=experiment/'latest-result.txt'
    previous=Path(args.compare_run) if args.compare_run else (Path(pointer.read_text().strip()) if pointer.exists() else None)
    if previous and not (previous/f'swc/{cid}.swc').is_file():
        if args.compare_run:parser.error('compare-run does not contain this cell SWC')
        previous=None
    name=args.run_name or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'_'+cid
    if Path(name).name!=name:parser.error('run-name must be a directory name')
    out=experiment/'runs'/name;out.mkdir(parents=True,exist_ok=False)
    for sub in ['swc','mips','source']:(out/sub).mkdir()
    snapshots={}
    files=['algorithms/tracing/prune.py','algorithms/tracing/terminal_pruning.py','algorithms/tracing/connection_diagnostics.py',
           'algorithms/tracing/signal_support.py','algorithms/tracing/soma_signal_groups.py','algorithms/tracing/gcut_adapter.py','visualization/connection_diagnostics.py',
           'visualization/swc_comparison.py','visualization/pruning_summary.py','visualization/projection_viewer.py',
           'visualization/assets/projection_viewer.html','tools/run_connection_experiment.py',
           '11_prune_swc.py','12_guard_regression.py']
    for rel in files:
        src=ROOT/rel;dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        snapshots[rel]=hashlib.sha256(src.read_bytes()).hexdigest()
    inputs=dict(raw=str(data/f'inputs/raw/{cid}.tif'),mask=str(data/f'source_v083b/mask/{cid}.tif'),
                swc=str(data/f'source_v083b/gcut_output/selected_resampled_step1/{cid}.swc'),
                soma_json=str(data/f'inputs/soma_img/{cid}.json'),soma_mask=str(data/f'inputs/soma_seg/{cid}.tif'))
    save_json(out/'run.json',dict(policy=POLICY,cell=cid,source_job=provenance.get('source_job'),inputs=inputs,metadata=meta,
                               metadata_path=str(meta_path),copy_manifest=str(provenance_path),
                               metadata_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                               copy_manifest_sha256=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                               source_root=provenance.get('source_root','/data/disk/C6.0/v0.83b'),
                               source_sha256=snapshots,scope='pruning only; frozen v0.83b upstream input',output_root=str(out),
                               previous_run=str(previous) if previous else None))
    sys.modules['pipeline_config']=types.SimpleNamespace(BASE_DIR=out,PATHS={});mod=load_module('prune')
    from visualization.connection_diagnostics import configure_render_workers,finish_rendering
    configure_render_workers(args.render_workers)
    try:
        mod.process_neuron_morphology(inputs['mask'],inputs['swc'],inputs['raw'],str(out/'swc'),str(out/'mips'),
            pd.DataFrame([meta],index=[int(cid)]),sphere_zradius=10,show_steps=False,
            soma_json_path=inputs['soma_json'],soma_mask_path=inputs['soma_mask'])
    finally:finish_rendering()
    raw=tifffile.imread(inputs['raw'])
    versions=[(label,data/f'references/{label}/{cid}.swc') for label in ['v0.81','v0.83b']
              if (data/f'references/{label}/{cid}.swc').is_file()]
    if previous and (previous/f'swc/{cid}.swc').is_file():
        versions.append(('v0.84 previous',previous/f'swc/{cid}.swc'))
    versions.append(('v0.84 thin-gap + crossing-switch',out/f'swc/{cid}.swc'))
    render_swc_comparison(raw,versions,out/'comparison.png',title=f'Neuron {cid} | historical references / previous / optimized')
    if len(versions)==4:
        render_swc_comparison(raw,versions[-2:],out/'optimization_effect.png',title=f'Neuron {cid} | previous / optimized')
    stages=[('Pruning input',Path(inputs['swc'])),('After initial <7.75 um',out/f'mips/{cid}_connections/before_connection_cuts.swc'),
            ('After diagnosed cuts',out/f'mips/{cid}_connections/after_connection_cuts.swc'),
            ('After neighbor-soma pruning + geometry',out/f'mips/{cid}_connections/after_neighbor_soma_cuts.swc')]
    render_swc_comparison(raw,stages,out/'stages.png',title=f'Neuron {cid} | pruning stages')
    result={label:metrics(path) for label,path in versions};stage_metrics={label:metrics(path) for label,path in stages}
    overlap={label:dict(reference_covered_by_v084=proximity(path,versions[-1][1]),v084_covered_by_reference=proximity(versions[-1][1],path)) for label,path in versions[:-1]}
    audit=json.loads((out/f'mips/{cid}_pruning_reasons.json').read_text())
    diagnosis=json.loads((out/f'mips/{cid}_connections/diagnosis.json').read_text())
    cut_sites=[s for s in diagnosis['sites'] if s.get('proposed_cut_ids')]
    review_links=''.join(f'<li><a href="mips/{cid}_connections/site_{s["site_id"]}.html">位置 {s["site_id"]} · {s["kind"]}</a>：拟剪节点 {s["proposed_cut_ids"]}</li>' for s in cut_sites)
    summary=dict(versions=result,stages=stage_metrics,spatial_proximity=overlap,diagnostic_sites=len(diagnosis['sites']),
                 diagnostic_counts=diagnosis['counts'],initial_removed_nodes=len(audit['initial_short_leaf_audit']['removed_ids']),
                 connection_removed_nodes=len(audit['connection_audit']['removed_ids']),
                 neighbor_soma_removed_nodes=len(audit['neighbor_soma_audit']['removed_ids']),
                 neighbor_soma_entry_events=audit['neighbor_soma_audit']['events'],
                 note='Same image coordinates; v0.81 historical soma differs from v0.83b. No manual ground truth; length gain is not accuracy.')
    save_json(out/'comparison_metrics.json',summary)
    with (out/'comparison_metrics.csv').open('w',newline='') as f:
        keys=['version','nodes','length_um','branches','tips','branchpoints'];writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for label,m in result.items():writer.writerow(dict(version=label,**{k:m[k] for k in keys[1:]}))
    rows=''.join(f'<tr><td>{html.escape(label)}</td><td>{m["nodes"]}</td><td>{m["length_um"]:.2f}</td><td>{m["branchpoints"]}</td><td>{m["tips"]}</td></tr>' for label,m in result.items())
    (out/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{cid} 版本对照</title>
<style>body{{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px}}img{{width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse}}</style>
<h1>{cid}：先短末梢处理，再诊断与剪断</h1><p>复用已复制的 0.83b 重采样 SWC 和图像，仅改变剪枝。先保存局部断口和疑似串线诊断，再执行剪断，随后胞体清理、其他胞体进入剪枝、合并近点和 Zigzag 修正；末尾不再次执行 7.75 µm 过滤。</p>
<p>新增完整长分支窗口、随路径移动的薄层判定及连续弱纤维保护。串线规则要求三维及至少两个薄层视图支持，属于连接归属推断，不能代替人工真值。诊断共删除 {len(audit['connection_audit']['removed_ids'])} 个节点；请优先复核以下位置：</p><ul>{review_links}</ul>
<p>本次其他胞体规则删除 {len(audit["neighbor_soma_audit"]["removed_ids"])} 个节点。<a href="mips/{cid}_connections/neighbor_soma_cuts.json">进入点与删除记录</a></p>
<p><a href="mips/{cid}_connections/index.html">全部局部包围盒和三方向诊断</a> · <a href="comparison_metrics.json">对比指标与阶段统计</a> · <a href="swc/{cid}.swc">新 SWC</a></p>
<table><tr><th>版本</th><th>节点</th><th>总长度 µm</th><th>分叉点</th><th>末端</th></tr>{rows}</table>
<p>相同原图坐标，无平移。0.81 胞体根坐标与 0.83b 略有差别。长度/节点数增加不等于质量提高；uncertain 位置保留，需结合局部图复核。</p>
<h2>历史版本及优化前后对照</h2><img src="comparison.png"><h2>本次各阶段</h2><img src="stages.png"></html>''',encoding='utf-8')
    (experiment/'latest-result.txt').write_text(str(out)+'\n')
    (experiment/f'latest-result-{cid}.txt').write_text(str(out)+'\n')
    print(json.dumps(dict(output=str(out),**summary),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
