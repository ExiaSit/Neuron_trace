#!/usr/bin/env python3
"""Validate completed pruning experiments and build a multi-cell review page."""
from pathlib import Path
import argparse,hashlib,html,json,shutil,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.freeze_connection_inputs import digest


def validate_tree(path):
    tree=np.loadtxt(path,ndmin=2)
    assert np.isfinite(tree).all(),path
    nodes={int(n[0]):n for n in tree};assert len(nodes)==len(tree),path
    roots=[i for i,n in nodes.items() if int(n[6])==-1];assert len(roots)==1,path
    for start in nodes:
        seen=set();node=start
        while node!=-1:
            assert node in nodes and node not in seen,path
            seen.add(node);node=int(nodes[node][6])
    return dict(nodes=len(nodes),root=roots[0],root_xyz=nodes[roots[0]][2:5].tolist(),
                single_connected_tree=True,sha256=digest(path))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);ap.add_argument('runs',nargs='+');args=ap.parse_args()
    if Path(args.name).name!=args.name:ap.error('name must be a directory name')
    experiment=ROOT.parent/'test/0.84';records=[];sections=[];rows=[]
    for directory in args.runs:
        run=Path(directory).resolve();info=json.loads((run/'run.json').read_text());cid=info['cell']
        metrics=json.loads((run/'comparison_metrics.json').read_text());diag=run/f'mips/{cid}_connections'
        report=json.loads((diag/'diagnosis.json').read_text());cuts=json.loads((diag/'cuts.json').read_text())
        validation=validate_tree(run/f'swc/{cid}.swc')
        before=np.loadtxt(diag/'before_connection_cuts.swc',ndmin=2);after=np.loadtxt(diag/'after_connection_cuts.swc',ndmin=2)
        assert set(before[:,0].astype(int))-set(after[:,0].astype(int))==set(cuts['removed_ids'])
        for s in report['sites']:
            if s.get('image'):
                for path in [diag/s['image'],diag/f"site_{s['site_id']}.html"]:assert path.is_file() and path.stat().st_size>0,path
        for rel,sha in info['source_sha256'].items():assert digest(run/'source'/rel)==sha,rel
        manifest=json.loads(Path(info['copy_manifest']).read_text())
        for f in manifest['files']:assert digest(experiment/'data'/f['destination'])==f['sha256'],f['destination']
        assert digest(info['metadata_path'])==info['metadata_sha256']
        versions=metrics['versions'];current=next(v for k,v in versions.items() if k.startswith('v0.84 thin'))
        baseline=versions['v0.83b'];delta=current['length_um']-baseline['length_um']
        events=[dict(cut_node_id=e['cut_node_id'],removed_nodes=len(e['removed_ids']),evidence_sites=e['evidence_sites'],evidence_types=e['evidence_types']) for e in cuts['events']]
        record=dict(cell=cid,run=str(run),versions=versions,length_difference_from_v083b_um=delta,
                    length_difference_from_v083b_percent=100*delta/baseline['length_um'],
                    initial_removed_nodes=metrics['initial_removed_nodes'],connection_removed_nodes=metrics['connection_removed_nodes'],
                    neighbor_soma_removed_nodes=metrics['neighbor_soma_removed_nodes'],
                    diagnostic_sites=len(report['sites']),diagnostic_counts=report['counts'],connection_events=events,
                    input_files_verified=len(manifest['files']),validation=validation)
        records.append(record)
        rel='../'+run.name;links=''.join(f'<li>节点 {e["cut_node_id"]}，下游 {e["removed_nodes"]} 节点；'+
                  ' / '.join(f'<a href="{rel}/mips/{cid}_connections/site_{sid}.html">位置 {sid}</a>' for sid in e['evidence_sites'])+'</li>' for e in sorted(events,key=lambda e:-e['removed_nodes']))
        for label,m in versions.items():
            rows.append(f'<tr><td>{cid}</td><td>{html.escape(label)}</td><td>{m["nodes"]}</td><td>{m["length_um"]:.2f}</td><td>{m["branchpoints"]}</td><td>{m["tips"]}</td></tr>')
        sections.append(f'<section><h2>{cid}</h2><p><a href="{rel}/index.html">完整结果</a> · <a href="{rel}/mips/{cid}_connections/interactive.html">薄层交互诊断</a> · <a href="{rel}/swc/{cid}.swc">新 SWC</a></p>'
          f'<p>检查 {len(report["sites"])} 个位置；初始短末梢删除 {record["initial_removed_nodes"]} 节点，连接诊断删除 {record["connection_removed_nodes"]} 节点，其他胞体规则删除 {record["neighbor_soma_removed_nodes"]} 节点。</p>'
          f'<p>相比 v0.83b，总长度变化 {delta:+.2f} µm（{record["length_difference_from_v083b_percent"]:+.1f}%）。同一重采样输入，仅测试剪枝；节点/长度差异不代表准确率。</p>'
          f'<img loading="lazy" src="{rel}/comparison.png"><h3>连接剪断位置（按删除节点数排序）</h3><ul>{links or "<li>无连接诊断剪断</li>"}</ul></section>')
    out=experiment/'runs'/args.name;out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>89074 / 97117 剪枝测试</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#17212d}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}img{width:100%}p,li{line-height:1.6}</style>'
        '<h1>当前 v0.84 在新增样本上的剪枝测试</h1><p>复用各样本 v0.83b 的原图、分割和剪枝前重采样 SWC，沿用当前 0.84 规则及阈值。每个样本使用其原运行元数据；未重新运行 mask、APP2 或 G-Cut。</p>'
        '<p>复制文件 SHA256、源码快照及输出单根连通性均已验证。串线判断属于影像归属推断，需结合局部图复核；历史 0.81 根坐标略有差别，所有图均使用原图坐标且未平移。</p>'
        '<table><tr><th>样本</th><th>版本</th><th>节点</th><th>长度 µm</th><th>分叉</th><th>末端</th></tr>'+''.join(rows)+'</table>'+''.join(sections)+'</html>')
    shutil.copy2(__file__,out/'summarize_connection_experiments.py')
    (experiment/'latest-batch-result.txt').write_text(str(out)+'\n')
    print(json.dumps(dict(output=str(out),cells=[r['cell'] for r in records]),ensure_ascii=False))

if __name__=='__main__':main()
