#!/usr/bin/env python3
"""Compare paired G-Cut/pruning outputs, with local raw-image inspection."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys
from collections import Counter

import numpy as np
import tifffile
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import bootstrap
from utils.case_worker import save_json
from tools.run_connection_experiment import metrics, proximity, samples
from visualization.swc_comparison import render_swc_comparison
from visualization.ownership_audit import clipped_segments, PLANES
from visualization.projection_viewer import write_site_viewer
from algorithms.tracing.soma_branchwalk import tree_index


def read(path):return json.loads(Path(path).read_text())


def paths(path):
    rows=np.loadtxt(path,ndmin=2);nodes={int(r[0]):r[2:5] for r in rows}
    return [np.array([nodes[int(r[6])],r[2:5]]) for r in rows if int(r[6]) in nodes]


def report(out):
    manifest=read(out/'run.json');cell=manifest['cell'];data=ROOT.parent/'test/0.84/data'
    raw=tifffile.imread(manifest['inputs']['raw'])
    arms=['legacy_seeds','gemma_seeds']
    label={'legacy_seeds':'v0.84 automatic soma seeds','gemma_seeds':'v0.84 Gemma soma seeds'}
    all_metrics={};stage_metrics={};seed_stats={};pruning={}
    final={a:out/a/f'pruning_swcs/{cell}.swc' for a in arms}
    versions=[(v,data/f'references/{v}/{cell}.swc') for v in ['v0.81','v0.83b'] if (data/f'references/{v}/{cell}.swc').exists()]
    versions.extend((label[a],final[a]) for a in arms)
    for name,path in versions:all_metrics[name]=metrics(path)
    for arm in arms:
        root=out/arm;gcut=root/f'gcut_output/{cell}'
        if not (root/'complete.json').exists():raise ValueError('Incomplete arm: '+arm)
        stage_paths={'gcut':root/f'gcut_output/selected_for_pruning/{cell}.swc',
                     'resampled':root/f'gcut_output/selected_resampled_step1/{cell}.swc','final':final[arm]}
        stage_metrics[arm]={k:metrics(p) for k,p in stage_paths.items()}
        for p in stage_paths.values():tree_index(np.loadtxt(p,ndmin=2).tolist())
        match=read(gcut/'seed-matching.json');survival=read(gcut/'gcut-seed-survival.json')
        seed_stats[arm]=dict(proposed=len(read(gcut/'soma-context.json')['centers_zyx']),
              matching_counts=dict(Counter(r['status'] for r in match['records'])),
              unique_matched=len(match['unique_matched_nodes']),survival=survival,
              unmatched=[r for r in match['records'] if r['status']!='matched'])
        seed_stats[arm]['components']={str(sid):dict(metrics=metrics(gcut/f'{cell}_gcut_soma_{sid}.swc'),
              matched_seed_ids=[r['seed_id'] for r in match['records'] if r['status']=='matched' and r['nearest_node_id']==sid])
              for sid in survival['assembled_soma_ids']}
        audit=read(root/f'pruning_mips/{cell}_pruning_reasons.json')
        pruning[arm]=dict(initial_short_removed=len(audit['initial_short_leaf_audit']['removed_ids']),
              connection_removed=len(audit['connection_audit']['removed_ids']),
              neighbor_soma_removed=len(audit['neighbor_soma_audit']['removed_ids']),
              active_rules=audit['active_rules'])
    render_swc_comparison(raw,versions,out/'comparison.png',title=f'{cell}: historical / paired downstream comparison')
    render_swc_comparison(raw,[(label[a],final[a]) for a in arms],out/'paired_final.png',title='Same APP2 and same downstream rules')
    render_swc_comparison(raw,[(label[a]+' before pruning',out/a/f'gcut_output/selected_resampled_step1/{cell}.swc') for a in arms],out/'paired_before_pruning.png',title='Effect of soma seeds before pruning')
    overlap={name:dict(reference_covered_by_gemma=proximity(p,final['gemma_seeds']),
                        gemma_covered_by_reference=proximity(final['gemma_seeds'],p)) for name,p in versions[:-1]}
    # Select largest spatial changes for local review; proximity is not accuracy.
    centers=[];change_metrics={}
    for a,b in [('legacy_seeds','gemma_seeds'),('gemma_seeds','legacy_seeds')]:
        p,w=samples(final[a]);q,_=samples(final[b]);distance=cKDTree(q).query(p)[0]
        changed=distance>2;change_metrics[a]=dict(length_farther_than_2um=float(w[changed].sum()),fraction=float(w[changed].sum()/w.sum()))
        bins={}
        for point,weight in zip(p[changed],w[changed]):
            key=tuple(np.floor(point/30).astype(int));bins.setdefault(key,[]).append((point,weight))
        chosen=0
        for group in sorted(bins.values(),key=lambda g:-sum(t[1] for t in g)):
            center=np.average([t[0] for t in group],axis=0,weights=[t[1] for t in group])
            if all(np.linalg.norm(center-c['center'])>30 for c in centers):
                centers.append(dict(center=center,reason=a+' structure >2um from other result',length=sum(t[1] for t in group)))
                chosen+=1
                if chosen>=3:break
    cat=read(Path(manifest['seed_run'])/'normalized/soma_catalogue.json')
    old=np.array(read(out/'legacy_seeds'/f'gcut_output/{cell}/soma-context.json')['centers_zyx'])[:,::-1]
    matches=read(out/'gemma_seeds'/f'gcut_output/{cell}/seed-matching.json')['records']
    target=np.asarray(next(c['xyz_um'] for c in cat['accepted'] if c['candidate_id']=='T'))
    for match in matches:
        center=np.asarray(match['center_zyx'])[::-1]
        if match['status']=='matched' and match['seed_id']!='T' and np.linalg.norm(old-center,axis=1).min()>6:
            centers.append(dict(center=(center+target)/2,reason='New matched seed '+match['seed_id']+' versus target T',length=None,radius=32))
    unmatched=sorted((m for m in matches if m['status']=='unmatched'),key=lambda m:m['distance_um'])
    if unmatched:
        m=unmatched[0]
        centers.append(dict(center=np.asarray(m['center_zyx'])[::-1],reason=f"Unmatched seed {m['seed_id']}: distance {m['distance_um']:.2f} > tolerance {m['tolerance_um']:.2f} um",length=None))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    folder=out/'local_review';folder.mkdir(exist_ok=True)
    routes={a:paths(final[a]) for a in arms};local=[]
    for i,case in enumerate(centers,1):
        center=case['center'];radius=case.get('radius',25);lo=np.maximum(0,np.floor(center-radius)).astype(int);hi=np.minimum(np.array(raw.shape[::-1])-1,np.ceil(center+radius)).astype(int)
        crop=raw[lo[2]:hi[2]+1,lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        fig,axes=plt.subplots(3,3,figsize=(12,12),layout='constrained')
        for col,(plane,axis,xy) in enumerate(PLANES):
            a,b=xy;bg=crop.max(axis);extent=[lo[a]-.5,hi[a]+.5,hi[b]+.5,lo[b]-.5]
            for row,arm in enumerate([None,*arms]):
                ax=axes[row,col];ax.imshow(bg,cmap='gray',vmin=0,vmax=180,extent=extent,interpolation='nearest')
                if arm:
                    lines=clipped_segments(routes[arm],lo,hi,xy)
                    if lines:ax.add_collection(LineCollection(lines,colors='#ffcc55' if arm=='legacy_seeds' else '#44ddff',linewidths=1))
                for soma in cat['accepted']:
                    position=np.asarray(soma['xyz_um'])
                    if np.all(position>=lo) and np.all(position<=hi):
                        color='#ff6464' if soma['candidate_id']=='T' else '#99ff66'
                        ax.plot(position[a],position[b],marker='+',color=color,ms=7)
                        ax.annotate(soma['candidate_id'],(position[a],position[b]),xytext=(4,4),textcoords='offset points',color=color,fontsize=8)
                ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',xlabel='XYZ'[a]+' (um)',ylabel='XYZ'[b]+' (um)')
                ax.set_title(('Raw only' if arm is None else label[arm])+' | '+plane,fontsize=10)
        fig.suptitle(f'Local {i}: same raw crop and intensity window; '+case['reason'])
        fig.savefig(folder/f'local_{i}.png',dpi=140);plt.close(fig)
        for arm in arms:
            clipped=clipped_segments(routes[arm],lo,hi,(0,1,2))
            site=dict(site_id=i,center_xyz=center.tolist(),kind=label[arm],status='comparison; not a correctness label',
                     bbox_xyz=[[int(lo[k]),int(hi[k]+1)] for k in range(3)],image=f'local_{i}.png',
                     arms=[dict(xyz=e.tolist()) for e in clipped])
            viewer=folder/f'local_{i}_{arm}.html';write_site_viewer(crop,lo,site,viewer)
            doc=viewer.read_text().replace('href="diagnosis.json"','href="local_cases.json"').replace('href="interactive.html"','href="../index.html"')
            viewer.write_text(doc)
        local.append(dict(id=i,center_xyz=center.tolist(),bbox_xyz=[lo.tolist(),hi.tolist()],reason=case['reason'],changed_length_in_bin=case['length']))
    save_json(folder/'local_cases.json',local)
    summary=dict(cell=cell,metrics=all_metrics,stages=stage_metrics,seeds=seed_stats,pruning=pruning,
                  spatial_proximity=overlap,paired_changed_length=change_metrics,
                  unresolved_seed_ids=cat['review'],local_cases=local,
                  biological_accuracy_evaluated=False,note='Node/length differences and historical references are not correctness labels')
    save_json(out/'comparison_metrics.json',summary)
    # Vaa3D exports share an unflipped raw stack with image-coordinate SWCs.
    from v3dpy.loaders import Raw
    vaa=out/'vaa3d';vaa.mkdir(exist_ok=True);Raw().save(str(vaa/f'{cell}.v3draw'),raw[None])
    for arm in arms:
        shutil.copy2(final[arm],vaa/f'{arm}.swc')
        (vaa/f'{arm}.ano').write_text(f'RAWIMG={cell}.v3draw\nSWCFILE={arm}.swc\n')
    component_links=[]
    component_ano=[f'RAWIMG={cell}.v3draw']
    for sid,record in seed_stats['gemma_seeds']['components'].items():
        name=f'gemma_component_{sid}.swc'
        shutil.copy2(out/'gemma_seeds'/f'gcut_output/{cell}/{cell}_gcut_soma_{sid}.swc',vaa/name)
        component_ano.append('SWCFILE='+name)
        component_links.append(f'<li><a href="vaa3d/{name}">组件 {sid}：'+html.escape(', '.join(record['matched_seed_ids']))+'</a> · '+str(record['metrics']['nodes'])+' 节点 · '+f'{record["metrics"]["length_um"]:.2f}'+' µm</li>')
    (vaa/'gemma_all_components.ano').write_text('\n'.join(component_ano)+'\n')
    allocation='<h2>Gemma 种子对应的 G-Cut 完整组件</h2><p>这些是剪枝前的分配结果；保留为其他胞体的分支并未销毁。</p><ul>'+''.join(component_links)+'</ul><p><a href="vaa3d/gemma_all_components.ano">Vaa3D 叠加全部分配组件</a></p>'
    (vaa/'paired.ano').write_text(f'RAWIMG={cell}.v3draw\nSWCFILE=legacy_seeds.swc\nSWCFILE=gemma_seeds.swc\n')
    seed_overview='<p>旧组：'+str(seed_stats['legacy_seeds']['proposed'])+' 个胞体点 → '+str(seed_stats['legacy_seeds']['unique_matched'])+' 个 APP2 种子；Gemma 组：'+str(seed_stats['gemma_seeds']['proposed'])+' 个胞体点 → '+str(seed_stats['gemma_seeds']['unique_matched'])+' 个 APP2 种子。未匹配点不会自动创建新的神经突路径。</p>'
    rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{m["nodes"]}</td><td>{m["length_um"]:.2f}</td><td>{m["branchpoints"]}</td><td>{m["tips"]}</td></tr>' for k,m in all_metrics.items())
    cases=''.join(f'<h3>位置 {c["id"]} · XYZ {c["center_xyz"]}</h3><p>{html.escape(c["reason"])} · <a href="local_review/local_{c["id"]}_legacy_seeds.html">旧种子：薄层/切片</a> · <a href="local_review/local_{c["id"]}_gemma_seeds.html">Gemma 种子：薄层/切片</a></p><img loading="lazy" src="local_review/local_{c["id"]}.png">' for c in local)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+cell+' 胞体结果接入后续流程</title><style>body{font:16px/1.65 system-ui;max-width:1400px;margin:30px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}td,th{padding:10px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style><h1>'+cell+'：Gemma 胞体结果 → G-Cut → 重采样 → 基础剪枝</h1><p>两组共享同一份原图、mask 和 APP2 SWC。旧组执行原胞体筛选；新组使用 Gemma 审核保留的 '+str(len(cat['accepted'])-1)+' 个胞体与已知目标 T，不重复运行 NMS/自动种子合并。'+str(len(cat['review']))+' 个待复核点未加入。显示标记的放大半径没有用于计算。</p><p>后续两组均使用相同的 G-Cut、1 µm 重采样及已恢复的基础剪枝。未重跑 nnU-Net、APP2 或 Gemma；历史 v0.81/v0.83b 仅作背景对照。</p><p><a href="gemma_seeds/pruning_swcs/'+cell+'.swc">新最终 SWC</a> · <a href="vaa3d/gemma_seeds.ano">新结果 Vaa3D 叠加</a> · <a href="vaa3d/paired.ano">两组 Vaa3D 对照</a> · <a href="comparison_metrics.json">完整指标</a> · <a href="validation.json">输入与结构校验</a></p><table><tr><th>版本</th><th>节点</th><th>长度 µm</th><th>分叉点</th><th>末梢</th></tr>'+rows+'</table><p>无人工真值，准确率未评估；删去的长度不等于误连接减少，保留的长度也不等于目标长度召回。</p><h2>胞体种子实际匹配情况</h2><p><a href="soma_comparison/index.html">新旧胞体点差异图与 marker 文件</a></p>'+seed_overview+'<details><summary>展开逐点匹配和组件信息</summary><pre>'+html.escape(json.dumps(seed_stats,ensure_ascii=False,indent=2))+'</pre></details><p><a href="gemma_seeds/gcut_output/'+cell+'/seed-matching.json">Gemma 逐点 APP2 匹配</a> · <a href="legacy_seeds/gcut_output/'+cell+'/seed-matching.json">旧组逐点匹配</a></p>'+allocation+'<h2>最终结果对照</h2><img src="paired_final.png"><h2>剪枝前的 G-Cut 分配结果</h2><img src="paired_before_pruning.png"><h2>历史算法参考</h2><img src="comparison.png"><h2>代表性局部原图与结构对照</h2>'+cases+'<h2>基础剪枝诊断</h2><p><a href="legacy_seeds/pruning_mips/'+cell+'_connections/index.html">旧组局部连接诊断</a> · <a href="gemma_seeds/pruning_mips/'+cell+'_connections/index.html">Gemma 组局部连接诊断</a></p><pre>'+html.escape(json.dumps(pruning,ensure_ascii=False,indent=2))+'</pre>')
    save_json(out/'status.json',dict(status='complete',report=str(out/'index.html')))
    print(json.dumps(dict(output=str(out),metrics=all_metrics,seeds=seed_stats,pruning=pruning),ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();report(args.run.resolve())
