#!/usr/bin/env python3
"""Export paired G-Cut soma coordinates, spatial differences, and Vaa3D markers.

Matching is descriptive only: no matched/unmatched point is filtered or fed back
into G-Cut. Inputs are the actual per-arm soma contexts and seed-matching audits.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import bootstrap

COLORS = {'old': [255, 160, 32], 'new': [32, 255, 120],
          'target': [255, 64, 64], 'review': [170, 120, 255], 'historical': [80, 180, 255]}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spatial_pairs(old, new, threshold=6.):
    """Maximum cardinality, then minimum total Euclidean distance, one-to-one."""
    old = np.asarray(old, dtype=float).reshape(-1, 3)
    new = np.asarray(new, dtype=float).reshape(-1, 3)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Positive finite threshold required')
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('Finite coordinates required')
    if not len(old) or not len(new):
        return [], list(range(len(old))), list(range(len(new)))
    distance = cdist(old, new)
    # Each old point can match a new point or one of len(old) dummy columns.
    # Penalty exceeds the possible sum of all legal distances, so cardinality
    # has priority over distance even in ambiguous clusters.
    penalty = (min(len(old), len(new)) + 1) * (threshold + 1)
    cost = np.full((len(old), len(new) + len(old)), penalty)
    cost[:, :len(new)] = np.where(distance <= threshold, distance, penalty * 3)
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j), float(distance[i, j])) for i, j in zip(rows, cols)
             if j < len(new) and distance[i, j] <= threshold]
    oi, ni = {p[0] for p in pairs}, {p[1] for p in pairs}
    return pairs, [i for i in range(len(old)) if i not in oi], [i for i in range(len(new)) if i not in ni]


def marker_coordinates(xyz, shape_zyx, for_tiff=False):
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3).copy()
    shape = np.asarray(shape_zyx)
    if not np.isfinite(xyz).all() or not np.all((xyz >= 0) & (xyz < shape[::-1])):
        raise ValueError('Out-of-image marker')
    if for_tiff:
        xyz[:, 1] = shape[1] - 1 - xyz[:, 1]
    return xyz + 1


def write_marker(path, points, shape, radius=8., for_tiff=False):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid display radius')
    coordinates = marker_coordinates([p['xyz'] for p in points], shape, for_tiff)
    with Path(path).open('w') as f:
        f.write('##x,y,z,radius,shape,name,comment,color_r,color_g,color_b\n')
        writer = csv.writer(f)
        for point, xyz in zip(points, coordinates):
            writer.writerow([*xyz, radius, 1, point['name'], point['comment'], *point['color']])
    # Read back the deliverable, including round-trip coordinates and colors.
    with Path(path).open() as f:
        next(f); rows = list(csv.reader(f))
    if len(rows) != len(points):
        raise ValueError('Marker count mismatch')
    for row, pos, point in zip(rows, coordinates, points):
        if not np.allclose(np.array(row[:3], float), pos) or float(row[3]) != radius or list(map(int, row[7:10])) != point['color']:
            raise ValueError('Marker round-trip mismatch')
    return dict(count=len(points),sha256=digest(path),coordinate_mode='TIFF Y flipped; XYZ one based' if for_tiff else 'V3DRAW XYZ one based',display_radius_um=radius)


def points_for_arm(out, cell, arm):
    folder = out / arm / 'gcut_output' / cell
    context = read(folder / 'soma-context.json')
    audit = read(folder / 'seed-matching.json')
    points = []
    group = 'old' if arm == 'legacy_seeds' else 'new'
    if len(context['centers_zyx']) != len(audit['records']):
        raise ValueError('Context/matching count mismatch')
    for center, record in zip(context['centers_zyx'], audit['records']):
        if not np.array_equal(center, record['center_zyx']):
            raise ValueError('Context/matching coordinate mismatch')
        target = np.array_equal(center, context['target_zyx'])
        points.append(dict(name=group+'_'+record['seed_id'], seed_id=record['seed_id'],
                           xyz=list(center[::-1]), target=target,
                           color=COLORS['target' if target else group],
                           comment=f"{arm}; {record['status']}; display diameter 16um only",
                           matching=record))
    return points, audit


def draw(raw, old, new, path, title, bounds=None, differences=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lo, hi = (np.zeros(3, int), np.asarray(raw.shape[::-1])-1) if bounds is None else bounds
    crop = raw[lo[2]:hi[2]+1, lo[1]:hi[1]+1, lo[0]:hi[0]+1]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout='constrained')
    vmax = max(1, float(raw.max()))  # Full dynamic range preserves bright soma structure.
    for col, (name, axis, a, b) in enumerate([('XY', 0, 0, 1), ('XZ', 1, 0, 2), ('YZ', 2, 1, 2)]):
        extent = [lo[a]-.5, hi[a]+.5, hi[b]+.5, lo[b]-.5]
        for row, (label, points) in enumerate([('Old seeds',old),('Gemma seeds',new),('Overlay',old+new)]):
            ax = axes[row, col]
            ax.imshow(crop.max(axis), cmap='gray', vmin=0, vmax=vmax, extent=extent, interpolation='nearest')
            for p in points:
                xyz = np.asarray(p['xyz'])
                if not np.all((xyz >= lo) & (xyz <= hi)):
                    continue
                color = np.asarray(p['color'])/255.
                symbol = '+' if p['name'].startswith('new_') else 'o'
                ax.scatter([xyz[a]], [xyz[b]], marker=symbol, s=90 if bounds is not None else 50,
                           facecolors='none' if symbol=='o' else [color], edgecolors=[color], linewidths=1.1)
                if bounds is not None:
                    ax.annotate(p['seed_id'], (xyz[a],xyz[b]), xytext=(3,4),textcoords='offset points',color=color,fontsize=7)
            if row == 2 and differences:
                for o, n in differences:
                    p, q = np.asarray(old[o]['xyz']), np.asarray(new[n]['xyz'])
                    if np.all((p>=lo)&(p<=hi)) and np.all((q>=lo)&(q<=hi)):
                        ax.plot([p[a],q[a]],[p[b],q[b]],color='#dddddd',lw=.7,ls=':')
            ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',xlabel='XYZ'[a]+' (um)',ylabel='XYZ'[b]+' (um)',title=label+' | '+name)
    fig.suptitle(title+'\nOrange circles: old; green +: Gemma; red: known target. Same crop/window; 3D-filtered landmarks.')
    fig.savefig(path,dpi=150);plt.close(fig)
    return dict(xyz_bounds=[lo.tolist(),hi.tolist()],intensity_window=[0,vmax],array_axes='ZYX',coordinates='zero-based XYZ um',voxel_size_um_xyz=[1,1,1])


def report(out, threshold=6., radius=8.):
    out=Path(out).resolve(); manifest=read(out/'run.json');cell=manifest['cell']
    seed_run=Path(manifest['seed_run']); bundle=read(seed_run/'candidates.json')
    if bundle['voxel_size_um_xyz'] != [1,1,1]:
        raise ValueError('Exporter requires the recorded 1 um isotropic image')
    if digest(manifest['inputs']['raw']) != manifest['hashes']['raw']:
        raise ValueError('Changed raw input')
    raw=tifffile.imread(manifest['inputs']['raw'])
    old,old_audit=points_for_arm(out,cell,'legacy_seeds')
    new,new_audit=points_for_arm(out,cell,'gemma_seeds')
    # Known metadata target must pair to itself, not a nearby proposed point.
    oi=[i for i,p in enumerate(old) if p['target']]; ni=[i for i,p in enumerate(new) if p['target']]
    if len(oi)!=1 or len(ni)!=1 or not np.array_equal(old[oi[0]]['xyz'],new[ni[0]]['xyz']):
        raise ValueError('Known target is missing or ambiguous')
    orest=[i for i in range(len(old)) if i not in oi]; nrest=[i for i in range(len(new)) if i not in ni]
    pairs,only_o,only_n=spatial_pairs([old[i]['xyz'] for i in orest],[new[i]['xyz'] for i in nrest],threshold)
    pairs=[(oi[0],ni[0],0.)]+[(orest[i],nrest[j],d) for i,j,d in pairs]
    only_o=[orest[i] for i in only_o];only_n=[nrest[i] for i in only_n]
    dest=out/'soma_comparison';dest.mkdir(exist_ok=True)
    groups={'old_all':old,'gemma_all':new,'overlay':old+new,
            'old_only':[old[i] for i in only_o],'gemma_only':[new[i] for i in only_n]}
    for group,points in [('old',old),('gemma',new)]:
        groups[group+'_matched_soma']=[p for p in points if p['matching']['status'] in ['matched','same_app2_node']]
        groups[group+'_unmatched_soma']=[p for p in points if p['matching']['status']=='unmatched']
        groups[group+'_shared_app2_soma']=[p for p in points if p['matching']['status']=='same_app2_node']
        anchors={}
        for p in groups[group+'_matched_soma']:
            r=p['matching']; nid=r['nearest_node_id']
            if nid not in anchors:
                anchors[nid]=dict(p,name=group+'_APP2_'+str(nid),xyz=r['nearest_xyz'],
                                  comment=f'Actual G-Cut APP2 anchor {nid}; seed {p["seed_id"]}; display only')
        groups[group+'_app2_anchors']=list(anchors.values())
    catalogue=read(seed_run/'normalized/soma_catalogue.json')
    candidates={p['candidate_id']:p for p in bundle['candidates']}
    groups['review_not_used']=[dict(name='REVIEW_'+cid,seed_id=cid,xyz=candidates[cid]['xyz_um'],
                                   color=COLORS['review'],comment='Unresolved; NOT used by G-Cut') for cid in catalogue['review']]
    historical=ROOT.parent/'test/0.84/data/source_v083b/logs'/cell/'soma-context.json'
    if historical.exists():
        groups['historical_v083b']=[dict(name=f'historical_{i}',seed_id=f'historical_{i}',xyz=p[::-1],color=COLORS['historical'],comment='Archived v0.83b soma context; not paired current baseline') for i,p in enumerate(read(historical)['centers_zyx'])]
    exported={}
    for name,points in groups.items():
        for tiff in [False,True]:
            filename=name+('_for_tiff' if tiff else '')+'.marker'
            exported[filename]=write_marker(dest/filename,points,raw.shape,radius,tiff)
    mapping=dict(threshold_um=threshold,policy='known target paired first; other points maximum-cardinality/minimum-distance one-to-one within threshold; comparison ONLY; no suppression or identity truth',
                 pairs=[dict(old_id=old[i]['seed_id'],new_id=new[j]['seed_id'],old_xyz=old[i]['xyz'],new_xyz=new[j]['xyz'],distance_um=d) for i,j,d in pairs],
                 old_only=[old[i] for i in only_o],gemma_only=[new[i] for i in only_n],old_points=old,gemma_points=new)
    decisions=read(seed_run/'normalized/decisions.json')
    # The old workflow also proposes auxiliary seeds. A spatially unmatched old
    # seed must not be described as a Gemma rejection if Gemma never reviewed it.
    proposal_context=[]
    for point in old:
        nearby=[]
        for cid,c in candidates.items():
            distance=float(np.linalg.norm(np.asarray(point['xyz'])-c['xyz_um']))
            if distance<=threshold:
                decision=decisions[cid]['decision']
                nearby.append(dict(candidate_id=cid,distance_um=distance,decision=decision['decision'],duplicate_of=decision['duplicate_of']))
        nearby.sort(key=lambda x:x['distance_um'])
        proposal_context.append(dict(old_seed_id=point['seed_id'],known_target=point['target'],
                                     candidates_within_comparison_radius=nearby,
                                     note='spatial neighborhood only; not confirmed same soma; no nearby proposal does not mean model rejection'))
    mapping['old_point_proposal_context']=proposal_context
    save(dest/'spatial_correspondence.json',mapping)
    views={'overview':draw(raw,old,new,dest/'overview.png',cell+': soma seed positions',differences=[(i,j) for i,j,_ in pairs])}
    # Actual matched changes first, followed by largest coordinate shifts and unmatched differences.
    cases=[]
    proposals=[(new[i], 'Gemma-only coordinate: '+new[i]['seed_id']+'; '+new[i]['matching']['status']) for i in only_n]
    proposals += [(old[i],'Old-only coordinate: '+old[i]['seed_id']+'; '+old[i]['matching']['status']) for i in only_o]
    proposals.sort(key=lambda x:x[0]['matching']['status']!='matched')
    proposals += [(new[j],f'Paired coordinate shift {d:.2f} um: '+old[i]['seed_id']+' / '+new[j]['seed_id']) for i,j,d in sorted(pairs,key=lambda p:-p[2]) if d>0]
    for point,reason in proposals:
        center=np.asarray(point['xyz'])
        if any(np.linalg.norm(center-np.asarray(c['center_xyz']))<20 for c in cases):continue
        lo=np.maximum(0,np.floor(center-25)).astype(int);hi=np.minimum(np.asarray(raw.shape[::-1])-1,np.ceil(center+25)).astype(int)
        idx=len(cases)+1; filename=f'local_{idx}.png'
        views[filename]=draw(raw,old,new,dest/filename,cell+': '+reason,(lo,hi),[(i,j) for i,j,_ in pairs])
        cases.append(dict(file=filename,center_xyz=center.tolist(),reason=reason))
        if len(cases)>=6:break
    save(dest/'views.json',views)
    summary=dict(cell=cell,old_count=len(old),gemma_count=len(new),spatial_pairs=len(pairs),old_only=len(only_o),gemma_only=len(only_n),
                 old_actual_anchors=len(old_audit['unique_matched_nodes']),gemma_actual_anchors=len(new_audit['unique_matched_nodes']),
                 review_not_used=len(catalogue['review']),threshold_um=threshold,display_diameter_um=2*radius,
                 exports=exported,local_cases=cases,biological_accuracy_evaluated=False)
    save(dest/'summary.json',summary)
    save(dest/'manifest.json',dict(source_code_sha256=digest(__file__),downstream_manifest_sha256=digest(out/'run.json'),
                                  inputs={str(p):digest(p) for p in [seed_run/'normalized/soma_catalogue.json',*[out/a/'gcut_output'/cell/f for a in ['legacy_seeds','gemma_seeds'] for f in ['soma-context.json','seed-matching.json']]]},
                                  display_radius_used_for_gcut=False,marker_roundtrip_passed=True))
    (dest/'source.py').write_text(Path(__file__).read_text())
    instructions=f'''{cell} Vaa3D 胞体点比较\n\n推荐打开 ../vaa3d/{cell}.v3draw，然后通过 Vaa3D 的 marker 导入/拖入 overlay.marker，或分别载入 old_all.marker 与 gemma_all.marker。\n普通 .marker 使用 XYZ+1，配套 v3draw 不翻转 Y。若打开原 TIFF，使用 *_for_tiff.marker（X+1, height-Y, Z+1），不可混用。\n橙色=旧筛选；绿色=Gemma；红色=已知目标T；紫色=待复核且未参与 G-Cut。显示半径 {radius:g} µm，直径 {2*radius:g} µm；仅为显示。\nold_only/gemma_only 是 {threshold:g} µm 内一对一空间匹配后未配对的点，不代表真正新增/漏检胞体。\n*_matched_soma 是成功匹配 APP2 的胞体位置；*_app2_anchors 是实际参与 G-Cut 的 APP2 节点位置，两者可能不同。*_unmatched_soma 没有参与种子分配；*_shared_app2_soma 与另一个胞体点映射到同一个 APP2 节点，不增加独立种子。\n旧流程还包含辅助提名：仅旧点不一定被 Gemma 拒绝，也可能未进入本轮候选集，详见 spatial_correspondence.json 的 old_point_proposal_context；邻近候选只表示空间接近。旧组来自相同输入下当前原胞体流程；historical_v083b（如有）为归档历史胞体点，另列以免混淆。\n后续完整 SWC、全部新分配组件和原图见 ../vaa3d；两组 G-Cut、重采样、基础剪枝报告见 ../index.html。\n没有人工真值；点数、空间匹配和结构变化都不是准确率。\n'''
    (dest/'README.txt').write_text(instructions)
    rows=''.join('<tr><td>'+name+'</td><td>'+str(len(points))+'</td><td><a href="'+name+'.marker">V3DRAW marker</a> · <a href="'+name+'_for_tiff.marker">TIFF marker</a></td></tr>' for name,points in groups.items())
    local=''.join('<h3>'+html.escape(c['reason'])+'</h3><img loading="lazy" src="'+c['file']+'">' for c in cases)
    seed_link=os.path.relpath(seed_run/'normalized/index.html',dest)
    point_rows=[]
    for p in old+new:
        r=p['matching'];cid=p['seed_id']
        label=html.escape(p['name'])
        if p['name'].startswith('new_') and cid!='T':label='<a href="'+seed_link+'#'+cid+'">'+label+'</a>'
        point_rows.append('<tr><td>'+label+'</td><td>'+str(p['xyz'])+'</td><td>'+r['status']+'</td><td>'+str(r['nearest_node_id'])+'</td><td>'+f"{r['distance_um']:.2f} / {r['tolerance_um']:.2f}"+'</td></tr>')
    point_table='<h2>逐点坐标与实际 APP2 匹配</h2><p>点击 Gemma 点 ID 查看原始三视图、连续切片和模型回答。T 为已知目标，不由模型新检出。</p><table><tr><th>点 ID</th><th>XYZ µm</th><th>匹配状态</th><th>最近 APP2 节点</th><th>距离 / 容差 µm</th></tr>'+''.join(point_rows)+'</table>'

    proposal_rows=[]
    for p in proposal_context:
        near=p['candidates_within_comparison_radius']
        text='已知目标 T' if p['known_target'] else ('；'.join(n['candidate_id']+' '+n['decision']+(' → '+n['duplicate_of'] if n['duplicate_of'] else '')+f" ({n['distance_um']:.2f} µm)" for n in near) or '比较半径内无候选：不能称为被 Gemma 拒绝')
        proposal_rows.append('<tr><td>'+p['old_seed_id']+'</td><td>'+html.escape(text)+'</td></tr>')
    point_table+='<h2>旧点附近有哪些点真正送给了 Gemma</h2><p>旧流程包含辅助提名，本表仅追踪 '+str(threshold)+' µm 内的候选与程序应用结果；相近坐标不是同一胞体的真值。</p><table><tr><th>旧点</th><th>附近审核候选</th></tr>'+''.join(proposal_rows)+'</table>'
    (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+cell+' 胞体点差异</title><style>body{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto}img{width:100%}td,th{padding:10px;border:1px solid #ccc}pre{white-space:pre-wrap}</style><h1>'+cell+'：旧胞体点与 Gemma 胞体点</h1><p>旧 '+str(len(old))+' 点，新 '+str(len(new))+' 点；空间配对 '+str(len(pairs))+' 对，仅旧 '+str(len(only_o))+' 点，仅新 '+str(len(only_n))+' 点。实际 APP2 种子 '+str(summary['old_actual_anchors'])+' → '+str(summary['gemma_actual_anchors'])+'。</p><p><a href="../index.html">G-Cut 与剪枝对照</a> · <a href="../vaa3d/'+cell+'.v3draw">配套 V3DRAW 原图</a> · <a href="spatial_correspondence.json">逐点坐标对应和 APP2 匹配</a> · <a href="summary.json">统计</a></p><pre>'+html.escape(instructions)+'</pre><table><tr><th>点集</th><th>点数</th><th>文件</th></tr>'+rows+'</table><img src="overview.png">'+point_table+local)
    print(json.dumps(summary,ensure_ascii=False))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--threshold',type=float,default=6.)
    parser.add_argument('--radius',type=float,default=8.)
    args=parser.parse_args();report(args.run,args.threshold,args.radius)
