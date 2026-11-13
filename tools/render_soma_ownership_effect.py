#!/usr/bin/env python3
"""Validate and display the completed whole-soma ownership experiment."""
from pathlib import Path
import argparse,hashlib,html,json,sys
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import bootstrap
from algorithms.tracing.soma_ownership_context import apply_ownership_plan,ownership_gate
from algorithms.tracing.connection_diagnostics import save_json
from tools.run_soma_ownership_experiment import verify_gate
from visualization.projection_viewer import write_site_viewer,write_viewer_index


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();out=Path(args.run).resolve()
    identity=json.loads((out/'input.json').read_text());report=json.loads((out/'diagnosis.json').read_text());summary=json.loads((out/'summary.json').read_text());audit=json.loads((out/'cuts.json').read_text())
    raw=tifffile.imread(identity['raw']);tree=np.loadtxt(identity['swc'],ndmin=2);final=np.loadtxt(out/'swc/97117.swc',ndmin=2)
    assert all(c.get('status') for c in report['cases'])
    for k in ['swc','raw']:assert hashlib.sha256(Path(identity[k]).read_bytes()).hexdigest()==identity[k+'_sha256']
    reapplied,re_audit=apply_ownership_plan(tree.tolist(),report)
    np.testing.assert_allclose(final,np.asarray(reapplied),rtol=0,atol=5.1e-6);assert audit==re_audit
    nodes={int(n[0]):n for n in tree};kept=set(map(int,final[:,0]));removed=set(audit['removed_ids']);assert kept.isdisjoint(removed);assert kept|removed==set(nodes)
    roots=[int(n[0]) for n in final if int(n[6])<0];assert len(roots)==1
    assert all(int(n[6]) in kept for n in final if int(n[6])>=0)
    assert {int(n[0]) for n in tree if int(n[1])==1}.issubset(kept)
    target_runs=[c for c in report['cases'] if c.get('judgment',{}).get('ownership')=='target' and c['judgment'].get('soma_id')==0]
    assert all(not removed.intersection(c['supported_run_ids']) for c in target_runs)
    bycase={c['case_id']:c for c in report['cases']};attributions=[]
    for event in audit['events']:
        c=bycase[event['case_id']];cut=event['cut_node_id'];assert ownership_gate(c,c['judgment'])['accepted'];assert verify_gate(c['verification'],cut)
        assert np.linalg.norm(nodes[cut][2:5]-report['somas'][0]['center_xyz'])>14
        arc=sum(np.linalg.norm(nodes[n][2:5]-nodes[int(nodes[n][6])][2:5]) for n in event['removed_ids'])
        attributions.append(dict(case_id=c['case_id'],cut_node_id=cut,parent_id=int(nodes[cut][6]),cut_xyz=nodes[cut][2:5].tolist(),competing_soma_id=c['competing_soma_id'],competing_soma_xyz=report['somas'][c['competing_soma_id']]['center_xyz'],removed_nodes=len(event['removed_ids']),removed_length_um=float(arc),review=f'cases/{c["case_id"]}/diagnosis.json'))
    save_json(out/'attributions.json',attributions)
    final_effects=[]
    for c in report['cases']:
        owners=[e['case_id'] for e in audit['events'] if c['representative_id'] in e['removed_ids']]
        final_effects.append(dict(case_id=c['case_id'],local_review_status=c['status'],representative_removed=bool(owners),
            covering_actual_cut_cases=owners,supported_run_removed_nodes=sum(n in removed for n in c['supported_run_ids'])))
    save_json(out/'case_final_effects.json',final_effects)
    validation=dict(input_hashes_unchanged=True,complete_candidate_reviews=len(report['cases']),reapplied_plan_matches_saved_swc=True,all_actual_cuts_pass_ownership_and_independent_recheck=True,root_count=len(roots),parent_integrity=True,type1_somas_preserved=True,minimum_soma_protection_um=14,model_target_runs_preserved=len(target_runs),remaining_deletion_vs_target_judgment_conflicts=0,accuracy=None)
    save_json(out/'validation.json',validation)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    color_by_owner={s['soma_id']:plt.cm.tab10(s['soma_id']%10) for s in report['somas']}
    fig,axes=plt.subplots(2,3,figsize=(16,11),layout='constrained')
    all_xyz=tree[:,2:5];lo=np.maximum(0,np.floor(all_xyz.min(0)-18)).astype(int);hi=np.minimum(raw.shape[::-1],np.ceil(all_xyz.max(0)+19)).astype(int)
    for s in report['somas']:
        if s['soma_id'] in {a['competing_soma_id'] for a in attributions}|{0}:
            lo=np.minimum(lo,np.maximum(0,np.asarray(s['center_xyz'])-12));hi=np.maximum(hi,np.minimum(raw.shape[::-1],np.asarray(s['center_xyz'])+13))
    crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
    for axis,(plane,name) in enumerate([([0,1],'XY'),([0,2],'XZ'),([1,2],'YZ')]):
        bg=crop.max(axis);extent=[lo[plane[0]]-.5,hi[plane[0]]-.5,hi[plane[1]]-.5,lo[plane[1]]-.5]
        for row in range(2):
            ax=axes[row,axis];ax.imshow(bg,cmap='gray',vmin=0,vmax=200,extent=extent,interpolation='nearest')
            keep_edges=np.array([[n[2:5],nodes[int(n[6])][2:5]] for n in final if int(n[6]) in nodes])
            ax.add_collection(LineCollection(keep_edges[:,:,plane],colors='#64dcff',linewidths=.6,alpha=.85))
            if row==0:
                for event in audit['events']:
                    edges=np.array([[nodes[n][2:5],nodes[int(nodes[n][6])][2:5]] for n in event['removed_ids']])
                    ax.add_collection(LineCollection(edges[:,:,plane],colors=[color_by_owner[event['competing_soma_id']]],linewidths=1))
            for s in report['somas']:
                p=np.asarray(s['center_xyz'])
                if np.all((p>=lo)&(p<hi)):
                    color='lime' if s['is_target'] else color_by_owner[s['soma_id']]
                    ax.scatter(p[plane[0]],p[plane[1]],s=60,facecolors='none',edgecolors=[color]);ax.annotate('T' if s['is_target'] else 'S'+str(s['soma_id']),(p[plane[0]],p[plane[1]]),xytext=(3,3),textcoords='offset points',color=color,fontsize=9)
            ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal');ax.set_title(name+(' | cyan kept; other colors removed by proposed soma' if row==0 else ' | final retained trace'));ax.set_xlabel('XYZ'[plane[0]]+' (um)');ax.set_ylabel('XYZ'[plane[1]]+' (um)')
    fig.suptitle('97117: whole-upstream / competing-soma experiment\nSoma labels and removed-branch ownership are model hypotheses, not manually verified ground truth.')
    fig.savefig(out/'removed_overlay.png',dpi=140);plt.close(fig)
    # Interactive raw corridor viewer for actual cuts, including both somas.
    review=out/'review';review.mkdir(exist_ok=True);sites=[]
    for a in attributions:
        c=bycase[a['case_id']];lo,hi=np.asarray(c['bbox_xyz']);center=a['cut_xyz'];soma=report['somas'][a['competing_soma_id']]['center_xyz']
        site=dict(site_id=c['case_id'],kind='完整上游与胞体路径（假设）',status='已剪断节点 '+str(a['cut_node_id']),center_xyz=center,bbox_xyz=np.stack([lo,hi],axis=1).tolist(),image='../cases/'+c['case_id']+'/xy.png',context_xyz=c['upstream_xyz'],arms=[dict(xyz=c['other_route_xyz'],diagnosis=dict(status='suspected_switch')),dict(xyz=c['target_route_xyz'],diagnosis=dict(status='supported'))],pair_xyz=[report['somas'][0]['center_xyz'],soma])
        write_site_viewer(raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]],lo,site,review/f'site_{c["case_id"]}.html')
        p=review/f'site_{c["case_id"]}.html';doc=p.read_text().replace('定位交错分支 ${i+1}',"${i===0?'定位目标胞体':'定位候选胞体'}").replace('叠加 SWC','叠加路径假设').replace('<p class="note">这是原始体素','<p>青色：目标胞体出发 SWC；粉色：候选胞体路径；绿色：目标胞体图像路径。颜色仅标识路线，不代表确认归属。可定位两个胞体并逐层追踪。</p><p class="note">这是原始体素')
        p.write_text(doc);sites.append(site)
    save_json(review/'diagnosis.json',dict(sites=sites));write_viewer_index(dict(sites=sites),review)
    rows=''.join(f'<tr><td><a href="#case-{a["case_id"]}">{a["case_id"]}</a></td><td>{a["cut_node_id"]}</td><td>S{a["competing_soma_id"]} {a["competing_soma_xyz"]}</td><td>{a["removed_nodes"]}</td><td>{a["removed_length_um"]:.1f}</td></tr>' for a in attributions)
    block=f'<div id="effect"><h2>实际执行的剪断</h2><p><a href="review/interactive.html">交互检查：完整路径、胞体定位、逐层切片</a> · <a href="validation.json">结构与重放校验</a> · <a href="attributions.json">归属记录</a></p><table><tr><th>候选</th><th>剪断子节点</th><th>疑似其他胞体 XYZ</th><th>删除节点</th><th>删除长度 µm</th></tr>{rows}</table><img src="removed_overlay.png"></div>'
    p=out/'index.html';doc=p.read_text();assert 'id="effect"' not in doc;doc=doc.replace('<section ',block+'<section ',1)
    for effect in final_effects:
        marker='<section id="case-'+effect['case_id']+'">'
        note=('此候选的代表节点已随上游剪断删除，实际剪断记录：'+', '.join(effect['covering_actual_cut_cases'])) if effect['representative_removed'] else '此候选的代表节点最终保留。'
        doc=doc.replace(marker,marker+'<p>'+html.escape(note)+'（局部未提出剪断，并不排除更上游通过了本轮复核。）</p>')
    p.write_text(doc)
    print(json.dumps(dict(validation=validation,attributions=attributions),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
