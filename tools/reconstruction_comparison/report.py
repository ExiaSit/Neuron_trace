"""Historical numeric definitions, paired summaries, MIPs and a local gallery."""
from __future__ import annotations
import concurrent.futures
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import time
import traceback

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ['MPLBACKEND']='Agg'
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import tifffile

from . import legacy_prec_rec as evaluator
from .metrics import read_swc,swc_total_length,estimate_prec_rec_single

COLORS=['#b76055','#218eaa','#9873b2','#b38c37','#64778c','#25865d']
METRICS=('length','branches','nodes','precision','recall','mean_snr','mean_branch_length')


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.writing.json');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n');temporary.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def swc_path(cfg,version,cid):
    base=Path(cfg['output_root'])/'reconstruction'/version['product'] if 'product' in version else Path(version['swc_root'])
    return base/version['template'].format(id=cid)


def structural(swc,shape):
    xyz=np.array([v[:3] for v in swc.nodes.values()]);children=dict.fromkeys(swc.nodes,0)
    for parent in swc.parents.values():
        if parent in children:children[parent]+=1
    lengths=branch_lengths(swc)
    return dict(roots=sum(p<0 for p in swc.parents.values()),tips=sum(c==0 for c in children.values()),
       branchpoints=sum(c>1 for c in children.values()),branches=sum(c==0 or c>1 for c in children.values()),
       branch_segments=len(lengths),mean_branch_length=float(np.mean(lengths)) if lengths else 0.,
       out_of_image_nodes=int(np.sum(np.any((xyz<0)|(xyz>=np.array(shape)[::-1]),axis=1))))


def branch_lengths(swc):
    """3D arc lengths of maximal root/fork-to-fork/tip paths, in micrometres.

    Each edge belongs to exactly one segment. Single-node trees have no
    segments; zero-length paths with distinct nodes are retained.
    """
    validate_topology(swc)
    children={node:[] for node in swc.nodes}
    for node,parent in swc.parents.items():
        if parent!=-1:children[parent].append(node)
    lengths=[]
    for start,descendants in children.items():
        if swc.parents[start]!=-1 and len(descendants)==1:continue
        for node in descendants:
            previous=start;length=0.
            while True:
                length+=math.dist(swc.nodes[previous][:3],swc.nodes[node][:3])
                if len(children[node])!=1:break
                previous,node=node,children[node][0]
            lengths.append(length)
    return lengths


def validate_topology(swc):
    """Reject broken parents/cycles before the legacy traversal can loop."""
    done=set()
    for node in swc.parents:
        seen=set()
        while node!=-1 and node not in done:
            if node not in swc.parents:raise ValueError(f'Missing parent node {node}')
            if node in seen:raise ValueError(f'Parent cycle at node {node}')
            seen.add(node);node=swc.parents[node]
        done.update(seen)


def summarize(rows,versions):
    valid={v['key']:{r['id'] for r in rows if r['version_key']==v['key'] and not r['error']} for v in versions}
    common=set.intersection(*valid.values()) if valid else set()
    summary=[]
    for scope in ('available','paired_all_versions'):
        for v in versions:
            group=[r for r in rows if r['version_key']==v['key'] and not r['error'] and (scope=='available' or r['id'] in common)]
            result=dict(scope=scope,version_key=v['key'],version=v['label'],count=len(group))
            for key in METRICS:
                name=key if key in ('mean_snr','mean_branch_length') else 'mean_'+key
                result[name]=float(np.mean([float(r[key]) for r in group])) if group else None
            for key in ('length','branches','nodes'):result['total_'+key]=sum(float(r[key]) for r in group)
            summary.append(result)
    return summary,sorted(common,key=int)


def panel(cid,raw,versions,paths,rows,destination):
    mip=raw.max(axis=0);background=mip.astype(np.float32)
    lo,hi=np.percentile(background,[1,99.8])
    if hi<=lo:hi=float(background.max()) or lo+1
    fig,axes=plt.subplots(1,len(versions)+1,figsize=(3.2*(len(versions)+1),3.8),dpi=150,squeeze=False)
    axes=axes[0]
    for ax in axes:
        ax.imshow(background,cmap='gray',vmin=lo,vmax=hi);ax.axis('off')
    axes[0].set_title(f'{cid}\nOriginal XY MIP',fontsize=10)
    by_key={r['version_key']:r for r in rows}
    for ax,v,path,color in zip(axes[1:],versions,paths,COLORS):
        row=by_key[v['key']]
        if path.exists():
            swc=read_swc(path);segments=[]
            for node,parent in swc.parents.items():
                if parent in swc.nodes:segments.append([swc.nodes[parent][:2],swc.nodes[node][:2]])
            ax.add_collection(LineCollection(segments,colors=color,linewidths=.65,alpha=.95))
        else:ax.text(.5,.5,'Missing SWC',transform=ax.transAxes,color='red',ha='center')
        length=row.get('length');label=f'{length:.1f} um' if isinstance(length,(int,float)) else 'unavailable'
        ax.set_title(v['label'].replace(' historical','\nhistorical')+'\n'+label,fontsize=9)
    fig.suptitle(f'{cid}: same image and display range; SWCs in image XYZ coordinates',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.88));destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,bbox_inches='tight');plt.close(fig)


def evaluate_one(cfg,cid):
    out=Path(cfg['output_root']);manifest=json.loads((out/'manifests'/f'{cid}.json').read_text())
    for key in ('raw','mask'):
        path=Path(manifest['inputs'][key]);stat=path.stat();old=manifest['input_stat'][key]
        if (stat.st_size!=old['size'] or stat.st_mtime_ns!=old['mtime_ns']) and sha(path)!=manifest['input_sha256'][key]:
            raise ValueError(f'{cid}: {key} changed since reconstruction preflight')
    raw=tifffile.imread(manifest['inputs']['raw']);raw_float=raw.astype(np.float32)
    rows=[];paths=[]
    for version in cfg['versions']:
        path=swc_path(cfg,version,cid);paths.append(path)
        row=dict(id=cid,version_key=version['key'],version=version['label'],swc_path=str(path),error='',
                 raw_sha256=manifest['input_sha256']['raw'],mask_sha256=manifest['input_sha256']['mask'])
        try:
            row['swc_sha256']=sha(path);swc=read_swc(path)
            length,nodes,edges,missing=swc_total_length(swc)
            row.update(length=length,nodes=nodes,edges=edges,missing_parents=missing,**structural(swc,raw.shape))
            validate_topology(swc)
            row.update(estimate_prec_rec_single(evaluator,raw_float,path))
            if not all(math.isfinite(float(row[k])) for k in METRICS):raise ValueError('Nonfinite metric')
        except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
        rows.append(row)
    panel(cid,raw,cfg['versions'],paths,rows,out/'comparisons'/f'{cid}.png')
    save(out/'metrics'/f'{cid}.json',rows)
    return rows


def charts(summary,out):
    paired=[r for r in summary if r['scope']=='paired_all_versions'];labels=[r['version'].replace(' historical','\nhistorical').replace(' current','\ncurrent') for r in paired]
    fig,axes=plt.subplots(2,4,figsize=(21,10),dpi=160)
    items=[('mean_length','Mean length (um)'),('total_length','Total length (um)'),('mean_branches','Mean critical-point branch count'),('mean_nodes','Mean SWC nodes'),
           ('mean_branch_length','Mean branch length (um; cell-weighted)'),
           ('mean_precision','Legacy heuristic Precision'),('mean_recall','Legacy endpoint Recall'),('mean_snr','Mean SNR')]
    for ax,(key,title) in zip(axes.flat,items):
        values=[float(r[key]) if r[key] is not None else 0 for r in paired]
        bars=ax.bar(labels,values,color=COLORS[:len(paired)]);ax.set_title(title,fontsize=11)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3,fontsize=7)
        ax.tick_params(axis='x',labelsize=7,rotation=20)
        ax.set_ylim(0,1.1 if key in ('mean_precision','mean_recall') else max(max(values)*1.22,1.))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    count=paired[0]['count'] if paired else 0
    fig.suptitle(f'Original 200 cohort: {count} shared valid cases | current v0.82 GSDT vs historical versions')
    fig.tight_layout();fig.savefig(out/'summary_metrics.png');plt.close(fig)


def gallery(cfg,rows,summary,common,complete):
    out=Path(cfg['output_root']);by_id={}
    for r in rows:by_id.setdefault(r['id'],[]).append(r)
    paired=[r for r in summary if r['scope']=='paired_all_versions']
    headers=['版本','共同有效例数','平均长度 (um)','总长度 (um)','平均分枝数','平均分支长度 (um)','平均节点数','历史 Precision','历史 Recall','SNR']
    columns=['version','count','mean_length','total_length','mean_branches','mean_branch_length','mean_nodes','mean_precision','mean_recall','mean_snr']
    table='<tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr>'
    for r in paired:
        table+='<tr>'+''.join('<td>'+html.escape(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in columns)+'</tr>'
    cards=[]
    for cid,group in sorted(by_id.items(),key=lambda p:int(p[0])):
        status='共同有效样本' if cid in common else '有缺失或指标不可用，见逐例记录'
        links=[]
        for r in group:
            if r.get('swc_path') and Path(r['swc_path']).is_file():
                # A local symlink keeps the exported HTML self-contained in its links.
                link=out/'swc_links'/r['version_key']/(cid+'.swc');link.parent.mkdir(parents=True,exist_ok=True)
                if not link.exists():link.symlink_to(Path(r['swc_path']).resolve())
                links.append(f'<a href="swc_links/{r["version_key"]}/{cid}.swc">{html.escape(r["version"])} SWC</a>')
        cards.append(f'<section id="cell-{cid}"><h2>{cid}</h2><p>{status}</p><img loading="lazy" src="comparisons/{cid}.png"><p>'+ ' | '.join(links)+'</p></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>v0.82 GSDT 固定样本比较</title>
<style>body{font:15px system-ui;margin:24px;background:#eee;color:#222}section,header{background:white;padding:18px;margin-bottom:20px}img{max-width:100%}table{border-collapse:collapse}th,td{padding:8px;border:1px solid #bbb;text-align:right}th:first-child,td:first-child{text-align:left}a{color:#155b96}input{padding:7px}section:target{outline:3px solid #218eaa}</style>
<header><h1>v0.82 GSDT 与历史版本：固定原始 200 例</h1>'''
    page+=f'<p>当前已分析 {len(by_id)}/200 例；共同有效 {len(common)} 例；状态：'+('已完成' if complete else '计算中')+'</p>'
    page+='<p>历史 v0.82 指参考会话中修正 Y 坐标后的旧版；当前版是本次 Git 分支 v0.82 的 GSDT 流程，使用原始 mask，不做 mask 补全。APP2 为本次重新生成的剪枝前输出。</p>'
    page+='<p>PR 沿用参考会话的固定启发式实现，不是对人工真值的准确率。节点、分枝、长度描述结构规模，不能单独作为优劣判断。</p>'
    page+='<p>平均分支长度：沿三维 SWC 将根节点/分叉点至下一分叉点/末梢点作为一段，累加路径长度；先计算每例平均值，再对共同样本等权平均。分母为实际分支段数（逐例 CSV 的 branch_segments），与旧分枝计数不同。无边树取 0。</p>'
    page+='<p><a href="summary_metrics.csv">汇总 CSV</a> | <a href="per_cell_metrics.csv">逐例 CSV</a> | <a href="paired_deltas.csv">配对差值与区间</a> | <a href="missing_or_failed.csv">缺失指标</a> | <a href="reconstruction_failures.csv">重建失败原因</a> | <a href="run_manifest.json">运行来源</a></p>'
    page+='<p><input id="query" placeholder="输入编号筛选"><button onclick="location.hash=\'cell-\'+document.getElementById(\'query\').value">跳转</button></p>'
    page+='<table>'+table+'</table><img src="summary_metrics.png"></header>'+''.join(cards)+'</html>'
    (out/'index.html').write_text(page)


def build(cfg):
    out=Path(cfg['output_root']);ids=json.loads((out/'preflight.json').read_text())['ids']
    provenance=json.loads((Path(__file__).resolve().parents[2]/'comparison/metric_provenance.json').read_text())
    actual=sha(Path(__file__).with_name('legacy_prec_rec.py'))
    if actual!=provenance['legacy_prec_rec_sha256']:raise ValueError('Historical PR source changed; give this evaluator a new identity')
    rows=[];started=time.monotonic();errors=[]
    workers=int(cfg.get('analysis_workers',8))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        waiting=set(ids);active={};processed=0
        while waiting or active:
            for cid in sorted(waiting,key=int):
                if len(active)>=2*workers:break
                status=out/'status'/f'{cid}.json'
                if not status.exists():continue
                state=json.loads(status.read_text())
                if state['state'] not in ('complete','failed'):continue
                active[pool.submit(evaluate_one,cfg,cid)]=cid;waiting.remove(cid)
            done=[f for f in active if f.done()]
            if not done:
                if time.monotonic()-started>cfg.get('report_timeout_seconds',21600):
                    raise TimeoutError('Reconstruction did not finish for pending IDs')
                time.sleep(.5);continue
            for future in done:
                cid=active.pop(future)
                try:rows.extend(future.result())
                except Exception:errors.append(dict(id=cid,error=traceback.format_exc()))
                processed+=1
                save(out/'analysis_progress.json',dict(planned=len(ids),finished=len({r['id'] for r in rows}),errors=errors,seconds=round(time.monotonic()-started,3)))
                print(f'[{len({r["id"] for r in rows})}/{len(ids)}] evaluated {cid}',flush=True)
                if processed%20==0:
                    partial,common=summarize(rows,cfg['versions'])
                    write_csv(out/'per_cell_metrics.csv',rows);write_csv(out/'summary_metrics.csv',partial)
                    write_csv(out/'missing_or_failed.csv',[r for r in rows if r['error']],fields=['id','version_key','version','swc_path','error'])
                    charts(partial,out);gallery(cfg,rows,partial,common,complete=False)
    return publish(cfg,rows,ids,errors,actual)


def publish(cfg,rows,ids,errors,actual):
    """Write shared numerical and visual reports after evaluation or refresh."""
    out=Path(cfg['output_root'])
    rows.sort(key=lambda r:(int(r['id']),[v['key'] for v in cfg['versions']].index(r['version_key'])))
    summary,common=summarize(rows,cfg['versions'])
    write_csv(out/'per_cell_metrics.csv',rows);write_csv(out/'summary_metrics.csv',summary)
    deltas=paired_deltas(rows,cfg['versions'],common)
    write_csv(out/'paired_deltas.csv',deltas)
    failed=[r for r in rows if r['error']]
    write_csv(out/'missing_or_failed.csv',failed,fields=['id','version_key','version','swc_path','error'])
    charts(summary,out);gallery(cfg,rows,summary,common,complete=not errors and len({r['id'] for r in rows})==len(ids))
    report=dict(requested=len(ids),analyzed=len({r['id'] for r in rows}),version_count=len(cfg['versions']),
                paired_count=len(common),paired_ids=common,missing_or_failed=len(failed),analysis_errors=errors,
                metric_sha256=actual,summary=summary)
    save(out/'comparison_summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('paired_ids','summary')},ensure_ascii=False),flush=True)
    if errors:raise RuntimeError('Some case analyses failed; see analysis_progress.json')
    return report


def paired_deltas(rows,versions,common,seed=20261022):
    """Cell-level paired changes; do not treat SWC nodes as independent samples."""
    target=versions[-1]['key'];indexed={(r['id'],r['version_key']):r for r in rows if not r['error']}
    result=[];rng=np.random.default_rng(seed)
    for version_index,version in enumerate(versions[:-1]):
        for metric in METRICS:
            values=np.array([float(indexed[cid,target][metric])-float(indexed[cid,version['key']][metric]) for cid in common])
            if not len(values):continue
            # Extending the metric set must not change existing bootstrap draws.
            metric_rng=np.random.default_rng(np.random.SeedSequence([seed,version_index,1])) if metric=='mean_branch_length' else rng
            means=values[metric_rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
            lo,hi=np.quantile(means,[.025,.975])
            result.append(dict(scope='paired_all_versions',reference=version['label'],target=versions[-1]['label'],metric=metric,
                count=len(values),mean_delta=float(values.mean()),median_delta=float(np.median(values)),
                bootstrap_95_low=float(lo),bootstrap_95_high=float(hi),
                increased=int(np.sum(values>1e-9)),decreased=int(np.sum(values < -1e-9)),unchanged=int(np.sum(np.abs(values)<=1e-9)),
                bootstrap_seed=seed,bootstrap_replicates=2000))
    return result


def render_one(cfg,cid):
    out=Path(cfg['output_root']);manifest=json.loads((out/'manifests'/f'{cid}.json').read_text())
    rows=json.loads((out/'metrics'/f'{cid}.json').read_text())
    raw=tifffile.imread(manifest['inputs']['raw'])
    paths=[swc_path(cfg,v,cid) for v in cfg['versions']]
    panel(cid,raw,cfg['versions'],paths,rows,out/'comparisons'/f'{cid}.png')
    return cid


def render_existing(cfg):
    out=Path(cfg['output_root']);report=json.loads((out/'comparison_summary.json').read_text())
    ids=json.loads((out/'preflight.json').read_text())['ids']
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(cfg.get('analysis_workers',8))) as pool:
        list(pool.map(render_one,[cfg]*len(ids),ids))
    rows=[r for cid in ids for r in json.loads((out/'metrics'/f'{cid}.json').read_text())]
    charts(report['summary'],out);gallery(cfg,rows,report['summary'],report['paired_ids'],complete=True)


def refresh_structure(cfg):
    """Extend cached metrics from unchanged SWCs, without tracing or PR work."""
    out=Path(cfg['output_root'])
    comparison=json.loads((out/'comparison_summary.json').read_text())
    if comparison['analysis_errors']:raise ValueError('Resolve earlier analysis errors before refreshing')
    if sha(Path(__file__).with_name('legacy_prec_rec.py'))!=comparison['metric_sha256']:
        raise ValueError('Historical PR source changed; cached metrics cannot be reused')
    ids=json.loads((out/'preflight.json').read_text())['ids'];rows=[]
    updated={}
    for cid in ids:
        manifest=json.loads((out/'manifests'/f'{cid}.json').read_text())
        group=json.loads((out/'metrics'/f'{cid}.json').read_text())
        if {r['version_key'] for r in group}!={v['key'] for v in cfg['versions']}:
            raise ValueError(f'{cid}: cached versions do not match configuration')
        for row in group:
            if row['error']:continue
            path=Path(row['swc_path'])
            if sha(path)!=row['swc_sha256']:raise ValueError(f'{cid}: SWC changed: {path}')
            row.update(structural(read_swc(path),manifest['shape']))
        updated[cid]=group;rows.extend(group)
    for cid,group in updated.items():save(out/'metrics'/f'{cid}.json',group)
    return publish(cfg,rows,ids,[],comparison['metric_sha256'])


def write_text_report(cfg,comparison,validation):
    """Human-readable outcome with paired denominators and explicit failures."""
    out=Path(cfg['output_root']);count=comparison['paired_count']
    paired=[r for r in comparison['summary'] if r['scope']=='paired_all_versions']
    lines=['# v0.82 GSDT 与历史版本：固定原始样本比较','',
       f"本次对原始 {comparison['requested']} 个编号全部尝试重建，成功 {validation['reconstruction_succeeded']} 例、失败 {validation['reconstruction_failed']} 例。{comparison['version_count']} 组结果共同有效样本为 **{count} 例**。",
       '', '重建代码使用 v0.82 工作目录；原始 mask 直接输入 APP2，随后执行 G-Cut、重采样和剪枝。没有运行 mask 补全。',
       '历史 v0.82 是参考会话中修正 Y 坐标后的旧结果；本次 v0.82 为 GSDT 版。两者名称相同但流程不同。',
       '本次是完整版本对比，不能把全部差异单独归因于 GSDT。APP2 列为本次重跑的剪枝前结构。','',
       f'## 共同有效 {count} 例','',
       '| 版本 | 平均长度 µm | 总长度 µm | 平均分枝数 | 平均分支长度 µm | 平均节点数 | 历史 Precision | 历史 Recall | SNR |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f"| {r['version']} | {r['mean_length']:.3f} | {r['total_length']:.3f} | {r['mean_branches']:.3f} | {r['mean_branch_length']:.3f} | {r['mean_nodes']:.3f} | {r['mean_precision']:.6f} | {r['mean_recall']:.6f} | {r['mean_snr']:.6f} |")
    lines+=['','平均分支长度以根节点/分叉点至下一分叉点/末梢点的三维路径为一段；先计算每例的段长均值，再对细胞等权平均。',
       '逐例字段 `branch_segments` 为实际分支段数，与历史“末梢＋分叉点”的 `branches` 不同；无边树的段数与均长均为 0。']
    lines+=['','## 与历史 v0.82 的配对变化','',
       '以同一细胞为单位计算“当前版 − 历史版”。bootstrap 为本批样本的描述性区间。','',
       '| 指标 | 平均变化 | 95% bootstrap 区间 | 增加 / 减少 / 不变例数 |','|---|---:|---|---|']
    with (out/'paired_deltas.csv').open() as f:delta=list(csv.DictReader(f))
    for r in delta:
        if r['reference'].startswith('v0.82'):
            lines.append(f"| {r['metric']} | {float(r['mean_delta']):.6f} | [{float(r['bootstrap_95_low']):.6f}, {float(r['bootstrap_95_high']):.6f}] | {r['increased']} / {r['decreased']} / {r['unchanged']} |")
    lines+=['','PR 沿用固定的历史图像启发式实现，不是人工真值准确率。分枝数、节点数和长度描述结构规模；分枝增多不能单独解释为重建更准确。',
       '旧 Recall 的端点检查位置仍会随 SWC 改变，因此不能用它单独证明真实完整性。','',
       '## 失败与缺失','']
    for r in validation['failures']:
        lines.append(f"- {r['id']}：{r['stage']}，{r['reason']}")
    with (out/'missing_or_failed.csv').open() as f:
        missing=list(csv.DictReader(f))
    failed_ids={r['id'] for r in validation['failures']}
    current_key=cfg['versions'][-1]['key']
    for r in missing:
        if r['version_key']!=current_key or r['id'] not in failed_ids:
            lines.append(f"- {r['id']}（{r['version']}）：{r['error']}")
    lines+=['','失败案例未用旧 SWC 替代。所有失败、缺失和共同样本口径均保留在 CSV 中。','',
       '## 核对','',
       f"- 重新计算的 {validation['legacy_records_reproduced']} 条历史有效记录，与原报告的长度、节点、分枝、PR 和 SNR 逐项一致。",
       f"- 核对 {validation['input_files_verified']} 项源文件；原始输入未改变，运行使用的流水线源码未改变。",
       f"- {validation['valid_comparison_pngs']} 张逐例图、1 张统计图及 {validation['valid_html_links']} 个本地网页链接通过检查。",
       '- PR 源码 SHA-256：`'+comparison['metric_sha256']+'`。','',
       '## 文件','',f'- 结果目录：`{out}`',
       '- `index.html`：逐例图和 SWC 下载。',
       '- `per_cell_metrics.csv`、`summary_metrics.csv`：逐例与汇总。',
       '- `paired_deltas.csv`：配对差值和区间。',
       '- `reconstruction_failures.csv`、`missing_or_failed.csv`：失败与缺失。',
       '- `run_manifest.json`、`validation.json`：来源、哈希和独立核对。',
       '- 可复用方法、固定样本清单和运行参数已保存在 Git 的 `comparison/` 与 `tools/reconstruction_comparison/`。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
