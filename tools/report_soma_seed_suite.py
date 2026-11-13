#!/usr/bin/env python3
"""Summarize a completed multi-cell soma seed and paired downstream experiment."""
import argparse
from pathlib import Path
import html
import json
import os
import zipfile
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit


def read(path):return json.loads(Path(path).read_text())
def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2))


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for name,value in attrs:
            if name in ['href','src'] and value:self.links.append(value)


def check_links(pages):
    missing=[];checked=0
    for page in pages:
        parser=Links();parser.feed(page.read_text())
        for link in parser.links:
            parsed=urlsplit(link)
            if parsed.scheme or not parsed.path:continue
            path=(page.parent/unquote(parsed.path)).resolve();checked+=1
            if not path.exists():missing.append(dict(page=str(page),link=link))
    return dict(checked=checked,missing=missing,passed=not missing)


def build(out,cells):
    out=Path(out).resolve();runs=out.parent
    records=[];pages=[];commands=[]
    for cell in cells:
        seed=runs/f'20261113_{cell}_gemma_soma_seed_suite'
        down=runs/f'20261113_{cell}_gemma_soma_downstream_suite'
        soma=read(down/'soma_comparison/summary.json');model=read(seed/'normalized/summary.json')
        audit=read(seed/'engineering_audit.json');comparison=read(down/'comparison_metrics.json')
        if not audit['passed'] or not read(down/'validation.json')['arms_completed']:
            raise ValueError('Incomplete or unaudited cell: '+cell)
        baseline=comparison['stages']['legacy_seeds']['final'];new=comparison['stages']['gemma_seeds']['final']
        record=dict(cell=cell,candidates=model['union_candidates'],model_decisions=model['decisions'],
                    soma={k:soma[k] for k in ['old_count','gemma_count','spatial_pairs','old_only','gemma_only','old_actual_anchors','gemma_actual_anchors','review_not_used']},
                    legacy_final=baseline,gemma_final=new,model_calls=model['model_calls'],model_seconds=model['model_seconds'],
                    model_input_audit=audit,seed_run=str(seed),downstream_run=str(down),biological_accuracy_evaluated=False)
        records.append(record)
        # All display deliverables in a portable per-cell package. Reports remain
        # in the experiment folders; package README intentionally references local files.
        package=out/f'{cell}_vaa3d_markers_and_swc.zip'
        with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for p in sorted((down/'vaa3d').iterdir()):
                if p.is_file():z.write(p,cell+'/vaa3d/'+p.name)
            for p in sorted((down/'soma_comparison').glob('*.marker')):
                z.write(p,cell+'/markers/'+p.name)
            for path in sorted((down/'removed_connection_components').rglob('*.swc')):
                z.write(path,cell+'/removed_connection_components/'+str(path.relative_to(down/'removed_connection_components')))
            z.write(down/'removed_connection_components/audit.json',cell+'/removed_connection_components/audit.json')
            for filename in ['spatial_correspondence.json','summary.json']:
                z.write(down/'soma_comparison'/filename,cell+'/markers/'+filename)
            z.writestr(cell+'/README.txt',f'打开 vaa3d/{cell}.v3draw，然后导入 markers/overlay.marker 或 old_all.marker、gemma_all.marker。普通 marker 仅配套 v3draw；*_for_tiff.marker 仅用于原 TIFF。半径8µm仅显示。橙色旧点，绿色Gemma，红色已知目标。only 表示6µm一对一坐标比较未配对，不是生物学真值。matched_soma 是成功映射的胞体位置，app2_anchors 是实际G-Cut种子节点；unmatched_soma 不参与分配，shared_app2_soma 共享一个节点。两组最终SWC和Gemma全部分配组件均在vaa3d中。\n')
        pages += [down/'index.html', down/'soma_comparison/index.html', down/'removed_connection_components/index.html', seed/'normalized/index.html']
        pages += list((down/'local_review').glob('*.html'))
        for arm in ['legacy_seeds','gemma_seeds']:
            pages += list((down/arm/'pruning_mips'/f'{cell}_connections').glob('*.html'))
        commands.append(f'''# {cell}: use a NEW run name for a fresh experiment; exact inputs/runtime in input.json/runtime.json.
python tools/run_gemma_soma_seeds.py --cell {cell} --run-name <new_{cell}_seed_run>
python tools/finish_gemma_soma_seed_review.py <new_{cell}_seed_run_absolute_path>
python tools/run_gemma_seed_downstream.py --seed-run <new_{cell}_seed_run_absolute_path> --run-name <new_{cell}_downstream_run> --render-workers 4
python tools/report_gemma_seed_downstream.py <new_{cell}_downstream_run_absolute_path>
python tools/compare_soma_seed_markers.py <new_{cell}_downstream_run_absolute_path>
''')
    save(out/'comparison.json',dict(cells=records,biological_accuracy_evaluated=False))
    launch='/data/disk/Neuron_tracing_method/test/0.84/runtime/start_gemma.sh --ctx-size 16384 --parallel 1 --n-gpu-layers 54 --batch-size 1280 --ubatch-size 1280 --threads 16 --reasoning-budget 256 --image-min-tokens 1120 --no-context-shift --verbose'
    commands.insert(0,'# Start the existing model in a separate terminal (no reinstall):\nGEMMA_REQUIRED_FREE_MIB=21000 '+launch+'\n# Record a separate server log or byte range for each cell, then audit with tools/validate_gemma_soma_seed_run.py.\n# The batch server was stopped after all 238 requests to release GPU memory.\n')
    (out/'REPRODUCE.txt').write_text('Repository: /data/disk/Neuron_tracing_method/v0.84\nPython: /home/pzy/anaconda3/envs/hb_seg/bin/python\nEnvironment: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RECONNECT_COMPUTE_BACKEND=cpu MPLCONFIGDIR=/tmp/neuron_soma_mpl\nNo nnU-Net or APP2 rerun. Frozen inputs and hashes per downstream run.json. Exact batch commands: run_suite.py and suite.log. Server command: seed runtime.json; server_log_segment.json records actual log byte ranges.\n\n'+'\n'.join(commands))
    rows=[];sections=[]
    for r in records:
        cell=r['cell'];s=r['soma'];prefix='../'+Path(r['downstream_run']).name;seedprefix='../'+Path(r['seed_run']).name
        rows.append(f'<tr><td>{cell}</td><td>{r["candidates"]}</td><td>{s["old_count"]} → {s["gemma_count"]}</td><td>{s["spatial_pairs"]} / {s["old_only"]} / {s["gemma_only"]}</td><td>{s["old_actual_anchors"]} → {s["gemma_actual_anchors"]}</td><td>{r["legacy_final"]["nodes"]} → {r["gemma_final"]["nodes"]}</td><td>{r["legacy_final"]["length_um"]:.1f} → {r["gemma_final"]["length_um"]:.1f}</td><td>{s["review_not_used"]}</td></tr>')
        links=' · '.join(f'<a href="{prefix}/soma_comparison/{filename}.marker">{label}</a>' for filename,label in [('old_all','旧点 marker'),('gemma_all','Gemma 点 marker'),('overlay','新旧叠加 marker'),('old_only','仅旧 marker'),('gemma_only','仅新 marker')])
        sections.append(f'<h2>{cell}</h2><p><a href="{prefix}/soma_comparison/index.html">胞体差异与全部 marker</a> · <a href="{prefix}/index.html">G-Cut / 剪枝完整对照</a> · <a href="{prefix}/removed_connection_components/index.html">逐次剪断原图及移除组件</a> · <a href="{seedprefix}/normalized/index.html">逐候选原图与 Gemma 请求/回答</a> · <a href="{cell}_vaa3d_markers_and_swc.zip">下载 Vaa3D 包</a></p><p>{links}</p><p>Gemma 模型/程序审核统计：{html.escape(json.dumps(r["model_decisions"],ensure_ascii=False))}；调用 {r["model_calls"]} 次，模型用时 {r["model_seconds"]/60:.1f} 分钟。</p><img loading="lazy" src="{prefix}/soma_comparison/overview.png"><p><a href="{prefix}/vaa3d/gemma_seeds.swc">新最终 SWC</a> · <a href="{prefix}/vaa3d/legacy_seeds.swc">旧种子最终 SWC</a> · <a href="{prefix}/vaa3d/gemma_all_components.ano">所有新 G-Cut 组件</a></p>')
    impact='<p><b>106245 需优先复核：</b>新旧最终长度 4456.76 → 902.65 µm。变化既来自 G-Cut 分配，也来自后续既有局部剪枝；其中一次启发式剪断影响 2316 节点，并非 Gemma 已确认这些分支属于其他神经元。<a href="../20261113_106245_gemma_soma_downstream_suite/removed_connection_components/index.html">查看剪断原图及移除组件</a>。前三个样本的最终结构变化较小。所有结果均为实验结果，不能把减少量当改善证据。</p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>四个 neuron 胞体与后续流程对照</title><style>body{font:16px/1.7 system-ui;max-width:1500px;margin:30px auto;padding:15px}img{width:100%}td,th{padding:9px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>四个 neuron：Gemma 胞体筛选、G-Cut 与基础剪枝</h1><p>每个样本的新旧两组共享冻结的原图、mask、APP2，以及后续 G-Cut / 重采样 / 基础剪枝规则。旧组运行原胞体筛选；新组放宽 GSDT（严格 &gt;5 → &gt;4 µm），保留两档峰值并集，由 Gemma 判断是否胞体及是否重复，不使用 NMS 过滤新点。T 为元数据已知目标；待复核点未加入种子。没有重跑 nnU-Net、APP2。</p><p>橙色旧点、绿色新点、红色已知目标。marker 显示直径16µm，不参与计算。空间配对使用6µm一对一匹配，仅描述坐标差异，不能据此认定是同一个生物学胞体。实际APP2种子数另列。</p><p><b>没有人工真值，生物学准确率未评估。</b>胞体点数、删除量、SWC长度与格式测试都不是正确率。Gemma 接受的点仍可能误判；未匹配 APP2 的点不会自动生成新分支。</p><table><tr><th>Neuron</th><th>候选</th><th>胞体点旧→新</th><th>配对/仅旧/仅新</th><th>实际种子旧→新</th><th>最终节点旧→新</th><th>最终长度µm旧→新</th><th>待复核</th></tr>'+''.join(rows)+'</table>'+impact+'<p><a href="comparison.json">完整统计</a> · <a href="delivery_validation.json">交付链接校验</a> · <a href="artifact_validation.json">图像、marker 和 SWC 校验</a> · <a href="tests.log">针对性测试</a> · <a href="REPRODUCE.txt">复现说明</a></p>'+''.join(sections))
    pages.append(out/'index.html')
    save(out/'delivery_validation.json',dict(status='checking'))
    validation=check_links(pages)
    validation.update(cells=len(records),model_audits_passed=True,biological_accuracy_evaluated=False)
    save(out/'delivery_validation.json',validation)
    if not validation['passed']:raise ValueError('Broken deliverable links: '+json.dumps(validation['missing']))
    print(json.dumps(dict(cells=cells,link_validation=validation),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('--cells',nargs='+',default=['60003','89074','97117','106245']);a=p.parse_args();build(a.out,a.cells)
