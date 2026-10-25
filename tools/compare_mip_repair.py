#!/usr/bin/env python3
"""Paired, resumable comparison of live mask-repair implementations.

The cohort JSON contains cases with id, metadata, inputs and input_identity.
Results/diagnostics belong outside Git. No historical publication guard is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import tifffile
from scipy import ndimage as ndi

PLOT_LOCK = threading.Lock()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.writing.json')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def saved_records(output):
    # Atomic-write scratch files can survive an interrupted driver; they are
    # not extra cases and must never be counted or parsed as completed records.
    return [json.loads(p.read_text()) for p in (output/'cases').glob('*.json')
            if p.stem.isdecimal()]


def valid_reconstruction_pair(record):
    return record.get('status') == 'complete' and all(
        'final' in record.get(side+'_swc',{}) and side+'_final_sha256' in record
        for side in ('baseline','candidate'))


def code_hash(root):
    files = sorted([*root.glob('*.py'), *root.glob('algorithms/**/*.py'),
                    *root.glob('utils/**/*.py'), *root.glob('vendor/**/*.py'),
                    *root.glob('visualization/**/*.py')])
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def launch(command, log, timeout):
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    start = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if result:
        raise RuntimeError(f'Exit {result}: {log}')
    return time.monotonic()-start


def swc_metrics(path):
    a = np.loadtxt(path, comments='#', ndmin=2)
    if a.shape[1] < 7 or not np.isfinite(a[:, :7]).all():
        raise ValueError(f'Invalid SWC: {path}')
    by = {int(n[0]): n for n in a}
    if len(by) != len(a):
        raise ValueError(f'Duplicate SWC nodes: {path}')
    children = {i: [] for i in by}
    lengths = {}
    roots = []
    for i, n in by.items():
        parent = int(n[6])
        if parent == -1:
            roots.append(i)
        elif parent not in by:
            raise ValueError(f'Missing SWC parent: {path}')
        else:
            children[parent].append(i)
            lengths[i] = float(np.linalg.norm(n[2:5]-by[parent][2:5]))
    seen = set()
    queue = roots[:]
    while queue:
        i = queue.pop()
        if i in seen:
            raise ValueError(f'SWC cycle: {path}')
        seen.add(i)
        queue.extend(children[i])
    if len(seen) != len(by):
        raise ValueError(f'Disconnected SWC cycle: {path}')
    branches = []
    for i in by:
        if i not in roots and len(children[i]) == 1:
            continue
        for child in children[i]:
            length = lengths[child]
            while len(children[child]) == 1:
                child = children[child][0]
                length += lengths[child]
            branches.append(length)
    return dict(nodes=len(a), length_um=sum(lengths.values()), roots=len(roots),
                branches=len(branches), mean_branch_length_um=float(np.mean(branches)) if branches else 0.,
                tips=sum(not v for v in children.values()))


def comparison_image(raw, original, baseline, candidate, path, cid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    extra, lost = candidate & ~baseline, baseline & ~candidate
    for axis, plane in enumerate(('XY', 'XZ', 'YZ')):
        bg = raw.max(axis)
        vmax = max(25, float(np.percentile(bg, 99.5)))
        for col, (mask, name) in enumerate(zip((original, baseline, candidate, baseline),
                                              ('Input mask', 'v0.83', 'v0.83a', 'v0.83a - v0.83'))):
            ax = axes[axis, col]
            ax.imshow(bg, cmap='gray', vmin=0, vmax=vmax)
            rgba = np.zeros((*bg.shape, 4), np.float32)
            rgba[mask.any(axis)] = [0, 1, 1, .5]
            if col == 3:
                rgba[extra.any(axis)] = [1, .1, 0, .95]
                rgba[lost.any(axis)] = [1, 1, 0, .95]
            ax.imshow(rgba)
            ax.set_title(f'{plane} | {name}')
            ax.axis('off')
    fig.suptitle(f'{cid} | red: extra / yellow: absent relative to v0.83; 3-D changes projected')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def analyze_mask(rec, roots, output):
    cid = rec['id']
    raw = tifffile.imread(rec['inputs']['raw'])
    original = tifffile.imread(rec['inputs']['mask']) > 0
    masks = [tifffile.imread(root/'mask'/f'{cid}.tif') > 0 for root in roots]
    if any(mask.shape != raw.shape for mask in masks):
        raise ValueError('Shape mismatch')
    for mask in masks:
        if not np.all(mask[original]):
            raise ValueError('Original mask foreground lost')
    contrast = ndi.gaussian_filter(raw.astype(np.float32), (0, .6, .6))-ndi.gaussian_filter(raw.astype(np.float32), (0, 4, 4))
    from algorithms.tracing.regression_guard import foreign_context
    m = rec['metadata']
    target = np.rint([m['soma_z']*m['z_resolution']/1000,
                     m['soma_y']*m['xy_resolution']/1000,
                     m['soma_x']*m['xy_resolution']/1000]).astype(int)
    foreign, origin = foreign_context(raw, target, rec['inputs']['soma_json'], rec['inputs']['soma_mask'])
    lo = np.maximum(0, origin)
    hi = np.minimum(raw.shape, origin+np.array(foreign.shape))
    stats = {}
    for key, root, mask in zip(('baseline', 'candidate'), roots, masks):
        added = mask & ~original
        values = contrast[added]
        reports = {}
        for letter in 'bd':
            p = root/'mask_steps'/cid/letter/'report.json'
            r = json.loads(p.read_text())
            reports[letter] = dict(connections=len(r.get('accepted', [])),
                                   backmap=r.get('backmap', {}),
                                   paths=[dict(kind=e.get('kind'),path_zyx=e['path_zyx'],
                                               raw_signal_fraction=e.get('raw_signal_fraction'),
                                               reinforced_fraction=e.get('reinforced_fraction'))
                                          for e in r.get('accepted', [])])
        labels, count = ndi.label(mask, np.ones((3, 3, 3)))
        own = int(labels[tuple(target)])
        touched = []
        if own and np.all(hi > lo):
            dst = tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
            src = tuple(slice(int(a),int(b)) for a,b in zip(lo-origin,hi-origin))
            touched = [int(x) for x in np.unique(foreign[src][labels[dst] == own]) if x]
        stats[key] = dict(mask_voxels=int(mask.sum()), added_voxels=int(added.sum()),
                          components=int(count), target_in_mask=bool(own),
                          connected_foreign_soma_cores=touched,
                          measured_contrast_fraction=float(np.mean(values > 2)) if len(values) else None,
                          original_preserved=True, reports=reports)
        del labels
    a, b = masks
    stats.update(different_voxels=int(np.count_nonzero(a != b)),
                 candidate_extra_voxels=int(np.count_nonzero(b & ~a)),
                 candidate_absent_voxels=int(np.count_nonzero(a & ~b)),
                 mask_iou=float(np.count_nonzero(a & b)/max(1, np.count_nonzero(a | b))))
    stats['new_foreign_mask_contacts'] = sorted(set(stats['candidate']['connected_foreign_soma_cores'])-
                                              set(stats['baseline']['connected_foreign_soma_cores']))
    with PLOT_LOCK:
        comparison_image(raw, original, a, b, output/'comparisons'/f'{cid}.png', cid)
    return stats


def run_case(rec, args, signatures):
    cid = rec['id']
    output = args.output
    record_path = output/'cases'/f'{cid}.json'
    previous = json.loads(record_path.read_text()) if record_path.exists() else {}
    expected = [output/side/'mask'/f'{cid}.tif' for side in ('baseline','candidate')]
    if args.full:
        expected += [output/side/'pruning_swcs'/f'{cid}.swc' for side in ('baseline','candidate')]
    if previous.get('status') == 'complete' and all(p.is_file() and p.stat().st_size for p in expected):
        return previous
    result = dict(id=cid, selection=rec.get('selection'), status='running', stages={}, errors={})
    if 'mask' in previous:
        result['mask'] = previous['mask']
    save(record_path, result)
    roots = [output/'baseline', output/'candidate']
    manifests = []
    codes = [args.baseline_code, args.candidate_code]
    for key, code, root in zip(('baseline', 'candidate'), codes, roots):
        cfg = json.loads((code/'config.example.json').read_text())
        cfg.update(output_root=str(root), output_mask_dir=str(root/'mask'),
                   output_mip_dir=str(root/'mask_mips'), comparison_dir=str(root/'comparisons'),
                   direct_inputs=True, previous_swc_dirs=[], previous_app2_dirs=[],
                   validated_baseline_registry=None, baseline_root=None,
                   mask_mip=dict(mode='legacy' if key == 'baseline' else 'multiscale',
                                 thicknesses=[8,16,32,64],overlap=.5,min_axes=2))
        fingerprint = hashlib.sha256(json.dumps(dict(code=signatures[key], inputs=rec['input_identity'],
                                                    config=cfg, metadata=rec['metadata']),sort_keys=True).encode()).hexdigest()
        spec = dict(id=cid, **rec['inputs'], metadata=rec['metadata'], config=cfg,
                    old_swcs=[], fingerprint=fingerprint, selection_reason='paired_mip_candidate_no_baseline_substitution')
        manifest = root/'manifests'/f'{cid}.json'
        save(manifest, spec)
        manifests.append(manifest)
        try:
            result['stages'][key+'_mask_seconds'] = launch(
                [sys.executable, str(code/'07_reconnect_mask.py'), '--spec', str(manifest)],
                output/'logs'/f'{cid}-{key}-mask.log', args.timeout)
        except Exception as exc:
            result['errors'][key+'_mask'] = str(exc)
            save(record_path, result)
    if not result['errors'] and 'mask' not in result:
        try:
            result['mask'] = analyze_mask(rec, roots, output)
        except Exception as exc:
            result['errors']['mask_analysis'] = str(exc)
    save(record_path, result)
    if args.full:
        for key, code, manifest, root in zip(('baseline', 'candidate'), codes, manifests, roots):
            if key+'_mask' in result['errors']:
                continue
            metrics = {}
            try:
                result['stages'][key+'_reconstruction_seconds'] = launch(
                    [sys.executable, str(code/'utils/case_worker.py'), '--spec', str(manifest)],
                    output/'logs'/f'{cid}-{key}-full.log', args.timeout)
                for stage, sub in [('app2','trace_app2/down_sampled_swcs_app2'),
                                   ('gcut','gcut_output/selected_for_pruning'),
                                   ('resample','gcut_output/selected_resampled_step1'),('final','pruning_swcs')]:
                    metrics[stage] = swc_metrics(root/sub/f'{cid}.swc')
                result[key+'_swc'] = metrics
                result[key+'_final_sha256'] = sha(root/'pruning_swcs'/f'{cid}.swc')
            except Exception as exc:
                result['errors'][key+'_full'] = str(exc)
                if metrics:result[key+'_swc_partial'] = metrics
            save(record_path, result)
        if all('final' in result.get(side+'_swc',{}) for side in ('baseline','candidate')):
            from algorithms.tracing.regression_guard import foreign_context, foreign_contacts
            m = rec['metadata']
            target = [m['soma_z']*m['z_resolution']/1000,
                      m['soma_y']*m['xy_resolution']/1000,m['soma_x']*m['xy_resolution']/1000]
            raw = tifffile.imread(rec['inputs']['raw'])
            labels, origin = foreign_context(raw,target,rec['inputs']['soma_json'],rec['inputs']['soma_mask'])
            contacts = []
            tables = []
            for root in roots:
                nodes = np.loadtxt(root/'pruning_swcs'/f'{cid}.swc',comments='#',ndmin=2)
                tables.append(nodes[np.argsort(nodes[:,0]),:7])
                contacts.append(foreign_contacts(nodes,labels,origin))
            result['new_foreign_swc_contacts'] = sorted(contacts[1]-contacts[0])
            result['final_tables_equal'] = bool(tables[0].shape == tables[1].shape and
                                                np.allclose(tables[0],tables[1],rtol=0,atol=1e-6))
    # Both the source and staged bytes are immutable inputs to this comparison.
    for key, identity in rec['input_identity'].items():
        p = Path(identity['source'])
        s = p.stat()
        if (s.st_size, s.st_mtime_ns) != (identity['size'], identity['mtime_ns']) or sha(rec['inputs'][key]) != identity['sha256']:
            result['errors']['input_identity'] = key
    result['status'] = 'complete' if not result['errors'] else 'failed'
    save(record_path, result)
    return result


def summary(output, records, signatures):
    masks = [r for r in records if 'mask' in r]
    paired = [r for r in records if valid_reconstruction_pair(r)]
    data = dict(processed=len(records), complete=sum(r['status']=='complete' for r in records),
                failed=sum(r['status']=='failed' for r in records),
                paired_masks=len(masks), changed_masks=sum(r['mask']['different_voxels']>0 for r in masks),
                paired_reconstructions=len(paired),
                baseline_valid_reconstructions=sum('final' in r.get('baseline_swc',{}) for r in records),
                candidate_valid_reconstructions=sum('final' in r.get('candidate_swc',{}) for r in records),
                new_foreign_mask_contact_cases=[r['id'] for r in masks if r['mask'].get('new_foreign_mask_contacts')],
                new_foreign_swc_contact_cases=[r['id'] for r in records if r.get('new_foreign_swc_contacts')],
                changed_final_swcs=sum(not r.get('final_tables_equal',
                                            r['baseline_final_sha256']==r['candidate_final_sha256']) for r in paired),
                code=signatures, errors={r['id']:r['errors'] for r in records if r['errors']},
                note='No manual ground truth; voxel/length differences are not biological accuracy improvements.')
    retention_path = output/'branch_retention.json'
    if retention_path.exists():
        retention = json.loads(retention_path.read_text())
        data['branch_retention'] = {k:v for k,v in retention.items() if k != 'records'}
    save(output/'summary.json', data)
    rows = []
    for r in records:
        row = dict(id=r['id'], status=r['status'], selection=r['selection'], errors=json.dumps(r['errors']))
        row.update(r.get('stages',{}))
        audit_path = output/'branch_audits'/f'{r["id"]}.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            row['baseline_supported_coverage_by_candidate'] = audit['baseline_supported_coverage_by_candidate']
            row['candidate_supported_coverage_by_baseline'] = audit['candidate_supported_coverage_by_baseline']
        if 'mask' in r:
            m = r['mask']
            for field in ('different_voxels','candidate_extra_voxels','candidate_absent_voxels','mask_iou'):
                row[field] = m[field]
            for side in ('baseline','candidate'):
                for field in ('mask_voxels','added_voxels','components','measured_contrast_fraction'):
                    row[f'{side}_{field}'] = m[side][field]
        for side in ('baseline','candidate'):
            for stage, values in r.get(side+'_swc',{}).items():
                for field, value in values.items():
                    row[f'{side}_{stage}_{field}'] = value
        rows.append(row)
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output/'per_case.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
    write_index(output, records, data)
    return data


def write_index(output, records, data):
    rows = []
    def link(relative,label):
        if (output/relative).is_file():
            return f'<a href="{relative}">{label}</a>'
        return f'<span>{label}未生成</span>'
    def ordering(r):
        audit_path = output/'branch_audits'/f'{r["id"]}.json'
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        foreign = r.get('new_foreign_swc_contacts') or r.get('mask',{}).get('new_foreign_mask_contacts')
        return (r['status'] != 'failed',not bool(foreign),not audit.get('below_98_percent_retention',False),
                -r.get('mask',{}).get('different_voxels',0),int(r['id']))
    ordered = sorted(records,key=ordering)
    for r in ordered:
        cid = html.escape(r['id'])
        m = r.get('mask',{})
        a = r.get('baseline_swc',{}).get('final',{})
        b = r.get('candidate_swc',{}).get('final',{})
        def number(values, key):
            v = values.get(key)
            return '-' if v is None else f'{v:,.2f}'
        flag = []
        audit_path = output/'branch_audits'/f'{r["id"]}.json'
        retention = '-'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            retention = f'{100*audit["baseline_supported_coverage_by_candidate"]:.2f}%'
            if audit['below_98_percent_retention']:flag.append('旧版信号支持分支覆盖<98%，需复核')
        if m.get('new_foreign_mask_contacts'):flag.append('新 mask 胞体接触')
        if r.get('new_foreign_swc_contacts'):flag.append('新 SWC 胞体接触')
        if r['errors']:flag.append(html.escape(json.dumps(r['errors'],ensure_ascii=False)))
        rows.append(f'<tr><td>{cid}</td><td>{html.escape(r["status"])}</td>'
                    f'<td>{m.get("different_voxels","-")}</td>'
                    f'<td>{a.get("nodes","-")} / {b.get("nodes","-")}</td>'
                    f'<td>{number(a,"length_um")} / {number(b,"length_um")}</td>'
                    f'<td>{number(a,"mean_branch_length_um")} / {number(b,"mean_branch_length_um")}</td>'
                    f'<td>{retention}</td>'
                    f'<td>{link(f"comparisons/{cid}.png","mask三视图")} · '
                    f'{link(f"baseline/comparisons/{cid}.png","旧重建")} · '
                    f'{link(f"candidate/comparisons/{cid}.png","新重建")} · '
                    f'{link(f"baseline/pruning_swcs/{cid}.swc","旧SWC")} · '
                    f'{link(f"candidate/pruning_swcs/{cid}.swc","新SWC")} · '
                    f'{link(f"cases/{cid}.json","逐例记录")}</td>'
                    f'<td>{"；".join(flag)}</td></tr>')
    title = (f'v0.83 → v0.83a：已处理 {data["processed"]}，完成 {data["complete"]}，'
             f'失败 {data["failed"]}，mask变化 {data["changed_masks"]}，最终SWC变化 {data["changed_final_swcs"]}')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>MIP 对比</title>'
            '<style>body{font:14px sans-serif;padding:24px;color:#17202a}table{border-collapse:collapse;width:100%}'
            'td,th{border-bottom:1px solid #ddd;padding:10px;text-align:left}th{position:sticky;top:0;background:#edf2f7}'
            'a{color:#145ea8}tr:hover{background:#f7fafc}td:last-child{max-width:360px;overflow-wrap:anywhere}</style>'
            f'<h2>{title}</h2><p>数值顺序：v0.83 / v0.83a。长度单位 µm。'
            '分支是根、分叉与末端之间的完整路径。没有人工真值，长度增加不代表质量提高。</p>'
            '<p>mask 图中红色为新版本多出的体素，黄色为相对旧版本缺少的体素；均为三维差异再投影。'
            '其他胞体接触依赖辅助分割，仅作风险提示。失败案例保留在表内。</p>'
            '<p><a href="per_case.csv">逐例统计 CSV</a> · <a href="summary.json">汇总 JSON</a> · '
            '<a href="branch_retention.json">信号支持分支保留检查</a> · '
            '<a href="summary_metrics.png">统计图</a> · <a href="aggregate_metrics.json">阶段汇总</a></p>'
            '<table><thead><tr><th>ID</th><th>状态</th><th>mask差异体素</th><th>最终节点</th>'
            '<th>最终总长度</th><th>平均分支长度</th><th>旧分支覆盖</th><th>结果</th><th>提示</th></tr></thead><tbody>'
            +''.join(rows)+'</tbody></table></html>')
    (output/'index.html').write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--baseline-code', type=Path, required=True)
    ap.add_argument('--candidate-code', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--timeout', type=int, default=2400)
    ap.add_argument('--full', action='store_true', help='Also rerun APP2, G-Cut, resampling and pruning in both arms')
    ap.add_argument('--report-only',action='store_true',help='Rebuild reports from saved records without rerunning or changing tested code identity')
    ap.add_argument('--ids', help='Optional smoke-test subset, comma separated')
    args = ap.parse_args()
    sys.path.insert(0, str(args.candidate_code))
    from utils import bootstrap
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'.comparison.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.report_only:
        experiment = json.loads((args.output/'experiment.json').read_text())
        records = saved_records(args.output)
        print(json.dumps(summary(args.output,records,experiment['code']),indent=2),flush=True)
        return 0
    signatures = dict(baseline=code_hash(args.baseline_code), candidate=code_hash(args.candidate_code))
    cohort = json.loads(args.cohort.read_text())['cases']
    if args.ids:
        cohort = [r for r in cohort if r['id'] in args.ids.split(',')]
    identity = dict(code=signatures, cohort_sha256=sha(args.cohort), full=args.full)
    manifest = args.output/'experiment.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Code/cohort/mode changed; use a new output directory')
    save(manifest, identity)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, r, args, signatures):r['id'] for r in cohort}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                traceback.print_exc()
                result = dict(id=cid,status='failed',selection=None,errors={'runner':str(exc)})
                save(args.output/'cases'/f'{cid}.json', result)
            records.append(result)
            data = summary(args.output, records, signatures)
            print(f"{len(records)}/{len(cohort)} | {cid}: {result['status']} | masks changed={data['changed_masks']} | errors={result['errors']}",flush=True)
    current = dict(baseline=code_hash(args.baseline_code),candidate=code_hash(args.candidate_code))
    if current != signatures:
        raise RuntimeError('Live code changed during experiment; comparison is invalid')
    data = summary(args.output,records,signatures)
    print(json.dumps(data,indent=2),flush=True)
    return int(bool(data['failed']))


if __name__ == '__main__':
    sys.exit(main())
