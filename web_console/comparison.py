"""Read-only views of explicitly registered paired reconstruction reports."""
import json
from pathlib import Path
from web_console.artifact_metadata import with_storage_paths


def comparison_paths(cid):
    if not cid.isdigit() or len(cid) > 12:
        raise ValueError('ID 无效')
    return {
        f'comparisons/{cid}.png': ('三方向 Mask 对照', True),
        'summary_metrics.png': ('300 例总体统计', True),
        'per_case.csv': ('300例逐例指标.csv', False),
        'summary.json': ('300例汇总.json', False),
        'aggregate_metrics.json': ('300例重建统计.json', False),
        'branch_retention.json': ('300例分支保留统计.json', False),
        f'cases/{cid}.json': (f'{cid}_对照指标.json', False),
        f'branch_audits/{cid}.json': (f'{cid}_分支保留核查.json', False),
        f'baseline/pruning_swcs/{cid}.swc': (f'v0.83_{cid}.swc', False),
        f'candidate/pruning_swcs/{cid}.swc': (f'v0.83a_{cid}.swc', False),
    }


def registered_root(record, cid):
    if cid not in record['request']['ids']:
        raise ValueError('样本不属于本任务')
    source = record.get('comparison_source')
    if not source or not record.get('read_only'):
        raise ValueError('该实验没有登记跨版本对照')
    return Path(source)


def comparison_artifacts(record, cid):
    root = registered_root(record, cid)
    result = []
    for relative, (name, image) in comparison_paths(cid).items():
        p = root / relative
        if p.is_file():
            result.append(dict(path=relative, name=name, group='mip_comparison', image=image,
                               scope='comparison', cell=cid, size=p.stat().st_size,
                               mtime=p.stat().st_mtime_ns,
                               description=('三行依次为 XY/XZ/YZ；四列为原 mask、v0.83、v0.83a、两版差异。红色为新版多出的体素，黄色为仅旧版存在的体素。' if relative.startswith('comparisons/') else '已完成的配对实验统计。')))
    return with_storage_paths(root,result)


def comparison_summary(record, cid):
    root = registered_root(record, cid)
    path = root / 'cases' / f'{cid}.json'
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    mask = data.get('mask', {})
    result = dict(errors=data.get('errors', {}), versions={},
                  different_voxels=mask.get('different_voxels'),
                  new_foreign_mask_contacts=mask.get('new_foreign_mask_contacts', []),
                  new_foreign_swc_contacts=data.get('new_foreign_swc_contacts', []))
    for key, label in [('baseline', 'v0.83'), ('candidate', 'v0.83a')]:
        swcs = data.get(key + '_swc', {})
        status_path = root / key / 'status' / f'{cid}.json'
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            if status.get('status') == 'failed':
                result['errors'][key + '_full'] = status.get('error', '源实验失败')[-1200:]
        result['versions'][label] = dict(mask_voxels=mask.get(key, {}).get('mask_voxels'),
            added_voxels=mask.get(key, {}).get('added_voxels'), final=swcs.get('final'))
    audit = root / 'branch_audits' / f'{cid}.json'
    if audit.is_file():
        result['branch_retention'] = json.loads(audit.read_text()).get('baseline_supported_coverage_by_candidate')
    return result
