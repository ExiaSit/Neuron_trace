#!/usr/bin/env python3
"""Register completed paired runs as read-only console experiments (admin CLI only)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.case_worker import save_json

STAGE_KEYS = {7: 'continuity', 8: 'app2', 9: 'gcut', 10: 'resample',
              11: 'prune', 12: 'guard', 13: 'comparison'}


def imported_progress(source, ids, variant):
    completed, failed = {}, {}
    for cid in ids:
        status = json.loads((source / variant / 'status' / f'{cid}.json').read_text())
        case = json.loads((source / 'cases' / f'{cid}.json').read_text())
        for stage, key in STAGE_KEYS.items():
            if key in status.get('completed_stages', []):
                seconds = status.get('seconds', {}).get(key, 0)
                if stage == 7:
                    seconds += sum(v for k, v in status.get('seconds', {}).items() if k.startswith('mask_'))
                completed[f'{cid}:{stage}'] = dict(seconds=round(seconds, 2), imported=True)
            elif key == status.get('current_stage') and status.get('status') == 'failed':
                failed[f'{cid}:{stage}'] = status.get('error', '源实验失败')
        error = case.get('errors', {}).get(variant + '_full')
        if error and status.get('status') == 'complete':
            # The stricter paired SWC validation can reject a nominally completed run.
            failed[f'{cid}:11'] = error
            completed.pop(f'{cid}:11', None)
    return dict(status='imported', completed=completed, failed=failed,
                error=f'只读历史结果；{len(ids)} 例，{len(failed)} 例存在失败或结构校验问题。')


def register(state_root, cohort_root):
    cohort_root = cohort_root.resolve()
    cohort = json.loads((cohort_root / 'cohort.json').read_text())
    ids = [str(c['id']) for c in cohort['cases']]
    if not ids or len(ids) != len(set(ids)) or any(not x.isdigit() for x in ids):
        raise ValueError('样本列表无效')
    source = cohort_root / 'comparison'
    jobs = state_root / 'jobs'; jobs.mkdir(parents=True, exist_ok=True)
    created = []
    for variant, version, title in [('baseline', 'v0.83', '原 XY MIP'),
                                     ('candidate', 'v0.83a', '三方向 · 重叠 · 多厚度 MIP')]:
        jid = hashlib.sha256(f'{source}:{variant}'.encode()).hexdigest()[:16]
        job = jobs / jid
        if job.exists():
            old = json.loads((job / 'job.json').read_text())
            if not old.get('read_only') or old.get('comparison_source') != str(source):
                raise ValueError('导入 ID 与现有实验冲突')
            created.append(jid); continue
        progress = imported_progress(source, ids, variant)
        cache = cohort_root / 'input_cache'
        request = dict(name=f'{version} · {title} · {len(ids)}例对照', ids=ids, start=7, stop=13,
                       gpu_id=0, method_version=version, metadata_file=cohort['source_metadata'],
                       raw_dir=str(cache/'raw'), mask_dir=str(cache/'mask'),
                       soma_crop_dir=str(cache/'soma_json'), soma_seg_dir=str(cache/'soma_mask'))
        job.mkdir()
        (job / 'reconstruction').symlink_to(source / variant, target_is_directory=True)
        save_json(job/'progress.json', progress)
        # Publish the registration last. No executable plan is invented for imported data.
        save_json(job/'job.json', dict(request=request, created_at=time.time(), read_only=True,
            comparison_source=str(source), imported_from=str(source/variant),
            note='已完成的配对实验，只读导入；不是网页刚刚重跑的结果。可用当前编号新建实验；失败与分支保留风险见下方。'))
        created.append(jid)
    return created


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state-root', required=True, type=Path)
    ap.add_argument('--cohort-root', required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(register(args.state_root, args.cohort_root)))
