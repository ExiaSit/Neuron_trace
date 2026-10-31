"""Read per-sample checkpoints for imported parallel batches and native jobs."""
import json
import re
import time
from collections import Counter

STAGES = {'continuity': 7, 'app2': 8, 'gcut': 9, 'resample': 10,
          'prune': 11, 'guard': 12, 'comparison': 13}
NAMES = {'mask_a': 'Mask a · 方向分析', 'mask_b': 'Mask b · 信号与 MIP',
         'mask_c': 'Mask c · 胞体初补', 'mask_d': 'Mask d · 胞体连接',
         'mask_e': 'Mask e · 加粗', 'mask_f': 'Mask f · 胞体形状',
         'continuity': 'Mask g · 受约束延伸', 'app2': '骨架＋APP2',
         'gcut': 'G-Cut', 'resample': '重采样', 'prune': '剪枝',
         'guard': '结果检查', 'comparison': '比较图'}


def sample_status(job, request, progress, imported=False, active=False, now=None):
    now = time.time() if now is None else now
    try: stamp = (job/'progress.json').stat().st_mtime
    except OSError: stamp = now
    stale = bool(progress.get('stale')) or (imported and progress.get('status') in ('running', 'queued') and now-stamp > 90)
    completed = progress.get('completed', {})
    failed = progress.get('failed', {})
    current = progress.get('current') or {}
    rows = []
    for cid in request['ids']:
        cid = str(cid)
        if not re.fullmatch(r'\d{1,12}', cid):
            continue
        state = {}
        path = job/'reconstruction/status'/f'{cid}.json'
        if imported and path.is_file():
            try:
                state = json.loads(path.read_text())
            except (OSError, ValueError):
                state = {'status': 'unknown'}
        done = [n for n in range(request['start'], request['stop']+1) if f'{cid}:{n}' in completed]
        errors = [(key, value) for key, value in failed.items() if key == cid or key.startswith(cid+':')]
        status = state.get('status')
        stage = state.get('current_stage')
        stage_num = 7 if str(stage).startswith('mask_') else STAGES.get(stage)
        if not status:
            if errors:
                status = 'failed'
                try: stage_num = int(errors[0][0].split(':')[1])
                except (IndexError, ValueError): pass
            elif len(done) == request['stop']-request['start']+1:
                status = 'complete'
            elif current.get('id') == cid and (active or imported):
                status, stage_num = 'running', current.get('stage')
            elif done:
                status = 'paused' if progress.get('status') == 'paused' else 'interrupted'
            else:
                status = 'pending'
        if status in ('queued', 'mask_complete'): status = 'pending'
        if status == 'running' and (stale or progress.get('status') in ('paused', 'interrupted', 'failed')):
            status = 'paused' if progress.get('status') == 'paused' else 'interrupted'
        error = state.get('error') or (str(errors[0][1]) if errors else '')
        if status == 'failed' and state.get('timeout'): status = 'timeout'
        rows.append(dict(id=cid, status=status, stage=stage_num,
                         stage_name=NAMES.get(stage, current.get('name', '') if current.get('id') == cid else ''),
                         completed_stages=len(done), error=error,
                         elapsed_seconds=max(0, round(now-state['started_at'])) if status == 'running' and state.get('started_at') else None))
    return dict(samples=rows, counts=dict(Counter(x['status'] for x in rows)),
                stale=stale, updated_at=stamp,
                warning='进度已超过 90 秒未更新，原运行样本标为中断，请检查后台进程。' if stale else '')
