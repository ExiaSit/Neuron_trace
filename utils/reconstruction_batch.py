#!/usr/bin/env python3
"""Run v0.83 for all stable matched inputs, reading source files in place."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import tifffile
from utils import mask_batch as batch
from utils.case_worker import save_json, paths, check_swc

HERE = Path(__file__).resolve().parents[1]


def index_files(folder, suffixes):
    found = {}; duplicates = set()
    for entry in os.scandir(folder):
        if not entry.is_file() or Path(entry.name).suffix.lower() not in suffixes:
            continue
        match = re.fullmatch(r'(?:image_)?(\d+)(?:_0000)?', Path(entry.name).stem)
        if not match:
            continue
        cid = str(int(match[1]))
        if cid in found:
            duplicates.add(cid)
        else:
            found[cid] = Path(entry.path)
    for cid in duplicates:
        found.pop(cid)
    return found, duplicates


def file_stamp(path):
    p = Path(path); s = p.stat()
    return [str(p.resolve()), s.st_size, s.st_mtime_ns]


def stable(paths_to_check, minimum_age, now=None):
    now = time.time() if now is None else now
    for p in paths_to_check:
        s = Path(p).stat()
        if not s.st_size or now - max(s.st_mtime, s.st_ctime) < minimum_age:
            return False
    return True


def validate_tiffs(raw, mask, soma):
    shapes = []
    for p in [raw, mask, soma]:
        size = Path(p).stat().st_size
        with tifffile.TiffFile(p) as tif:
            series = tif.series[0]
            if len(series.shape) != 3:
                raise ValueError(f'Expected 3D TIFF: {p}')
            if p == raw and series.dtype != np.uint8:
                raise ValueError(f'Expected uint8 raw: {p}')
            # Shaped/contiguous TIFFs may retain a full declared shape even
            # when later IFDs or pixel data have not arrived yet.
            if series.dataoffset is not None and series.dataoffset + series.size * series.dtype.itemsize > size:
                raise ValueError(f'Contiguous TIFF data incomplete: {p}')
            # Check every strip/tile boundary without loading the whole volume.
            for page in series.pages:
                if any(o <= 0 or n <= 0 or o+n > size for o, n in zip(page.dataoffsets, page.databytecounts)):
                    raise ValueError(f'TIFF data incomplete / still transferring: {p}')
            shapes.append(series.shape)
    if shapes[0] != shapes[1]:
        raise ValueError('Raw/mask shape mismatch')


def main():
    ap = argparse.ArgumentParser(description='v0.83: 所有已匹配样本，缺失或迁移中先跳过')
    ap.add_argument('--config', required=True)
    ap.add_argument('--ids')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg.get('method_version') != 'v0.83' or not cfg.get('direct_inputs'):
        raise ValueError('This runner requires v0.83 and direct_inputs=true')
    batch.validate_parallel_config(cfg)
    root = Path(cfg['output_root']).resolve()
    sources = [Path(cfg[k]).resolve() for k in ['raw_dir','mask_dir','soma_crop_dir','soma_seg_dir']]
    for src in sources:
        if not src.is_dir() or batch.inside(root, src) or batch.inside(src, root):
            raise ValueError(f'Invalid or overlapping source: {src}')
    if Path(cfg['output_mask_dir']).resolve() != root/'mask':
        raise ValueError('Mask output must be output_root/mask')
    for k in ['output_mip_dir','comparison_dir']:
        dest = Path(cfg[k]).resolve()
        if dest == root or not batch.inside(dest, root) or batch.inside(dest, root/'mask'):
            raise ValueError(f'Invalid output path: {k}')
    for k in ['python_executable','vaa3d_app2','vaa3d_resample']:
        if not Path(cfg[k]).is_file():
            raise ValueError(f'Missing dependency: {k}')
    if Path(sys.executable).resolve() != Path(cfg['python_executable']).resolve():
        raise ValueError('Use the configured hb_seg Python')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'.run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root/'batch.pid').write_text(str(os.getpid())+'\n')
    indexes = []; duplicate_sets = []
    for directory, suffixes in zip(sources, [{'.tif','.tiff'}, {'.tif','.tiff'}, {'.json'}, {'.tif','.tiff'}]):
        index, duplicates = index_files(directory, suffixes)
        indexes.append(index); duplicate_sets.append(duplicates)
    raw_index, mask_index, crop_index, soma_index = indexes
    duplicates = set().union(*duplicate_sets)
    cols = ['cell_id','soma_x','soma_y','soma_z','xy_resolution','z_resolution']
    for encoding in ['utf-8-sig','gb18030']:
        try:
            meta = pd.read_csv(cfg['metadata_file'], usecols=cols, index_col='cell_id', encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('Cannot decode metadata')
    code_files = [*sorted(HERE.glob('*.py')), *sorted((HERE/'utils').rglob('*.py')),
                  *sorted((HERE/'algorithms').rglob('*.py')), *sorted((HERE/'visualization').rglob('*.py')),
                  *sorted((HERE/'vendor').rglob('*.py')), *sorted((HERE/'vendor').rglob('*.json'))]
    code_hash = hashlib.sha256(b''.join(p.read_bytes() for p in code_files)).hexdigest()
    save_json(root/'runtime.json', {'version':'v0.83','revision':cfg['method_revision'],'source':str(HERE),'sha256':code_hash})
    save_json(root/'run-config.json', cfg)
    wanted = {str(int(x)) for x in args.ids.split(',')} if args.ids else None
    all_ids = set(raw_index) | set(mask_index) | duplicates
    if wanted is not None:
        all_ids = wanted
    ready = {}; skipped = []
    for cid in sorted(all_ids, key=int):
        reason = None
        if cid in duplicates:
            reason = 'duplicate_input_id'
        elif cid not in raw_index or cid not in mask_index:
            reason = 'missing_raw' if cid not in raw_index else 'missing_mask'
        elif cid not in crop_index or cid not in soma_index:
            reason = 'missing_soma_auxiliary'
        elif int(cid) not in meta.index:
            reason = 'missing_metadata'
        else:
            record = meta.loc[int(cid)]
            if isinstance(record, pd.DataFrame) or not np.isfinite(record.to_numpy(dtype=float)).all():
                reason = 'invalid_or_duplicate_metadata'
            else:
                inputs = [index[cid] for index in indexes]
                try:
                    if not stable(inputs, cfg.get('minimum_stable_seconds',120)):
                        reason = 'still_transferring_or_recently_changed'
                    else:
                        ready[cid] = inputs
                except OSError:
                    reason = 'input_disappeared_during_scan'
        if reason:
            skipped.append(dict(id=cid, reason=reason))
    save_json(root/'skipped-inputs.json', {'records':skipped})
    plan = dict(version='v0.83', discovered_raw=len(raw_index), discovered_mask=len(mask_index),
                matched_raw_mask=len(set(raw_index)&set(mask_index)), ready=len(ready), skipped=len(skipped),
                ready_ids=list(ready), selection='all_matched_no_swc_screening')
    save_json(root/'input-scan.json', plan)
    print(f'v0.83：原图 {len(raw_index)}，mask {len(mask_index)}，本轮就绪 {len(ready)}，暂跳过 {len(skipped)}；{cfg["workers"]} 路并行', flush=True)
    if args.check:
        return 0
    counters = dict(total=len(ready), completed=0, resumed=0, failed=0, deferred=0)
    active = {}; guard = threading.Lock(); monitor_stop = threading.Event()
    def progress():
        with guard:
            data = dict(counters, active=dict(active), updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        stages = {}
        for cid in data['active']:
            p = root/'status'/f'{cid}.json'
            try:
                stage = json.loads(p.read_text()).get('current_stage','preflight')
            except (OSError, ValueError):
                stage = 'preflight'
            stages[stage] = stages.get(stage,0)+1
        data['stages'] = stages
        save_json(root/'progress.json', data)
        print(f'进度：完成 {data["completed"]}，续跑跳过 {data["resumed"]}，失败 {data["failed"]}，推迟 {data["deferred"]}，运行 {len(data["active"])}/{data["total"]}；阶段 {stages}', flush=True)
    def monitor():
        while not monitor_stop.wait(60):
            progress()
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    def launch(pair, config, stop):
        cid = pair[2].stem; inputs = ready[cid]
        with guard:
            active[cid] = 'running'
        child = None
        try:
            if not stable(inputs, cfg.get('minimum_stable_seconds',120)):
                with guard: counters['deferred'] += 1
                save_json(root/'deferred'/f'{cid}.json', {'id':cid,'reason':'inputs_not_stable_at_launch'})
                return '输入仍在迁移，本轮推迟'
            stamps = [file_stamp(p) for p in inputs]
            record = meta.loc[int(cid)].to_dict()
            override = cfg.get('soma_overrides_image_xyz',{}).get(cid)
            if override is not None:
                if len(override)!=3 or not np.isfinite(override).all():raise ValueError('Invalid soma override')
                for k,res,val in zip(['soma_x','soma_y','soma_z'],['xy_resolution','xy_resolution','z_resolution'],override):
                    record[k] = float(val)*1000/float(record[res])
            fingerprint = hashlib.sha256(json.dumps(dict(inputs=stamps, metadata=record, code=code_hash,
                tools={k:cfg[k] for k in ['vaa3d_app2','vaa3d_resample','python_executable']}),sort_keys=True).encode()).hexdigest()
            spec = dict(id=cid, raw=str(inputs[0]), mask=str(inputs[1]), soma_json=str(inputs[2]), soma_mask=str(inputs[3]),
                        old_swcs=[], selection_reason='all_matched_v083', metadata=record, config=cfg, fingerprint=fingerprint)
            state_path = root/'status'/f'{cid}.json'
            from utils.baseline_io import seed
            seed(spec,stamps,state_path,save_json,check_swc)
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get('fingerprint') != fingerprint:
                    raise ValueError('Input or runtime differs from saved result; existing result retained')
                p = paths(spec)
                expected = [p[k] for k in ['mask','app2','selected','resampled','final','mip','comparison']]
                expected += [root/'gcut_mips'/f'{cid}.png',root/'pruning_mips'/f'{cid}_combined_process.png']
                if state.get('status') == 'complete' and all(x.is_file() and x.stat().st_size for x in expected):
                    check_swc(p['final'])
                    with guard: counters['resumed'] += 1
                    return '完整结果已存在，跳过'
            validate_tiffs(inputs[0], inputs[1], inputs[3])
            json.loads(inputs[2].read_text())
            if stamps != [file_stamp(p) for p in inputs]:
                raise ValueError('Inputs changed during preflight')
            spec_path = root/'manifests'/f'{cid}.json'; save_json(spec_path,spec)
            log = root/'logs'/f'{cid}.log'; log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as stream:
                child = subprocess.Popen([cfg['python_executable'],str(HERE/'utils/case_worker.py'),'--spec',str(spec_path)],
                    stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                while child.poll() is None:
                    if stop.wait(1):raise InterruptedError('Batch stopped')
            if child.returncode:
                raise RuntimeError(f'阶段失败，见 {log}')
            if stamps != [file_stamp(p) for p in inputs]:
                state=json.loads(state_path.read_text());state.update(status='failed',error='Inputs changed during execution')
                save_json(state_path,state)
                raise ValueError('Inputs changed during execution; result not accepted as complete')
            with guard: counters['completed'] += 1
            return 'v0.83 mask、SWC 和对比图完成'
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            with guard: counters['failed'] += 1
            save_json(root/'failures'/f'{cid}.json', {'id':cid,'error':str(exc)})
            raise
        finally:
            with guard: active.pop(cid,None)
    batch.process = launch
    estimate = batch.estimate_memory_gb
    batch.estimate_memory_gb = lambda pair: estimate(pair)*1.5
    jobs = [(inputs[0],inputs[1],root/'mask'/f'{cid}.tif') for cid,inputs in ready.items()]
    try:
        successful, failures = batch.process_group(jobs,cfg,0,len(jobs))
        save_json(root/'run-summary.json', dict(counters, scheduler_failures=failures, finished=True))
        return 1 if failures else 0
    finally:
        monitor_stop.set();thread.join();progress()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('已停止；同一命令可断点续跑，并重新扫描新迁入文件。',flush=True)
        sys.exit(130)
