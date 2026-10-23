#!/usr/bin/env python3
"""Internal six-pass mask runner and memory-aware scheduler; use run_pipeline.py."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parents[1]
STAGES = ['skeleton_direction.py', 'signal_reconnect.py', 'soma_completion.py', 'soma_neurite_reconnect.py', 'connection_thickness.py', 'soma_root_repair.py']
STAGE_NAMES = ['骨架及方向分析', '三维信号验证与初次重连', '胞体补全', '胞体与分支重连', '连接厚度恢复', '胞体根部补全']


def inside(path, parent):
    return path == parent or parent in path.parents


def prepare(config_path, limit=None):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    for key in ['raw_dir', 'mask_dir', 'output_mask_dir', 'output_mip_dir']:
        p = Path(cfg[key]).expanduser()
        cfg[key] = (config_path.parent / p).resolve() if not p.is_absolute() else p.resolve()
    for key in ['raw_dir', 'mask_dir']:
        if not cfg[key].is_dir():
            raise ValueError(f'{key} 不存在: {cfg[key]}')
    outputs = [cfg['output_mask_dir'], cfg['output_mip_dir']]
    for out in outputs:
        for inp in [cfg['raw_dir'], cfg['mask_dir']]:
            if inside(out, inp) or inside(inp, out):
                raise ValueError('输出目录必须与输入目录分开，不能相互包含')
    if inside(outputs[0], outputs[1]) or inside(outputs[1], outputs[0]):
        raise ValueError('mask 与 MIP 输出目录不能相同或相互包含')
    every = cfg.get('mip_every', 500)
    if not isinstance(every, int) or isinstance(every, bool) or every < 1:
        raise ValueError('mip_every 必须是正整数')
    cfg['mip_every'] = every
    if not isinstance(cfg.get('skip_existing', True), bool):
        raise ValueError('skip_existing 必须是 true 或 false')
    if cfg.get('temp_dir'):
        p = Path(cfg['temp_dir']).expanduser()
        cfg['temp_dir'] = str((config_path.parent / p).resolve())
        for out in outputs:
            if inside(Path(cfg['temp_dir']), out) or inside(out, Path(cfg['temp_dir'])):
                raise ValueError('temp_dir 必须与结果目录分开')
    validate_parallel_config(cfg)
    masks = sorted(cfg['mask_dir'].glob(cfg.get('mask_glob', '*.tif')))
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit 必须大于 0')
        masks = masks[:limit]
    if not masks:
        raise ValueError('没有匹配的 mask 文件')
    pairs, seen = [], set()
    for mask in masks:
        if not mask.is_file():
            raise ValueError(f'mask 不是文件: {mask}')
        rel = mask.relative_to(cfg['mask_dir'])
        name = cfg.get('raw_name_template', '{stem}_0000.tif').format(stem=mask.stem, name=mask.name)
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('raw_name_template 必须生成文件名，不能包含目录')
        raw = cfg['raw_dir'] / rel.parent / name
        if not raw.is_file():
            raise ValueError(f'原图不存在: {raw}（mask: {mask}）')
        out = cfg['output_mask_dir'] / rel.with_suffix('.tif')
        if out in seen:
            raise ValueError(f'输出文件名重复: {out}')
        seen.add(out)
        pairs.append((raw, mask, out))
    return cfg, pairs


def validate_parallel_config(cfg):
    workers = cfg.setdefault('workers', 1)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError('workers 必须是正整数')
    for key, default in [('memory_budget_gb', 256), ('memory_reserve_gb', 32)]:
        value = cfg.setdefault(key, default)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not np.isfinite(value) or value < 0 or (key == 'memory_budget_gb' and value == 0):
            raise ValueError(f'{key} 必须是有效的非负数字，内存预算必须大于 0')


def available_memory_gb():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) / 1024 ** 2


def estimate_memory_gb(pair):
    # Conservative scheduling estimate, not a hard RSS cap. Algorithms may have
    # different peak usage with different fragment complexity.
    with tifffile.TiffFile(pair[0]) as tif:
        series = tif.series[0]
        if len(series.shape) != 3 or series.dtype != np.uint8:
            raise ValueError('原图必须是 uint8 三维 ZYX TIFF')
        return .5 + int(np.prod(series.shape)) * 96 / 1024 ** 3


def process_group(group, cfg, start, total):
    """Bounded parallel subprocess pipelines; results returned in input order."""
    successes = []
    failures = 0
    stop = threading.Event()
    pending = {}
    index = 0
    estimates = {}
    workers = cfg.get('workers', 1)
    def job(pair, position):
        began = time.monotonic()
        print(f'[{position}/{total}] {pair[1].name} 开始', flush=True)
        status = process(pair, cfg, stop)
        print(f'[{pair[1].name}] {status}，{time.monotonic()-began:.1f}s', flush=True)
        return pair
    pool = ThreadPoolExecutor(max_workers=workers)
    waiting_since = None
    try:
        while index < len(group) or pending:
            while index < len(group) and len(pending) < workers:
                pair = group[index]
                try:
                    if index not in estimates:
                        estimates[index] = estimate_memory_gb(pair)
                    estimate = estimates[index]
                except Exception as exc:
                    failures += 1
                    print(f'[{pair[1].name}] 输入检查失败: {exc}', file=sys.stderr, flush=True)
                    index += 1
                    continue
                used = sum(item[1] for item in pending.values())
                # Conservative: available RAM already excludes allocated memory;
                # subtract whole reservations too, to cover jobs still ramping up.
                room = min(cfg['memory_budget_gb'], available_memory_gb() - cfg['memory_reserve_gb'])
                if used + estimate > room:
                    if not pending:
                        failures += 1
                        print(f'[{pair[1].name}] 内存不足以启动：估计 {estimate:.1f} GiB，可调度 {max(0,room):.1f} GiB；请调整预算/预留或释放内存后重试', file=sys.stderr, flush=True)
                        index += 1
                        continue
                    if waiting_since is None or time.monotonic() - waiting_since >= 30:
                        print(f'调度等待内存：运行 {len(pending)} 个，已预估 {used:.1f} GiB，下一个需 {estimate:.1f} GiB', flush=True)
                        waiting_since = time.monotonic()
                    break
                print(f'启动调度：{pair[1].name}，估计峰值 {estimate:.1f} GiB，同时运行 {len(pending)+1}/{workers}', flush=True)
                future = pool.submit(job, pair, start + index + 1)
                pending[future] = (index, estimate)
                index += 1
            if pending:
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    position, _ = pending.pop(future)
                    try:
                        successes.append((position, future.result()))
                    except Exception as exc:
                        failures += 1
                        print(f'[{group[position][1].name}] 失败: {exc}', file=sys.stderr, flush=True)
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
    return [pair for _, pair in sorted(successes)], failures


def load_inputs(raw_path, mask_path):
    raw = tifffile.imread(raw_path)
    mask = tifffile.imread(mask_path) > 0
    if raw.ndim != 3 or raw.shape != mask.shape or raw.dtype != np.uint8:
        raise ValueError('要求原图为 uint8 三维 ZYX TIFF，mask 与原图尺寸相同；不自动缩放灰度')
    return raw, mask


def validate_result(result, mask):
    if result.shape != mask.shape or result.dtype != np.uint8:
        raise ValueError('结果的尺寸或类型不正确')
    if not np.all((result == 0) | (result == 255)) or not np.all(result[mask] > 0):
        raise ValueError('结果不是 0/255 二值 mask，或初始前景有丢失')


def atomic_tiff(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tif', delete=False) as f:
        tmp = Path(f.name)
    try:
        tifffile.imwrite(tmp, result, compression='zlib', photometric='minisblack', metadata={'axes': 'ZYX'})
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def process(pair, cfg, stop=None):
    raw_path, mask_path, out = pair
    def progress(message):
        print(f'[{mask_path.name}] {message}', flush=True)
    def check_stop():
        if stop is not None and stop.is_set():
            raise InterruptedError('任务已取消')
    check_stop()
    raw, mask = load_inputs(raw_path, mask_path)
    if out.exists() and cfg.get('skip_existing', True):
        validate_result(tifffile.imread(out), mask)
        return '已存在，校验后跳过'
    progress(f'输入 ZYX={raw.shape}，{raw.size / 1e6:.1f} 百万体素')
    del raw
    env = {k: v for k, v in os.environ.items() if not k.startswith('RECONNECT_')}
    env.update(RECONNECT_RAW=str(raw_path), RECONNECT_MASK=str(mask_path),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='mask-reconnect-', dir=cfg.get('temp_dir')) as work:
        env['RECONNECT_WORK'] = work
        progress(f'临时工作目录: {work}')
        for number, stage in enumerate(STAGES, 1):
            check_stop()
            title = STAGE_NAMES[number - 1]
            started = time.monotonic()
            progress(f'[{number}/6] {title} 开始')
            last_heartbeat = started
            log_path = Path(work) / f'{stage}.log'
            with log_path.open('w') as log:
                child = subprocess.Popen([sys.executable, str(HERE / 'algorithm' / stage)],
                                         cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while True:
                        check_stop()
                        try:
                            code = child.wait(timeout=1)
                            break
                        except subprocess.TimeoutExpired:
                            now = time.monotonic()
                            if now - last_heartbeat >= 30:
                                progress(f'[{number}/6] {title} 进程尚未结束，已用 {now-started:.0f}s（PID {child.pid}）')
                                last_heartbeat = now
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
            if code:
                raise RuntimeError(f'{stage} 失败:\n{log_path.read_text()[-4000:]}')
            progress(f'[{number}/6] {title} 完成，{time.monotonic()-started:.1f}s')
        result = tifffile.imread(Path(work) / 'f/connected-mask.tif')
        validate_result(result, mask)
        check_stop()
        atomic_tiff(out, result)
    return '完成'


from visualization.mask_mip import make_mip



# Compatibility entry retained for the original scheduler regression tests.
# The public v0.83 entry is run_pipeline.py.
def main():
    parser = argparse.ArgumentParser(description='halo-guard-v3：批量生成新的 mask TIFF 与抽样 MIP')
    parser.add_argument('--config', default=str(HERE / 'config.json'))
    parser.add_argument('--workers', type=int, help='覆盖配置中的并行样本数')
    parser.add_argument('--check', action='store_true', help='只检查配置和输入文件配对，不执行算法')
    parser.add_argument('--limit', type=int, help='只处理排序后的前 N 个样本，供试跑使用')
    args = parser.parse_args()
    cfg, pairs = prepare(args.config, args.limit)
    if args.workers is not None:
        cfg['workers'] = args.workers
    validate_parallel_config(cfg)
    print(f'找到 {len(pairs)} 对输入；mask 输出: {cfg["output_mask_dir"]}；MIP 输出: {cfg["output_mip_dir"]}', flush=True)
    if args.check:
        print('文件配对检查通过；运行时会检查每个 TIFF 的类型和尺寸。')
        return 0
    if cfg.get('temp_dir'):
        Path(cfg['temp_dir']).mkdir(parents=True, exist_ok=True)
    print(f'并行数 {cfg["workers"]}；估计内存预算 {cfg["memory_budget_gb"]} GiB；系统预留 {cfg["memory_reserve_gb"]} GiB', flush=True)
    rng = random.Random(cfg.get('random_seed', 20261012))
    failures = 0
    for start in range(0, len(pairs), cfg['mip_every']):
        successful = []
        if cfg['workers'] > 1:
            successful, errors = process_group(pairs[start:start + cfg['mip_every']], cfg, start, len(pairs))
            failures += errors
        else:
            for offset, pair in enumerate(pairs[start:start + cfg['mip_every']], start + 1):
                started = time.monotonic()
                print(f'[{offset}/{len(pairs)}] {pair[1].name} 开始', flush=True)
                try:
                    status = process(pair, cfg)
                    successful.append(pair)
                    print(f'  {status}，{time.monotonic()-started:.1f}s', flush=True)
                except Exception as exc:
                    failures += 1
                    print(f'  失败: {exc}', file=sys.stderr, flush=True)
        if successful:
            selected = rng.choice(successful)
            path = cfg['output_mip_dir'] / f'batch_{start // cfg["mip_every"] + 1:05d}.png'
            try:
                make_mip(selected, path)
                print(f'抽样 MIP: {path}（{selected[1].name}）', flush=True)
            except Exception as exc:
                failures += 1
                print(f'MIP 失败: {exc}', file=sys.stderr, flush=True)
    print(f'处理结束：共 {len(pairs)} 个输入，失败事件 {failures} 次（详见上方错误）。' if failures else f'处理结束：全部 {len(pairs)} 个输入成功，MIP 已抽样生成。', flush=True)
    return 1 if failures else 0
