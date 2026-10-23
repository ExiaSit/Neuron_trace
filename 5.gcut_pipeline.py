"""v0.81 stage interface backed by the current G-Cut, without mask repair.

A subprocess per case isolates candidate backend scopes and per-case masks from
v0.81's thread scheduler. All inputs retain v0.81 image_<id> naming.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def _write_json(path, value):
    tmp = path.with_suffix('.writing.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda v: v.item() if hasattr(v, 'item') else str(v))+'\n')
    tmp.replace(path)


def worker_task(args):
    from pipeline_config import CONFIG_FILE, GCUT, PATHS
    img, mask, swc, output, percentile, threshold, errors, crop, soma, meta = args[:10]
    cid = int(Path(img).stem.replace('image_', '').replace('_0000', ''))
    selected = args[10] if len(args) == 11 else PATHS['gcut_selected_swc_dir']
    folder = Path(output) / str(cid)
    folder.mkdir(parents=True, exist_ok=True)
    job = dict(id=cid, paths=list(map(str, [img, mask, swc, folder, errors, crop, soma, selected])),
               percentile=float(percentile), thickness_um=float(threshold),
               metadata=meta.loc[cid].to_dict(), gcut=dict(GCUT))
    manifest = folder/'worker-spec.json'
    _write_json(manifest, job)
    env = dict(os.environ, NEURON_V082_CONFIG=str(CONFIG_FILE),
               RECONNECT_COMPUTE_BACKEND='cpu',
               RECONNECT_GPU_STATS=str(folder/'gcut-gpu.json'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    if GCUT.get('gpu_lock_path'):
        env['RECONNECT_GPU_LOCK_PATH'] = str(GCUT['gpu_lock_path'])
    started = time.monotonic()
    print(f'[gcut] {cid}: APP2 GSDT / 原始合并mask，开始', flush=True)
    with (folder/'worker.log').open('w') as stream:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                  '--job', str(manifest)], env=env, cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT)
    elapsed = time.monotonic()-started
    result_path = folder/'worker-result.json'
    if process.returncode:
        tail = (folder/'worker.log').read_text(errors='replace')[-3000:]
        raise RuntimeError(f'G-Cut {cid} failed; log: {folder/"worker.log"}\n{tail}')
    result = json.loads(result_path.read_text())
    if not result['success']:
        raise RuntimeError(result['message'])
    phases = json.loads((folder/'gcut-timing.json').read_text())['seconds']
    for name, seconds in phases.items():
        print(f'[耗时] G-Cut {cid} / {name}: {seconds:.2f} 秒', flush=True)
    print(f'[耗时] G-Cut {cid} 总计: {elapsed:.2f} 秒；结果: {folder}', flush=True)
    return result['result']


def _run_job(manifest):
    import importlib.util
    import numpy as np
    import pandas as pd
    import tifffile
    from algorithms.tracing.point_exports import write_points
    job = json.loads(Path(manifest).read_text())
    img, mask, swc, folder, errors, crop, soma, selected = map(Path, job['paths'])
    errors.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('gcut_engine', ROOT/'algorithms/tracing/gcut_stage.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    if not engine.GCUT_AVAILABLE:
        raise RuntimeError(str(engine.GCUT_IMPORT_ERROR))
    cfg = job['gcut']
    engine.CANDIDATE_EVIDENCE = cfg.get('candidate_evidence', {'enabled': True})
    engine.CANDIDATE_SUPPORT_MASK = str(mask)
    engine.GCUT_CANDIDATE_METHOD = cfg.get('candidate_method', 'app2_gsdt')
    engine.GCUT_CANDIDATE_BACKEND = cfg.get('candidate_backend', 'cuda')
    if engine.GCUT_CANDIDATE_METHOD not in ('app2_gsdt', 'edt'):
        raise ValueError('gcut.candidate_method must be app2_gsdt or edt')
    if engine.GCUT_CANDIDATE_BACKEND not in ('cpu', 'cuda'):
        raise ValueError('gcut.candidate_backend must be cpu or cuda')
    # Keep v0.81's compatibility names for its resampling/pruning stages.
    engine.selected_swc_path = lambda directory, image: Path(directory)/(Path(image).stem.replace('_0000', '')+'.swc')
    original_summary = engine.save_comprehensive_summary
    def summary(*args, **kwargs):
        kwargs['individual_output_dir'] = None
        return original_summary(*args, **kwargs)
    engine.save_comprehensive_summary = summary
    meta = pd.DataFrame([job['metadata']], index=[job['id']])
    result = engine.worker_task((img, mask, swc, folder, job['percentile'], job['thickness_um'],
                                 errors, crop, soma, meta, selected))
    target_file = selected/(img.stem.replace('_0000', '')+'.swc')
    targets = list(folder.glob('*_gcut_soma_*_TARGET.swc'))
    if result[1] is None or len(targets) != 1 or not target_file.exists():
        raise RuntimeError(f'No unique G-Cut target produced: {result}')
    nms = json.loads((folder/'nms-audit.json').read_text())
    context = json.loads((folder/'soma-context.json').read_text())
    evidence = json.loads((folder/'candidate-evidence.json').read_text())
    with tifffile.TiffFile(img) as tif:
        shape = np.asarray(tif.series[0].shape)
    for coords, name in [
        (evidence['before_filter_zyx'], 'before_evidence_points.swc'),
        (evidence['rejected_zyx'], 'rejected_candidates.swc'),
        (nms['pre_nms_zyx'], 'pre_nms_points.swc'),
        (nms['after_distance_nms_zyx'], 'distance_nms_only_points.swc'),
        (nms['after_signal_merge_zyx'], 'post_nms_points.swc')]:
        write_points(folder/name, coords, context['target_zyx'], shape, img, name)
    shutil.copy2(target_file, folder/'gcut_target.swc')
    shutil.copy2(folder/(img.stem+'_comprehensive_vis.png'), folder/'overview.png')
    for name, source in [('original_image.tif', img), ('input_mask.tif', mask)]:
        link = folder/name
        if link.is_symlink(): link.unlink()
        if not link.exists(): link.symlink_to(source.resolve())
    _write_json(folder/'worker-result.json', dict(success=True, result=result,
                method=engine.GCUT_CANDIDATE_METHOD, mask_repair=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    try:
        _run_job(args.job)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
