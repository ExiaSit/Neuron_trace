"""Isolated per-ID incremental reconstruction; all stages are persisted."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import types

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parents[1]

from algorithms.tracing import one_sided
from utils import mask_batch as mask_run
from utils.output_cleanup import cleanup


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.writing.json')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str)+'\n')
    os.replace(tmp, path)


def load_module(name):
    spec = importlib.util.spec_from_file_location('incremental_'+name, HERE/'algorithms/tracing'/f'{"gcut_stage" if name=="gcut" else name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paths(spec):
    root = Path(spec['config']['output_root']); cid = spec['id']
    direct=spec['config'].get('direct_inputs',False)
    return dict(root=root, raw=Path(spec['raw']) if direct else root/'img'/f'{cid}_0000.tif', mask=root/'mask'/f'{cid}.tif',
                original_mask=root/'mask_original'/f'{cid}.tif',
                soma_json=Path(spec['soma_json']) if direct else root/'soma_img'/f'{cid}.json',
                soma_mask=Path(spec['soma_mask']) if direct else root/'soma_seg'/f'{cid}.tif',
                app2=root/'trace_app2/down_sampled_swcs_app2'/f'{cid}.swc',
                selected=root/'gcut_output/selected_for_pruning'/f'{cid}.swc',
                resampled=root/'gcut_output/selected_resampled_step1'/f'{cid}.swc',
                final=root/'pruning_swcs'/f'{cid}.swc',
                mip=Path(spec['config']['output_mip_dir'])/f'{cid}.png',
                comparison=Path(spec['config']['comparison_dir'])/f'{cid}.png')


def setup_config(spec):
    cfg=spec['config'];p=paths(spec);r=p['root']
    mod=types.ModuleType('pipeline_config');mod.BASE_DIR=r
    mod.GCUT=dict(target_percentile=55.,gsdt_threshold_x=5.)
    mod.PATHS=dict(image_1um_dir=r/'img',merged_mask_dir=r/'mask',trace_output_dir=r/'trace_app2',
        trace_swc_dir=p['app2'].parent,trace_marker_dir=r/'trace_app2/markers',trace_vis_dir=r/'trace_app2/vis_mip',
        pipeline_skip_log=r/'logs/pipeline_skipped.log',trace_timeout_log=r/'logs/app2_timeout.log',
        trace_error_log=r/'logs/app2_error.log',trace_cost_log=r/'logs/app2_cost.log',
        vaa3d_path=cfg['vaa3d_app2'],vaa3d_resample_path=cfg['vaa3d_resample'],gcut=str(HERE/'vendor/gcut'))
    sys.modules['pipeline_config']=mod
    for key in ['mask','app2','selected','resampled','final','mip','comparison']:p[key].parent.mkdir(parents=True,exist_ok=True)
    return p


def check_swc(path):
    nodes, _ = one_sided.parse_swc(path)
    if any(not np.isfinite([n.x,n.y,n.z,n.radius]).all() for n in nodes.values()):
        raise ValueError(f'Non-finite SWC: {path}')
    return nodes


from visualization.comparison import comparison


def execute_stage(spec, stage):
    from utils.stages import load_stage
    return load_stage(stage).run(spec)


def run_worker(spec_path, mask_only=False):
    spec=json.loads(Path(spec_path).read_text());p=paths(spec);r=p['root'];cid=spec['id']
    status_path=r/'status'/f'{cid}.json';state={'id':cid,'fingerprint':spec['fingerprint'],'completed_stages':[],'status':'running'}
    if status_path.exists():
        previous=json.loads(status_path.read_text())
        if previous.get('fingerprint')!=spec['fingerprint']:
            raise ValueError('Inputs/config/code changed for existing ID; choose a fresh output_root instead of mixing versions')
        state=previous;state['status']='running'
    if state.get('comparison_version') != 2:
        state['completed_stages']=[x for x in state['completed_stages'] if x!='comparison']
    work=r/'mask_steps'/cid;work.mkdir(parents=True,exist_ok=True)
    logs=r/'logs'/cid;logs.mkdir(parents=True,exist_ok=True)
    from algorithms.mask_repair.multiscale_mip import configuration
    mip_config=configuration(spec['config'].get('mask_mip', {}))
    env=os.environ.copy();env.update(RECONNECT_RAW=spec['raw'],RECONNECT_MASK=spec['mask'],RECONNECT_WORK=str(work),RECONNECT_MIP_CONFIG=json.dumps(mip_config),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    from algorithms.mask_repair.local_pair_mip import configuration as pair_configuration
    env['RECONNECT_LOCAL_PAIR_MIP_CONFIG']=json.dumps(pair_configuration(spec['config'].get('local_pair_mip',{})))
    backend=spec['config'].get('mask_compute_backend','cuda')
    minimum=spec['config'].get('mask_gpu_min_voxels',262144)
    if backend not in ('cpu','cuda'):raise ValueError('mask_compute_backend must be cpu or cuda')
    if type(minimum) is not int or minimum<1:raise ValueError('mask_gpu_min_voxels must be a positive integer')
    env.update(RECONNECT_COMPUTE_BACKEND=backend,RECONNECT_GPU_MIN_VOXELS=str(minimum))
    def stage_run(name, command, expected):
        if name in state['completed_stages'] and all(x.is_file() and x.stat().st_size>0 for x in expected):return
        state['current_stage']=name;save_json(status_path,state)
        print(f'[{cid}] {name} 开始',flush=True);start=time.monotonic()
        stage_env=env.copy()
        stage_env['RECONNECT_COMPUTE_BACKEND']=backend if name.startswith('mask_') or name=='continuity' else 'cpu'
        stage_env['RECONNECT_GPU_STATS']=str(logs/f'{name}-gpu.json')
        with (logs/f'{name}.log').open('w') as stream:
            child=subprocess.run(command,env=stage_env,stdout=stream,stderr=subprocess.STDOUT,cwd=work)
        if child.returncode:raise RuntimeError(f'{name} failed: {(logs/f"{name}.log").read_text()[-2500:]}')
        if not all(x.is_file() and x.stat().st_size>0 for x in expected):raise RuntimeError(f'{name} missing output')
        if name not in state['completed_stages']:state['completed_stages'].append(name)
        state.setdefault('seconds',{})[name]=round(time.monotonic()-start,2);save_json(status_path,state)
    try:
        from utils.stages import load_stage
        load_stage('continuity').prepare(spec,state,stage_run)
        stage_run('continuity',[sys.executable,str(Path(__file__).resolve()),'--spec',str(spec_path),'--stage','continuity'],[p['mask'],r/'logs'/cid/'continuity.json'])
        mask_run.make_mip((Path(spec['raw']),Path(spec['mask']),p['mask']),p['mip'])
        if mask_only:
            state['status']='mask_complete';save_json(status_path,state)
            return
        for stage,expected in [('app2',p['app2']),('gcut',p['selected']),('resample',p['resampled']),('prune',p['final']),('guard',r/'regression_guard'/f'{cid}.json'),('comparison',p['comparison'])]:
            required=[expected]
            if stage=='gcut':required.append(r/'gcut_mips'/f'{cid}.png')
            if stage=='prune':required.append(r/'pruning_mips'/f'{cid}_combined_process.png')
            stage_run(stage,[sys.executable,str(Path(__file__).resolve()),'--spec',str(spec_path),'--stage',stage],required)
        check_swc(p['final']);state['comparison_version']=2;state['status']='complete';state.pop('error',None);save_json(status_path,state);cleanup(r,cid)
    except BaseException as exc:
        state['status']='failed';state['error']=str(exc);save_json(status_path,state)
        try:
            comparison(spec, failure=state.get('current_stage','unknown'))
            state['comparison_version']=2;save_json(status_path,state)
        except Exception as plotting_error:
            state['comparison_error']=str(plotting_error);save_json(status_path,state)
        cleanup(r,cid)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--stage');args=parser.parse_args()
    try:
        if args.stage:execute_stage(json.loads(Path(args.spec).read_text()),args.stage)
        else:run_worker(Path(args.spec).resolve())
    except Exception:
        traceback.print_exc();sys.exit(1)
