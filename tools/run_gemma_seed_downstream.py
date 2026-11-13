#!/usr/bin/env python3
"""Paired downstream experiment from identical frozen APP2, raw and mask inputs."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import bootstrap
from utils.case_worker import setup_config, load_module, save_json
from utils.stages import load_stage
from algorithms.tracing.soma_branchwalk import tree_index


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-run', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--render-workers', type=int, default=1)
    args = parser.parse_args()
    if args.render_workers < 1:
        parser.error('render-workers must be positive')
    if Path(args.run_name).name != args.run_name:
        parser.error('run-name must be a single directory name')
    base = ROOT.parent/'test/0.84'
    data = base/'data'
    seed_run = args.seed_run.resolve()
    source = read(seed_run/'input.json')
    cell = source['cell']
    catalogue = read(seed_run/'normalized/soma_catalogue.json')
    context = dict(target_zyx=next(p['zyx'] for p in catalogue['accepted'] if p['candidate_id']=='T'),
                   centers_zyx=[p['zyx'] for p in catalogue['accepted']],
                   radii=[p.get('gsdt_score') for p in catalogue['accepted']],
                   seed_ids=[p['candidate_id'] for p in catalogue['accepted']],
                   source_catalogue=str(seed_run/'normalized/soma_catalogue.json'),
                   unresolved_candidates_not_used=catalogue['review'],
                   policy='accepted Gemma soma seeds plus known T; no display radii')
    inputs = dict(raw=Path(source['raw']),mask=Path(source['mask']),
                  app2=data/f'source_v083b/trace_app2/down_sampled_swcs_app2/{cell}.swc',
                  soma_json=data/f'inputs/soma_img/{cell}.json',
                  soma_mask=data/f'inputs/soma_seg/{cell}.tif',
                  catalogue=seed_run/'normalized/soma_catalogue.json')
    for key in ['raw','mask']:
        if sha(inputs[key]) != source[key+'_sha256']:
            raise ValueError(f'{key} changed since Gemma inference')
    out = base/'runs'/args.run_name
    out.mkdir(exist_ok=True)
    files = ['tools/run_gemma_seed_downstream.py','algorithms/tracing/gcut_stage.py',
             'algorithms/tracing/soma_seed_integration.py','algorithms/tracing/gcut_adapter.py',
             'algorithms/tracing/soma_signal_groups.py','algorithms/tracing/soma_seed_ownership.py',
             'algorithms/tracing/resample.py','algorithms/tracing/prune.py',
             'algorithms/tracing/terminal_pruning.py','algorithms/tracing/connection_diagnostics.py',
             'visualization/gcut_summary.py','10_resample_swc.py','11_prune_swc.py','12_guard_regression.py']
    config_path=data/f'provenance/{cell}/run-config.json'
    if not config_path.exists() and cell=='60003':
        config_path=data/'provenance/job-config.json'
    original_cfg=read(config_path)
    manifest = dict(cell=cell,seed_run=str(seed_run),inputs={k:str(v) for k,v in inputs.items()},
                    hashes={k:sha(v) for k,v in inputs.items()},code={f:sha(ROOT/f) for f in files},
                    configuration_path=str(config_path),configuration_sha256=sha(config_path),render_workers=args.render_workers,
                    comparison='same APP2/raw/mask and downstream rules; automatic seeds vs Gemma accepted seeds',
                    nnunet_rerun=False,app2_rerun=False,new_model_calls=0)
    if (out/'run.json').exists() and read(out/'run.json') != manifest:
        raise ValueError('Input/code changed: use a new run')
    save_json(out/'run.json',manifest)
    save_json(out/'gemma_seed_context.json',context)
    for file in files:
        dest=out/'source'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,dest)
    for arm in ['legacy_seeds','gemma_seeds']:
        arm_root=out/arm
        arm_root.mkdir(exist_ok=True)
        if (arm_root/'complete.json').exists():
            continue
        cfg=dict(original_cfg,output_root=str(arm_root),direct_inputs=True,
                 output_mip_dir=str(arm_root/'mask_mips'),comparison_dir=str(arm_root/'comparisons'),
                 method_revision='gemma-seed-downstream-paired-20261113')
        spec=dict(id=cell,config=cfg,metadata=source['metadata'],raw=str(inputs['raw']),
                  soma_json=str(inputs['soma_json']),soma_mask=str(inputs['soma_mask']))
        p=setup_config(spec)
        shutil.copy2(inputs['mask'],p['mask'])
        shutil.copy2(inputs['app2'],p['app2'])
        (arm_root/'logs'/cell).mkdir(parents=True,exist_ok=True)
        save_json(arm_root/'spec.json',spec)
        timing={}
        print(f'{arm}: G-Cut starting',flush=True)
        save_json(out/'status.json',dict(arm=arm,stage='gcut',status='running'))
        start=time.monotonic()
        with (arm_root/'logs/gcut.log').open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            mod=load_module('gcut')
            if not mod.GCUT_AVAILABLE:
                raise RuntimeError(str(mod.GCUT_IMPORT_ERROR))
            folder=arm_root/'gcut_output'/cell;folder.mkdir(parents=True,exist_ok=True)
            result=mod.process_image_to_gcut(inputs['raw'],p['mask'],p['app2'],folder,
                      context['target_zyx'],55.,5.,inputs['soma_json'],inputs['soma_mask'],
                      soma_seed_override=context if arm=='gemma_seeds' else None)
            print(result,flush=True)
            targets=list(folder.glob('*_TARGET.swc'))
            if len(targets)!=1 or '失败' in result[1]:
                raise RuntimeError(f'G-Cut failed to identify one target: {result[1]}')
            shutil.copy2(targets[0],p['selected'])
            for name in ['soma-context.json','soma-merge-audit.json','target-extension-ownership.json']:
                shutil.copy2(folder/name,arm_root/'logs'/cell/name)
        timing['gcut']=time.monotonic()-start
        for stage,key in [('resample','resampled'),('prune','final'),('guard','final')]:
            print(f'{arm}: {stage} starting',flush=True)
            save_json(out/'status.json',dict(arm=arm,stage=stage,status='running'))
            start=time.monotonic()
            with (arm_root/'logs'/f'{stage}.log').open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                if stage=='prune':
                    from visualization.connection_diagnostics import configure_render_workers,finish_rendering
                    configure_render_workers(args.render_workers)
                    try:load_stage(stage).run(spec)
                    finally:finish_rendering()
                else:
                    load_stage(stage).run(spec)
                tree_index(np.loadtxt(p[key],ndmin=2).tolist())
            timing[stage]=time.monotonic()-start
        save_json(arm_root/'complete.json',dict(seconds=timing,final=str(p['final']),
                   target_tree_valid=True,selected_seed_count=len(read(folder/'soma-context.json')['centers_zyx'])))
        print(f'{arm}: complete {timing}',flush=True)
    unchanged={k:sha(v)==manifest['hashes'][k] for k,v in inputs.items()}
    if not all(unchanged.values()):
        raise ValueError('Source input changed during execution')
    save_json(out/'validation.json',dict(inputs_unchanged=unchanged,arms_completed=True,
              biological_accuracy_evaluated=False,display_radius_used_for_gcut=False))
    save_json(out/'status.json',dict(status='downstream_complete',report_pending=True))
    print(str(out),flush=True)


if __name__=='__main__':
    main()
