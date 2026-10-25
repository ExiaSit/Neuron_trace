"""Backfill a finished job's visualizations while preserving reconstruction checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils import bootstrap
import numpy as np
import tifffile
from utils.case_worker import paths, save_json, setup_config, load_module
from visualization.diagnostics import record_mask_step, mask_transition, MASK_TITLES, write_short_leaf_audit
from visualization.comparison import comparison


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def replay_continuation_steps(spec, raw, before):
    """Replay the steps recorded by that run, never silently add a removed step."""
    from algorithms.mask_repair.guarded_continuation import extend
    p=paths(spec)
    audit_path=p['root']/'logs'/spec['id']/'continuity.json'
    audit=json.loads(audit_path.read_text())
    if audit.get('root_geodesic_enabled') is False:
        historical_h=False
    elif 'geodesic_supplement' in audit:
        historical_h=True
    else:
        raise ValueError('Original continuity audit does not identify mask steps; refusing to guess')
    m=spec['metadata'];xyz=[float(m[k])*float(m[res])/1000 for k,res in
                           [('soma_x','xy_resolution'),('soma_y','xy_resolution'),('soma_z','z_resolution')]]
    if historical_h and spec['config'].get('review_supplement_only'):
        from algorithms.mask_repair.root_geodesic import supplement
        final,_=supplement(raw,before,xyz,p['soma_json'],p['soma_mask'])
        return [(7,before,final)]
    middle,_=extend(raw,before,xyz,p['soma_json'],p['soma_mask'])
    steps=[(6,before,middle)]
    if historical_h:
        from algorithms.mask_repair.root_geodesic import supplement
        final,_=supplement(raw,middle,xyz,p['soma_json'],p['soma_mask'])
        steps.append((7,middle,final))
    return steps


def backfill(spec, replay=False):
    p=paths(spec);root=p['root'];cid=spec['id'];work=root/'mask_steps'/cid
    core=[p[k] for k in ['mask','app2','selected','resampled','final'] if p[k].is_file()]
    original={str(path):sha(path) for path in core};results={}
    for index,letter in enumerate('abcdef'):
        before=Path(spec['mask']) if index==0 else work/'abcdef'[index-1]/'connected-mask.tif'
        after=work/letter/'connected-mask.tif'
        if not before.is_file() or not after.is_file():raise ValueError(f'Missing snapshot: {before} or {after}')
        results[letter]=record_mask_step(spec,index,before,after,'retained historical snapshots')
        print(f'{cid}: mask {letter}: {results[letter]}',flush=True)
    raw=tifffile.imread(p['raw']);before=tifffile.imread(work/'f/connected-mask.tif')>0
    if replay:
        steps=replay_continuation_steps(spec,raw,before)
        if not np.array_equal(steps[-1][2],tifffile.imread(p['mask'])>0):
            raise ValueError('Replay differs from saved final mask; refusing to publish separated steps')
        for index,prev,next_mask in steps:
            letter='abcdefgh'[index]
            results[letter]=mask_transition(raw,prev,next_mask,root/'mask_step_mips'/f'{cid}_{index+1:02d}_{letter}.png',
                f'{cid} | {index+1:02d} {MASK_TITLES[index]}',
                dict(replay_input=str(work/'f/connected-mask.tif'),verified_final=str(p['mask']),
                     verified_final_sha256=original[str(p['mask'])],final_voxel_equality=True),
                'replayed; final mask exactly matches saved result')
            print(f'{cid}: replay mask {letter}: {results[letter]}',flush=True)
    else:
        mask_transition(raw,before,tifffile.imread(p['mask']),root/'mask_step_mips'/f'{cid}_07_combined.png',
                        f'{cid} | Final repair changes since step f',provenance='combined; intermediate unavailable')
    setup_config(spec);load_module('gcut')
    from algorithms.tracing.gcut_adapter import make_segmentor
    vis=root/'gcut_output'/cid
    injected=vis/f'{p["raw"].stem}_temp_injected.swc';somas=vis/f'{p["raw"].stem}_soma_ids.txt'
    segmentor=make_segmentor(str(injected),str(somas))
    audit=dict(segmentor.short_leaf_audit,inputs={str(x):sha(x) for x in [injected,somas]})
    write_short_leaf_audit(audit,raw,root,cid,'replayed from retained G-Cut input; preprocessing only')
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        spec=dict(spec,config=dict(spec['config'],comparison_dir=tmp))
        comparison(spec)
        Path(tmp,f'{cid}.png').replace(root/'comparisons'/f'{cid}_diagnostics.png')
    if {str(path):sha(path) for path in core}!=original:raise RuntimeError('Reconstruction output changed during refresh')
    save_json(root/'comparison_sources'/f'{cid}_diagnostic_refresh.json',dict(
        core_sha256=original,core_unchanged=True,mask_steps=results,gcut_removed_nodes=len(audit['removed_ids']),
        comparison='separate diagnostic PNG; original stage-13 checkpoint preserved'))
    print('Verified: original mask and all four SWC outputs unchanged.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec',required=True)
    ap.add_argument('--previous-app2-dir',action='append')
    ap.add_argument('--replay-continuation',action='store_true')
    args=ap.parse_args();spec=json.loads(Path(args.spec).read_text())
    if args.previous_app2_dir:spec['config']['previous_app2_dirs']=args.previous_app2_dir
    backfill(spec,args.replay_continuation)
