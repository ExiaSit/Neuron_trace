"""Numbered v0.83 reconstruction stage; run_pipeline.py is the batch entry."""
from pathlib import Path
import json, shutil, sys, time
import numpy as np
import tifffile
from utils import bootstrap
from utils.case_worker import setup_config, load_module, check_swc, save_json, comparison
from utils import mask_batch as mask_run

def prepare(spec, state, stage_run):
    """Six ordered mask repair passes; each pass has its own resume checkpoint."""
    p=setup_config(spec);work=p['root']/'mask_steps'/spec['id']
    if not state.get('mask_saved') or not p['mask'].is_file():
        from visualization.diagnostics import record_mask_step
        for index,file in enumerate(mask_run.STAGES):
            letter='abcdef'[index]
            stage_run('mask_'+letter,
                      [sys.executable,str(bootstrap.ROOT/'algorithms/mask_repair'/file)],
                      [work/letter/'connected-mask.tif',work/letter/'report.json'])
            before=Path(spec['mask']) if index==0 else work/'abcdef'[index-1]/'connected-mask.tif'
            record_mask_step(spec,index,before,work/letter/'connected-mask.tif')
        original=tifffile.imread(spec['mask'])>0
        result=tifffile.imread(work/'f/connected-mask.tif')
        mask_run.validate_result(result,original)
        mask_run.atomic_tiff(p['mask'],result)
        state['mask_saved']=True
        save_json(p['root']/'status'/f"{spec['id']}.json",state)

def run(spec):
    p=setup_config(spec);cfg=spec['config'];r=p['root'];cid=spec['id']
    import pandas as pd
    meta=pd.DataFrame([spec['metadata']], index=[int(cid)])
    from algorithms.mask_repair.guarded_continuation import extend
    m=spec['metadata'];xyz=[float(m[k])*float(m[res])/1000 for k,res in [('soma_x','xy_resolution'),('soma_y','xy_resolution'),('soma_z','z_resolution')]]
    raw=tifffile.imread(p['raw']);mask=tifffile.imread(p['mask'])>0
    from visualization.diagnostics import record_mask_step
    work=r/'mask_steps'/cid
    mask_run.atomic_tiff(work/'continuation_input/connected-mask.tif',mask.astype('uint8')*255)
    if cfg.get('review_supplement_only'):
        from algorithms.mask_repair.root_geodesic import supplement
        result,audit=supplement(raw,mask,xyz,p['soma_json'],p['soma_mask'])
        report=dict(geodesic_supplement=audit,original_preserved=bool(np.all(result[mask])),added_voxels=int(np.count_nonzero(result&~mask)))
    else:
        result,report=extend(raw,mask,xyz,p['soma_json'],p['soma_mask'])
        mask_run.atomic_tiff(work/'g/connected-mask.tif',result.astype('uint8')*255)
        record_mask_step(spec,6,work/'continuation_input/connected-mask.tif',work/'g/connected-mask.tif')
        # Preserve the validated curved soma-root repair in fresh runs too.
        from algorithms.mask_repair.root_geodesic import supplement
        result,root_audit=supplement(raw,result,xyz,p['soma_json'],p['soma_mask'])
        report['geodesic_supplement']=root_audit
        report['added_voxels']=int(np.count_nonzero(result&~mask))
    mask_run.validate_result(result.astype('uint8')*255,mask)
    mask_run.atomic_tiff(work/'h/connected-mask.tif',result.astype('uint8')*255)
    record_mask_step(spec,7,work/('continuation_input' if cfg.get('review_supplement_only') else 'g')/'connected-mask.tif',work/'h/connected-mask.tif')
    mask_run.atomic_tiff(p['mask'],result.astype('uint8')*255)
    save_json(r/'logs'/cid/'continuity.json',report)

if __name__=="__main__":
    import argparse
    from utils.case_worker import run_worker
    parser=argparse.ArgumentParser(description='六轮 mask 修复、受约束延伸与根部补全')
    parser.add_argument('--spec',required=True)
    args=parser.parse_args()
    run_worker(Path(args.spec).resolve(),mask_only=True)
