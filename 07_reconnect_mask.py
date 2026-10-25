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
            if letter in ('b','d'):
                from collections import Counter
                report=json.loads((work/letter/'report.json').read_text())
                save_json(p['root']/'logs'/spec['id']/f'mask-{letter}-mip.json',dict(
                    backmap=report.get('backmap',{}),connections=len(report.get('accepted',[])),
                    rejected=dict(Counter(x.get('reason','unknown') for x in report.get('rejected',[])))))
        original=tifffile.imread(spec['mask'])>0
        result=tifffile.imread(work/'f/connected-mask.tif')
        mask_run.validate_result(result,original)
        mask_run.atomic_tiff(p['mask'],result)
        state['mask_saved']=True
        save_json(p['root']/'status'/f"{spec['id']}.json",state)

def run(spec):
    """Finish mask repair with guarded continuation (g); no geodesic pass (h)."""
    p=setup_config(spec);r=p['root'];cid=spec['id']
    from algorithms.mask_repair.guarded_continuation import extend
    m=spec['metadata'];xyz=[float(m[k])*float(m[res])/1000 for k,res in [('soma_x','xy_resolution'),('soma_y','xy_resolution'),('soma_z','z_resolution')]]
    raw=tifffile.imread(p['raw']);mask=tifffile.imread(p['mask'])>0
    from visualization.diagnostics import record_mask_step
    work=r/'mask_steps'/cid
    before=work/'continuation_input/connected-mask.tif'
    after=work/'g/connected-mask.tif'
    mask_run.atomic_tiff(before,mask.astype('uint8')*255)
    result,report=extend(raw,mask,xyz,p['soma_json'],p['soma_mask'])
    result=result.astype('uint8')*255
    mask_run.validate_result(result,mask)
    mask_run.atomic_tiff(after,result)
    record_mask_step(spec,6,before,after)
    mask_run.atomic_tiff(p['mask'],result)
    report.update(root_geodesic_enabled=False,final_mask_step='g',
                  original_preserved=bool(np.all(result[mask]>0)),
                  added_voxels=int(np.count_nonzero((result>0)&~mask)))
    save_json(r/'logs'/cid/'continuity.json',report)

if __name__=="__main__":
    import argparse
    from utils.case_worker import run_worker
    parser=argparse.ArgumentParser(description='六轮 mask 修复与受约束延伸（不执行根部测地线补全）')
    parser.add_argument('--spec',required=True)
    args=parser.parse_args()
    run_worker(Path(args.spec).resolve(),mask_only=True)
