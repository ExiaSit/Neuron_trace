"""Numbered v0.83 reconstruction stage; run_pipeline.py is the batch entry."""
from pathlib import Path
import json, shutil, sys, time
import numpy as np
import tifffile
from utils import bootstrap
from utils.case_worker import setup_config, load_module, check_swc, save_json, comparison
from utils import mask_batch as mask_run

def run(spec):
    p=setup_config(spec);cfg=spec['config'];r=p['root'];cid=spec['id']
    import pandas as pd
    meta=pd.DataFrame([spec['metadata']], index=[int(cid)])
    mod=load_module('gcut')
    if not mod.GCUT_AVAILABLE:raise RuntimeError(str(mod.GCUT_IMPORT_ERROR))
    summary_path=r/'gcut_mips'/f'{cid}.png';summary_path.parent.mkdir(parents=True,exist_ok=True)
    original_summary=mod.save_comprehensive_summary
    def save_summary(*args, **kwargs):
        args=list(args);args[10]=summary_path
        kwargs['individual_output_dir']=None
        return original_summary(*args, **kwargs)
    mod.save_comprehensive_summary=save_summary
    vis=r/'gcut_output'/cid
    if vis.exists():vis.rename(vis.with_name(f'{cid}_retry_{time.time_ns()}'))
    if p['selected'].exists():p['selected'].rename(p['selected'].with_name(f'{cid}_retry_{time.time_ns()}.swc'))
    vis.mkdir(parents=True,exist_ok=True)
    result=mod.worker_task((p['raw'],p['mask'],p['app2'],vis,55.,5.,r/'logs'/f'{cid}_gcut_errors.log',p['soma_json'],p['soma_mask'],meta,p['selected'].parent))
    if (vis/'soma-merge-audit.json').exists():shutil.copy2(vis/'soma-merge-audit.json',r/'logs'/cid/'soma-merge-audit.json')
    if (vis/'soma-context.json').exists():shutil.copy2(vis/'soma-context.json',r/'logs'/cid/'soma-context.json')
    if (vis/'short-leaf-audit.json').is_file():
        from visualization.diagnostics import write_short_leaf_audit
        write_short_leaf_audit(json.loads((vis/'short-leaf-audit.json').read_text()),tifffile.imread(p['raw']),r,cid)
    if result[1] is None:raise RuntimeError(str(result))
    # Require an explicitly identified target instead of accepting an arbitrary split.
    targets=list(vis.glob(f'{p["raw"].stem}_gcut_soma_*_TARGET.swc'))
    if len(targets)!=1:raise RuntimeError(f'Expected one G-Cut target, found {len(targets)}: {result}')
    shutil.copy2(targets[0],p['selected']);check_swc(p['selected'])

if __name__=="__main__":
    from utils.stage_cli import run_spec
    run_spec(run)
