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
    mod=load_module('resample');work=r/'gcut_output/resample_intermediates'/cid;work.mkdir(parents=True,exist_ok=True)
    raw=work/'resampled.swc';sorted_path=work/'sorted_plugin.swc'
    mod.resample_swc(p['selected'],raw,1.,cfg['vaa3d_resample'])
    mod._run_vaa3d(cfg['vaa3d_resample'],'sort_neuron_swc','sort_swc',raw,sorted_path,timeout=300)
    frame=mod._read_swc_dataframe(sorted_path);frame['type']=3;frame.loc[frame.index[0],'type']=1
    frame.to_csv(p['resampled'],sep=' ',index=False);check_swc(p['resampled'])

if __name__=="__main__":
    from utils.stage_cli import run_spec
    run_spec(run)
