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
    comparison(spec)

if __name__=="__main__":
    from utils.stage_cli import run_spec
    run_spec(run)
