from pathlib import Path
import json
import numpy as np
import tifffile
from utils import bootstrap
from utils.mask_batch import atomic_tiff
from utils.case_worker import save_json

def run(spec):
    from algorithms.preprocessing import resample_1um
    m=spec['metadata'];image=tifffile.imread(spec['normalized'])
    output=resample_1um(image,m['xy_resolution'],m['z_resolution'])
    atomic_tiff(Path(spec['raw']),output)
    save_json(Path(spec['raw']).with_name(spec['id']+'.json'),{'spacing':[1.,1.,1.]})

if __name__=="__main__":
    from utils.preparation import cli
    cli(2)
