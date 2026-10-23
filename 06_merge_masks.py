from pathlib import Path
import json
import numpy as np
import tifffile
from utils import bootstrap
from utils.mask_batch import atomic_tiff
from utils.case_worker import save_json

def run(spec):
    neurite=tifffile.imread(spec['neurite_mask'])>0
    soma=tifffile.imread(spec['soma_mask'])>0
    b=json.loads(Path(spec['soma_json']).read_text())['bounds_1um_zxy']
    region=tuple(slice(b[i],b[i+1]) for i in (0,2,4))
    shape=tuple(b[i+1]-b[i] for i in (0,2,4))
    if neurite[region].shape!=shape or any(n>s for n,s in zip(shape,soma.shape)):raise ValueError('Soma bounds/shape mismatch')
    neurite[region]|=soma[tuple(slice(0,n) for n in shape)]
    atomic_tiff(Path(spec['mask']),neurite.astype(np.uint8)*255)

if __name__=="__main__":
    from utils.preparation import cli
    cli(6)
