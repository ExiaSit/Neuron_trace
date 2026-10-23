from pathlib import Path
import json
import numpy as np
import tifffile
from utils import bootstrap
from utils.mask_batch import atomic_tiff
from utils.case_worker import save_json

def run(spec):
    from utils.v3d_io import load_image
    image=load_image(spec['source'],normalize=True)
    if image.ndim!=3 or image.dtype!=np.uint8:raise ValueError('V3D must decode to a 3D uint8 volume')
    atomic_tiff(Path(spec['normalized']),image)

if __name__=="__main__":
    from utils.preparation import cli
    cli(1)
