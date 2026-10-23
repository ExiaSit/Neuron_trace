from pathlib import Path
import json
import numpy as np
import tifffile
from utils import bootstrap
from utils.mask_batch import atomic_tiff
from utils.case_worker import save_json

def run(spec):
    from utils.preparation import inference
    inference([spec],'soma')

if __name__=="__main__":
    from utils.preparation import cli
    cli(4)
