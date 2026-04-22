from neuroutils.swc.parser import resample_swc_file, rescale_swc
import os
from joblib import Parallel, delayed
from neuroutils.config.settings import DEFAULT_NUM_THREADS
from tqdm import tqdm
from neuroutils.meta.mapping import extract_neuron_id
from neuroutils.meta.neuron import get_neuron_meta
from neuroutils.swc.io import load_swc, save_swc

origin_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled/retrace_swcs_filtered"
target_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled/retrace_swcs_rescaled"
os.makedirs(target_dir, exist_ok=True)
def do_rescale(swc_in, swc_out):
    neuron_id = extract_neuron_id(swc_in)
    meta = get_neuron_meta(neuron_id)
    xy_resolution, z_resolution = meta['xy_resolution'].values[0], meta['z_resolution'].values[0]
    xy_resolution = float(xy_resolution) / 1000 * 2
    z_resolution = float(z_resolution) / 1000 * 2

    swc_points = load_swc(swc_in)
    rescale_swc(
        swc_points,
        xy_resolution=xy_resolution,
        z_resolution=z_resolution,
    )
    save_swc(swc_points, swc_out)

swc_files = [f for f in os.listdir(origin_dir) if f.endswith('.swc')]

Parallel(n_jobs=DEFAULT_NUM_THREADS)(
    delayed(do_rescale)(
        os.path.join(origin_dir, swc_file),
        os.path.join(target_dir, swc_file),
    ) for swc_file in tqdm(swc_files)
)


origin_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled/retrace_swcs_rescaled"
target_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled/retrace_swcs_resampled"
os.makedirs(target_dir, exist_ok=True)

swc_files = [f for f in os.listdir(origin_dir) if f.endswith('.swc')]

Parallel(n_jobs=DEFAULT_NUM_THREADS)(
    delayed(resample_swc_file)(
        os.path.join(origin_dir, swc_file),
        os.path.join(target_dir, swc_file),
    ) for swc_file in tqdm(swc_files)
)



