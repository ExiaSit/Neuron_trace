from neuroutils.swc.parser import eswc2swc
import os
from tqdm import tqdm
from neuroutils.swc.l_measure import calc_global_features_from_folder
from neuroutils.swc.parser import standardize_swc
from neuroutils.meta.mapping import extract_neuron_id
from neuroutils.meta.neuron import get_neuron_meta
from joblib import Parallel, delayed

source_eswc_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/origin_manual_eswc_sorted"
swc_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/origin_manual_swc_sorted"
os.makedirs(swc_dir, exist_ok=True)
# eswc to swc
eswc_files = [f for f in os.listdir(source_eswc_dir) if f.endswith('.eswc')]
# for eswc_file in tqdm(eswc_files):
#     source_path = os.path.join(source_eswc_dir, eswc_file)
#     target_path = os.path.join(swc_dir, eswc_file.replace('.eswc', '.swc'))
#     eswc2swc(source_path, target_path)

# stand


target_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/rescaled_1um_swc"
swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc')]
os.makedirs(target_dir, exist_ok=True)
def c_task(swc_file):
    source_path = os.path.join(swc_dir, swc_file)
    target_path = os.path.join(target_dir, swc_file)
    cell_id = extract_neuron_id(swc_file)
    meta = get_neuron_meta(cell_id, use_cache=False)
    xy_resolution, z_resolution = meta['xy拍摄分辨率(*10e-3μm/px)'].values[0], meta['z拍摄分辨率(*10e-3μm/px)'].values[0]
    xy_resolution, z_resolution = float(xy_resolution) / 1000, float(z_resolution) / 1000  # convert to μm/px
    standardize_swc(source_path, target_path, rescale_flag=True, xy_resolution=xy_resolution, z_resolution=z_resolution)
# for swc_file in tqdm(swc_files):
Parallel(n_jobs=8)(delayed(c_task)(swc_file) for swc_file in tqdm(swc_files))


