import os
from neuroutils.visualization.plot import plot_img_and_swc
from joblib import Parallel, delayed
from tqdm import tqdm
from neuroutils.swc.io import load_swc, save_swc
import matplotlib.pyplot as plt

img_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/full_res_img"
swc_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled/retrace_swcs_resampled"
plt_save_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled/plot"

swc_files = os.listdir(swc_dir)
# print(swc_files)

def do_plot_img_and_swc(img_file, swc_file, save_file):
    if(os.path.exists(save_file)):
        # print(f"{save_file} already exists, skip")
        return
    try:
        plot_img_and_swc(
            img_file, swc_file, save_file,
            rescale_img_flag=True,
            flip_y=True,
            down_sample_flag=False,

        )
    except Exception as e:
        print(f"Error in {img_file} and {swc_file}: {e}")
        return

Parallel(n_jobs=8)(
    delayed(do_plot_img_and_swc)(
        # img_file, swc_file, save_file
        os.path.join(img_dir, swc_file.replace(".swc", "_0000.tif")),
        os.path.join(swc_dir, swc_file),
        os.path.join(plt_save_dir, swc_file.replace(".swc", ".png")),
    )
    for swc_file in tqdm(swc_files)

)

