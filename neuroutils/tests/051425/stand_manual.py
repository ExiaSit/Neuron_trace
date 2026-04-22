import os
from neuroutils.swc.parser import standardize_swc
from joblib import Parallel, delayed
from tqdm import tqdm

manual_swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab"
# standardized_swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_standardized"
standardized_swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample" # 不带sort的版本，sort可能有bug

def do_standardize_swc(swc_path, save_path):
    if(not os.path.exists(save_path)):
        try:
            standardize_swc(swc_path, save_path)
        except Exception as e:
            print(f"Error processing {swc_path}: {e}")

if __name__ == "__main__":
    if(not os.path.exists(standardized_swc_dir)):
        os.makedirs(standardized_swc_dir)
    swc_paths = []
    for root, dirs, files in os.walk(manual_swc_dir):
        for file in files:
            if file.endswith(".swc"):
                swc_paths.append(os.path.join(root, file))

    print(f"Total {len(swc_paths)} swc files to process")

    Parallel(n_jobs=8, backend="loky")(
        delayed(do_standardize_swc)(swc_path, os.path.join(standardized_swc_dir, os.path.basename(swc_path)))
        for swc_path in tqdm(swc_paths)
    )

