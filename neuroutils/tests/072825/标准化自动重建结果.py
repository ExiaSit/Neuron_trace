import os
from neuroutils.swc.parser import standardize_swc
from joblib import Parallel, delayed
from tqdm import tqdm

todo_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/h01-guided-reconstruction/data/swc_refined"
standardized_swc_dir = todo_dir + "_standardized"

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
    for root, dirs, files in os.walk(todo_dir):
        for file in files:
            if file.endswith(".swc"):
                swc_paths.append(os.path.join(root, file))
                # print(swc_paths[-1])

    print(f"Total {len(swc_paths)} swc files to process")

    Parallel(n_jobs=8, backend="loky")(
        delayed(do_standardize_swc)(swc_path, swc_path.replace(todo_dir, standardized_swc_dir))
        for swc_path in tqdm(swc_paths)
    )