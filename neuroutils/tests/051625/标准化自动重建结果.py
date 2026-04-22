import os
from neuroutils.swc.parser import standardize_swc
from joblib import Parallel, delayed
from tqdm import tqdm

todo_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um"
standardized_swc_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_standardized"

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

    print(f"Total {len(swc_paths)} swc files to process")

    Parallel(n_jobs=8, backend="loky")(
        delayed(do_standardize_swc)(swc_path, os.path.join(standardized_swc_dir, os.path.basename(swc_path)))
        for swc_path in tqdm(swc_paths)
    )