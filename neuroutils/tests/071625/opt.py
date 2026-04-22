import os
import shutil
from simple_swc_tool.opt_topology_analyse import my_opt
from tqdm import tqdm

# todo_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/rescaled_1um_swc_origin_name"
# new_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/rescaled_1um_swc"
# os.makedirs(new_dir, exist_ok=True)
# swc_files = [f for f in os.listdir(todo_dir) if f.endswith('.swc')]
# # rename
# for swc_file in swc_files:
#     id = int(swc_file.split('_')[0])
#     new_name = f"{id}.swc"
#     shutil.copy(os.path.join(todo_dir, swc_file), os.path.join(new_dir, new_name))
todo_dir = "/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/data/auto8.4k_0510_resample1um_mergedBranches0712"
new_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/h01-guided-reconstruction/data/auto8.4k_0510_pruned_resample1um_renamed_0716"
os.makedirs(new_dir, exist_ok=True)
swc_files = [f for f in os.listdir(todo_dir) if f.endswith('.swc')]
# # rename
for swc_file in tqdm(swc_files):
    id = int(swc_file.split('_')[0])
    new_name = f"{id}.swc"
    shutil.copy(os.path.join(todo_dir, swc_file), os.path.join(new_dir, new_name))


manual_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/rescaled_1um_swc"
auto_dir = new_dir
save_file = "/data2/kfchen/tracing_ws/14k_raw_img_data/rescaled_1um_swc_origin_name_opt_result_0716.csv"
my_opt(manual_dir, auto_dir, save_file)



