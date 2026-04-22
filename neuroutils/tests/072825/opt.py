import os
import shutil
from simple_swc_tool.opt_topology_analyse import my_opt
from tqdm import tqdm


manual_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample_prune"
auto_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/h01-guided-reconstruction/data/swc_refined_standardized/baseline"
# save_file = "/data2/kfchen/tracing_ws/14k_raw_img_data/rescaled_1um_swc_origin_name_opt_result_0716.csv"
save_file = auto_dir + "_opt_result.csv"
my_opt(manual_dir, auto_dir, save_file)

manual_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample_prune"
auto_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/h01-guided-reconstruction/data/swc_refined_standardized/cldice"
# save_file = "/data2/kfchen/tracing_ws/14k_raw_img_data/rescaled_1um_swc_origin_name_opt_result_0716.csv"
save_file = auto_dir + "_opt_result.csv"
my_opt(manual_dir, auto_dir, save_file)

manual_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample_prune"
auto_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/h01-guided-reconstruction/data/swc_refined_standardized/ours"
# save_file = "/data2/kfchen/tracing_ws/14k_raw_img_data/rescaled_1um_swc_origin_name_opt_result_0716.csv"
save_file = auto_dir + "_opt_result.csv"
my_opt(manual_dir, auto_dir, save_file)

manual_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample_prune"
auto_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/h01-guided-reconstruction/data/swc_refined_standardized/skelrec"
# save_file = "/data2/kfchen/tracing_ws/14k_raw_img_data/rescaled_1um_swc_origin_name_opt_result_0716.csv"
save_file = auto_dir + "_opt_result.csv"
my_opt(manual_dir, auto_dir, save_file)



