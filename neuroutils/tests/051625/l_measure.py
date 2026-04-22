from neuroutils.swc.l_measure import calc_global_features_from_folder

todo_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um"
save_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_global_features_from_pylib.csv"

todo_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab"
save_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_global_features_from_pylib.csv"

# todo_dir =

calc_global_features_from_folder(todo_dir, save_dir)
