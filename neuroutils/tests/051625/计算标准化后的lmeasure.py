from neuroutils.swc.l_measure import calc_global_features_from_folder

todo_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_standardized"
save_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_standardized_global_features.csv"

calc_global_features_from_folder(todo_dir, save_file)