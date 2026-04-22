from neuroutils.swc.l_measure import calc_global_features_from_folder

todo_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample_prune"
save_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample_prune_l_measure.csv"

# todo_dir =

calc_global_features_from_folder(todo_dir, save_dir)
