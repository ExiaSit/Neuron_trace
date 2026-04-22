from neuroutils.swc.l_measure import compare_features


file1 = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_global_features_from_pylib.csv"
file2 = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/test/auto8.4k_0510_pruned_resample1um.csv"

save_file = "/home/kfchen/neuron_seg_human/neuroutils/tests/051625/l_measure_plot.png"


compare_features(file1, file2, save_file, label1="no_re", label2="resampled")