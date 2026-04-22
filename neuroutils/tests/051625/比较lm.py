from neuroutils.swc.l_measure import compare_features


file1 = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_global_features_from_pylib.csv"
# file2 = r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_standardized_l_measure.csv"
file2 = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample_l_measure.csv"# 不带sort的版本

save_file = "/home/kfchen/neuron_seg_human/neuroutils/tests/051625/比较_nosort.png"


compare_features(file1, file2, save_file, label1="un_re_gt", label2="re_gt_no_sort")