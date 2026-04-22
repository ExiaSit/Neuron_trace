from neuroutils.swc.l_measure import compare_features


file1 = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_standardized_l_measure.csv"
file2 = "/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/auto8.4k_0510_pruned_resample1um.csv"

save_file = "/home/kfchen/neuron_seg_human/neuroutils/tests/051425/l_measure_plot.png"


compare_features(file1, file2, save_file, label1="8k_manual", label2="H01_pruned")