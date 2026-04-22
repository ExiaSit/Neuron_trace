import os
from neuroutils.meta.mapping import extract_neuron_id
manual_dir = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/Different_versions_human_dendrite_reconstructed/humanNeuron_manual/humanNeuron_all_swc_manual_Vaa3D&CAR_latestVersion_V20241231/all_swc/swc"

swc_files = [f for f in os.listdir(manual_dir) if f.endswith('.eswc')]
swc_nums = [extract_neuron_id(f) for f in swc_files]
swc_nums = set(swc_nums)
print(list(swc_nums)[:10])  # Print first 10 neuron IDs for verification

recon_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um"
recon_files = [f for f in os.listdir(recon_dir) if f.endswith('.swc')]
recon_nums = [extract_neuron_id(f) for f in recon_files]
recon_nums = set(recon_nums)
print(list(recon_nums)[:10])  # Print first 10 neuron IDs for verification

print(f"not in manual: {len(recon_nums - swc_nums)}")
print(f"not in recon: {len(swc_nums - recon_nums)}")
print(f"total manual: {len(swc_nums)}")
print(f"total recon: {len(recon_nums)}")
print("common: ", len(swc_nums & recon_nums))