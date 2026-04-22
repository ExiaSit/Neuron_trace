import pandas as pd

neuron_meta_info = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
tissue_meta_info = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/sample_info_e.csv"

df1 = pd.read_csv(neuron_meta_info, encoding="gbk")
df2 = pd.read_csv(tissue_meta_info, encoding="gbk")

df1["pt"] = df1["patient_number"].astype(str) + "_" + df1["tissue_block_number"].astype(str)
df2["pt"] = df2["Patient Number"].astype(str) + "_" + df2["Tissue Block Number"].astype(str)

pt1 = df1["pt"].tolist()
pt2 = df2["pt"].tolist()

pt1 = set(pt1)
pt2 = set(pt2)

# 找到不一致的pt
# in df1 but not in df2
pt1_not_in_pt2 = pt1 - pt2
print("pt1_not_in_pt2", pt1_not_in_pt2)

print(df1[df1["pt"].isin(pt1_not_in_pt2)])

