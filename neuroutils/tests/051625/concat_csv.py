# 把两个csv拼起来

import pandas as pd

file1 = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/tissue_info.csv"
file2 = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/sample_info_from_database.xlsx"

df1 = pd.read_csv(file1)
df2 = pd.read_excel(file2)

print(df1.columns)
print(df2.columns)

# rename df2
df2.rename(columns={'tissue_id': 'tissue_block_number'}, inplace=True)

# 新加一列 patient+tissue
df1["pt"] = df1["patient_number"].astype(str) + "_" + df1["tissue_block_number"].astype(str)
df2["pt"] = df2["patient_number"].astype(str) + "_" + df2["tissue_block_number"].astype(str)

# find common rows
df1_pt = df1["pt"].tolist()
df2_pt = df2["pt"].tolist()
common_pt = list(set(df1_pt) & set(df2_pt))

# filter df1 and df2
df1 = df1[df1["pt"].isin(common_pt)]
df2 = df2[df2["pt"].isin(common_pt)]

# merge df1 and df2
df = pd.merge(df1, df2, on="pt", how="inner")
df = df.drop(columns=["pt"])

# rename
df.rename(columns={"gender_x": "gender",
                     "tissue_block_number_x": "tissue_block_number",
                        "patient_number_x": "patient_number",
                   }, inplace=True)
# df["hospital"] = df["sample_id"]
df["hospital"] = df["sample_id"].apply(lambda x: x.split("-")[1])

'''
patient_number_x	tissue_block_number_x	age	gender_x	brain_region	neuron_number	total_id		sample_id
tumor_location	intracranial_location
pathological_diagnosis	tissue_type		comment1
'''
# instersted columns
instersted_columns = [
    "patient_number",
    "tissue_block_number",
    "age",
    "gender",
    "brain_region",
    "neuron_number",
    "total_id",
    "hospital",
    "tumor_location",
    "intracranial_location",
    "pathological_diagnosis",
    "tissue_type",
    "comment1",
]
# filter columns
df = df[instersted_columns]

# save to csv
df.to_csv("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/sample_info.csv", index=False, encoding="gbk")