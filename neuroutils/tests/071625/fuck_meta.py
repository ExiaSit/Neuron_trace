#%%
import pandas as pd

# 读取CSV文件
file_path = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path, encoding='gbk')

mask = data['patient_number'] == "P00014" # 创建一个布尔掩码
data.loc[mask, 'brain_region'] = "(S,M)FG"
print(data.shape)

mask = data['patient_number'] == "P00016"  # 创建一个布尔掩码
data.loc[mask, 'brain_region'] = "PL.L"
print(data.shape)

mask = data['patient_number'] == "P00020"  # 创建一个布尔掩码
data.loc[mask, 'brain_region'] = "PL.R"
print(data.shape)

# save
data.to_csv("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta_0716.csv",
            index=False, encoding="gbk")
print(data.shape)
