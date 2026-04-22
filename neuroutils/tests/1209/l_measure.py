from neuroutils.swc.l_measure import calc_global_features_from_folder
import pandas as pd

todo_dir = "/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/data/auto8.4k_0510_resample1um_mergedBranches0712_crop100"
save_file = "/data2/kfchen/tracing_ws/14k_raw_img_data/crop100/auto8.4k_0510_resample1um_mergedBranches0712_crop100_l_measure.csv"

# todo_dir =

calc_global_features_from_folder(todo_dir, save_file)

# rename
meta = pd.read_csv(save_file)
print(meta.columns)
# 重命名第一列的所有内容，转成int类型
for i in range(len(meta)):
    meta.iloc[i, 0] = int(meta.iloc[i, 0].split("_")[0].split(".")[0])
# meta.rename(columns={meta.columns[0]: "id"}, inplace=True)
# save
meta.to_csv(save_file, index=False)