import glob
import os
import subprocess
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d.proj3d import transform
from neurom.features.morphology import feature
from torch.cuda import current_blas_handle
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from simple_swc_tool.swc_io import read_swc

from nnUNet.scripts.mip import get_mip_swc, get_mip
import tifffile
import numpy as np
import pingouin as pg
from matplotlib.lines import Line2D

import scipy.stats as stats
from matplotlib.colors import LinearSegmentedColormap

# def calc_global_features(swc_file, vaa3d=r'D:\Vaa3D_V4.001_Windows_MSVC_64bit\vaa3d_msvc.exe'):
#     cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x global_neuron_feature -f compute_feature -i "{swc_file}"'
#     # cmd_str = f"{vaa3d} /x global_neuron_feature /f compute_feature /i {swc_file}"
#     p = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     output, err = p.communicate()
#     output_copy = output
#     output = output.decode().splitlines()[35:-2]
#     id = os.path.split(swc_file)[-1].split('_')[0].split('.')[0]
#
#     info_dict = {}
#     for s in output:
#         s_s = s.split(':')
#         if(len(s_s) < 2):
#             continue
#         it1, it2 = s_s
#         it1 = it1.strip()
#         it2 = it2.strip()
#         if (it2 == '-1.#IND'):
#             it2 = '-1'
#         info_dict[it1] = float(it2)
#
#     try:
#         features = {
#             'ID': id,
#             'N_node': int(info_dict['N_node']),
#             'Soma_surface': info_dict['Soma_surface'],
#             'N_stem': int(info_dict['N_stem']),
#             'Number of Bifurcatons': int(info_dict['Number of Bifurcatons']),
#             'Number of Branches': int(info_dict['Number of Branches']),
#             'Number of Tips': int(info_dict['Number of Tips']),
#             'Overall Width': info_dict['Overall Width'],
#             'Overall Height': info_dict['Overall Height'],
#             'Overall Depth': info_dict['Overall Depth'],
#             'Average Diameter': info_dict['Average Diameter'],
#             'Total Length': info_dict['Total Length'],
#             'Total Surface': info_dict['Total Surface'],
#             'Total Volume': info_dict['Total Volume'],
#             'Max Euclidean Distance': info_dict['Max Euclidean Distance'],
#             'Max Path Distance': info_dict['Max Path Distance'],
#             'Max Branch Order': info_dict['Max Branch Order'],
#             'Average Contraction': info_dict['Average Contraction'],
#             'Average Fragmentation': info_dict['Average Fragmentation'],
#             'Average Parent-daughter Ratio': info_dict['Average Parent-daughter Ratio'],
#             'Average Bifurcation Angle Local': info_dict['Average Bifurcation Angle Local'],
#             'Average Bifurcation Angle Remote': info_dict['Average Bifurcation Angle Remote'],
#             'Hausdorff Dimension': info_dict['Hausdorff Dimension']
#         }
#     except Exception as e:
#         # 记录具体错误信息
#         print(f"Error processing file {swc_file}: {str(e)}")
#         # 可以打印出更多的诊断信息
#         print("Command string:", cmd_str)
#         print("Output copy:", output_copy)
#
#         features = {
#             'ID': id,
#             'N_node': None,
#             'Soma_surface': None,
#             'N_stem': None,
#             'Number of Bifurcatons': None,
#             'Number of Branches': None,
#             'Number of Tips': None,
#             'Overall Width': None,
#             'Overall Height': None,
#             'Overall Depth': None,
#             'Average Diameter': None,
#             'Total Length': None,
#             'Total Surface': None,
#             'Total Volume': None,
#             'Max Euclidean Distance': None,
#             'Max Path Distance': None,
#             'Max Branch Order': None,
#             'Average Contraction': None,
#             'Average Fragmentation': None,
#             'Average Parent-daughter Ratio': None,
#             'Average Bifurcation Angle Local': None,
#             'Average Bifurcation Angle Remote': None,
#             'Hausdorff Dimension': None
#         }
#
#     return features


# def plot_violin(df_gt, df_pred, violin_png):
#     feature_names = ['N_node', 'Soma_surface', 'N_stem', 'Number of Bifurcatons',
#                     'Number of Branches', 'Number of Tips', 'Overall Width', 'Overall Height',
#                     'Overall Depth', 'Average Diameter', 'Total Length', 'Total Surface',
#                     'Total Volume', 'Max Euclidean Distance', 'Max Path Distance',
#                     'Max Branch Order', 'Average Contraction', 'Average Fragmentation',
#                     'Average Parent-daughter Ratio', 'Average Bifurcation Angle Local',
#                     'Average Bifurcation Angle Remote', 'Hausdorff Dimension']
#
#     # plt.figure(figsize=(20, 20))
#
#     num_features = len(feature_names)
#     cols = 5  # 每行显示3个子图
#     rows = (num_features + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=False)
#     axes = axes.flatten()
#
#     df_gt['Type'] = 'manual traced'  # "GT"
#     df_pred['Type'] = 'auto traced'  # "Pred"
#
#     df = pd.concat([df_gt, df_pred], axis=0)
#     df_long = pd.melt(df, id_vars=['Type'], value_vars=feature_names, var_name='Feature', value_name='Value')
#
#     for idx, feature in enumerate(feature_names):
#         ax = axes[idx]
#
#         sns.violinplot(x='Feature', y='Value', hue='Type', data=df_long[df_long['Feature'] == feature], split=True,
#                        ax=ax)
#         ax.set_title(feature)
#         ax.set_xlabel('')  # 清除x轴标签
#         ax.set_ylabel('')  # 清除y轴标签
#         ax.legend().set_visible(False)  # 在每个子图中隐藏图例
#
#         if idx == 0:  # 只在第一个子图中显示图例
#             ax.legend(title='Data Type', loc='upper right')
#
#         # 隐藏空余的子图
#     for ax in axes[num_features:]:
#         ax.axis('off')
#
#     plt.tight_layout()
#     plt.savefig(violin_png)
#     plt.close()

# def merge_bins(observed, expected, min_freq=5):
#     # 合并频数小于 min_freq 的 bin
#     # observed, expected = np.histogram(type_a_values, bins=int(np.sqrt(len(type_a_values))), range=current_range)
#     # # to list
#     # observed = observed[0].tolist()
#     # expected = expected[0].tolist()
#     # 合并频数小于 min_freq 的 bin
#     new_observed = []
#     new_expected = []
#     current_observed = 0
#     current_expected = 0
#     for i in range(len(observed)):
#         current_observed += observed[i]
#         current_expected += expected[i]
#         if current_observed >= min_freq and current_expected >= min_freq:
#             new_observed.append(current_observed)
#             new_expected.append(current_expected)
#             current_observed = 0
#             current_expected = 0
#     # 最后一个
#     if current_observed > 0 or current_expected > 0:
#         new_observed[-1] += current_observed
#         new_expected[-1] += current_expected
#     return new_observed, new_expected

# def plot_violin(df_a, df_b, violin_file=None, labels=['GS', 'Auto'],
#                 feature_names=['Stems', 'Bifurcations', 'Branches', 'Tips',
#                                'OverallWidth', 'OverallHeight', 'OverallDepth', 'Length',
#                                'MaxEuclideanDistance', 'MaxPathDistance', 'MaxBranchOrder', ]
#                 ):
#
#     test_id_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv"
#     test_id = pd.read_csv(test_id_file)
#     test_id = test_id['id'].tolist()
#
#     # feature_names = ['N_stem', 'Number of Branches', 'Number of Tips', 'Total Length', 'Max Branch Order']
#     df_a.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
#     df_b.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
#
#     # rename id
#     # for i in range(len(df_b)):
#     #     df_b['ID'][i] = int(df_b['ID'][i].split('_')[0])
#
#     ids1 = df_a['ID'].tolist()
#     ids2 = df_b['ID'].tolist()
#
#     common_ids = list(set(ids1) & set(ids2))
#     # common_ids = list(set(common_ids) & set(test_id))
#
#     df_a = df_a[df_a['ID'].isin(common_ids)]
#     df_b = df_b[df_b['ID'].isin(common_ids)]
#     # sort
#     df_a = df_a.sort_values(by='ID')
#     df_b = df_b.sort_values(by='ID')
#     print(df_a.shape, df_b.shape)
#     # print(df_a.columns)
#     # print(df_b.columns)
#     print(df_a.columns.tolist())
#     print(df_b.columns.tolist())
#     # 列名映射
#
#     # 定义列名映射字典: {原始列名: 目标列名}
#     column_mapping = {
#         'N_stem': 'Stems',
#         'Number of Bifurcations': 'Bifurcations',
#         'Number of Branches': 'Branches',
#         'Number of Tips': 'Tips',
#         'Overall Width': 'OverallWidth',
#         'Overall Height': 'OverallHeight',
#         'Overall Depth': 'OverallDepth',
#         'Total Length': 'Length',
#         'Max Euclidean Distance': 'MaxEuclideanDistance',
#         'Max Path Distance': 'MaxPathDistance',
#         'Max Branch Order': 'MaxBranchOrder'
#     }
#     df_a = df_a.rename(columns=column_mapping)
#     df_b = df_b.rename(columns=column_mapping)
#
#     existing_features = [f for f in feature_names + ["ID"] if f in df_a.columns and f in df_b.columns]
#     df_a = df_a[existing_features]
#     df_b = df_b[existing_features]
#     print(df_a.shape, df_b.shape)
#
#     length_weighted_lm_file = "/data2/kfchen/tracing_ws/14k_raw_img_data/crop100/comp_lm_df_crop100_1209.csv"
#     lw_df = pd.read_csv(length_weighted_lm_file)
#     lw_df = lw_df[lw_df['id'].isin(common_ids)]
#
#
#
#     feature_name_maps = {
#         'Branches': 'No. of Branches*',
#         'Length': 'Total Length (μm)',
#         'MaxPathDistance': 'Max Path Dist. (μm)',
#         'Stems': 'No. of Stems',
#         'Tips': 'No. of Tips*',
#         'MaxBranchOrder': 'Max Branch Order',
#         'Nodes': 'No. of Nodes',
#         'Bifurcations': 'No. of Bifurcations*',
#         'OverallWidth': 'Width (μm)',
#         'OverallHeight': 'Height (μm)',
#         'OverallDepth': 'Depth (μm)',
#         'MaxEuclideanDistance': 'Max Euclidean Dist. (μm)',
#         # 'Max Branch Order': 'Max Branch Order',
#         # 'Average Bifurcation Angle Remote': 'Avg. Remote BA (°)'
#     }
#
#     num_features = len(feature_names)
#     cols = 11
#     rows = (num_features + cols - 1) // cols
#     # fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, 4 * rows), dpi=300)  # 调整figsize和dpi提高清晰度
#     fig = plt.figure(figsize=(6, 4), dpi=800)
#     # axes = axes.flatten()
#
#     df_a['Type'], df_b['Type'] = labels
#     df = pd.concat([df_a, df_b], axis=0)
#
#     # 1. 检查 'Type' 是否真的在列名里
#     # if 'Type' not in df.columns:
#     #     print(f"❌ 错误：列名 'Type' 不存在！\n当前的列名有：{df.columns.tolist()}")
#     #     # 提示：有时候是因为列名里有空格，比如 'Type '
#     #
#     # # 2. 检查 feature_names 里的列是否都在 df 里
#     # # 确保 feature_names 是个列表
#     # if not isinstance(feature_names, list):
#     #     print(f"⚠️ 警告：feature_names 不是列表，它是：{type(feature_names)}")
#     #
#     # # 找出缺少的列
#     # missing_cols = [col for col in feature_names if col not in df.columns]
#     # if missing_cols:
#     #     print(f"❌ 错误：feature_names 中有 {len(missing_cols)} 个列在 df 中找不到！")
#     #     print(f"缺失的列名示例：{missing_cols}")
#     #     # df.columns
#     #     print(f"当前的列名有：{df.columns.tolist()}")
#     # else:
#     #     print("✅ 所有列名检查通过，可能是数据类型或其他问题。")
#
#     df_long = pd.melt(df, id_vars=['Type'], value_vars=feature_names, var_name='Feature', value_name='Value')
#     print(df_long)
#
#     # save_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/test/violin_test_weighted_0716.csv"
#     # df_long.to_csv(save_file, index=False)
#
#
#     bugn_colors = plt.get_cmap('BuGn')
#     # yellow
#     # colors = [plt.get_cmap('YlOrBr')(x) for x in np.linspace(0.2, 0.3, len(feature_names))]
#     # green
#     colors = [plt.get_cmap('BuGn')(x) for x in np.linspace(0.2, 0.3, len(feature_names))]
#
#     posision = np.arange(11)
#     plt.legend().set_visible(False)
#     # 绘图
#     for idx, feature in enumerate(feature_names):
#         # ax = axes[idx]
#         ax = fig
#
#
#         # 筛选当前特征的数据
#         feature_data = df_long[df_long['Feature'] == feature]
#         type_a_values = feature_data[feature_data['Type'] == labels[0]]['Value'].to_numpy().astype(float)
#         type_b_values = feature_data[feature_data['Type'] == labels[1]]['Value'].to_numpy().astype(float)
#
#         data_comp = type_b_values / type_a_values
#
#
#         if(feature == 'Branches'):
#             data_comp = lw_df["Length_Weighted_Number_of_Branches"].to_numpy()
#             print("????")
#             print(data_comp, np.max(data_comp))
#         elif(feature == 'Tips'):
#             data_comp = lw_df["Length_Weighted_Number_of_Tips"].to_numpy()
#             print("????")
#         elif(feature == 'Bifurcations'):
#             data_comp = lw_df["Length_Weighted_Number_of_Bifurcatons"].to_numpy()
#             print("????")
#
#         current_data = data_comp
#         print(feature, "median: ", np.median(current_data), np.min(current_data), np.max(current_data))
#         violin_parts = plt.violinplot(current_data,
#                                       positions=[posision[idx]], widths=0.8,
#                                       showmeans=False, showmedians=False, showextrema=False,
#                                       )
#         for partname in ['bodies']:
#             for part in violin_parts[partname]:
#                 part.set_edgecolor('black')  # 设置边缘线的颜色
#                 part.set_linewidth(1)  # 设置边缘线的宽度
#                 part.set_facecolor(colors[idx])  # 设置填充颜色
#                 # alpha
#                 part.set_alpha(1)
#         # current_legend = ax.legend(labels, loc='center left', bbox_to_anchor=(0.5, 0.5), fontsize=12)
#
#         for i in range(1):
#             plt.boxplot(current_data,
#                         positions=[posision[idx]], widths=0.4,
#                         patch_artist=True,
#                         showfliers=True,
#                         boxprops=dict(color='black', linewidth=1, facecolor='white'),
#
#                         capprops=dict(color='black'),
#                         medianprops=dict(color='black'),
#                         flierprops=dict(marker='o', color='black', markersize=3)
#                         )
#         xticks = [feature_name_maps[f] for f in feature_names]
#
#         plt.xticks(posision, xticks, rotation=45, fontsize=13, ha='right')
#         # plt.yticks(fontsize=13)
#
#         # 轴线的粗细
#         plt.gca().spines['left'].set_linewidth(1)
#         plt.gca().spines['bottom'].set_linewidth(1)
#
#         plt.axhline(y=1.1, color='gray', linestyle='--', linewidth=1)
#         plt.axhline(y=0.6, color='gray', linestyle='--', linewidth=1)
#         plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
#
#         # 关闭上面和右边的坐标轴
#         plt.gca().spines['top'].set_visible(False)
#         plt.gca().spines['right'].set_visible(False)
#
#         plt.ylim(0.25, 1.75)
#         # 关闭legend
#
#     # ytick 15
#     plt.yticks(fontsize=13)
#     # 隐藏不需要的子图
#     plt.tight_layout()  #
#     # plt.show()
#     plt.savefig(violin_file)
#     plt.close()


# # def l_measure_swc_file(swc_file, v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
# #     return calc_global_features(swc_file, vaa3d=v3d_path)
#
# def l_measure_swc_dir(swc_dir, result_csv, v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
#     feature_names = pd.DataFrame(columns=['ID', 'N_node', 'Soma_surface', 'N_stem', 'Number of Bifurcatons',
#                                           'Number of Branches', 'Number of Tips', 'Overall Width', 'Overall Height',
#                                           'Overall Depth', 'Average Diameter', 'Total Length', 'Total Surface',
#                                           'Total Volume', 'Max Euclidean Distance', 'Max Path Distance',
#                                           'Max Branch Order', 'Average Contraction', 'Average Fragmentation',
#                                           'Average Parent-daughter Ratio', 'Average Bifurcation Angle Local',
#                                           'Average Bifurcation Angle Remote', 'Hausdorff Dimension'])
#     if(os.path.exists(result_csv)):
#         os.remove(result_csv)
#
#     feature_names.to_csv(result_csv, float_format='%g', index=False)
#
#     swc_files = glob.glob(os.path.join(swc_dir, '*swc'))
#     # swc_files.sort()
#
#     l_measure_results = []
#     swc_paths = [os.path.join(swc_dir, f) for f in swc_files]
#     progress_bar = tqdm(total=len(swc_paths), desc='Processing')
#
#     # for swc_path in swc_paths:
#     #     l_measure_results.append(l_measure_swc_file(swc_path, v3d_path))
#     #    progress_bar.update(1)
#     # 多线程
#     with ThreadPoolExecutor(max_workers=12) as executor:  # 可以根据你的系统调整 max_workers
#         future_to_files = {executor.submit(l_measure_swc_file, swc_path, v3d_path): swc_path for swc_path in swc_paths}
#         for future in as_completed(future_to_files):
#             result = future.result()
#             l_measure_results.append(result)
#             progress_bar.update(1)
#
#     progress_bar.close()
#
#     df_gt = pd.DataFrame(l_measure_results)
#     df_gt = df_gt.sort_values(by='ID')
#     df_gt.to_csv(result_csv, float_format='%g', index=False, mode='a', header=False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Union

# =============================================================================
# 1. 全局配置与常量定义 (Global Configuration)
# =============================================================================

# 原始CSV列名 -> 内部标准列名
# 目的：无论原始CSV叫什么，内部统一使用左侧的标准名称处理
RAW_TO_STD_COL_MAP = {
    'N_stem': 'Stems',
    'Number of Bifurcations': 'Bifurcations',
    'Number of Branches': 'Branches',
    'Number of Tips': 'Tips',
    'Overall Width': 'OverallWidth',
    'Overall Height': 'OverallHeight',
    'Overall Depth': 'OverallDepth',
    'Total Length': 'Length',
    'Max Euclidean Distance': 'MaxEuclideanDistance',
    'Max Path Distance': 'MaxPathDistance',
    'Max Branch Order': 'MaxBranchOrder'
}

# 内部标准列名 -> 绘图显示的X轴标签
STD_COL_TO_LABEL_MAP = {
    'Branches': 'No. of Branches*',
    'Length': 'Total Length (μm)',
    'MaxPathDistance': 'Max Path Dist. (μm)',
    'Stems': 'No. of Stems',
    'Tips': 'No. of Tips*',
    'MaxBranchOrder': 'Max Branch Order',
    'Bifurcations': 'No. of Bifurcations*',
    'OverallWidth': 'Width (μm)',
    'OverallHeight': 'Height (μm)',
    'OverallDepth': 'Depth (μm)',
    'MaxEuclideanDistance': 'Max Euclidean Dist. (μm)',
}

# 特殊逻辑配置：需要从 Length-Weighted 表中读取的特征
# 键为内部标准列名，值为目标CSV中可能的列名列表（支持容错）
WEIGHTED_METRICS_CONFIG = {
    'Branches': ['Length_Weighted_Number_of_Branches_ratio'],
    'Tips': ['Length_Weighted_Number_of_Tips_ratio'],
    'Bifurcations': ['Length_Weighted_Number_of_Bifurcations_ratio', 'Length_Weighted_Number_of_Bifurcatons_ratio']
}

# 绘图样式配置
STYLE_CONFIG = {
    'figure_size': (8, 5),
    'dpi': 300,
    'violin_width': 0.8,
    'box_width': 0.4,
    'colormap': 'BuGn',
    'y_limit': (0.25, 1.75)
}


# =============================================================================
# 2. 数据处理层 (Data Processing Layer)
# =============================================================================

def preprocess_and_align_datasets(
        df_ground_truth: pd.DataFrame,
        df_prediction: pd.DataFrame,
        df_weighted: Optional[pd.DataFrame] = None,
        id_column: str = 'ID'
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    预处理数据：重命名列，基于共同ID对齐数据，并排序。

    Args:
        df_ground_truth: 金标准数据 (作为分母)
        df_prediction: 预测/自动重建数据 (作为分子)
        df_weighted: 长度加权数据 (可选)
        id_column: ID列的名称

    Returns:
        对齐且排序后的 (df_gt, df_pred, df_wt)
    """
    # 1. 统一 ID 列名并设为 Index
    # 使用 .copy() 避免 SettingWithCopyWarning
    df_gt = df_ground_truth.rename(columns={'Unnamed: 0': id_column}).set_index(id_column, drop=False).copy()
    df_pred = df_prediction.rename(columns={'Unnamed: 0': id_column}).set_index(id_column, drop=False).copy()

    # 2. 计算共同 ID (交集)
    common_ids = df_gt.index.intersection(df_pred.index)

    if df_weighted is not None:
        df_wt = df_weighted.rename(columns={'id': id_column}).set_index(id_column, drop=False).copy()
        common_ids = common_ids.intersection(df_wt.index)
    else:
        df_wt = None

    # 3. 基于共同 ID 过滤并排序 (Critical Step)
    df_gt = df_gt.loc[common_ids].sort_index()
    df_pred = df_pred.loc[common_ids].sort_index()

    if df_wt is not None:
        df_wt = df_wt.loc[common_ids].sort_index()

    # 4. 应用列名映射
    df_gt.rename(columns=RAW_TO_STD_COL_MAP, inplace=True)
    df_pred.rename(columns=RAW_TO_STD_COL_MAP, inplace=True)

    print(f"[Info] Datasets aligned. Common samples: {len(common_ids)}")
    return df_gt, df_pred, df_wt


def extract_metric_values(
        metric_name: str,
        df_gt: pd.DataFrame,
        df_pred: pd.DataFrame,
        df_wt: Optional[pd.DataFrame]
) -> np.ndarray:
    """
    提取特定指标的数值。如果是拓扑特征，尝试从加权表中读取；
    否则计算 Ratio = Prediction / GroundTruth。
    """
    # 策略 A: 检查是否需要使用加权数据 (Weighted Metrics)
    if metric_name in WEIGHTED_METRICS_CONFIG and df_wt is not None:
        possible_columns = WEIGHTED_METRICS_CONFIG[metric_name]
        for col in possible_columns:
            if col in df_wt.columns:
                return df_wt[col].to_numpy().astype(float)

        # 如果配置了要读加权但没找到列，记录警告并回退到普通比率计算
        print(f"[Warning] Weighted column for '{metric_name}' not found in weighted dataframe. Falling back to ratio.")

    # 策略 B: 计算比率 (Ratio Calculation)
    try:
        # 确保转换为 float 类型
        values_gt = df_gt[metric_name].to_numpy().astype(float)
        values_pred = df_pred[metric_name].to_numpy().astype(float)

        # 处理分母为 0 的情况
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = values_pred / values_gt

        # 将 inf 替换为 nan，方便后续处理（可选）
        ratios[np.isinf(ratios)] = np.nan
        return ratios

    except KeyError:
        raise KeyError(f"Metric '{metric_name}' not found in input dataframes.")


# =============================================================================
# 3. 可视化层 (Visualization Layer)
# =============================================================================

def _render_violin_element(
        ax: plt.Axes,
        data: np.ndarray,
        x_pos: int,
        color_hex: Union[str, Tuple]
):
    """[内部函数] 绘制单个小提琴和箱线图组件"""

    # 过滤 NaN 值，防止绘图报错
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return

    # 1. 绘制小提琴图 (Violin Plot)
    violin_parts = ax.violinplot(
        valid_data,
        positions=[x_pos],
        widths=STYLE_CONFIG['violin_width'],
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body in violin_parts['bodies']:
        body.set_facecolor(color_hex)
        body.set_edgecolor('black')
        body.set_linewidth(1)
        body.set_alpha(1.0)

    # 2. 绘制箱线图 (Box Plot)
    ax.boxplot(
        valid_data,
        positions=[x_pos],
        widths=STYLE_CONFIG['box_width'],
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor='white', color='black', linewidth=1),
        capprops=dict(color='black'),
        medianprops=dict(color='black'),
        flierprops=dict(marker='o', color='black', markersize=3)
    )


def _configure_axes(ax: plt.Axes, x_positions: np.ndarray, x_labels: List[str]):
    """[内部函数] 设置坐标轴样式、网格线和边框"""

    # X 轴设置
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)

    # Y 轴设置
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(STYLE_CONFIG['y_limit'])

    # 参考线 (Reference Lines)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.2, label='Perfect Match')
    ax.axhline(y=0.6, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.axhline(y=1.1, color='gray', linestyle=':', linewidth=1, alpha=0.6)

    # 边框美化 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)


def plot_morphology_comparison(
        df_ground_truth: pd.DataFrame,
        df_prediction: pd.DataFrame,
        df_weighted: Optional[pd.DataFrame] = None,
        metrics_list: Optional[List[str]] = None,
        save_path: Optional[str] = None
):
    """
    主绘图函数：生成形态学特征对比图。

    Args:
        df_ground_truth: 金标准数据 DataFrame
        df_prediction: 预测数据 DataFrame
        df_weighted: 加权数据 DataFrame (用于拓扑特征修正)
        metrics_list: 需要绘制的特征列表 (内部标准名)
        save_path: 图片保存路径 (包含文件名)
    """

    # 默认特征列表
    if metrics_list is None:
        metrics_list = [
            'Stems', 'Bifurcations', 'Branches', 'Tips',
            'Length', 'MaxPathDistance'
        ]

    # 1. 数据预处理
    df_gt, df_pred, df_wt = preprocess_and_align_datasets(
        df_ground_truth, df_prediction, df_weighted
    )

    # 2. 初始化画布
    fig, ax = plt.subplots(figsize=STYLE_CONFIG['figure_size'], dpi=STYLE_CONFIG['dpi'])

    # 生成颜色映射
    cmap = plt.get_cmap(STYLE_CONFIG['colormap'])
    colors = [cmap(i) for i in np.linspace(0.2, 0.5, len(metrics_list))]

    x_positions = np.arange(len(metrics_list))
    display_labels = []

    # 3. 循环绘制每个特征
    for idx, metric_name in enumerate(metrics_list):
        try:
            # 获取数据
            metric_data = extract_metric_values(metric_name, df_gt, df_pred, df_wt)

            # 绘制组件
            _render_violin_element(ax, metric_data, x_positions[idx], colors[idx])

            # 统计信息输出 (Console Log)
            median_val = np.nanmedian(metric_data)
            print(f"Plotting {metric_name:20s} | Median Ratio: {median_val:.3f}")

            # 记录标签
            label = STD_COL_TO_LABEL_MAP.get(metric_name, metric_name)
            display_labels.append(label)

        except Exception as e:
            print(f"[Error] Failed to plot metric '{metric_name}': {e}")
            display_labels.append(metric_name) # 占位防止错位

    # 4. 设置样式与保存
    _configure_axes(ax, x_positions, display_labels)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Success] Figure saved to: {save_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# 4. 调用示例 (Usage Example)
# =============================================================================
if __name__ == "__main__":
    df_gs = pd.read_csv(r"/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/manual_resampled1um_crop100_renamed.csv")
    df_auto = pd.read_csv(r"/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/auto8.4k_0510_resample1um_mergedBranches0712_crop100_renamed.csv")
    df_lw = pd.read_csv(r"/data2/kfchen/tracing_ws/14k_raw_img_data/crop100/comp_lm_df_crop100_1209.csv")
    violin_file = r"/home/kfchen/neuron_seg_human/neuroutils/tests/1209/lmeasure_violin_weighted.png"


    # 示例调用:
    plot_morphology_comparison(
        df_gs, df_auto, df_lw,
        metrics_list=['Stems', 'Bifurcations', 'Branches', 'Tips',
                      'OverallWidth', 'OverallHeight', 'OverallDepth', 'Length',
                      'MaxEuclideanDistance', 'MaxPathDistance', 'MaxBranchOrder', ],
        save_path=violin_file
    )
