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

def calc_global_features(swc_file, vaa3d=r'D:\Vaa3D_V4.001_Windows_MSVC_64bit\vaa3d_msvc.exe'):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x global_neuron_feature -f compute_feature -i "{swc_file}"'
    # cmd_str = f"{vaa3d} /x global_neuron_feature /f compute_feature /i {swc_file}"
    p = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()
    output_copy = output
    output = output.decode().splitlines()[35:-2]
    id = os.path.split(swc_file)[-1].split('_')[0].split('.')[0]

    info_dict = {}
    for s in output:
        s_s = s.split(':')
        if(len(s_s) < 2):
            continue
        it1, it2 = s_s
        it1 = it1.strip()
        it2 = it2.strip()
        if (it2 == '-1.#IND'):
            it2 = '-1'
        info_dict[it1] = float(it2)

    try:
        features = {
            'ID': id,
            'N_node': int(info_dict['N_node']),
            'Soma_surface': info_dict['Soma_surface'],
            'N_stem': int(info_dict['N_stem']),
            'Number of Bifurcatons': int(info_dict['Number of Bifurcatons']),
            'Number of Branches': int(info_dict['Number of Branches']),
            'Number of Tips': int(info_dict['Number of Tips']),
            'Overall Width': info_dict['Overall Width'],
            'Overall Height': info_dict['Overall Height'],
            'Overall Depth': info_dict['Overall Depth'],
            'Average Diameter': info_dict['Average Diameter'],
            'Total Length': info_dict['Total Length'],
            'Total Surface': info_dict['Total Surface'],
            'Total Volume': info_dict['Total Volume'],
            'Max Euclidean Distance': info_dict['Max Euclidean Distance'],
            'Max Path Distance': info_dict['Max Path Distance'],
            'Max Branch Order': info_dict['Max Branch Order'],
            'Average Contraction': info_dict['Average Contraction'],
            'Average Fragmentation': info_dict['Average Fragmentation'],
            'Average Parent-daughter Ratio': info_dict['Average Parent-daughter Ratio'],
            'Average Bifurcation Angle Local': info_dict['Average Bifurcation Angle Local'],
            'Average Bifurcation Angle Remote': info_dict['Average Bifurcation Angle Remote'],
            'Hausdorff Dimension': info_dict['Hausdorff Dimension']
        }
    except Exception as e:
        # 记录具体错误信息
        print(f"Error processing file {swc_file}: {str(e)}")
        # 可以打印出更多的诊断信息
        print("Command string:", cmd_str)
        print("Output copy:", output_copy)

        features = {
            'ID': id,
            'N_node': None,
            'Soma_surface': None,
            'N_stem': None,
            'Number of Bifurcatons': None,
            'Number of Branches': None,
            'Number of Tips': None,
            'Overall Width': None,
            'Overall Height': None,
            'Overall Depth': None,
            'Average Diameter': None,
            'Total Length': None,
            'Total Surface': None,
            'Total Volume': None,
            'Max Euclidean Distance': None,
            'Max Path Distance': None,
            'Max Branch Order': None,
            'Average Contraction': None,
            'Average Fragmentation': None,
            'Average Parent-daughter Ratio': None,
            'Average Bifurcation Angle Local': None,
            'Average Bifurcation Angle Remote': None,
            'Hausdorff Dimension': None
        }

    return features


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

def merge_bins(observed, expected, min_freq=5):
    # 合并频数小于 min_freq 的 bin
    # observed, expected = np.histogram(type_a_values, bins=int(np.sqrt(len(type_a_values))), range=current_range)
    # # to list
    # observed = observed[0].tolist()
    # expected = expected[0].tolist()
    # 合并频数小于 min_freq 的 bin
    new_observed = []
    new_expected = []
    current_observed = 0
    current_expected = 0
    for i in range(len(observed)):
        current_observed += observed[i]
        current_expected += expected[i]
        if current_observed >= min_freq and current_expected >= min_freq:
            new_observed.append(current_observed)
            new_expected.append(current_expected)
            current_observed = 0
            current_expected = 0
    # 最后一个
    if current_observed > 0 or current_expected > 0:
        new_observed[-1] += current_observed
        new_expected[-1] += current_expected
    return new_observed, new_expected

def plot_violin(df_a, df_b, violin_file=None, labels=['GS', 'Auto'],
                feature_names=['Stems', 'Bifurcations', 'Branches', 'Tips',
                               'OverallWidth', 'OverallHeight', 'OverallDepth', 'Length',
                               'MaxEuclideanDistance', 'MaxPathDistance', 'MaxBranchOrder', ]
                ):
    test_id_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv"
    test_id = pd.read_csv(test_id_file)
    test_id = test_id['id'].tolist()

    # feature_names = ['N_stem', 'Number of Branches', 'Number of Tips', 'Total Length', 'Max Branch Order']
    df_a.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    df_b.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

    # rename id
    # for i in range(len(df_b)):
    #     df_b['ID'][i] = int(df_b['ID'][i].split('_')[0])

    ids1 = df_a['ID'].tolist()
    ids2 = df_b['ID'].tolist()

    common_ids = list(set(ids1) & set(ids2))
    common_ids = list(set(common_ids) & set(test_id))

    df_a = df_a[df_a['ID'].isin(common_ids)]
    df_b = df_b[df_b['ID'].isin(common_ids)]
    # sort
    df_a = df_a.sort_values(by='ID')
    df_b = df_b.sort_values(by='ID')
    print(len(df_a))


    length_weighted_lm_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/test/comp_lm_df_pruned.csv"
    lw_df = pd.read_csv(length_weighted_lm_file)
    lw_df = lw_df[lw_df['id'].isin(common_ids)]



    feature_name_maps = {
        'Branches': 'No. of Branches*',
        'Length': 'Total Length (μm)',
        'MaxPathDistance': 'Max Path Dist. (μm)',
        'Stems': 'No. of Stems',
        'Tips': 'No. of Tips*',
        'MaxBranchOrder': 'Max Branch Order',
        'Nodes': 'No. of Nodes',
        'Bifurcations': 'No. of Bifurcations*',
        'OverallWidth': 'Width (μm)',
        'OverallHeight': 'Height (μm)',
        'OverallDepth': 'Depth (μm)',
        'MaxEuclideanDistance': 'Max Euclidean Dist. (μm)',
        # 'Max Branch Order': 'Max Branch Order',
        # 'Average Bifurcation Angle Remote': 'Avg. Remote BA (°)'
    }

    num_features = len(feature_names)
    cols = 11
    rows = (num_features + cols - 1) // cols
    # fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, 4 * rows), dpi=300)  # 调整figsize和dpi提高清晰度
    fig = plt.figure(figsize=(6, 4), dpi=300)
    # axes = axes.flatten()

    df_a['Type'], df_b['Type'] = labels
    df = pd.concat([df_a, df_b], axis=0)
    df_long = pd.melt(df, id_vars=['Type'], value_vars=feature_names, var_name='Feature', value_name='Value')

    save_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/test/violin_test.csv"
    df_long.to_csv(save_file, index=False)


    bugn_colors = plt.get_cmap('BuGn')
    # yellow
    # colors = [plt.get_cmap('YlOrBr')(x) for x in np.linspace(0.2, 0.3, len(feature_names))]
    # green
    colors = [plt.get_cmap('BuGn')(x) for x in np.linspace(0.2, 0.3, len(feature_names))]

    posision = np.arange(11)
    plt.legend().set_visible(False)
    # 绘图
    for idx, feature in enumerate(feature_names):
        # ax = axes[idx]
        ax = fig

        # if(feature == 'Branches'):
        #     data_comp = lw_df["Length_Weighted_Number_of_Branches"].to_numpy()
        # elif(feature == 'Tips'):
        #     data_comp = lw_df["Length_Weighted_Number_of_Tips"].to_numpy()
        # elif(feature == 'Bifurcations'):
        #     data_comp = lw_df["Length_Weighted_Number_of_Bifurcatons"].to_numpy()
        if(False):pass

        else:
            # 筛选当前特征的数据
            feature_data = df_long[df_long['Feature'] == feature]
            # print(len(feature_data))

            # 计算人工标注和自动重建结果的相关系数
            type_a_values = feature_data[feature_data['Type'] == labels[0]]['Value'].to_numpy().astype(float)
            type_b_values = feature_data[feature_data['Type'] == labels[1]]['Value'].to_numpy().astype(float)

            # print("--------------")
            # print(feature)
            # arr = type_a_values
            # nan_mask = np.isnan(arr)
            # print("NaN values:", nan_mask)
            # print("Indices with NaN:", np.where(nan_mask))
            #
            # arr = type_b_values
            # nan_mask = np.isnan(arr)
            # print("NaN values:", nan_mask)
            # print("Indices with NaN:", np.where(nan_mask))

            data_comp = type_b_values / type_a_values


        for i in range(1):
            current_data = data_comp
            print(feature, "median: ", np.median(current_data))
            violin_parts = plt.violinplot(current_data,
                                          positions=[posision[idx]], widths=0.8,
                                          showmeans=False, showmedians=False, showextrema=False,
                                          )
            for partname in ['bodies']:
                for part in violin_parts[partname]:
                    part.set_edgecolor('black')  # 设置边缘线的颜色
                    part.set_linewidth(1)  # 设置边缘线的宽度
                    part.set_facecolor(colors[idx])  # 设置填充颜色
                    # alpha
                    part.set_alpha(1)
        # current_legend = ax.legend(labels, loc='center left', bbox_to_anchor=(0.5, 0.5), fontsize=12)

        for i in range(1):
            plt.boxplot(current_data,
                        positions=[posision[idx]], widths=0.4,
                        patch_artist=True,
                        showfliers=True,
                        boxprops=dict(color='black', linewidth=1, facecolor='white'),

                        capprops=dict(color='black'),
                        medianprops=dict(color='black'),
                        flierprops=dict(marker='o', color='black', markersize=3)
                        )
        xticks = [feature_name_maps[f] for f in feature_names]

        plt.xticks(posision, xticks, rotation=45, fontsize=13, ha='right')
        # plt.yticks(fontsize=13)

        # 轴线的粗细
        plt.gca().spines['left'].set_linewidth(1)
        plt.gca().spines['bottom'].set_linewidth(1)

        plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
        # plt.axhline(y=0.9, color='gray', linestyle='--', linewidth=1)
        # plt.axhline(y=1.1, color='gray', linestyle='--', linewidth=1)

        # 关闭上面和右边的坐标轴
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.ylim(0.25, 1.75)
        # 关闭legend

    # ytick 15
    plt.yticks(fontsize=13)
    # 隐藏不需要的子图
    plt.tight_layout()  #
    # plt.show()
    plt.savefig(violin_file)
    plt.close()



def plot_box(df_a, df_b, box_file, labels=[]):
    # feature_names = ['N_node', 'Soma_surface', 'N_stem', 'Number of Bifurcatons',
    #                 'Number of Branches', 'Number of Tips', 'Overall Width', 'Overall Height',
    #                 'Overall Depth', 'Average Diameter', 'Total Length', 'Total Surface',
    #                 'Total Volume', 'Max Euclidean Distance', 'Max Path Distance',
    #                 'Max Branch Order', 'Average Contraction', 'Average Fragmentation',
    #                 'Average Parent-daughter Ratio', 'Average Bifurcation Angle Local',
    #                 'Average Bifurcation Angle Remote', 'Hausdorff Dimension']
    feature_names = ['N_stem', 'Number of Branches', 'Number of Tips', 'Total Length']
    feature_name_maps = {'Number of Branches': 'Number of Branches', 'Total Length': 'Total Length (μm)',
                         'Max Path Distance': 'Max Path Distance (μm)', 'N_stem': 'Number of Stems',
                         'Number of Tips': 'Number of Tips', 'Max Branch Order': 'Max Branch Order'}

    num_features = len(feature_names)
    cols = 4
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, 1.5 * rows), dpi=300)  # 调整figsize和dpi提高清晰度
    axes = axes.flatten()
    plt.rcParams.update({'font.size': 20})  # 更新字体大小
    # 设置字体 times new roman
    plt.rcParams['font.family'] = 'Times New Roman'

    df_a['Type'], df_b['Type'] = labels
    df = pd.concat([df_a, df_b], axis=0)
    df_long = pd.melt(df, id_vars=['Type'], value_vars=feature_names, var_name='Feature', value_name='Value')

    # 绘图
    for idx, feature in enumerate(feature_names):
        ax = axes[idx]
        if feature == 'Number of Branches':
            ax.set_ylim(-1.5, 150)
        elif feature == 'Total Length':
            ax.set_ylim(-50, 5000)
        sns.boxplot(x='Feature', y='Value', hue='Type', data=df_long[df_long['Feature'] == feature], ax=ax,
                    palette="viridis", linewidth=0.8, gap=.2, fliersize=0, native_scale=True)
        # ax.set_title(feature_name_maps[feature], fontsize=15)
        ax.set_title("")
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=10)  # 调整刻度标签大小
        ax.legend().set_visible(False)


    # 隐藏不需要的子图
    for ax in axes[num_features:]:
        ax.axis('off')

    plt.tight_layout(pad=1.0)  # 调整布局
    plt.savefig(box_file)
    plt.close()

def get_common_rows_from_dfs(dfs):
    # 获取所有 DataFrame 第一列的共同项
    # 假设 df 列名为 'col1'
    common_items = set(dfs[0].iloc[:, 0])  # 假设所有 DataFrame 第一列都是一样的列名
    for df in dfs[1:]:
        common_items &= set(df.iloc[:, 0])  # 交集操作，找出共同的元素

    # 将共有项作为索引过滤每个 DataFrame
    common_df_list = []
    for df in dfs:
        filtered_df = df[df.iloc[:, 0].isin(common_items)]  # 根据第一列的共有项筛选
        # 找到有多少行
        print(len(filtered_df))
        common_df_list.append(filtered_df)

    # 返回包含共同项的所有 DataFrame
    return common_df_list



def process_files(gt_file, pred_file, v3d_path):
    features_gt = calc_global_features(gt_file, vaa3d=v3d_path)
    features_pred = calc_global_features(pred_file, vaa3d=v3d_path)
    if features_gt is not None and features_pred is not None:
        return (features_gt, features_pred)
    return None


def l_measure_gt_and_pred(gt_dir, pred_dir, gt_csv, pred_csv, violin_png,
                          v3d_path=r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x", debug=False):
    features_all = pd.DataFrame(columns=['ID', 'N_node', 'Soma_surface', 'N_stem', 'Number of Bifurcatons',
                                         'Number of Branches', 'Number of Tips', 'Overall Width', 'Overall Height',
                                         'Overall Depth', 'Average Diameter', 'Total Length', 'Total Surface',
                                         'Total Volume', 'Max Euclidean Distance', 'Max Path Distance',
                                         'Max Branch Order', 'Average Contraction', 'Average Fragmentation',
                                         'Average Parent-daughter Ratio', 'Average Bifurcation Angle Local',
                                         'Average Bifurcation Angle Remote', 'Hausdorff Dimension'])

    features_all.to_csv(gt_csv, float_format='%g', index=False)
    features_all.to_csv(pred_csv, float_format='%g', index=False)

    gt_files = glob.glob(os.path.join(gt_dir, '*swc'))
    pred_files = glob.glob(os.path.join(pred_dir, '*swc'))
    gt_files.sort()
    pred_files.sort()

    gt_ids = [int(os.path.split(f)[-1].split('_')[0].split('.')[0]) for f in gt_files]
    pred_ids = [int(os.path.split(f)[-1].split('_')[0].split('.')[0]) for f in pred_files]
    shared_ids = list(set(gt_ids) & set(pred_ids))

    # debug
    if (debug):
        shared_ids = shared_ids[:10]

    filtered_gt_files = [f for f, id in zip(gt_files, gt_ids) if id in shared_ids]
    filtered_pred_files = [f for f, id in zip(pred_files, pred_ids) if id in shared_ids]

    features_all_gt = []
    features_all_pred = []

    with ThreadPoolExecutor(max_workers=12) as executor:  # 可以根据你的系统调整 max_workers
        # 设置进度条
        progress_bar = tqdm(total=len(filtered_gt_files), desc='Processing_gt')

        # 提交任务到线程池
        future_to_files = {executor.submit(process_files, gt, pred, v3d_path): (gt, pred) for gt, pred in
                           zip(filtered_gt_files, filtered_pred_files)}

        # 处理线程池的结果
        for future in as_completed(future_to_files):
            result = future.result()
            if result is not None:
                features_gt, features_pred = result
                features_all_gt.append(features_gt)
                features_all_pred.append(features_pred)
            progress_bar.update(1)

    progress_bar.close()
    # print(features_all_gt)
    df_gt = pd.DataFrame(features_all_gt)
    df_gt = df_gt.sort_values(by='ID')
    df_gt.to_csv(gt_csv, float_format='%g', index=False, mode='a', header=False)

    df_pred = pd.DataFrame(features_all_pred)
    df_pred = df_pred.sort_values(by='ID')
    df_pred.to_csv(pred_csv, float_format='%g', index=False, mode='a', header=False)
    progress_bar.close()

    plot_violin(df_gt, df_pred, violin_png)

def compare_l_measure():
    # gt_dir = r"/data/kfchen/trace_ws/gt_seg_downsample/v3dswc" # gt segment traced
    # gt_dir = r"/data/kfchen/trace_ws/to_gu/lab/2_sort"  # manual traced()sorted
    # gt_dir = r"/data/kfchen/trace_ws/to_gu/origin_swc" # manual traced

    # gt_dir = r"/data/kfchen/trace_ws/to_gu/lab/2_flip_after_sort"
    # gt_dir = r"/data/kfchen/trace_ws/neurom_ws/new_sort/pruned_swc"
    gt_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_unified_GS"



    # gt_dir = r"/data/kfchen/trace_ws/result500_fold0_source/v3dswc"
    # pred_dir = r"/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/10847_auto_v1.4_12k/swc"
    # gt_dir = (r"/data/kfchen/trace_ws/result500_164_500_aug_noptls/v3dswc")
    # pred_dir = r"/data/kfchen/trace_ws/result500_fold0_source/v3dswc"
    # pred_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/validation_traced/pruned_v3dswc"
    pred_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_unified_Auto"

    # gt_csv = r"/data/kfchen/nnUNet/gt_swc.csv"
    # pred_csv = r"/data/kfchen/nnUNet/pred_swc.csv"
    # violin_png = r"/data/kfchen/nnUNet/violin.png"
    gt_csv = pred_dir.replace('unified_Auto', 'gt_swc.csv')
    pred_csv = pred_dir.replace('unified_Auto', 'pred_swc.csv')
    violin_png = pred_dir.replace('unified_Auto', 'violin_man_nnunet.png')
    v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"

    if (os.path.exists(pred_csv)):
        os.remove(pred_csv)
    if (os.path.exists(gt_csv)):
        os.remove(gt_csv)
    if (os.path.exists(violin_png)):
        os.remove(violin_png)

    l_measure_gt_and_pred(gt_dir, pred_dir, gt_csv, pred_csv, violin_png, v3d_path=v3d_path)

def compare_tip_to_soma(traced_dir1 = r"/data/kfchen/trace_ws/result500_new_resized_test_noptls/connswc",
                        traced_dir2 = r"/data/kfchen/trace_ws/result500_new_resized_test_ptls/connswc"):
    dir1_files = glob.glob(os.path.join(traced_dir1, '*swc'))
    dir2_files = glob.glob(os.path.join(traced_dir2, '*swc'))
    dir1_files.sort()
    dir2_files.sort()

    dir1_ids = [int(os.path.split(f)[-1].split('_')[0].split('.')[0]) for f in dir1_files]
    dir2_ids = [int(os.path.split(f)[-1].split('_')[0].split('.')[0]) for f in dir2_files]
    shared_ids = list(set(dir1_ids) & set(dir2_ids))

    dir1_mean_tip_to_soma_dist_list = []
    dir2_mean_tip_to_soma_dist_list = []
    better_list = []

    for idx in shared_ids:
        dir1_swc_file = [f for f, id in zip(dir1_files, dir1_ids) if id == idx][0]
        dir2_swc_file = [f for f, id in zip(dir2_files, dir2_ids) if id == idx][0]

        point_l1 = read_swc(dir1_swc_file)
        point_l2 = read_swc(dir2_swc_file)

        file1_tip_to_soma_dist_list = []
        file2_tip_to_soma_dist_list = []

        for p1 in point_l1.p:
            if(p1.n == 0 or p1.n == 1):
                continue
            if(len(p1.s) == 0): # tip
                file1_tip_to_soma_dist_list.append(point_l1.calc_p_to_soma(p1.n))

        for p2 in point_l2.p:
            if (p2.n == 0 or p2.n == 1):
                continue
            if(len(p2.s) == 0): # tip
                file2_tip_to_soma_dist_list.append(point_l2.calc_p_to_soma(p2.n))

        # print(len(file1_tip_to_soma_dist_list), len(file2_tip_to_soma_dist_list))
        mean1 = sum(file1_tip_to_soma_dist_list) / len(file1_tip_to_soma_dist_list)
        mean2 = sum(file2_tip_to_soma_dist_list) / len(file2_tip_to_soma_dist_list)

        dir1_mean_tip_to_soma_dist_list.append(mean1)
        dir2_mean_tip_to_soma_dist_list.append(mean2)

        print(mean1, mean2)
        if(mean1 < mean2):
            better_list.append(1)
        else:
            better_list.append(0)

    print(f"mean dir1_mean_tip_to_soma_dist_list: {sum(dir1_mean_tip_to_soma_dist_list) / len(dir1_mean_tip_to_soma_dist_list)}")
    print(f"mean dir2_mean_tip_to_soma_dist_list: {sum(dir2_mean_tip_to_soma_dist_list) / len(dir2_mean_tip_to_soma_dist_list)}")
    print("better rate: ", sum(better_list) / len(better_list))
    print(len(better_list))


def l_measure_swc_file(swc_file, v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
    return calc_global_features(swc_file, vaa3d=v3d_path)

def l_measure_swc_dir(swc_dir, result_csv, v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
    feature_names = pd.DataFrame(columns=['ID', 'N_node', 'Soma_surface', 'N_stem', 'Number of Bifurcatons',
                                          'Number of Branches', 'Number of Tips', 'Overall Width', 'Overall Height',
                                          'Overall Depth', 'Average Diameter', 'Total Length', 'Total Surface',
                                          'Total Volume', 'Max Euclidean Distance', 'Max Path Distance',
                                          'Max Branch Order', 'Average Contraction', 'Average Fragmentation',
                                          'Average Parent-daughter Ratio', 'Average Bifurcation Angle Local',
                                          'Average Bifurcation Angle Remote', 'Hausdorff Dimension'])
    if(os.path.exists(result_csv)):
        os.remove(result_csv)

    feature_names.to_csv(result_csv, float_format='%g', index=False)

    swc_files = glob.glob(os.path.join(swc_dir, '*swc'))
    # swc_files.sort()

    l_measure_results = []
    swc_paths = [os.path.join(swc_dir, f) for f in swc_files]
    progress_bar = tqdm(total=len(swc_paths), desc='Processing')

    # for swc_path in swc_paths:
    #     l_measure_results.append(l_measure_swc_file(swc_path, v3d_path))
    #    progress_bar.update(1)
    # 多线程
    with ThreadPoolExecutor(max_workers=12) as executor:  # 可以根据你的系统调整 max_workers
        future_to_files = {executor.submit(l_measure_swc_file, swc_path, v3d_path): swc_path for swc_path in swc_paths}
        for future in as_completed(future_to_files):
            result = future.result()
            l_measure_results.append(result)
            progress_bar.update(1)

    progress_bar.close()

    df_gt = pd.DataFrame(l_measure_results)
    df_gt = df_gt.sort_values(by='ID')
    df_gt.to_csv(result_csv, float_format='%g', index=False, mode='a', header=False)


if __name__ == '__main__':
    # df_a = pd.read_csv(r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_standardized_l_measure.csv")# resample后的版本
    # df_a = pd.read_csv("/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_global_features_from_pylib.csv")# 没有resample的版本
    df_a = pd.read_csv(r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample_prune_l_measure.csv") # prune后的版本

    df_b = pd.read_csv(r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/test/auto8.4k_0510_pruned_resample1um.csv") # liu的版本
    # df_b = pd.read_csv(r"/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_standardized_global_features.csv")# 只resample的版本

    print(df_a.columns)

    violin_file = r"/home/kfchen/neuron_seg_human/neuroutils/tests/051725/prune.png"
    plot_violin(df_a, df_b, violin_file, labels=['Auto', 'Manual'])
