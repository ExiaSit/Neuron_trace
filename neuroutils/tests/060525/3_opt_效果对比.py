import os.path
import time
import glob
from simple_swc_tool.Topology_scoring import metrics_delin as md
import tifffile
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel


def main_opt(G_gt_path, out_dir):
    result_list, failed_files = [], []

    # num = os.path.splitext(os.path.split(G_gt_path)[-1])[0]
    # out_swc_name = num + ".swc"
    out_swc_name = os.path.split(G_gt_path)[-1]
    G_pred_path = os.path.join(out_dir, out_swc_name)
    if (not os.path.exists(G_pred_path)):
        return result_list, failed_files
    id = os.path.split(G_gt_path)[-1].split(".")[0].split("_")[0]
    id = str(int(id))
    result_list.append(id) # ["ID"]

    G_gt = md.load_graph_swc(G_gt_path)
    G_pred = md.load_graph_swc(G_pred_path)

    try:
        # --------------------------------------------------opt_j
        f1, precision, recall, \
            tp, pp, ap, \
            matches_g, matches_hg, \
            g_gt_snap, g_pred_snap = md.opt_j(G_gt,
                                              G_pred,
                                              th_existing=1,  # 在捕获过程中，只有当该边的所有端点都不在th_existing范围内时，才会将一个附加节点插入到该边中
                                              th_snap=25,  # 如果一个点到最近的边的距离小于th_snap，那么它就被折入图中
                                              alpha=100)  # 鼓励匹配具有相似顺序的两个节点
        result_list.append(precision) # ["optj_precision"]
        result_list.append(recall) # ["optj_recall"]
        result_list.append(f1) # ["optj_f1"]

        # --------------------------------------------------opt_p
        n_conn_precis, n_conn_recall, \
            n_inter_precis, n_inter_recall, \
            con_prob_precis, con_prob_recall, con_prob_f1 = md.opt_p(G_gt, G_pred)

        result_list.append(con_prob_precis) # ["optp_con_prob_precis"]
        result_list.append(con_prob_recall) # ["optp_con_prob_recall"]
        result_list.append(con_prob_f1) # ["optp_con_prob_f1"]

        # --------------------------------------------------opt_g
        f1, spurious, missings, \
            n_preds_sum, n_gts_sum, \
            n_spurious_marbless_sum, \
            n_empty_holess_sum = md.opt_g(G_gt, G_pred,
                                          spacing=10,
                                          dist_limit=300,
                                          dist_matching=25,
                                          N=50,  # to speed up this script
                                          verbose=False)
        result_list.append(spurious) # ["optg_spurious"]
        result_list.append(missings) # ["optg_missings"]
        result_list.append(f1) # ["optg_f1"]

        # print(f"{file_num} files done!\n")
    except:
        failed_files.append(G_gt_path)
        return [], failed_files

    # print(result_list)
    return result_list, failed_files

def my_opt(G_gt_path, out_dir, csv_file):
    # flip_flag = False
    # if (flip_flag):
    #     csv_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
    #     df = pd.read_csv(csv_file, encoding='gbk')
    #     from nnUNet.scripts.resolution_unifier import find_resolution
    #     fliped_out_dir = out_dir.replace("Auto", "Auto_fliped")
    #     if (not os.path.exists(fliped_out_dir)):
    #         os.makedirs(fliped_out_dir)
    #     swc_files = glob.glob(os.path.join(out_dir, '*.swc'))
    #     for swc_file in swc_files:
    #         fliped_swc_file = swc_file.replace("Auto", "Auto_fliped")
    #         if (os.path.exists(fliped_swc_file)):
    #             continue
    #         img_file = os.path.join(img_dir, os.path.split(swc_file)[-1].replace(".swc", ".tif"))
    #         img = tifffile.imread(img_file)
    #         xy_resolution = find_resolution(df, os.path.split(swc_file)[-1])
    #         result_lines = []
    #         with open(swc_file, 'r') as f:
    #             lines = f.readlines()
    #             for line in lines:
    #                 if line.startswith("#"):
    #                     result_lines.append(line)
    #                 else:
    #                     line = line.strip().split()
    #                     y = img.shape[1] * xy_resolution / 1000 - float(line[3])
    #                     result_lines.append(f"{line[0]} {line[1]} {line[2]} {y} {line[4]} {line[5]} {line[6]}\n")
    #         with open(fliped_swc_file, 'w') as f:
    #             f.writelines(result_lines)
    #     out_dir = fliped_out_dir
    # rename_flag = False
    # if (rename_flag):
    #     swc_files = glob.glob(os.path.join(out_dir, '*.swc'))
    #     for swc_file in swc_files:
    #         new_swc_file = str(int(swc_file.split("/")[-1].split("_")[0])) + ".swc"
    #         # print(new_swc_file)
    #         os.rename(swc_file, os.path.join(out_dir, new_swc_file))
    #
    #     # swc_files = glob.glob(os.path.join(gt_dir, '*.swc'))
    #     # for swc_file in swc_files:
    #     #     new_swc_file = str(int(swc_file.split("/")[-1].split("_")[0])) + ".swc"
    #     #     # print(new_swc_file)
    #     #     os.rename(swc_file, os.path.join(gt_dir, new_swc_file))

    result_dict = {
        "ID": [],

        "optj_precision": [],
        "optj_recall": [],
        "optj_f1": [],

        "optp_con_prob_precis": [],
        "optp_con_prob_recall": [],
        "optp_con_prob_f1": [],

        "optg_spurious": [],
        "optg_missings": [],
        "optg_f1": []
    }
    result_list = []
    failed_files = []
    swc_files = glob.glob(os.path.join(gt_dir, '*.swc'))
    # swc_files = swc_files[:10]

    # for G_gt_path in swc_files:  # 读所有的swc文件
    #     result_list, failed_files = main_opt(G_gt_path, out_dir, result_list, failed_files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        progress = tqdm(total=len(swc_files), desc='Processing SWC Files', unit='file')
        future_to_file = {executor.submit(main_opt, G_gt_path, out_dir): G_gt_path for
                          G_gt_path in swc_files}
        for future in concurrent.futures.as_completed(future_to_file):
            progress.update(1)  # 更新进度条
            try:
                res_list, fail_files = future.result()
                if (not res_list == []): result_list.append(res_list)
                if (not fail_files == []): failed_files.append(fail_files)
            except Exception as e:
                print(f"Error processing file {future_to_file[future]}: {e}")
        progress.close()

    # 整理result_list
    for result in result_list:
        # print(result)
        result_dict["ID"].append(result[0])
        result_dict["optj_precision"].append(result[1])
        result_dict["optj_recall"].append(result[2])
        result_dict["optj_f1"].append(result[3])
        result_dict["optp_con_prob_precis"].append(result[4])
        result_dict["optp_con_prob_recall"].append(result[5])
        result_dict["optp_con_prob_f1"].append(result[6])
        result_dict["optg_spurious"].append(result[7])
        result_dict["optg_missings"].append(result[8])
        result_dict["optg_f1"].append(result[9])

    print("mean optj_precision: ", np.mean(result_dict["optj_precision"]),
          "mean optj_recall: ", np.mean(result_dict["optj_recall"]),
          "mean optj_f1: ", np.mean(result_dict["optj_f1"]))
    print("mean optp_con_prob_precis: ", np.mean(result_dict["optp_con_prob_precis"]),
          "mean optp_con_prob_recall: ", np.mean(result_dict["optp_con_prob_recall"]),
          "mean optp_con_prob_f1: ", np.mean(result_dict["optp_con_prob_f1"]))
    print("mean optg_spurious: ", np.mean(result_dict["optg_spurious"]),
          "mean optg_missings: ", np.mean(result_dict["optg_missings"]),
          "mean optg_f1: ", np.mean(result_dict["optg_f1"]))

    print("failed_files: ", failed_files)

    with open(csv_file, 'w') as f:
        f.write(
            "ID,optj_precision,optj_recall,optj_f1,optp_con_prob_precis,optp_con_prob_recall,optp_con_prob_f1,optg_spurious,optg_missings,optg_f1\n")
        for i in range(len(result_dict["ID"])):
            f.write("{},{},{},{},{},{},{},{},{}, {}\n".format(
                result_dict["ID"][i],
                result_dict["optj_precision"][i],
                result_dict["optj_recall"][i],
                result_dict["optj_f1"][i],
                result_dict["optp_con_prob_precis"][i],
                result_dict["optp_con_prob_recall"][i],
                result_dict["optp_con_prob_f1"][i],
                result_dict["optg_spurious"][i],
                result_dict["optg_missings"][i],
                result_dict["optg_f1"][i]
            ))

def find_good_sample_opt(df):
    gold_sample_ids = []
    silver_sample_ids = []
    other_sample_ids = []
    # 遍历所有行
    for i in range(len(df)):
        current_optj_f1 = df.iloc[i, 3]
        current_optp_con_prob_f1 = df.iloc[i, 6]
        current_optg_f1 = df.iloc[i, 9]

        if(current_optj_f1 > 0.9 and current_optp_con_prob_f1 > 0.9 and current_optg_f1 > 0.9):
            gold_sample_ids.append(df.iloc[i, 0])
        elif(current_optj_f1 > 0.8 and current_optp_con_prob_f1 > 0.8 and current_optg_f1 > 0.8):
            silver_sample_ids.append(df.iloc[i, 0])
        else:
            other_sample_ids.append(df.iloc[i, 0])

    print(f"gold_sample number: {len(gold_sample_ids)}, "
          f"silver_sample number: {len(silver_sample_ids)}, "
          f"other_sample number: {len(other_sample_ids)}")
    return gold_sample_ids, silver_sample_ids, other_sample_ids


def find_good_sample_l_measure(df, df_gs):
    gold_sample_ids = []
    silver_sample_ids = []
    other_sample_ids = []
    # 遍历所有行
    for i in range(len(df)):
        # Total Length
        current_total_length = df['Total Length'][i]
        current_gs_total_length = df_gs['Total Length'][df_gs['ID'] == df['ID'][i]].values[0]

        # Number of Branches
        current_num_branches = df['Number of Branches'][i]
        current_gs_num_branches = df_gs['Number of Branches'][df_gs['ID'] == df['ID'][i]].values[0]

        # Number of Tips
        current_num_tips = df['Number of Tips'][i]
        current_gs_num_tips = df_gs['Number of Tips'][df_gs['ID'] == df['ID'][i]].values[0]

        if(current_total_length > 0.9 * current_gs_total_length and current_num_branches > 0.9 * current_gs_num_branches and current_num_tips > 0.9 * current_gs_num_tips):
            gold_sample_ids.append(df.iloc[i, 0])
        elif(current_total_length > 0.8 * current_gs_total_length and current_num_branches > 0.8 * current_gs_num_branches and current_num_tips > 0.8 * current_gs_num_tips):
            silver_sample_ids.append(df.iloc[i, 0])
        else:
            other_sample_ids.append(df.iloc[i, 0])

    print(f"gold_sample number: {len(gold_sample_ids)}, "
          f"silver_sample number: {len(silver_sample_ids)}, "
          f"other_sample number: {len(other_sample_ids)}")
    return gold_sample_ids, silver_sample_ids, other_sample_ids

import pandas as pd

def find_good_sample_combined(df_opt, df_l_measure, df_gs, gold_threshold=0.8, silver_threshold=0.7):
    gold_sample_ids = []
    silver_sample_ids = []
    other_sample_ids = []

    # 合并 df_opt 和 df_l_measure，确保按 'ID' 对齐
    df_combined = pd.merge(df_opt, df_l_measure, on='ID', how='inner')

    # 遍历所有行
    for i in range(len(df_combined)):
        current_id = df_combined.iloc[i]['ID']

        # 从 df_opt 中获取三个指标
        current_optj_f1 = df_combined.iloc[i]['optj_f1']
        current_optp_con_prob_f1 = df_combined.iloc[i]['optp_con_prob_f1']
        current_optg_f1 = df_combined.iloc[i]['optg_f1']

        # 从 df_l_measure 中获取三个指标
        current_total_length = df_combined.iloc[i]['Total Length']
        current_num_branches = df_combined.iloc[i]['Number of Branches']
        current_num_tips = df_combined.iloc[i]['Number of Tips']

        # 从 df_gs（黄金标准）中获取对应的指标
        gs_row = df_gs[df_gs['ID'] == current_id]
        if gs_row.empty:
            print(f"警告：未找到 ID 为 {current_id} 的黄金标准数据。")
            continue

        current_gs_total_length = gs_row.iloc[0]['Total Length']
        current_gs_num_branches = gs_row.iloc[0]['Number of Branches']
        current_gs_num_tips = gs_row.iloc[0]['Number of Tips']

        # 检查金样本条件（所有六个条件都 > 0.9）
        conditions_gold = [
            current_optj_f1 > gold_threshold,
            current_optp_con_prob_f1 > gold_threshold,
            current_optg_f1 > gold_threshold,
            current_total_length > gold_threshold * current_gs_total_length,
            current_num_branches > gold_threshold * current_gs_num_branches,
            current_num_tips > gold_threshold * current_gs_num_tips
        ]

        # 检查银样本条件（所有六个条件都 > 0.8）
        conditions_silver = [
            current_optj_f1 > silver_threshold,
            current_optp_con_prob_f1 > silver_threshold,
            current_optg_f1 > silver_threshold,
            current_total_length > silver_threshold * current_gs_total_length,
            current_num_branches > silver_threshold * current_gs_num_branches,
            current_num_tips > silver_threshold * current_gs_num_tips
        ]

        if all(conditions_gold):
            gold_sample_ids.append(current_id)
        elif all(conditions_silver):
            silver_sample_ids.append(current_id)
        else:
            other_sample_ids.append(current_id)

    print(f"金样本数量: {len(gold_sample_ids)}, "
          f"银样本数量: {len(silver_sample_ids)}, "
          f"其他样本数量: {len(other_sample_ids)}")

    return gold_sample_ids, silver_sample_ids, other_sample_ids

def plot_box_of_swc_list(opt_result_files, labels, box_file):
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
    feature_names = ['optj_f1', 'optp_con_prob_f1', 'optg_f1']
    feature_name_maps = {'optj_f1': 'OPT-J F1',
                         'optp_con_prob_f1': 'OPT-P F1',
                         'optg_f1': 'OPT-G F1', }

    num_features = len(feature_names)
    cols = 3
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, 3 * rows), dpi=300)
    axes = axes.flatten()
    # plt.rcParams.update({'font.size': 20})  # 更新字体大小
    # 设置字体 Arial
    # plt.rcParams['font.family'] = 'Arial'

    dfs = [pd.read_csv(f) for f in opt_result_files]
    dfs = get_common_rows_from_dfs(dfs)
    for i, df in enumerate(dfs):
        df['Type'] = labels[i]

    df = pd.concat(dfs, axis=0)
    average_values = df.groupby('Type')[feature_names].mean()
    print("各类各特征的平均值：")
    print(average_values)
    df_long = pd.melt(df, id_vars=['Type'], value_vars=feature_names, var_name='Feature', value_name='Value')

    colors = plt.get_cmap('Set3').colors
    colors = [colors[8], colors[2], colors[5], colors[3], colors[0]]

    # 绘图
    for idx, feature in enumerate(feature_names):
        ax = axes[idx]
        if feature == 'Number of Branches':
            ax.set_ylim(-1.5, 150)
        elif feature == 'Total Length':
            ax.set_ylim(-50, 5000)

        feature_data = df_long[df_long['Feature'] == feature]

        # 计算positions, 每个Feature会有多个箱体
        # positions的数量要等于每个Feature和Type的组合数量
        positions = [0, 0.9, 1.5, 2.1]
        hue_order = feature_data['Type'].unique()  # 获取所有的hue分类，通常是['Type 1', 'Type 2']

        print(positions)

        # # 绘制箱线图
        # sns.boxplot(x='Feature', y='Value', hue='Type', data=feature_data, ax=ax,
        #             palette=colors, dodge=False, fliersize=0, native_scale=True, positions=positions)
        current_data = []
        for i, hue in enumerate(hue_order):
            current_data.append(feature_data[feature_data['Type'] == hue]['Value'].to_numpy())
            ax.boxplot(current_data[i], positions=[positions[i]], widths=0.5, patch_artist=True,
                       showfliers=True, boxprops=dict(facecolor=colors[i], color='black'),
                       medianprops=dict(color='black'), flierprops=dict(marker='o', color='black', markersize=3))

        # 计算p值并标注
        p_values = []
        baseline_data = current_data[0]  # 第一列作为baseline
        mean_differences = []

        for i in range(1, len(current_data)):
            compared_data = current_data[i]
            t_stat, p_value = ttest_rel(baseline_data, compared_data)
            p_values.append(p_value)
            print(f"p-value for {hue_order[0]} vs {hue_order[i]}: {p_value:.2e}")

            mean_diff = np.mean(compared_data) - np.mean(baseline_data)
            mean_differences.append(mean_diff)

        # 在图中标注p值
        y_max = max([max(data) for data in current_data])
        y_min = min([min(data) for data in current_data])
        y_range = y_max - y_min

        # 为每个比较添加p值标注
        for i, p_value in enumerate(p_values):
            if(i==0 or i == 1):continue
            x1 = positions[0]  # baseline位置
            x2 = positions[i + 1]  # 比较方法位置

            # 计算标注高度，避免重叠
            y_height = y_max + y_range * 0.1 + (i-2) * y_range * 0.15

            # 画连接线
            ax.plot([x1, x2], [y_height, y_height], color='black', lw=1)
            ax.plot([x1, x1], [y_height - y_range * 0.02, y_height], color='black', lw=1)
            ax.plot([x2, x2], [y_height - y_range * 0.02, y_height], color='black', lw=1)

            # 标注p值
            p_text = f"p={p_value:.2e}"  # 使用科学计数法显示p值
            # 平均差异百分比
            delta_text = f"{mean_differences[i] / np.mean(baseline_data) * 100:.2f}%"
            if(mean_differences[i] > 0):
                p_text += f"\nΔ=+{delta_text}"
            else:
                p_text += f"\nΔ={delta_text}"

            # ax.text((x1 + x2) / 2, y_height + y_range * 0.02, p_text,
            #         ha='center', va='bottom', fontsize=10)
            ax.text(1.05, y_height + y_range * 0.02, p_text,
                    ha='center', va='bottom', fontsize=12)

        # 调整y轴范围以容纳p值标注
        if p_values:  # 如果有p值需要标注
            ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * (len(p_values) * 0.15))

        ax.set_title("")
        ax.get_xaxis().set_visible(False)
        ax.legend().set_visible(False)
        ax.set_ylabel(feature_name_maps[feature], fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)

    # 隐藏不需要的子图
    for ax in axes[num_features:]:
        ax.axis('off')

    plt.tight_layout(pad=2.0)
    plt.savefig(box_file)
    plt.close()


if __name__ == '__main__':
    num_false_set = [1, 5, 10, 15, 20, 25, 30, 35, 40]

    # gt_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_unified_GS"
    # out_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_unified_Auto"

    gt_dir = r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab"

    work_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet"
    loss_list = ['baseline', 'cldice', 'skelrec', 'newcel_0.1']
    out_dirs = [os.path.join(work_dir, loss) + "/8_estimated_radius_swc" for loss in loss_list]

    # for out_dir in out_dirs:
    #     print('processing:', out_dir)
    #     # out_dir = r"/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/8_estimated_radius_swc"
    #     # img_dir = r"/data/kfchen/trace_ws/to_gu/img"
    #     # save_path = r"/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/result.txt"
    #     csv_file = out_dir + "_opt_result.csv"
    #
    #     # my_opt(gt_dir, out_dir, csv_file)
    #
    #     # 重新处理id
    #     # df = pd.read_csv(csv_file)
    #     # new_id = []
    #     # for i in range(len(df)):
    #     #     new_id.append(str(int(df.iloc[i, 0].split('.')[0].split('_')[0])))
    #     # df['ID'] = new_id
    #     # df.to_csv(csv_file.replace('.csv', '_new.csv'), index=False)
    #
    #     csv_file = out_dir + "_opt_result_new.csv"
    #     df_opt = pd.read_csv(csv_file)
    #     # gold_sample_ids1, silver_sample_ids1, other_sample_ids1 = find_good_sample_opt(df)
    #
    #     l_measure_file = out_dir + "_l_measure.csv"
    #     df_l = pd.read_csv(l_measure_file)
    #     gs_l_measure_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_l_measure.csv"
    #     df_gs_l = pd.read_csv(gs_l_measure_file)
    #     # gold_sample_ids2, silver_sample_ids2, other_sample_ids2 = find_good_sample_l_measure(df_l, df_gs_l)
    #
    #     find_good_sample_combined(df_opt, df_l, df_gs_l)

    plot_box_of_swc_list([out_dir + "_opt_result.csv" for out_dir in out_dirs],
                         ['Baseline', 'clDice', 'SkelRec', 'Proposed'],
                         "/home/kfchen/neuron_seg_human/neuroutils/tests/060525/opt_box.png")










