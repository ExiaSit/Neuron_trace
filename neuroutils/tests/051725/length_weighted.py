import numpy as np
import pandas as pd
import networkx as nx
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import tifffile
from skimage.transform import resize
import shutil

feature_name_maps = {
    'Number_of_Branches': 'No. of Branches',
    'Total_Length': 'Length',
    'Max_Path_Distance': 'Max Path Dist.',
    'N_stem': 'No. of Stems',
    'Number_of_Tips': 'No. of Tips',
    'Max_Branch_Order': 'Max Branch Order',
    'N_node': 'No. of Nodes',
    'Number_of_Bifurcatons': 'No. of Bifurcations',
    'Overall_Width': 'Width',
    'Overall_Height': 'Height',
    'Overall_Depth': 'Depth',
    'Max_Euclidean Distance': 'Max Euclidean',
    "Distence_Weighted_Number_of_Tips": "DW No. of Tips",
    "Distence_Weighted_Number_of_Branches": "DW No. of Branches",
    "Distence_Weighted_Number_of_Bifurcatons": "DW No. of Bifurcations",
    "Distence_Weighted_Total_Length": "DW Length",
    "Length_Weighted_Number_of_Tips": "No. of Tips*",
    "Length_Weighted_Number_of_Branches": "No. of Branches*",
    "Length_Weighted_Number_of_Bifurcatons": "No. of Bifurcations*",
}

def dist_weighted_map(current_x, x_max=200):
    # x = np.linspace(0, x_max, 1000)
    # y = np.where(x <= x_max, 1 - x / x_max, 0)
    # return np.interp(current_x, x, y)
    # x_max = 100
    if(current_x <= x_max):
        return 1 - current_x / x_max
    else:
        return 0

def length_weighted_map(current_x, x_max=200):
    return current_x / x_max

class L_measure_calculator:
    def __init__(self, G, img_shape=None):
        self.G = G
        self.nodes = list(G.nodes())
        self.edges = list(G.edges())
        self.num_of_nodes = len(self.nodes)
        self.num_of_edges = len(self.edges)
        self.max_dist = np.sqrt(img_shape[0]**2 + img_shape[1]**2 + img_shape[2]**2) / 2

        self.l_measure = {
            'Number_of_Tips': self.get_num_of_tips(),
            'Length_Weighted_Number_of_Tips': self.get_length_weighted_num_of_tips(),
            'Number_of_Branches': self.get_num_of_branches(),
            'Length_Weighted_Number_of_Branches': self.get_length_weighted_num_of_branches(),
            'Number_of_Bifurcatons': self.get_num_of_bifurcations(),
            # 'Total_Length': self.get_total_length(),
            # 'Distence_Weighted_Number_of_Tips': self.get_dist_weighted_num_of_tips(),
            # 'Distence_Weighted_Number_of_Branches': self.get_dist_weighted_num_of_branches(),
            # 'Distence_Weighted_Number_of_Bifurcatons': self.get_dist_weighted_num_of_bifurcations(),
            # 'Distence_Weighted_Total_Length': self.get_dist_weighted_total_length(),


            'Length_Weighted_Number_of_Bifurcatons': self.get_length_weighted_num_of_bifurcations(),
        }

    def get_num_of_tips(self):
        num_of_tips = 0
        for node in self.nodes:
            if self.G.out_degree(node) == 0:
                num_of_tips += 1.0
        return num_of_tips

    def get_dist_weighted_num_of_tips(self):
        soma = self.G.nodes[1]
        tip_node_list = []
        max_dist = self.max_dist
        for node in self.nodes:
            if self.G.out_degree(node) == 0:
                tip_node_list.append(node)

        dist_weighted_num_of_tips = 0
        for tip_node in tip_node_list:
            x, y, z = self.G.nodes[tip_node]['x'], self.G.nodes[tip_node]['y'], self.G.nodes[tip_node]['z']
            dist = np.sqrt((x-soma['x'])**2 + (y-soma['y'])**2 + (z-soma['z'])**2)
            # print(dist, max_dist, dist_weighted_map(dist, max_dist))
            dist_weighted_num_of_tips += dist_weighted_map(dist, max_dist) * 1.0
        return dist_weighted_num_of_tips

    def get_length_weighted_num_of_tips(self):
        tip_node_list = []
        max_dist = self.max_dist
        for node in self.nodes:
            if self.G.out_degree(node) == 0:
                tip_node_list.append(node)

        num_of_tips = 0
        for tip_node in tip_node_list:
            current_branch_nodes = [tip_node]
            branch_length = 0
            x_list, y_list, z_list = ([self.G.nodes[tip_node]['x']],
                                      [self.G.nodes[tip_node]['y']],
                                      [self.G.nodes[tip_node]['z']])
            while self.G.out_degree(current_branch_nodes[-1]) == 1 or self.G.out_degree(current_branch_nodes[-1]) == 0:
                if(len(list(self.G.predecessors(current_branch_nodes[-1]))) == 0):
                    break
                current_branch_nodes.append(list(self.G.predecessors(current_branch_nodes[-1]))[0])
                x_list.append(self.G.nodes[current_branch_nodes[-1]]['x'])
                y_list.append(self.G.nodes[current_branch_nodes[-1]]['y'])
                z_list.append(self.G.nodes[current_branch_nodes[-1]]['z'])
                branch_length += np.sqrt((x_list[-1]-x_list[-2])**2 + (y_list[-1]-y_list[-2])**2 + (z_list[-1]-z_list[-2])**2)

            num_of_tips += length_weighted_map(branch_length, max_dist)
        return num_of_tips



    def get_num_of_branches(self):
        num_of_branches = 0
        for node in self.nodes:
            if self.G.out_degree(node) > 1:
                num_of_branches += self.G.out_degree(node)
        return num_of_branches

    def get_dist_weighted_num_of_branches(self):
        soma = self.G.nodes[1]
        branch_start_node_list = []
        for node in self.nodes:
            if self.G.out_degree(node) > 1:
                branch_start_node_list.append(node)
        dist_weighted_num_of_branches = 0
        for branch_start_node in branch_start_node_list:
            current_branch_nodes = [branch_start_node]
            x_list, y_list, z_list = ([self.G.nodes[branch_start_node]['x']],
                                      [self.G.nodes[branch_start_node]['y']],
                                      [self.G.nodes[branch_start_node]['z']])
            while self.G.out_degree(current_branch_nodes[-1]) == 1:
                current_branch_nodes.append(list(self.G.successors(current_branch_nodes[-1]))[0])
                x_list.append(self.G.nodes[current_branch_nodes[-1]]['x'])
                y_list.append(self.G.nodes[current_branch_nodes[-1]]['y'])
                z_list.append(self.G.nodes[current_branch_nodes[-1]]['z'])
            mean_x, mean_y, mean_z = np.mean(x_list), np.mean(y_list), np.mean(z_list)
            dist = np.sqrt((mean_x-soma['x'])**2 + (mean_y-soma['y'])**2 + (mean_z-soma['z'])**2)
            dist_weighted_num_of_branches += dist_weighted_map(dist, self.max_dist)
        return dist_weighted_num_of_branches

    def get_length_weighted_num_of_branches(self):
        branch_start_node_list = []
        for node in self.nodes:
            if self.G.out_degree(node) > 1 and node != 1:
                # node 的 子节点
                for child in self.G.successors(node):
                    branch_start_node_list.append(child)
        num_of_branches = 0
        for branch_start_node in branch_start_node_list:
            current_branch_nodes = [branch_start_node]
            previous_node = list(self.G.predecessors(branch_start_node))[0]
            branch_length = np.sqrt((self.G.nodes[branch_start_node]['x']-self.G.nodes[previous_node]['x'])**2 +
                                    (self.G.nodes[branch_start_node]['y']-self.G.nodes[previous_node]['y'])**2 +
                                    (self.G.nodes[branch_start_node]['z']-self.G.nodes[previous_node]['z'])**2)
            x_list, y_list, z_list = ([self.G.nodes[branch_start_node]['x']],
                                      [self.G.nodes[branch_start_node]['y']],
                                      [self.G.nodes[branch_start_node]['z']])
            while self.G.out_degree(current_branch_nodes[-1]) == 1:
                current_branch_nodes.append(list(self.G.successors(current_branch_nodes[-1]))[0])
                x_list.append(self.G.nodes[current_branch_nodes[-1]]['x'])
                y_list.append(self.G.nodes[current_branch_nodes[-1]]['y'])
                z_list.append(self.G.nodes[current_branch_nodes[-1]]['z'])
                branch_length += np.sqrt((x_list[-1]-x_list[-2])**2 + (y_list[-1]-y_list[-2])**2 + (z_list[-1]-z_list[-2])**2)
            num_of_branches += length_weighted_map(branch_length, self.max_dist)
        return num_of_branches

    def get_num_of_bifurcations(self):
        num_of_bifurcations = 0
        for node in self.nodes:
            if self.G.out_degree(node) > 1:
                num_of_bifurcations += 1
        return num_of_bifurcations

    def get_dist_weighted_num_of_bifurcations(self):
        soma = self.G.nodes[1]
        bifurcation_node_list = []
        for node in self.nodes:
            if self.G.out_degree(node) > 1:
                bifurcation_node_list.append(node)
        dist_weighted_num_of_bifurcations = 0
        for bifurcation_node in bifurcation_node_list:
            x, y, z = self.G.nodes[bifurcation_node]['x'], self.G.nodes[bifurcation_node]['y'], self.G.nodes[bifurcation_node]['z']
            dist = np.sqrt((x-soma['x'])**2 + (y-soma['y'])**2 + (z-soma['z'])**2)
            dist_weighted_num_of_bifurcations += dist_weighted_map(dist, self.max_dist)
        return dist_weighted_num_of_bifurcations

    def get_length_weighted_num_of_bifurcations(self):
        bifurcation_node_list = []
        for node in self.nodes:
            if self.G.out_degree(node) > 1:
                bifurcation_node_list.append(node)
        num_of_bifurcations = 0
        for bifurcation_node in bifurcation_node_list:
            total_connected_length = 0
            for child in self.G.successors(bifurcation_node):
                current_branch_nodes = [child]
                branch_length = np.sqrt((self.G.nodes[child]['x']-self.G.nodes[bifurcation_node]['x'])**2 +
                                        (self.G.nodes[child]['y']-self.G.nodes[bifurcation_node]['y'])**2 +
                                        (self.G.nodes[child]['z']-self.G.nodes[bifurcation_node]['z'])**2)
                x_list, y_list, z_list = ([self.G.nodes[child]['x']],
                                          [self.G.nodes[child]['y']],
                                          [self.G.nodes[child]['z']])
                while self.G.out_degree(current_branch_nodes[-1]) == 1:
                    current_branch_nodes.append(list(self.G.successors(current_branch_nodes[-1]))[0])
                    x_list.append(self.G.nodes[current_branch_nodes[-1]]['x'])
                    y_list.append(self.G.nodes[current_branch_nodes[-1]]['y'])
                    z_list.append(self.G.nodes[current_branch_nodes[-1]]['z'])
                    branch_length += np.sqrt((x_list[-1]-x_list[-2])**2 + (y_list[-1]-y_list[-2])**2 + (z_list[-1]-z_list[-2])**2)
                total_connected_length += branch_length
            num_of_bifurcations += length_weighted_map(total_connected_length, self.max_dist)
        return num_of_bifurcations

    def get_total_length(self):
        total_length = 0
        for edge in self.edges:
            n1 = edge[0]
            n2 = edge[1]
            x1, y1, z1 = self.G.nodes[n1]['x'], self.G.nodes[n1]['y'], self.G.nodes[n1]['z']
            x2, y2, z2 = self.G.nodes[n2]['x'], self.G.nodes[n2]['y'], self.G.nodes[n2]['z']
            total_length += np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        return total_length

    def get_dist_weighted_total_length(self):
        soma = self.G.nodes[1]
        total_length = 0
        for edge in self.edges:
            n1 = edge[0]
            n2 = edge[1]
            x1, y1, z1 = self.G.nodes[n1]['x'], self.G.nodes[n1]['y'], self.G.nodes[n1]['z']
            x2, y2, z2 = self.G.nodes[n2]['x'], self.G.nodes[n2]['y'], self.G.nodes[n2]['z']
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            dist_to_soma = np.sqrt(((x1+x2)/2-soma['x'])**2 + ((y1+y2)/2-soma['y'])**2 + ((z1+z2)/2-soma['z'])**2)
            total_length += dist_weighted_map(dist_to_soma, self.max_dist) * dist
        return total_length

def read_swc(swc_file):
    swc = pd.read_csv(swc_file, sep=' ', header=None, comment='#')
    swc.columns = ['n', 'type', 'x', 'y', 'z', 'r', 'parent']
    return swc

def generate_tree_from_swc_file(swc_file, img_file):
    img = tifffile.imread(img_file)
    img_shape = img.shape

    swc = read_swc(swc_file)
    G = nx.DiGraph()
    for i in range(swc.shape[0]):
        n = swc.iloc[i]['n']
        x = swc.iloc[i]['x']
        y = swc.iloc[i]['y']
        z = swc.iloc[i]['z']
        r = swc.iloc[i]['r']
        parent = swc.iloc[i]['parent']
        G.add_node(n, x=x, y=y, z=z, r=r)
        if parent != -1:
            G.add_edge(parent, n)
    return G, img_shape

def l_measure_swc_file(swc_file, img_file):
    if(not os.path.exists(swc_file) or not os.path.exists(img_file)):
        # print(f"{swc_file} or {img_file} not exist")
        return None
    try:
        G, img_shape = generate_tree_from_swc_file(swc_file, img_file)
        l_measure_calculator = L_measure_calculator(G, img_shape)
        result = {
            'id': str(int(os.path.basename(swc_file).split('.')[0].split('_')[0])),
        }
        result.update(l_measure_calculator.l_measure)

        return result
    except Exception as e:
        print(f"Error in {swc_file}: {e}")
        return None

def l_measure_swc_dir(swc_dir, img_dir, todo_list):
    swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc')]
    # filter swc_files by todo_list
    swc_files = [f for f in swc_files if int(os.path.basename(f).split('.')[0].split('_')[0]) in todo_list]

    results = Parallel(n_jobs=20)(
        delayed(l_measure_swc_file)(
            os.path.join(swc_dir, swc_file),
            os.path.join(img_dir, swc_file.replace('.swc', '.tif'))
        ) for swc_file in tqdm(swc_files)
    )
    # filter None
    results = [result for result in results if result is not None]
    results = pd.DataFrame(results)

    return results

def get_comp_l_measure_df(pred_lm_df, gt_lm_df):
    pred_id_list = pred_lm_df['id'].tolist()
    gt_id_list = gt_lm_df['id'].tolist()
    common_id_list = list(set(pred_id_list) & set(gt_id_list))
    print(len(common_id_list))

    comp_lm_df = pd.DataFrame()
    for id in tqdm(common_id_list):
        # add row
        comp_lm_df.loc[id, 'id'] = id
        pred_row = pred_lm_df[pred_lm_df['id'] == id]
        gt_row = gt_lm_df[gt_lm_df['id'] == id]
        for feature in pred_row.columns:
            if feature == 'id':
                continue
            comp_lm_df.loc[id, feature] = pred_row[feature].values[0] / gt_row[feature].values[0]

    comp_lm_df = comp_lm_df.reset_index(drop=True)
    return comp_lm_df

def plot_comp_l_measure_df(comp_lm_df):
    todo_list_file = "/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/test232.csv"
    todo_list = pd.read_csv(todo_list_file)
    todo_list = todo_list['ID'].tolist()

    tab20_colors = plt.cm.get_cmap('tab20c', 20).colors
    colors = [tab20_colors[9], tab20_colors[5],
              tab20_colors[10], tab20_colors[6],
              tab20_colors[11], tab20_colors[7],
              ]

    comp_lm_df = comp_lm_df.dropna()
    comp_lm_df = comp_lm_df[comp_lm_df['id'].isin(todo_list)]
    # drop id
    comp_lm_df = comp_lm_df.drop(columns=['id'])
    print(len(comp_lm_df))


    fig = plt.figure(figsize=(5, 5))
    #
    position = range(comp_lm_df.shape[1])
    # print(position)
    for i, feature in enumerate(comp_lm_df.columns):
        current_feature = comp_lm_df[feature].values
        # print(f"{feature}: {current_feature}")
        print(f"{feature}: {np.median(current_feature):.2f}")

        # 检查是否有nan
        if np.isnan(current_feature).any():
            print(f"{feature} has nan")
        # 检查是否有inf
        if np.isinf(current_feature).any():
            print(f"{feature} has inf")
            print(current_feature)
        # violin
        violin_parts = plt.violinplot(current_feature, positions=[position[i]], widths=0.8,
                                      showmeans=False, showmedians=False, showextrema=False,
                                      )
        for partname in ['bodies']:
            for part in violin_parts[partname]:
                part.set_edgecolor('black')  # 设置边缘线的颜色
                part.set_linewidth(1)  # 设置边缘线的宽度
                part.set_facecolor(colors[i])  # 设置填充颜色
                # alpha
                part.set_alpha(1)

        plt.boxplot(current_feature,
                    positions=[position[i]], widths=0.4,
                    patch_artist=True,
                    showfliers=True,
                    boxprops=dict(color='black', linewidth=1, facecolor='white'),

                    capprops=dict(color='black'),
                    medianprops=dict(color='black'),
                    flierprops=dict(marker='o', color='black', markersize=3)
                    )

        # print(f"{feature}: {np.mean(current_feature)}")


    plt.xticks(position, [feature_name_maps[feature] for feature in comp_lm_df.columns if feature != 'id'],
               rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=12)
    # 在y=1
    plt.axhline(y=1, color='gray', linestyle='--')
    plt.axhline(y=0.9, color='gray', linestyle='--')
    plt.axhline(y=0.8, color='gray', linestyle='--')
    # plt.xticks(position, [feature for feature in comp_lm_df.columns if feature != 'id'], rotation=45)
    plt.ylim(0.25, 1.75)
    # plt.title('Comparison of L-measure')
    plt.tight_layout()
    plt.show()
    plt.close()

def prepare_rescaled_img(source_img_dir, target_img_dir):
    neuron_info_df = pd.read_csv(
        "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv",
        encoding='gbk')
    df = neuron_info_df
    def current_task(source_img_file, target_img_file):
        if(not os.path.exists(source_img_file) or os.path.exists(target_img_file)):
            return
        id = int(os.path.basename(source_img_file).split('.')[0].split('_')[0])
        xy_resolution = df.loc[df.iloc[:, 0] == id, 'xy拍摄分辨率(*10e-3μm/px)'].values[0]
        xy_resolution = float(xy_resolution) / 1000

        img = tifffile.imread(source_img_file)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = resize(img, (img.shape[0], int(img.shape[1]*xy_resolution), int(img.shape[2]*xy_resolution)), order=2)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype('uint8')
        print(img.shape)
        tifffile.imwrite(target_img_file, img)

    img_files = [f for f in os.listdir(source_img_dir) if f.endswith('.tif')]
    Parallel(n_jobs=20)(delayed(current_task)(os.path.join(source_img_dir, img_file), os.path.join(target_img_dir, img_file)) for img_file in tqdm(img_files))

def rename_swcs():
    source_dir = "/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/data/auto8.4k_0510_pruned_resample1um"
    target_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/test/auto8.4k_0510_pruned_resample1um"
    os.makedirs(target_dir, exist_ok=True)
    swc_files = [f for f in os.listdir(source_dir) if f.endswith('.swc')]
    for swc_file in tqdm(swc_files):
        id = int(os.path.basename(swc_file).split('.')[0].split('_')[0])
        new_name = f"{id}.swc"
        shutil.copy(os.path.join(source_dir, swc_file), os.path.join(target_dir, new_name))
        print(f"rename {swc_file} to {new_name}")

    source_csv = "/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/auto8.4k_0510_pruned_resample1um_renamed.csv"
    target_csv = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/test/auto8.4k_0510_pruned_resample1um.csv"
    df = pd.read_csv(source_csv)
    for i in range(len(df)):
        id = int(df.iloc[i, 0].split('.')[0].split('_')[0])
        new_name = id
        df.iloc[i, 0] = new_name
    df.to_csv(target_csv, index=False)


if __name__ == '__main__':
    # rename_swcs()


    img_dir = "/data/kfchen/trace_ws/to_gu/rescaled_img"
    # os.makedirs(img_dir, exist_ok=True)
    # prepare_rescaled_img("/data/kfchen/trace_ws/to_gu/img", img_dir)
    # exit()
    todo_list_file = "/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/test232.csv"
    todo_list = pd.read_csv(todo_list_file)
    todo_list = todo_list['ID'].tolist()

    result_comp_lm_df_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/test/comp_lm_df_pruned.csv"
    if os.path.exists(result_comp_lm_df_file):
        comp_lm_df = pd.read_csv(result_comp_lm_df_file)
    else:
        print("!!!!!!!!!!!!!!")
        swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/test/auto8.4k_0510_pruned_resample1um"
        auto_lm_df = l_measure_swc_dir(swc_dir, img_dir, todo_list)
        print(auto_lm_df.shape)

        swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample_prune"
        manual_lm_df = l_measure_swc_dir(swc_dir, img_dir, todo_list)
        print(manual_lm_df.shape)

        comp_lm_df = get_comp_l_measure_df(auto_lm_df, manual_lm_df)
        # save comp_lm_df
        comp_lm_df.to_csv(result_comp_lm_df_file, index=False)

    # print full df
    pd.set_option('display.max_rows', None)  # 不限制显示行数
    # pd.set_option('display.max_columns', None)  # 不限制显示列数
    pd.set_option('display.width', None)  # 自动调整宽度
    pd.set_option('display.max_colwidth', None)  # 显示列的最大宽度
    print(comp_lm_df)

    plot_comp_l_measure_df(comp_lm_df)





