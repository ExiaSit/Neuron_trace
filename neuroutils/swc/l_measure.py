##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-07
#Description:
##########################################################
import os, glob
import subprocess
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
from subprocess import TimeoutExpired
import shutil
import tempfile
import atexit
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Patch


__FEAT_NAMES22__ = [
    'Nodes', 'SomaSurface', 'Stems', 'Bifurcations',
    'Branches', 'Tips', 'OverallWidth', 'OverallHeight', 'OverallDepth',
    'AverageDiameter', 'Length', 'Surface', 'Volume',
    'MaxEuclideanDistance', 'MaxPathDistance', 'MaxBranchOrder',
    'AverageContraction', 'AverageFragmentation',
    'AverageParent-daughterRatio', 'AverageBifurcationAngleLocal',
    'AverageBifurcationAngleRemote', 'HausdorffDimension'
]

FEAT_NAME_DICT = {
    'N_node': 'Nodes',
    'Soma_surface': 'SomaSurface',
    'N_stem': 'Stems',
    'Number of Bifurcatons': 'Bifurcations',
    'Number of Branches': 'Branches',
    'Number of Tips': 'Tips',
    'Overall Width': 'OverallWidth',
    'Overall Height': 'OverallHeight',
    'Overall Depth': 'OverallDepth',
    'Average Diameter': 'AverageDiameter',
    'Total Length': 'Length',
    'Total Surface': 'Surface',
    'Total Volume': 'Volume',
    'Max Euclidean Distance': 'MaxEuclideanDistance',
    'Max Path Distance': 'MaxPathDistance',
    'Max Branch Order': 'MaxBranchOrder',
    'Average Contraction': 'AverageContraction',
    'Average Fragmentation': 'AverageFragmentation',
    'Average Parent-daughter Ratio': 'AverageParent-daughterRatio',
    'Average Bifurcation Angle Local': 'AverageBifurcationAngleLocal',
    'Average Bifurcation Angle Remote': 'AverageBifurcationAngleRemote',
    'Hausdorff Dimension': 'HausdorffDimension'
}



# 全局临时目录（进程退出时自动清理）
TEMP_DIR = tempfile.mkdtemp(prefix="vaa3d_tmp_")
atexit.register(lambda: shutil.rmtree(TEMP_DIR, ignore_errors=True))

def _create_temp_copy(src_swc):
    """创建临时副本，返回无空格的安全路径"""
    # 生成随机文件名（确保无空格和特殊字符）
    temp_name = f"tmp_{os.urandom(4).hex()}.swc"
    temp_path = os.path.join(TEMP_DIR, temp_name)

    # 创建硬链接（比复制更快，且不占双倍空间）
    #try:
    #    os.link(src_swc, temp_path)  # 硬链接
    #except OSError:
    shutil.copy2(src_swc, temp_path)  # 回退到实际复制

    return temp_path


def calc_global_features(swc_file, vaa3d='/opt/Vaa3D_x.1.1.4_ubuntu/Vaa3D-x', timeout=60):
    if ' ' in swc_file:
        # The naming is extremely stupid, but I must handle it anyway!
        temp_path = _create_temp_copy(swc_file)
        cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x global_neuron_feature -f compute_feature -i "{temp_path}"'
    else:
        cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x global_neuron_feature -f compute_feature -i "{swc_file}"'

    try:
        # 启动子进程并设置超时
        p = subprocess.run(
            cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,  # 单位：秒
            text=True
        )
    except TimeoutExpired:
        # 超时后强制终止进程
        print(f"Timeout ({timeout}s) for file: {swc_file}")
        raise RuntimeError(f"Vaa3D timed out on {swc_file}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Vaa3D failed on {swc_file}: {e.stderr.decode().strip()}")

    # 解析输出，跳过无效行
    output = p.stdout.splitlines()
    info_dict = {}
    for s in output:
        if ':' not in s:  # 跳过不含冒号的行
            continue
        try:
            it1, it2 = s.split(':', 1)  # 仅分割第一个冒号
            it1 = it1.strip()
            it2 = it2.strip()
            if it1 and it2:  # 确保非空
                info_dict[it1] = float(it2) if it2.replace('.', '', 1).isdigit() else it2
        except ValueError:
            print(f"Ignoring malformed line in {swc_file}: {s}")
            continue

    # 检查必要字段是否存在
    required_keys = [
        'N_node', 'Soma_surface', 'N_stem', 'Number of Bifurcatons',
        'Number of Branches', 'Number of Tips', 'Overall Width',
        'Overall Height', 'Overall Depth', 'Average Diameter',
        'Total Length', 'Total Surface', 'Total Volume',
        'Max Euclidean Distance', 'Max Path Distance', 'Max Branch Order',
        'Average Contraction', 'Average Fragmentation',
        'Average Parent-daughter Ratio', 'Average Bifurcation Angle Local',
        'Average Bifurcation Angle Remote', 'Hausdorff Dimension'
    ]
    for key in required_keys:
        if key not in info_dict:
            raise ValueError(f"Missing required key '{key}' in Vaa3D output for {swc_file}")

    # 按固定顺序返回特征值
    features = [
        int(info_dict['N_node']),
        info_dict['Soma_surface'],
        int(info_dict['N_stem']),
        int(info_dict['Number of Bifurcatons']),
        int(info_dict['Number of Branches']),
        int(info_dict['Number of Tips']),
        info_dict['Overall Width'],
        info_dict['Overall Height'],
        info_dict['Overall Depth'],
        info_dict['Average Diameter'],
        info_dict['Total Length'],
        info_dict['Total Surface'],
        info_dict['Total Volume'],
        info_dict['Max Euclidean Distance'],
        info_dict['Max Path Distance'],
        int(info_dict['Max Branch Order']),
        info_dict['Average Contraction'],
        info_dict['Average Fragmentation'],
        info_dict['Average Parent-daughter Ratio'],
        info_dict['Average Bifurcation Angle Local'],
        info_dict['Average Bifurcation Angle Remote'],
        info_dict['Hausdorff Dimension']
    ]
    return features


# helper function
def _wrapper(swcfile, prefix, out_dict, robust=True, timeout=60):
    # print(f'Processing: {prefix}')

    try:  # 修改2：统一异常处理逻辑
        features = calc_global_features(swcfile, timeout=timeout)
        out_dict[prefix] = features
    except Exception as e:
        if robust:
            print(f"Error processing {swcfile}: {str(e)}")
        else:
            raise
#########


def calc_global_features_from_folder(swc_dir, outfile=None, robust=True, nprocessors=8, timeout=60):

    ################## Helper functions #################
    def is_valid_swc(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # 非空且非注释行
                    return True
            return False  # 文件全为空或注释

    ############ End of helper functions ################

    with Manager() as manager:  # 修改3：使用Manager上下文
        out_dict = manager.dict()  # 创建共享字典
        arg_list = []

        # 构建参数列表（包含robust参数）
        iswc = 0
        for swcfile in glob.glob(os.path.join(swc_dir, '*.swc')):
            prefix = os.path.splitext(os.path.basename(swcfile))[0]
            # debug
            iswc += 1
            #if iswc % 10 == 0:
            #    break
            if not is_valid_swc(swcfile):
                print(prefix)
                continue

            arg_list.append((swcfile, prefix, out_dict, robust, timeout))  # 修改4：添加robust参数

        # 修改5：使用with自动管理Pool生命周期
        print('Starts to calculate...')
        with Pool(processes=nprocessors) as pool:
            pool.starmap(_wrapper, arg_list)

        # 从共享字典提取数据
        print('Aggregationg all features')
        features_all = [[k, *v] for k, v in out_dict.items()]

        # 后续处理保持不变
        df = pd.DataFrame(features_all, columns=['', *__FEAT_NAMES22__])
        if outfile is not None:
            df.to_csv(outfile, float_format='%g', index=False)
        return df

def compare_features(gf_file1, gf_file2, outfile=None, features=None, label1=None, label2=None, details=True):
    """
    Compare two global feature files and save the differences to a new file.
    :param gf_file1: Path to the first global feature file (CSV).
    :param gf_file2: Path to the second global feature file (CSV).
    :param outfile: Path to save the comparison results (CSV).
    :param features: List of feature names to compare. If None, all features will be compared.
    """
    # 读取两个CSV文件
    df1 = pd.read_csv(gf_file1, index_col=0)
    df2 = pd.read_csv(gf_file2, index_col=0)
    if(label1 is None):
        label1 = os.path.basename(gf_file1)[:5]
        label2 = os.path.basename(gf_file2)[:5]

    # 提取特征列（排除第一列的名称列）
    if(features is None):
        features = df1.columns
    # features = ['Stems', 'AverageContraction', 'AverageFragmentation', 'AverageParent-daughterRatio',
    #             'AverageBifurcationAngleRemote', 'HausdorffDimension']

    # 设置图形大小
    sns.set_theme(style='ticks', font_scale=1.3)
    nrows = int(np.ceil(len(features) / 4))
    plt.figure(figsize=(12, 12 / 4 * nrows))

    # 为每个特征创建一个子图
    for i, feature in enumerate(features, 1):
        plt.subplot(nrows, 4, i)  # 6行4列的子图布局

        # 获取两个数据集的值
        values1 = df1[feature].values
        values2 = df2[feature].values

        if(details):
            # 输出平均值
            mean1 = np.mean(values1)
            mean2 = np.mean(values2)
            std1 = np.std(values1)
            std2 = np.std(values2)
            print(f"{feature} - {label1}: mean={mean1:.2f}, std={std1:.2f}")
            print(f"{feature} - {label2}: mean={mean2:.2f}, std={std2:.2f}")
            print("-----")

        # 确定合适的bins
        combined = np.concatenate([values1, values2])
        bins = np.linspace(min(combined), max(combined), 40)

        # 绘制直方图
        plt.hist(values1, bins=bins, alpha=0.5, label=label1, color='blue', density=True)
        plt.hist(values2, bins=bins, alpha=0.5, label=label2, color='orange', density=True)

        plt.title(feature)

    # 在最后一个子图上添加图例
    plt.subplot(nrows, 4, len(features) + 1)  # 留出一个空白子图
    plt.axis('off')  # 隐藏坐标轴
    legend_handles = [
        Patch(facecolor='blue', edgecolor='blue', alpha=0.5, label=label1),
        Patch(facecolor='orange', edgecolor='orange', alpha=0.5, label=label2)
    ]
    plt.legend(handles=legend_handles, loc='center', fontsize=12, title='Legend', title_fontsize='13')

    # 调整布局防止重叠
    plt.tight_layout()
    plt.suptitle(f'Feature Comparison between {label1} and {label2}', fontsize=16, y=1.02)
    plt.savefig(outfile, dpi=300)
    plt.close()
