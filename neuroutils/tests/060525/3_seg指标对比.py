import os
import time

from PIL import Image
import numpy as np
import cc3d
import tifffile

from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import skeletonize_3d

import concurrent.futures
from tqdm import tqdm

import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.stats import ttest_ind
import cv2
import gudhi as gd

from scipy.stats import ttest_rel


def close_operation_3d(image, structure=None):
    """
    对三维图像进行闭操作。

    :param image: 输入的三维numpy数组
    :param structure: 结构元素，用于定义邻域。如果为None，则使用默认的结构元素。
    :return: 经过闭操作的图像
    """

    # 先膨胀后腐蚀
    dilated = binary_dilation(image, structure=structure)
    closed_image = binary_erosion(dilated, structure=structure)

    return closed_image


def find_largest_connected_components(img_data, target_cc_num):
    # 使用cc3d计算连通块
    labels_out, N = cc3d.connected_components(img_data, connectivity=26, return_N=True)

    # 统计每个标签的体素数
    label_sizes = np.bincount(labels_out.ravel())
    label_sizes[0] = 0  # 忽略背景的体素

    # 找到体积最大的n个连通块的标签
    largest_labels = np.argsort(label_sizes)[-target_cc_num:]

    # 创建一个数组用来存储最大的n个连通块的数据
    largest_components = np.zeros_like(labels_out)

    # 提取每个连通块并放入largest_components数组
    for label in largest_labels:
        largest_components[labels_out == label] = label

    return largest_components


def get_single_c_relative_coverage(seg_data, target_cc_num):
    close_data = close_operation_3d(seg_data)
    max_cc_seg_data = find_largest_connected_components(close_data, target_cc_num)

    skel_seg = skeletonize_3d(seg_data > 0)
    skel_max_cc = skeletonize_3d(max_cc_seg_data > 0)

    if(np.sum(skel_seg > 0) > 0):
        return np.sum(skel_max_cc > 0) / np.sum(skel_seg > 0)
    else:
        return 0

# 计算相对前景比例
def get_relative_foreground_ratio(seg_data, gt_data):
# 计算前景比例
    seg_foreground_ratio = np.sum(seg_data > 0) / np.sum(seg_data >= 0)
    gt_foreground_ratio = np.sum(gt_data > 0) / np.sum(gt_data >= 0)

    # 计算相对前景比例
    relative_foreground_ratio = seg_foreground_ratio / gt_foreground_ratio
    return relative_foreground_ratio


def get_single_overlap(seg_data, gt_data):
    # 计算交集和并集
    intersection = np.sum((seg_data > 0) & (gt_data > 0))
    union = np.sum((seg_data > 0) | (gt_data > 0))

    # 计算Jaccard指数（Overlap）
    if union == 0:
        jaccard_index = 1.0  # 避免除以零的情况
    else:
        jaccard_index = intersection / union
    return jaccard_index

def get_single_c_overlap(seg_data, gt_data, target_cc_num):
    seg_data = close_operation_3d(seg_data)
    seg_data = find_largest_connected_components(seg_data, target_cc_num)
    # 计算交集和并集
    intersection = np.sum((seg_data > 0) & (gt_data > 0))
    union = np.sum((seg_data > 0) | (gt_data > 0))

    # 计算Jaccard指数（Overlap）
    if union == 0:
        jaccard_index = 1.0  # 避免除以零的情况
    else:
        jaccard_index = intersection / union
    return jaccard_index

def dice_coefficient(seg, gt):
    # print(seg.shape, gt.shape)
    intersection = np.sum(seg[gt > 0])
    seg_sum = np.sum(seg)
    gt_sum = np.sum(gt)
    dice = 2 * intersection / (seg_sum + gt_sum)
    return dice


def get_single_c_dice(seg, gt, target_cc_num):
    seg = close_operation_3d(seg)
    seg = find_largest_connected_components(seg, target_cc_num)
    return dice_coefficient(seg > 0, gt > 0)

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def get_cldice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    v_p = np.where(v_p > 0, 1, 0)
    v_l = np.where(v_l > 0, 1, 0)
    tprec = cl_score(v_p,skeletonize_3d(v_l))
    tsens = cl_score(v_l,skeletonize_3d(v_p))
    result = 2*tprec*tsens/(tprec+tsens)
    # print(tprec, tsens, result)
    return result

def get_fingures_single_result(seg_folder, gt_folder, seg_file, gt_file, target_cc_num):
    # Load the segmentation result and ground truth image
    seg_path = os.path.join(seg_folder, seg_file)
    gt_path = os.path.join(gt_folder, gt_file)

    if(seg_path[-4:] == ".tif"):
        seg_img = tifffile.imread(seg_path)
        gt_img = tifffile.imread(gt_path)
    elif(seg_path[-7:] == ".nii.gz"):
        seg_img = nib.load(seg_path)
        gt_img = nib.load(gt_path)
        seg_img = seg_img.get_fdata()
        gt_img = gt_img.get_fdata()

    # Convert images to NumPy arrays
    seg_data = np.array(seg_img).astype(np.uint8)
    gt_data = np.array(gt_img).astype(np.uint8)
    # print(seg_data.shape, gt_data.shape)

    # print(seg_file)
    if(seg_file[:5] == 'image'):
        seg_id = seg_file.split('_')[1].replace('.tif', '')
    else:
        seg_id = seg_file.split('_')[0]

    return (seg_id,
            dice_coefficient(seg_data > 0, gt_data > 0),
            get_single_c_dice(seg_data > 0, gt_data > 0, target_cc_num),
            get_single_overlap(seg_data, gt_data),
            get_single_c_overlap(seg_data, gt_data, target_cc_num),
            get_single_c_relative_coverage(seg_data, target_cc_num),
            get_relative_foreground_ratio(seg_data, gt_data),
            get_single_broken_points(seg_data, gt_data),
            get_skel_accuracy(seg_data, gt_data),
            # calc_betti(seg_data),
            get_cldice(seg_data, gt_data)
    )

def calc_neuron_radius(img, x, y, z):
    indices = np.argwhere(img)
    # print(indices)
    distances = np.sqrt(((indices - np.array([z, y, x])) ** 2).sum(axis=1))
    max_distance = np.max(distances)
    # print(max_distance, img.shape)
    return max_distance

def create_spherical_masks(center, shape, radius, dist_threshold=[0.25, 0.75]):
    z, y, x = np.indices(shape)
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    mask_inner = distances < (dist_threshold[0] * radius)
    mask_middle = (distances >= (dist_threshold[0] * radius)) & (distances < (dist_threshold[1] * radius))
    mask_outer = distances >= (dist_threshold[1] * radius)
    return mask_inner, mask_middle, mask_outer

def get_graded_fingures_single_result(seg_folder, gt_folder, soma_folder, seg_file, gt_file, soma_file, target_cc_num):
    # Load the segmentation result and ground truth image
    seg_path = os.path.join(seg_folder, seg_file)
    gt_path = os.path.join(gt_folder, gt_file)
    soma_path = os.path.join(soma_folder, soma_file)

    if(seg_path[-4:] == ".tif"):
        seg_img = tifffile.imread(seg_path)
        gt_img = tifffile.imread(gt_path)
    elif(seg_path[-7:] == ".nii.gz"):
        seg_img = nib.load(seg_path)
        gt_img = nib.load(gt_path)
        seg_img = seg_img.get_fdata()
        gt_img = gt_img.get_fdata()

    # Convert images to NumPy arrays
    seg_data = np.array(seg_img).astype(np.uint8)
    gt_data = np.array(gt_img).astype(np.uint8)
    # print(seg_data.shape, gt_data.shape)

    # print(seg_file)
    if(seg_file[:5] == 'image'):
        seg_id = seg_file.split('_')[1].replace('.tif', '')
    else:
        seg_id = seg_file.split('_')[0]

    # read soma marker
    with open(soma_path, 'r') as f:
        soma_data = f.readlines()
    # print(soma_data)
    soma_data = soma_data[0].split(',')
    # print(soma_data)
    soma_x, soma_y, soma_z = round(float(soma_data[0])), round(seg_img.shape[1] - 1 - float(soma_data[1][1:])), round(float(soma_data[2][1:]))
    neuron_radius = calc_neuron_radius(gt_data, soma_x, soma_y, soma_z)
    # print(neuron_radius, soma_x, soma_y, soma_z, gt_data.shape)
    mask_inner, mask_middle, mask_outer = create_spherical_masks((soma_x, soma_y, soma_z), gt_data.shape, neuron_radius)

    gt_inner = np.where(mask_inner, gt_data, 0)
    gt_middle = np.where(mask_middle, gt_data, 0)
    gt_outer = np.where(mask_outer, gt_data, 0)

    seg_inner = np.where(mask_inner, seg_data, 0)
    seg_middle = np.where(mask_middle, seg_data, 0)
    seg_outer = np.where(mask_outer, seg_data, 0)

    debug = False
    if (debug):
        break_points_inner, seg_data_inner, skel_gt_inner, tp_skel_inner, fn_skel_inner = get_single_broken_points(seg_inner, gt_inner, debug)
        break_points_middle, seg_data_middle, skel_gt_middle, tp_skel_middle, fn_skel_middle = get_single_broken_points(seg_middle, gt_middle, debug)
        break_points_outer, seg_data_outer, skel_gt_outer, tp_skel_outer, fn_skel_outer = get_single_broken_points(seg_outer, gt_outer, debug)

        skel_accuracy_inner = get_skel_accuracy(seg_inner, gt_inner)
        skel_accuracy_middle = get_skel_accuracy(seg_middle, gt_middle)
        skel_accuracy_outer = get_skel_accuracy(seg_outer, gt_outer)

        mip_list = [
            np.max(seg_data_inner, axis=0) * 255,
            np.max(skel_gt_inner, axis=0) * 255,
            np.max(tp_skel_inner, axis=0) * 255,
            np.max(fn_skel_inner, axis=0) * 255,

            np.max(seg_data_middle, axis=0) * 255,
            np.max(skel_gt_middle, axis=0) * 255,
            np.max(tp_skel_middle, axis=0) * 255,
            np.max(fn_skel_middle, axis=0) * 255,

            np.max(seg_data_outer, axis=0) * 255,
            np.max(skel_gt_outer, axis=0) * 255,
            np.max(tp_skel_outer, axis=0) * 255,
            np.max(fn_skel_outer, axis=0) * 255,
        ]
        mip_list = [f.astype(np.uint8) for f in mip_list]
        mip = [(f - np.min(f)) / (np.max(f) - np.min(f)) * 255 for f in mip_list]
        # mip = np.concatenate(mip, axis=1)
        mip_list = [
            np.concatenate(mip[:4], axis=1),
            np.concatenate(mip[4:8], axis=1),
            np.concatenate(mip[8:], axis=1),
        ]
        mip = np.concatenate(mip_list, axis=0)

        # text break_points_inner
        cv2.putText(mip, f"Break Points Inner: {break_points_inner}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(mip, f"Break Points Middle: {break_points_middle}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(mip, f"Break Points Outer: {break_points_outer}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(mip, f"Skel Accuracy Inner: {skel_accuracy_inner:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(mip, f"Skel Accuracy Middle: {skel_accuracy_middle:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(mip, f"Skel Accuracy Outer: {skel_accuracy_outer:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        mip_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/graded_mip"
        cv2.imwrite(os.path.join(mip_dir, f"{seg_id}.png"), mip)


    else:
        break_points_inner = get_single_broken_points(seg_inner, gt_inner)
        break_points_middle = get_single_broken_points(seg_middle, gt_middle)
        break_points_outer = get_single_broken_points(seg_outer, gt_outer)
        skel_accuracy_inner = get_skel_accuracy(seg_inner, gt_inner)
        skel_accuracy_middle = get_skel_accuracy(seg_middle, gt_middle)
        skel_accuracy_outer = get_skel_accuracy(seg_outer, gt_outer)





    return (
        seg_id,
        break_points_inner,
        break_points_middle,
        break_points_outer,
        skel_accuracy_inner,
        skel_accuracy_middle,
        skel_accuracy_outer,
    )

def get_single_broken_points(seg_data, gt_data, debug=False):
    # 计算骨架
    skel_gt = skeletonize_3d(gt_data > 0)
    skel_gt = (skel_gt > 0)
    seg_data = binary_dilation(seg_data) # 降低偏移对计算结果的影响

    tp_skel = (seg_data > 0) & skel_gt
    skel_gt = skel_gt.astype(np.uint8)
    tp_skel = tp_skel.astype(np.uint8)
    fn_skel = skel_gt - tp_skel

    # print(np.sum(skel_gt > 0), np.sum(tp_skel > 0), np.sum(fn_skel > 0))

    _, cc_num = cc3d.connected_components(fn_skel, connectivity=26, return_N=True)
    if(not debug):
        return cc_num
    else:
        return cc_num, seg_data, skel_gt, tp_skel, fn_skel


def get_skel_accuracy(seg_data, gt_data):
    skel_gt = skeletonize_3d(gt_data > 0)
    skel_gt = (skel_gt > 0)
    seg_data = binary_dilation(seg_data)

    tp_skel = (seg_data > 0) & skel_gt
    result = np.sum(tp_skel > 0) / np.sum(skel_gt)
    # print(np.sum(tp_skel > 0), np.sum(skel_gt), result)
    return result


def get_normalized_foreground_breaking_points(seg_data, gt_data, target_ratio):
    while(np.sum(seg_data > 0) > np.sum(gt_data > 0) * target_ratio):
        seg_data = binary_erosion(seg_data)
    return get_single_broken_points(seg_data, gt_data)

def process_file_pair(seg_folder, gt_folder, file_pair, target_cc_num):
    seg_file, gt_file = file_pair
    return get_fingures_single_result(seg_folder, gt_folder, seg_file, gt_file, target_cc_num)


def compute_metrics_for_all_pairs(seg_folder, gt_folder, target_cc_num, prefix=""):
    seg_files = [f for f in os.listdir(seg_folder) if f.endswith(prefix)]
    gt_files = [f for f in os.listdir(gt_folder) if f.endswith(prefix)]

    gt_files_copy = gt_files.copy()
    for gt_file in gt_files_copy:
        flag = False
        for seg_file in seg_files:
            if prefix == '.tif' and seg_file == gt_file:
                flag = True
                break
            if(prefix == '.nii.gz' and seg_file[-10:] == gt_file[-10:]):
                flag = True
                break
        if(flag == False):
            gt_files.remove(gt_file)
    # sort
    gt_files = sorted(gt_files)
    seg_files = sorted(seg_files)
    print(len(seg_files), len(gt_files))
    if(not len(seg_files) == len(gt_files)):
        return

    file_pairs = [(seg_file, gt_file) for seg_file, gt_file in zip(seg_files, gt_files)]

    # debug
    for i in range(len(file_pairs)):
        # print(file_pairs[i])
        if(file_pairs[i][0][:5] != file_pairs[i][1][:5]):
            print("error")
    # file_pairs = file_pairs[:10]

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file_pair, seg_folder, gt_folder, pair, target_cc_num) for pair in file_pairs]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Files"):
            results.append(future.result())

    return results

def process_graded_file_pair(seg_folder, gt_folder, soma_folder, file_pair, target_cc_num):
    seg_file, gt_file, soma_file = file_pair
    return get_graded_fingures_single_result(seg_folder, gt_folder, soma_folder, seg_file, gt_file, soma_file, target_cc_num)

def compute_graded_metrics_for_all_pairs(seg_folder, gt_folder, soma_folder, target_cc_num, prefix=""):
    seg_files = [f for f in os.listdir(seg_folder) if f.endswith(prefix)]
    gt_files = [f for f in os.listdir(gt_folder) if f.endswith(prefix)]
    soma_files = [f for f in os.listdir(soma_folder) if f.endswith(".marker")]

    gt_files_copy = gt_files.copy()
    for gt_file in gt_files_copy:
        flag = False
        for seg_file in seg_files:
            if prefix == '.tif' and seg_file == gt_file:
                flag = True
                break
            if(prefix == '.nii.gz' and seg_file[-10:] == gt_file[-10:]):
                flag = True
                break
        if(flag == False):
            gt_files.remove(gt_file)
    # sort
    gt_files = sorted(gt_files)
    seg_files = sorted(seg_files)
    soma_files = sorted(soma_files)
    print(len(seg_files), len(gt_files), len(soma_files))
    if(not len(seg_files) == len(gt_files)):
        return

    file_pairs = [(seg_file, gt_file, soma_file) for seg_file, gt_file, soma_file in zip(seg_files, gt_files, soma_files)]

    # debug
    for i in range(len(file_pairs)):
        # print(file_pairs[i])
        if(file_pairs[i][0][:5] != file_pairs[i][1][:5] or file_pairs[i][0][:5] != file_pairs[i][2][:5]):
            print("error")
    # file_pairs = file_pairs[:10]

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_graded_file_pair, seg_folder, gt_folder, soma_folder, pair, target_cc_num) for pair in file_pairs]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Files"):
            results.append(future.result())

    return results

def calc_betti(img):
    rips_complex = gd.RipsComplex(points=np.array(np.nonzero(img)).T, max_edge_length = 1.5)

    # 构建持久性同调对象
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

    # 计算持久性同调并获得 Betti 数字
    persistence = simplex_tree.persistence()

    # 计算 Betti 数字
    betti_0 = simplex_tree.betti_number(0)  # 连通分量
    betti_1 = simplex_tree.betti_number(1)  # 一维环

    print(f"Betti 0: {betti_0}")  # 连通分量的数量
    print(f"Betti 1: {betti_1}")  # 一维环的数量

# 把nii.gz转成tif
def nii2tif(nii_path, tif_path):
    img = nib.load(nii_path)
    img_data = img.get_fdata()
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255
    img_data = np.array(img_data).astype(np.uint8)
    tifffile.imsave(tif_path, img_data)

def move_img(source_folder, target_folder, move_ratio=0.2):
    img_files = os.listdir(source_folder)
    for img_file in img_files:
        if(np.random.rand() < move_ratio):
            os.rename(os.path.join(source_folder, img_file), os.path.join(target_folder, img_file))

def generate_mip_and_compare(seg_folder1, seg_folder2, mip_folder):
    if(not os.path.exists(mip_folder)):
        os.makedirs(mip_folder)
    files = os.listdir(seg_folder1)
    for file in files:
        seg_path1 = os.path.join(seg_folder1, file)
        seg_path2 = os.path.join(seg_folder2, file)
        seg_img1 = tifffile.imread(seg_path1)
        seg_img2 = tifffile.imread(seg_path2)
        seg_img1 = (seg_img1 > 0).astype(np.uint8)
        seg_img2 = (seg_img2 > 0).astype(np.uint8)
        mip1 = np.max(seg_img1, axis=0) * 255
        mip2 = np.max(seg_img2, axis=0) * 255

        cat_img = np.concatenate([mip1, mip2], axis=1)
        cat_img = Image.fromarray(cat_img)
        # save png
        cat_img.save(os.path.join(mip_folder, file[:-4] + "_mip.png"))

# 比较前景比例
def compare_foreground_ratio(seg_folder, gt_folder):
    seg_files = os.listdir(seg_folder)
    gt_files = os.listdir(gt_folder)
    seg_files = sorted(seg_files)
    gt_files = sorted(gt_files)

    seg_front_ratios = []
    gt_front_ratios = []

    for i in range(len(seg_files)):
        seg_path = os.path.join(seg_folder, seg_files[i])
        gt_path = os.path.join(gt_folder, gt_files[i])
        seg_img = tifffile.imread(seg_path)
        gt_img = tifffile.imread(gt_path)
        seg_foreground_ratio = np.sum(seg_img > 0) / np.sum(seg_img >= 0)
        gt_foreground_ratio = np.sum(gt_img > 0) / np.sum(gt_img >= 0)
        seg_front_ratios.append(seg_foreground_ratio)
        gt_front_ratios.append(gt_foreground_ratio)

    print("Mean Foreground Ratio of Segmentation:", np.mean(seg_front_ratios))
    print("Mean Foreground Ratio of Ground Truth:", np.mean(gt_front_ratios))

def normalized_foreground(seg_folder1, seg_folder2, result_folder): # ptls baseline
    seg_files1 = os.listdir(seg_folder1)
    num = 1
    for seg_file1 in seg_files1:
        if(seg_file1[-4:] != ".tif"):
            continue
        print(f"Processing {num}/{len(seg_files1)}")
        num += 1
        seg_path1 = os.path.join(seg_folder1, seg_file1)
        seg_path2 = os.path.join(seg_folder2, seg_file1)
        seg_img1 = tifffile.imread(seg_path1)
        seg_img2 = tifffile.imread(seg_path2)
        seg_img1 = (seg_img1 > 0).astype(np.uint8)
        seg_img2 = (seg_img2 > 0).astype(np.uint8)
        print(np.sum(seg_img1 > 0), np.sum(seg_img2 > 0))
        fg1, fg2 = np.sum(seg_img1 > 0), np.sum(seg_img2 > 0)
        print(fg1, fg2, fg1>fg2)
        while (fg1 * 0.77 > fg2):
            seg_img2 = binary_dilation(seg_img2)
            fg2 = np.sum(seg_img2 > 0)
        print(np.sum(seg_img1 > 0), np.sum(seg_img2 > 0))
        print("-----------")
        tifffile.imwrite(os.path.join(result_folder, seg_file1), seg_img2)

def plot_box(result_csv_a, result_csv_b, box_file, labels=["Baseline", "Proposed"], metric_names = ["Broken Points", "Skeleton Accuracy"]):
    feature_name_maps={
        "Broken Points": "Number of Breaks",
        "Skeleton Accuracy": "Skeleton Accuracy",
    }
    num_features = len(metric_names)
    cols = 2
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, 4 * rows), dpi=300)  # 调整figsize和dpi提高清晰度
    axes = axes.flatten()
    plt.rcParams.update({'font.size': 20})
    # 设置字体times new roman
    plt.rcParams['font.family'] = 'Times New Roman'

    df_a = pd.read_csv(result_csv_a)
    df_b = pd.read_csv(result_csv_b)
    df_a['Type'], df_b['Type'] = labels
    df = pd.concat([df_a, df_b], axis=0)
    df_long = pd.melt(df, id_vars=['Type'], value_vars=metric_names, var_name='Feature', value_name='Value')

    # 绘图
    for idx, feature in enumerate(metric_names):
        ax = axes[idx]
        current_data = df_long[df_long['Feature'] == feature]
        if feature == "Broken Points":
            ax.set_ylim(-0.8, 80)
        elif feature == "Skeleton Accuracy":
            ax.set_ylim(0.7, 1.0)
        sns.boxplot(x='Feature', y='Value', hue='Type', data=current_data, ax=ax,
                    palette="viridis", linewidth=0.8, gap=.2, fliersize=0)

        type_a_values = current_data[current_data['Type'] == labels[0]]['Value']
        type_b_values = current_data[current_data['Type'] == labels[1]]['Value']
        t_stat, p_value = ttest_ind(type_a_values, type_b_values)

        # 添加p值注释
        ax.text(0, ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1, f'p = {p_value:.2e}',
                horizontalalignment='center', color='black', fontsize=15)


        # ax.set_title(feature_name_maps[feature], fontsize=15)
        ax.set_xlabel('')
        ax.set_ylabel(feature_name_maps[feature], fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10)  # 调整刻度标签大小
        # if idx == 1:
        #     ax.legend(loc='upper right', fontsize=8, title_fontsize='10')
        # else:
        #     ax.legend().set_visible(False)
        ax.legend().set_visible(False)

    # 隐藏不需要的子图
    for ax in axes[num_features:]:
        ax.axis('off')

    plt.tight_layout(pad=1.0)  # 调整布局
    plt.savefig(box_file)
    plt.close()

def plot_box_graded(result_csv_a, result_csv_b, box_file, labels=["Baseline", "Proposed"], metric_names = []):
    feature_name_maps={
        "Broken Points Inner": "Low",
        "Broken Points Middle": "Middle",
        "Broken Points Outer": "High",

        "Skeleton Accuracy Inner": "Low",
        "Skeleton Accuracy Middle": "Middle",
        "Skeleton Accuracy Outer": "High",
    }

    group_1_indices = [0, 1, 2]  # 假设前三个为第一组
    group_2_indices = [3, 4, 5]  # 假设后三个为第二组


    num_features = len(metric_names)
    cols = 2
    rows = 1 # (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, 3 * rows), dpi=300)  # 调整figsize和dpi提高清晰度
    axes = axes.flatten()
    plt.rcParams.update({'font.size': 15})
    # 设置字体times new roman
    plt.rcParams['font.family'] = 'Times New Roman'

    df_a = pd.read_csv(result_csv_a)
    df_b = pd.read_csv(result_csv_b)
    df_a['Type'], df_b['Type'] = labels
    df = pd.concat([df_a, df_b], axis=0)
    df_long = pd.melt(df, id_vars=['Type'], value_vars=metric_names, var_name='Feature', value_name='Value')

    viridis_palette = sns.color_palette("viridis", n_colors=5)[:4]


    # 绘图
    for idx, feature in enumerate(metric_names):
        if idx in group_1_indices:
            ax = axes[0]
        else:
            ax = axes[1]

        current_data = df_long[df_long['Feature'] == feature]
        # mean
        a_mean = current_data[current_data['Type'] == labels[0]]['Value'].mean()
        b_mean = current_data[current_data['Type'] == labels[1]]['Value'].mean()
        print(f"{feature}: mean1 {a_mean:.2f}, mean2 {b_mean:.2f}, {b_mean/a_mean:.2f}")

        sns.boxplot(x='Feature', y='Value', hue='Type', data=current_data, ax=ax,
                    palette=viridis_palette, linewidth=0.8, gap=.2, fliersize=3, flierprops={"marker": "+"},)

        type_a_values = current_data[current_data['Type'] == labels[0]]['Value'].values
        type_b_values = current_data[current_data['Type'] == labels[1]]['Value'].values
        # print(len(type_a_values), type_a_values)
        # print(len(type_b_values), type_b_values)
        t_stat, p_value = ttest_ind(type_a_values, type_b_values)

        # 添加p值注释
        if("Inner" in feature):
            offset_x = 0
        elif("Middle" in feature):
            offset_x = 1
        else:
            offset_x = 2
        # if(ax == axes[0]):
        #     offset_y = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1
        # else:
        #     offset_y = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.2
        offset_y = ax.get_ylim()[1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
        # ax.text(offset_x, offset_y, f'p={p_value:.4}',
        #         horizontalalignment='center', color='black', fontsize=12)


        # ax.set_title(feature_name_maps[feature], fontsize=15)
        ax.set_xlabel('')
        # ax.set_ylabel(feature_name_maps[feature], fontsize=20)
        if("Broken" in feature):
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: feature_name_maps[metric_names[int(x)]]))
        elif("Skeleton" in feature):
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: feature_name_maps[metric_names[int(x)+3]]))
        ax.tick_params(axis='both', which='major', labelsize=12)  # 调整刻度标签大小
        ax.legend().set_visible(False)

    # 隐藏不需要的子图
    for ax in axes[num_features:]:
        ax.axis('off')

    axes[0].set_ylabel("Number of Breaks", fontsize=15)
    axes[1].set_ylabel("Skeleton Accuracy", fontsize=15)

    # axes[1].legend(loc='upper right', fontsize=8, title_fontsize='10')

    plt.tight_layout(pad=1.0)  # 调整布局
    plt.savefig(box_file)
    plt.close()

def plot_hist_graded(result_csv_a, result_csv_b, hist_file, labels=["Baseline", "Proposed"], metric_names = []):
    feature_name_maps = {
            "Broken Points Inner": "Low-level Breaks",
            "Broken Points Middle": "Middle-level Breaks",
            "Broken Points Outer": "High-level Breaks",
            "Skeleton Accuracy Inner": "Low-level Accuracy",
            "Skeleton Accuracy Middle": "Middle-level Accuracy",
            "Skeleton Accuracy Outer": "High-level Accuracy",
        }
    group_1_indices = [0, 1, 2]  # 假设前三个为第一组
    group_2_indices = [3, 4, 5]  # 假设后三个为第二组

    num_features = len(metric_names)
    fig, axes = plt.subplots(2, 3, figsize=(3*4, 2*3), dpi=300)
    axes = axes.flatten()
    plt.rcParams.update({'font.size': 20, 'font.family': 'Times New Roman'})

    df_a = pd.read_csv(result_csv_a)
    df_b = pd.read_csv(result_csv_b)
    df_a['Type'], df_b['Type'] = labels
    df = pd.concat([df_a, df_b], axis=0)
    df_long = pd.melt(df, id_vars=['Type'], value_vars=metric_names, var_name='Feature', value_name='Value')

    for idx, feature in enumerate(metric_names):
        ax = axes[idx]
        current_data = df_long[df_long['Feature'] == feature]

        # 绘制直方图
        sns.histplot(data=current_data, x='Value', hue='Type', kde=True, ax=ax, palette="viridis", element="step", stat="count")

        # 计算并注释p值
        type_a_values = current_data[current_data['Type'] == labels[0]]['Value']
        type_b_values = current_data[current_data['Type'] == labels[1]]['Value']
        t_stat, p_value = ttest_ind(type_a_values, type_b_values)
        ax.text(0.5, 0.95, f'p = {p_value:.4}', transform=ax.transAxes, horizontalalignment='center', verticalalignment='top', color='black', fontsize=12)

        ax.set_xlabel(feature_name_maps[feature])
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        # ax.legend(title=feature_name_maps[feature], title_fontsize='14', fontsize=12)
        ax.legend().set_visible(False)

    plt.tight_layout(pad=1.0)
    plt.savefig(hist_file)
    plt.close()


def plot_delta_histogram(result_csv_a, result_csv_b, hist_file, labels=["TypeA", "TypeB"], metric_names=[]):
    # 设定特征名称的映射
    feature_name_maps = {
        "Broken Points Inner": "Low-level Breaks",
        "Skeleton Accuracy Inner": "Low-level Accuracy",
        "Broken Points Middle": "Middle-level Breaks",
        "Skeleton Accuracy Middle": "Middle-level Accuracy",
        "Broken Points Outer": "High-level Breaks",
        "Skeleton Accuracy Outer": "High-level Accuracy",
    }

    # 读取CSV文件
    df_a = pd.read_csv(result_csv_a).sort_values(by='ID')
    df_b = pd.read_csv(result_csv_b).sort_values(by='ID')

    # 检查数据集长度是否一致
    assert len(df_a) == len(df_b), "Dataframes should have the same length."

    # 初始化图形
    num_features = len(metric_names)
    fig, axes = plt.subplots(3, 2, figsize=(3*2, 3*3), dpi=300)
    axes = axes.flatten()
    plt.rcParams.update({'font.size': 15, 'font.family': 'Times New Roman'})

    # 处理每个特征
    for idx, feature in enumerate(metric_names):
        ax = axes[idx]

        # 计算差值
        if("Accuracy" in feature):
            delta_values = df_b[feature] - df_a[feature]
        else:
            delta_values = df_a[feature] - df_b[feature]

        # 去掉0值
        delta_values = delta_values[delta_values != 0]

        if(feature == 'Broken Points Inner'):
            bins_num = 5
        elif (feature == 'Broken Points Middle'):
            bins_num = 13
        elif(feature == 'Broken Points Outer'):
            bins_num = 6
        else:
            bins_num = 30
        bins = np.linspace(delta_values.min(), delta_values.max(), bins_num)

        if(not "Outer" in feature):
            # 在左面添加30%的bin
            step = bins[1] - bins[0]
            times = round(0.3*len(bins))
            new_bins = np.array([])
            for i in range(times):
                new_bins = np.append(new_bins, bins[0] - (times - i) * step)
            bins = np.append(new_bins, bins)


        if(not 0 in bins):
            min_nature = np.max(bins)
            for bin in bins:
                if(bin < min_nature and bin > 0):
                    min_nature = bin
            bins = [f - min_nature for f in bins]
            bins.append(bins[-1] + bins[-1] - bins[-2])

        # 绘制直方图
        sns.histplot(delta_values, bins=bins, kde=True, ax=ax,
                     color='gray',
                     element="step", stat="count")
        # ax.set_title(feature_name_maps.get(feature, feature))
        if('Broken' in feature):
            ax.set_xlabel("-Δ " + feature_name_maps.get(feature, feature), fontsize=15)
        else:
            ax.set_xlabel("Δ " + feature_name_maps.get(feature, feature), fontsize=15)
        ax.set_ylabel('Frequency', fontsize=15)

        # left_side_ratio = np.sum(delta_values < 0) / len(delta_values)
        # right_side_ratio = np.sum(delta_values > 0) / len(delta_values)

        worse_ratio = np.sum(delta_values < 0) / len(delta_values)
        equal_ratio = np.sum(delta_values == 0) / len(delta_values)
        better_ratio = np.sum(delta_values > 0) / len(delta_values)


        # 在直方图上添加注释
        # ax.text(0, 0.5, f'{left_side_ratio:.2%}', transform=ax.transAxes, horizontalalignment='left',
        #         color='red')
        # ax.text(0.7, 0.5,
        #         f'Percentage\n'
        #         f'Improvement:\n{better_ratio:.2%}\n'
        #         f'Worsening:\n{worse_ratio:.2%}\n'
        #         f'No Change:\n{equal_ratio:.2%}',
        #         transform=ax.transAxes, horizontalalignment='center',
        #         color='green',
        #         fontsize=12)
        ax.text(0.9, 0.5, f"{better_ratio:.2%}", transform=ax.transAxes, horizontalalignment='right', color='red', fontsize=12)
        ax.text(0.1, 0.5, f"{worse_ratio:.2%}", transform=ax.transAxes, horizontalalignment='left', color='red', fontsize=12)


        # 绘制x=0的虚线
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)

        # # 计算统计显著性
        # t_stat, p_value = ttest_ind(df_a[feature], df_b[feature])
        # ax.text(0.7, 0.9, f'p={p_value:.4}', transform=ax.transAxes, horizontalalignment='center', color='red', fontsize=12)

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(hist_file)
    plt.close()


def main_calc_metrics():
    # dataset_list = {
    #     'hb_gt': '/data/kfchen/nnUNet/nnUNet_raw/Dataset169_hb_10k/labelsTr',
    #     'hb_seg_baseline': '/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/source500/validation',
    #     'hb_seg_ptls': '/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/validation',
    #
    #     'HepaticVessel_gt': "/data/kfchen/nnUNet/nnUNet_raw/Dataset202_HepaticVessel01/labelsTr",
    #     'HepaticVessel_seg_baseline': "/data/kfchen/nnUNet/nnUNet_results/Dataset202_HepaticVessel01/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/close_result/noptls500/validation",
    #     'HepaticVessel_seg_ptls': "/data/kfchen/nnUNet/nnUNet_results/Dataset202_HepaticVessel01/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/close_result/ptls500/validation",
    #
    #     'cas_gt': "/home/kfchen/nnUNet_raw/Dataset301_CAS/labelsTr",
    #     'cas_seg_baseline': "/home/kfchen/nnUNet_results/Dataset301_CAS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/source/validation",
    #     'cas_seg_ptls': "/home/kfchen/nnUNet_results/Dataset301_CAS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls/validation",
    # }
    # seg_folder = dataset_list['hb_seg_ptls']
    # gt_folder = dataset_list['hb_gt']

    seg_folder = "/data/kfchen/trace_ws/paper_trace_result/" + "nnunet/cldice" + "/0_seg"
    gt_folder = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/label"

    # generate_mip_and_compare(dataset_list['hb_seg_baseline'], dataset_list['hb_seg_ptls'],
    #                          "/data/kfchen/nnUNet/nnUNet_results/Dataset167_human_brain_10000_noptls/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls_full_weight200_250+250/mip")

    all_metrics = compute_metrics_for_all_pairs(seg_folder, gt_folder, target_cc_num=1, prefix=".tif")

    for i in range(len(all_metrics)):
        print(all_metrics[i])

    mean_dice = np.mean([metrics[1] for metrics in all_metrics])
    mean_c_dice = np.mean([metrics[2] for metrics in all_metrics])
    mean_overlap = np.mean([metrics[3] for metrics in all_metrics])
    mean_c_overlap = np.mean([metrics[4] for metrics in all_metrics])
    mean_c_relative_coverage = np.mean([metrics[5] for metrics in all_metrics])
    mean_relative_foreground_ratio = np.mean([metrics[6] for metrics in all_metrics])
    mean_broken_points = np.mean([metrics[7] for metrics in all_metrics])
    mean_skel_acc = np.mean([metrics[8] for metrics in all_metrics])
    mean_cldice = np.mean([metrics[9] for metrics in all_metrics])

    # 4位小数输出
    print("Mean Dice Coefficient:", round(mean_dice, 4))
    print("Mean Overlap:", round(mean_overlap, 4))
    print("Mean C-Dice Coefficient:", round(mean_c_dice, 4))
    print("Mean C-Overlap:", round(mean_c_overlap, 4))
    print("Mean C-Relative Coverage:", round(mean_c_relative_coverage, 4))
    print("Mean Relative Foreground Ratio:", round(mean_relative_foreground_ratio, 4))
    print("Mean Broken Points:", round(mean_broken_points, 4))
    print("Mean Skeleton Accuracy:", round(mean_skel_acc, 4))
    print("Mean clDice:", round(mean_cldice, 4))

    print(round(mean_dice, 4), round(mean_overlap, 4), round(mean_c_dice, 4), round(mean_c_overlap, 4),
          round(mean_c_relative_coverage, 4), round(mean_relative_foreground_ratio, 4), round(mean_broken_points, 4),
            round(mean_skel_acc, 4), round(mean_cldice, 4))

    result_csv = "/data/kfchen/trace_ws/paper_trace_result/" + "nnunet/cldice" + "_seg_metrics.csv"
    if(os.path.exists(result_csv)):
        os.remove(result_csv)
    # sort
    all_metrics = sorted(all_metrics, key=lambda x: x[0])
    with open(result_csv, "w") as f:
        f.write("ID,Dice,Overlap,C-Dice,C-Overlap,C-Relative Coverage,Relative Foreground Ratio,Broken Points,Skeleton Accuracy,clDice\n")
        f.write(f"Mean,{round(mean_dice, 4)},{round(mean_overlap, 4)},{round(mean_c_dice, 4)},"
                f"{round(mean_c_overlap, 4)},{round(mean_c_relative_coverage, 4)},{round(mean_relative_foreground_ratio, 4)},"
                f"{round(mean_broken_points, 4)},{round(mean_skel_acc, 4)},{round(mean_cldice, 4)}\n")
        for metrics in all_metrics:
            f.write(f"{metrics[0]},{metrics[1]},{metrics[3]},{metrics[2]},{metrics[4]},{metrics[5]},{metrics[6]},{metrics[7]},{metrics[8]},{metrics[9]}\n")



    # visualize
    # box_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/box.png"
    # plot_box("/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/source500.csv",
    #          "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/ptls10.csv",
    #          box_file, labels=["nnUNet", "Proposed"])

def calc_graded_metrics():
    seg_folder = "/data/kfchen/trace_ws/paper_trace_result/nnunet/cldice/0_seg"
    gt_folder = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/label"
    soma_folder = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/somamarker"

    """分段metrics"""
    all_metrics = compute_graded_metrics_for_all_pairs(seg_folder, gt_folder, soma_folder, target_cc_num=1, prefix=".tif")

    result_csv = "/data/kfchen/trace_ws/paper_trace_result/nnunet/" + seg_folder.split("/")[-1] + "_graded_seg_metrics.csv"
    if (os.path.exists(result_csv)):
        os.remove(result_csv)
    all_metrics = sorted(all_metrics, key=lambda x: x[0])
    with open(result_csv, "w") as f:
        f.write("ID,Broken Points Inner,Broken Points Middle,Broken Points Outer,Skeleton Accuracy Inner,Skeleton Accuracy Middle,Skeleton Accuracy Outer\n")
        for metrics in all_metrics:
            f.write(f"{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]},{metrics[4]},{metrics[5]},{metrics[6]}\n")





    # box_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/graded_box1.png"
    # plot_box_graded("/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/graded_source500.csv",
    #                 "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/graded_ptls10.csv",
    #                 box_file,
    #                 labels=["nnUNet", "Proposed"],
    #                 metric_names=["Broken Points Inner", "Broken Points Middle", "Broken Points Outer",
    #                               "Skeleton Accuracy Inner", "Skeleton Accuracy Middle", "Skeleton Accuracy Outer"])

    # hist_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/graded_hist1.png"
    # plot_hist_graded("/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/graded_source500.csv",
    #                  "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/graded_ptls10.csv",
    #                  hist_file,
    #                  labels=["nnUNet", "Proposed"],
    #                  metric_names=["Broken Points Inner", "Broken Points Middle", "Broken Points Outer",
    #                                "Skeleton Accuracy Inner", "Skeleton Accuracy Middle", "Skeleton Accuracy Outer"])


    # delta_hist_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/graded_delta_hist_source500_vs_ptls10_2.png"
    # plot_delta_histogram("/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/source500_graded_seg_metrics.csv",
    #                      "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/ptls10_graded_seg_metrics.csv",
    #                      delta_hist_file,
    #                      labels=["nnUNet", "Proposed"],
    #                      metric_names=["Broken Points Inner", "Skeleton Accuracy Inner",
    #                                    "Broken Points Middle", "Skeleton Accuracy Middle",
    #                                    "Broken Points Outer", "Skeleton Accuracy Outer"])

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
        # print(len(filtered_df))
        # print(filtered_df)
        common_df_list.append(filtered_df)

    # 返回包含共同项的所有 DataFrame
    return common_df_list

def plot_box_of_swc_list(seg_metrics_files, labels, box_file):
    feature_names = ['Dice', 'Broken Points', 'Skeleton Accuracy']
    feature_name_maps = {
        # 'Dice': 'Dice Coefficient ↑',
        # 'Broken Points': 'No. of Skel. Breaks ↓',
        # 'Skeleton Accuracy': 'Skel. Accuracy ↑',
        'Dice': 'Dice Coefficient',
        'Broken Points': 'No. of Skel. Breaks',
        'Skeleton Accuracy': 'Skel. Accuracy',
    }

    num_features = len(feature_names)
    cols = 3
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, 3 * rows), dpi=300)  # 调整figsize和dpi提高清晰度
    axes = axes.flatten()
    # plt.rcParams.update({'font.size': 20})  # 更新字体大小
    # 设置字体 Arial
    # plt.rcParams['font.family'] = 'Arial'

    dfs = [pd.read_csv(f) for f in seg_metrics_files]
    dfs = get_common_rows_from_dfs(dfs)
    for i, df in enumerate(dfs):
        df['Type'] = labels[i]

    df = pd.concat(dfs, axis=0)
    df_long = pd.melt(df, id_vars=['Type'], value_vars=feature_names, var_name='Feature', value_name='Value')

    colors = plt.get_cmap('Set3').colors
    colors = [colors[8], colors[2], colors[5], colors[3], colors[0]]

    # 绘图
    for idx, feature in enumerate(feature_names):
        ax = axes[idx]

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
            # ax.boxplot(current_data[i], positions=[positions[i]], widths=0.5, patch_artist=True,
            #            showfliers=False, boxprops=dict(facecolor=colors[i], color='black'),
            #            medianprops=dict(color='black'))
            ax.boxplot(current_data[i], positions=[positions[i]], widths=0.5, patch_artist=True,
                          showfliers=True, boxprops=dict(facecolor=colors[i], color='black'),
                          medianprops=dict(color='black'), flierprops=dict(marker='o', color='black', markersize=3))
            print(f"mean {hue}: {np.mean(current_data[i])}, {np.mean(current_data[i]) - np.mean(current_data[0])}")


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
            if(i==0 or i == 1):
                delta_text = f"{mean_differences[i] / np.mean(baseline_data) * 100:.2f}%"
                print(f"Mean difference for {hue_order[0]} vs {hue_order[i]}: , {delta_text}")
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
            print(f"Mean difference for {hue_order[0]} vs {hue_order[i]}: , {delta_text}")
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

# main
if __name__ == '__main__':
    # main_calc_metrics()

    root_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet"
    loss_list = ['baseline', 'cldice', 'skelrec', 'newcel_0.1']
    swc_dir_list = [os.path.join(root_dir, loss) for loss in loss_list]
    plot_box_of_swc_list([swc_dir + "_seg_metrics.csv" for swc_dir in swc_dir_list],
                         ['Baseline', 'clDice', 'SkelRec', 'Proposed', "Label"],
                         "/home/kfchen/neuron_seg_human/neuroutils/tests/060525/seg_metrics_box.png")

