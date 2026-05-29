import numpy as np
import tifffile
from pathlib import Path
import scipy.ndimage as ndimage
import sys
import os
import json
import shutil
import pandas as pd

# 必须在导入 pyplot 之前设置为 'Agg'，防止多进程画图时 GUI 后端崩溃
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # 引入用于绘制圆圈的模块

import concurrent.futures
from tqdm import tqdm
from pipeline_config import BASE_DIR, PATHS, GCUT


# =========================================================================
# 1. 动态加载本地 G-Cut 模块
# =========================================================================
try:
    sys.path.insert(0,PATHS["gcut"])  # 将 G-Cut 模块路径添加到 sys.path
    from neuron_segmentation import NeuronSegmentation
    GCUT_AVAILABLE = True
except ImportError as e:
    print(f"警告: 找不到原版 G-Cut 模块。请确保 gcut 文件夹在当前目录下。报错: {e}")
    GCUT_AVAILABLE = False


# =========================================================================
# 2. 独立实现的 SWC 读写函数
# =========================================================================
def parse_swc(swc_file):
    tree = []
    with open(swc_file, 'r') as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line or line.startswith('#'): 
                continue
            parts = line.split()
            idx, type_, x, y, z, r, p = parts[:7]
            tree.append((int(idx), int(type_), float(x), float(y), float(z), float(r), int(p)))
    return tree

def write_swc(tree, swc_file):
    with open(swc_file, 'w') as fp:
        fp.write('##n type x y z r parent\n')
        for node in tree:
            idx, type_, x, y, z, r, p = node
            fp.write(f'{int(idx)} {int(type_)} {float(x):.5f} {float(y):.5f} {float(z):.5f} {float(r):.1f} {int(p)}\n')


def swc_has_nodes(swc_file):
    swc_file = Path(swc_file)
    if not swc_file.exists() or swc_file.stat().st_size == 0:
        return False
    with open(swc_file, 'r', encoding='utf-8', errors='ignore') as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if len(line.split()) >= 7:
                return True
    return False


def selected_swc_path(result_dir, img_path):
    return Path(result_dir) / f"{Path(img_path).stem.replace('_0000', '')}.swc"


def find_gcut_target_swc(vis_dir, img_path):
    stem = Path(img_path).stem
    base_stem = stem.replace('_0000', '')
    vis_dir = Path(vis_dir)
    patterns = (
        f"{stem}_gcut_soma_*_TARGET.swc",
        f"{base_stem}_gcut_soma_*_TARGET.swc",
        f"{stem}_gcut_soma_*.swc",
        f"{base_stem}_gcut_soma_*.swc",
    )
    for pattern in patterns:
        candidates = sorted(vis_dir.glob(pattern))
        for candidate in candidates:
            if swc_has_nodes(candidate):
                return candidate
    return None


def prepare_selected_swc(result_dir, img_path, vis_dir):
    selected_path = selected_swc_path(result_dir, img_path)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    if swc_has_nodes(selected_path):
        return selected_path
    target_swc = find_gcut_target_swc(vis_dir, img_path)
    if target_swc is None:
        return None
    shutil.copy2(target_swc, selected_path)
    return selected_path


# =========================================================================
# 3. 核心算法组件
# =========================================================================
def remove_fake_roots(tree):
    roots = [node for node in tree if node[6] == -1]
    
    if len(roots) <= 1:
        return tree, 0
        
    parent_ids = set([node[6] for node in tree])
    
    cleaned_tree = []
    fake_roots_removed = 0
    
    for node in tree:
        idx, type_, x, y, z, r, p = node
        if p == -1 and idx not in parent_ids:
            fake_roots_removed += 1
            continue 
        cleaned_tree.append(node)
        
    return cleaned_tree, fake_roots_removed


def apply_nms(coords, scores, nms_radius):
    if len(coords) == 0:
        return []
    coords = np.array(coords)
    scores = np.array(scores)
    sorted_indices = np.argsort(scores)[::-1]
    kept_indices = []
    
    for idx in sorted_indices:
        current_coord = coords[idx]
        keep = True
        for kept_idx in kept_indices:
            kept_coord = coords[kept_idx]
            dist = np.linalg.norm(current_coord - kept_coord)
            if dist < nms_radius:
                keep = False 
                break
        if keep:
            kept_indices.append(idx)
            
    return coords[kept_indices].tolist()


def reindex_tree_to_zero_based(tree):
    id_map = {node[0]: i for i, node in enumerate(tree,start=1)}
    id_map[-1] = -1 
    new_tree = []
    for i, node in enumerate(tree,start=1):
        idx, type_, x, y, z, r, p = node
        new_idx = id_map.get(idx, i)
        new_p = id_map.get(p, -1)
        new_tree.append((new_idx, type_, x, y, z, r, new_p))
    return new_tree


def match_soma_nodes_in_swc(tree, soma_coords, img_height, is_3d=True, max_tolerance=15.0):
    somata_ids = []
    if len(soma_coords) == 0 or len(tree) == 0:
        return somata_ids
        
    tree_coords = np.array([[node[2], node[3], node[4]] for node in tree])
    tree_ids = np.array([node[0] for node in tree])
    
    for coord in soma_coords:
        if is_3d and len(coord) == 3:
            cz, cy, cx = coord
            target_pt_swc = np.array([cx, img_height - cy, cz])
        else:
            cy, cx = coord
            target_pt_swc = np.array([cx, img_height - cy, 0.0])
            
        distances = np.linalg.norm(tree_coords - target_pt_swc, axis=1)
        min_index = np.argmin(distances)
        min_dist = distances[min_index]
        
        if min_dist <= max_tolerance:
            closest_node_id = tree_ids[min_index]
            if closest_node_id not in somata_ids:
                somata_ids.append(closest_node_id)
                
    return somata_ids


def get_binarization_threshold(img, mask, percentile_value=50.0):
    if img.shape != mask.shape:
        raise ValueError("维度不匹配！")
    extracted_voxels = img[mask > 0]
    if extracted_voxels.size == 0:
        return None
    return np.percentile(extracted_voxels, percentile_value)


# =========================================================================
# 4. 全新综合可视化绘图模块 (U形排列，加入候选点展示)
# =========================================================================
def save_comprehensive_summary(img, binary_img, dist_transform, gsdt_mask, global_soma_mask, pre_nms_coords, post_nms_coords, 
                               app2_tree, split_trees, target_soma_id, 
                               save_path, nms_radius=20.0, title_prefix="", individual_output_dir=None):
    """
    生成 2x4 的网格图。
    排列顺序呈左开口的U形:
    ax[0]->ax[1]->ax[2]->ax[3] (向右)
                         |
    ax[4]<-ax[5]<-ax[6]<-ax[7] (向左)
    """
    if img.ndim == 3:
        img_show = np.max(img, axis=0)
        binary_show = np.max(binary_img, axis=0)
        dist_show = np.max(dist_transform, axis=0)
        mask_show = np.max(gsdt_mask, axis=0)
        soma_mask_show = np.max(global_soma_mask, axis=0)
    else:
        img_show, binary_show, dist_show, mask_show, soma_mask_show = img, binary_img, dist_transform, gsdt_mask, global_soma_mask

    img_height = img_show.shape[0]

    fig, axes = plt.subplots(2, 4, figsize=(30, 15))
    fig.suptitle(f"{title_prefix} - Comprehensive Pipeline Summary", fontsize=24)
    ax = axes.flatten()

    # ====== 上半部分: 1 -> 2 -> 3 -> 4 ======
    
    ax[0].imshow(img_show, cmap='gray')
    ax[0].set_title("1. Original Image (MIP)", fontsize=16)
    ax[0].axis('off')

    ax[1].imshow(binary_show, cmap='gray')
    ax[1].set_title("2. Binarization Mask", fontsize=16)
    ax[1].axis('off')

    ax[2].imshow(dist_show, cmap='magma')
    ax[2].set_title("3. GSDT Distance Map", fontsize=16)
    ax[2].axis('off')

    # 图 4：NMS 筛选前的所有候选点与抑制范围
    ax[3].imshow(mask_show, cmap='gray')
    soma_overlay = np.ma.masked_where(soma_mask_show == 0, soma_mask_show)
    ax[3].imshow(soma_overlay, cmap='cool', alpha=0.6)
    ax[3].set_title(f"4. Pre-NMS Candidates (N={len(pre_nms_coords)}) & Ranges", fontsize=16)
    ax[3].axis('off')
    
    # 画出所有候选的极值点（黄色小点）
    for coord in pre_nms_coords:
        y, x = (coord[1], coord[2]) if img.ndim == 3 else (coord[0], coord[1])
        ax[3].plot(x, y, 'yo', markersize=4, zorder=4)

    # 画出最终存活的点（红星）和对应的清场圆圈
    for coord in post_nms_coords:
        y, x = (coord[1], coord[2]) if img.ndim == 3 else (coord[0], coord[1])
        ax[3].plot(x, y, 'r*', markersize=12, markeredgecolor='white', zorder=5)
        circle = patches.Circle((x, y), nms_radius, color='red', fill=False, linestyle='--', linewidth=1.5, alpha=0.7, zorder=6)
        ax[3].add_patch(circle)


    # ====== 下半部分 (从右向左): 5 <- 6 <- 7 <- 8 ======
    cmap = plt.get_cmap('tab20')
    colors_dict = {}
    if split_trees:
        colors_dict = {sid: cmap(i % 20) for i, sid in enumerate(split_trees.keys())}

    # 图 5 (ax[7])：NMS 筛选后的最终结果
    ax[7].imshow(mask_show, cmap='gray')
    ax[7].imshow(soma_overlay, cmap='cool', alpha=0.6)
    ax[7].set_title(f"5. Post-NMS Filtered Centers (N={len(post_nms_coords)})", fontsize=16)
    ax[7].axis('off')
    
    # 这里只画红星，极其干净
    for coord in post_nms_coords:
        y, x = (coord[1], coord[2]) if img.ndim == 3 else (coord[0], coord[1])
        ax[7].plot(x, y, 'r*', markersize=12, markeredgecolor='white', zorder=5)

    # 图 6 (ax[6])：Original App2 Trace
    ax[6].imshow(img_show, cmap='gray')
    ax[6].set_title("6. Original App2 Trace (Raw)", fontsize=16)
    ax[6].axis('off')
    if app2_tree and len(app2_tree) > 0:
        node_dict = {node[0]: node for node in app2_tree}
        for node in app2_tree:
            p_id = node[6]
            if p_id in node_dict:
                p_node = node_dict[p_id]
                ax[6].plot([node[2], p_node[2]], [img_height - node[3], img_height - p_node[3]], color='cyan', linewidth=1.5, alpha=0.6)
    else:
        ax[6].text(0.5, 0.5, 'No App2 Trace Found', color='white', ha='center', va='center', fontsize=16, transform=ax[6].transAxes)

    # 图 7 (ax[5])：G-Cut All Segmented Neurons
    ax[5].imshow(img_show, cmap='gray')
    ax[5].set_title("7. G-Cut Segmented (All Context)", fontsize=16)
    ax[5].axis('off')
    if split_trees:
        for sid, tree in split_trees.items():
            color = colors_dict[sid]
            node_dict = {node[0]: node for node in tree}
            for node in tree:
                p_id = node[6]
                if p_id in node_dict:
                    p_node = node_dict[p_id]
                    ax[5].plot([node[2], p_node[2]], [img_height - node[3], img_height - p_node[3]], color=color, linewidth=1.5, alpha=0.8)
            soma_node = node_dict.get(sid)
            if soma_node:
                ax[5].plot(soma_node[2], img_height - soma_node[3], marker='*', color=color, markersize=15, markeredgecolor='white')
    else:
        ax[5].text(0.5, 0.5, 'No G-Cut Results\n(Skipped/Empty)', color='white', ha='center', va='center', fontsize=16, transform=ax[5].transAxes)

    # 图 8 (ax[4])：G-Cut Target Neuron Only
    ax[4].imshow(img_show, cmap='gray')
    ax[4].axis('off')
    target_title = f"8. Target Neuron (Soma ID: {target_soma_id})" if target_soma_id is not None else "8. Target Neuron Not Extracted"
    ax[4].set_title(target_title, fontsize=16)
    if target_soma_id is not None and target_soma_id in split_trees:
        tree = split_trees[target_soma_id]
        color = colors_dict[target_soma_id]
        node_dict = {node[0]: node for node in tree}
        for node in tree:
            p_id = node[6]
            if p_id in node_dict:
                p_node = node_dict[p_id]
                ax[4].plot([node[2], p_node[2]], [img_height - node[3], img_height - p_node[3]], color=color, linewidth=1.5, alpha=0.8)
        soma_node = node_dict.get(target_soma_id)
        if soma_node:
            ax[4].plot(soma_node[2], img_height - soma_node[3], marker='*', color=color, markersize=15, markeredgecolor='white')
    else:
        ax[4].text(0.5, 0.5, 'No Target Match', color='white', ha='center', va='center', fontsize=16, transform=ax[4].transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    if individual_output_dir is not None:
        individual_output_dir = Path(individual_output_dir)
        individual_output_dir.mkdir(parents=True, exist_ok=True)
        panel_names = {
            0: "01_original_image_mip",
            1: "02_binarization_mask",
            2: "03_gsdt_distance_map",
            3: "04_pre_nms_candidates",
            7: "05_post_nms_centers",
            6: "06_original_app2_trace",
            5: "07_gcut_segmented_all",
            4: "08_gcut_target_neuron",
        }
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for idx, name in panel_names.items():
            bbox = ax[idx].get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(individual_output_dir / f"{name}.png", dpi=300, bbox_inches=bbox.expanded(1.08, 1.15))

    plt.close(fig)


# =========================================================================
# 5. 主流水线流程
# =========================================================================
def process_image_to_gcut(img_path, mask_path, swc_path, vis_output_dir, 
                          target_coord=None, percentile_value=50.0, gsdt_threshold=5.0,
                          soma_json_path=None, soma_mask_path=None):
    img_path_str = str(img_path)
    log_messages = []
    
    img = tifffile.imread(img_path_str)
    mask = tifffile.imread(mask_path)
    img_height = img.shape[1] if img.ndim == 3 else img.shape[0]
    
    intensity_threshold = get_binarization_threshold(img, mask, percentile_value)
    if intensity_threshold is None:
        return None, "失败: Mask 中没有任何有效区域"
    
    binary_img = img > intensity_threshold
    dist_transform = ndimage.distance_transform_edt(binary_img)
    gsdt_mask = dist_transform > gsdt_threshold
    
    labeled_array, num_features = ndimage.label(gsdt_mask)
    raw_soma_coords = []
    raw_gsdt_scores = []
    
    if num_features > 0:
        for i in range(1, num_features + 1):
            max_coord = ndimage.maximum_position(dist_transform, labels=labeled_array, index=i)
            raw_soma_coords.append(max_coord)
            raw_gsdt_scores.append(dist_transform[max_coord])
            
    # =========================================================================
    # 胞体定位优化：基于外部裁剪的 Soma Mask 和 JSON 偏移量进行严格排他
    # =========================================================================
    final_coords = []
    final_scores = []
    
    # 强制塞入来自 Meta 的目标坐标，赋予极大分数确保它在最终 NMS 中存活
    if target_coord is not None:
        final_coords.append(target_coord)
        final_scores.append(float('inf')) 
        
    target_cc_id = 0
    soma_mask_ccs = None
    z_min, x_min, y_min = 0, 0, 0
    z_dim, y_dim, x_dim = 0, 0, 0
    
    # 建立一个与原图同尺寸的全局 Soma Mask 容器用于画图
    global_soma_mask = np.zeros_like(img, dtype=np.uint8)
    
    # 读取 Crop 的 Soma Mask 与 JSON 边界信息
    if soma_json_path and soma_json_path.exists() and soma_mask_path and soma_mask_path.exists():
        with open(soma_json_path, 'r') as f:
            meta_data = json.load(f)
        bounds = meta_data.get("bounds_1um_zxy", [0, 0, 0, 0, 0, 0])
        z_min, x_min, y_min = int(bounds[0]), int(bounds[2]), int(bounds[4])
        
        s_mask = tifffile.imread(str(soma_mask_path))
        soma_mask_ccs, _ = ndimage.label(s_mask > 0) # 区分不同的胞体连通域
        
        if s_mask.ndim == 3:
            z_dim, y_dim, x_dim = s_mask.shape
            
            # 将裁剪的 mask 放回全局坐标系用于画图展示
            z_start, z_end = max(0, z_min), min(img.shape[0], z_min + z_dim)
            y_start, y_end = max(0, y_min), min(img.shape[1], y_min + y_dim)
            x_start, x_end = max(0, x_min), min(img.shape[2], x_min + x_dim)
            
            mz_start, mz_end = z_start - z_min, z_end - z_min
            my_start, my_end = y_start - y_min, y_end - y_min
            mx_start, mx_end = x_start - x_min, x_end - x_min
            
            global_soma_mask[z_start:z_end, y_start:y_end, x_start:x_end] = (s_mask[mz_start:mz_end, my_start:my_end, mx_start:mx_end] > 0).astype(np.uint8)
            
        else:
            y_dim, x_dim = s_mask.shape
            y_start, y_end = max(0, y_min), min(img.shape[0], y_min + y_dim)
            x_start, x_end = max(0, x_min), min(img.shape[1], x_min + x_dim)
            my_start, my_end = y_start - y_min, y_end - y_min
            mx_start, mx_end = x_start - x_min, x_end - x_min
            global_soma_mask[y_start:y_end, x_start:x_end] = (s_mask[my_start:my_end, mx_start:mx_end] > 0).astype(np.uint8)
            
        # 计算 Meta 坐标在局部 Mask 中的连通域 ID
        if target_coord is not None:
            if len(target_coord) == 3:
                tz, ty, tx = target_coord
                local_z, local_y, local_x = int(round(tz - z_min)), int(round(ty - y_min)), int(round(tx - x_min))
                if 0 <= local_z < z_dim and 0 <= local_y < y_dim and 0 <= local_x < x_dim:
                    target_cc_id = soma_mask_ccs[local_z, local_y, local_x]
            else:
                ty, tx = target_coord
                local_y, local_x = int(round(ty - y_min)), int(round(tx - x_min))
                if 0 <= local_y < y_dim and 0 <= local_x < x_dim:
                    target_cc_id = soma_mask_ccs[local_y, local_x]

    # 遍历算法生成的所有极值点进行清洗
    for coord, score in zip(raw_soma_coords, raw_gsdt_scores):
        is_in_target_mask = False
        
        if soma_mask_ccs is not None:
            if len(coord) == 3:
                cz, cy, cx = coord
                local_z, local_y, local_x = int(round(cz - z_min)), int(round(cy - y_min)), int(round(cx - x_min))
                if 0 <= local_z < z_dim and 0 <= local_y < y_dim and 0 <= local_x < x_dim:
                    cc_id = soma_mask_ccs[local_z, local_y, local_x]
                    # 如果该点与 Meta 目标点处于同一个胞体连通域内
                    if cc_id > 0 and cc_id == target_cc_id:
                        is_in_target_mask = True
            else:
                cy, cx = coord
                local_y, local_x = int(round(cy - y_min)), int(round(cx - x_min))
                if 0 <= local_y < y_dim and 0 <= local_x < x_dim:
                    cc_id = soma_mask_ccs[local_y, local_x]
                    if cc_id > 0 and cc_id == target_cc_id:
                        is_in_target_mask = True
                        
        # 不在目标胞体 Mask 上的点才被保留；在目标胞体上的杂点直接抛弃
        if not is_in_target_mask:
            final_coords.append(coord)
            final_scores.append(score)

    # 最后进行 NMS，Meta 点因为分数为 inf 会自动吞噬掉周围靠得太近的野点
    NMS_RADIUS = 20.0 
    soma_coords = apply_nms(final_coords, final_scores, nms_radius=NMS_RADIUS)
    # =========================================================================

    app2_tree = None
    split_trees = {}
    target_soma_id = None
    
    if GCUT_AVAILABLE:
        if Path(swc_path).exists():
            try:
                tree_orig = parse_swc(swc_path)
                if len(tree_orig) == 0:
                    return None, f"跳过: App2 SWC 为空或没有有效节点: {swc_path}"
                tree_orig, removed_count = remove_fake_roots(tree_orig)
                if removed_count > 0:
                    log_messages.append(f"清理了{img_path} 中{removed_count} 个无子节点的假根节点。")
                
                app2_tree = tree_orig 
                tree = reindex_tree_to_zero_based(tree_orig)
                
                somata_ids = match_soma_nodes_in_swc(
                    tree, soma_coords, img_height,
                    is_3d=(img.ndim == 3), max_tolerance=15.0
                )
                
                if len(somata_ids) == 0:
                    log_messages.append("跳过: 所有候选坐标均未能在 15 像素内匹配到 App2 节点，疑似漏追。仅生成预处理图。")
                else:
                    temp_swc_path = Path(vis_output_dir) / f"{img_path.stem}_temp_injected.swc"
                    temp_soma_txt = Path(vis_output_dir) / f"{img_path.stem}_soma_ids.txt"
                    write_swc(tree, temp_swc_path)
                    with open(temp_soma_txt, 'w') as f:
                        for sid in somata_ids:
                            f.write(f"{sid}\n")
                            
                    segmentor = NeuronSegmentation(str(temp_swc_path), str(temp_soma_txt), scale_z=1.0, scale_ouput_z=False)
                    segmentor.segment(conditional='human', cell_type='neocortex')
                    gcut_neurons = segmentor.knife.assemble() 
                    
                    if target_coord is not None:
                        min_dist = float('inf')
                        for sid, n_tree in gcut_neurons.items():
                            if sid in n_tree.tree:
                                s_node = n_tree.tree[sid] 
                                target_x = target_coord[2] if len(target_coord) == 3 else target_coord[1]
                                target_y_swc = img_height - (target_coord[1] if len(target_coord) == 3 else target_coord[0])
                                target_z = target_coord[0] if len(target_coord) == 3 else 0.0
                                
                                dist = ((s_node.x - target_x)**2 + (s_node.y - target_y_swc)**2 + (s_node.z - target_z)**2)**0.5
                                if dist < min_dist:
                                    min_dist = dist
                                    target_soma_id = sid

                    for soma_id, neuron_tree_obj in gcut_neurons.items():
                        single_tree_list = []
                        for node_id, node in neuron_tree_obj.tree.items():
                            single_tree_list.append((node_id, node.node_type, node.x, node.y, node.z, node.radius, node.parent_id))
                            
                        split_trees[soma_id] = single_tree_list
                        
                        if target_soma_id is not None:
                            if soma_id == target_soma_id:
                                out_name = Path(vis_output_dir) / f"{img_path.stem}_gcut_soma_{soma_id}_TARGET.swc"
                                write_swc(single_tree_list, out_name)
                        else:
                            out_name = Path(vis_output_dir) / f"{img_path.stem}_gcut_soma_{soma_id}.swc"
                            write_swc(single_tree_list, out_name)
                    
                    temp_swc_path.unlink(missing_ok=True)
                    temp_soma_txt.unlink(missing_ok=True)
                    
                    if target_soma_id is not None:
                        log_messages.append(f"G-Cut 成功。已保存目标 ID: {target_soma_id}")
                    else:
                        log_messages.append(f"G-Cut 成功。已保存所有神经元。")
            except Exception as e:
                log_messages.append(f"失败: G-Cut 内部处理报错 -> {e}")
        else:
            log_messages.append(f"跳过: 追踪文件 {swc_path} 不存在。仅生成预处理图。")

    combined_vis_path = Path(vis_output_dir) / f"{img_path.stem}_comprehensive_vis.png"
    # 注意这里将 final_coords (NMS前的候选点) 和 soma_coords (NMS后的点) 都传进去了
    save_comprehensive_summary(
        img, binary_img, dist_transform, gsdt_mask, global_soma_mask, final_coords, soma_coords,
        app2_tree, split_trees, target_soma_id,
        combined_vis_path, nms_radius=NMS_RADIUS, title_prefix=img_path.name,
        individual_output_dir=None
    )

    return (intensity_threshold, soma_coords), "; ".join(log_messages)


def extract_id(filename):
    name_part = filename.split('.')[0]
    if name_part.endswith("_0000"):
        return name_part[:-5].replace("image_", "")
    return name_part

def worker_task(args):
    """多进程的任务包装器"""
    if len(args) == 11:
        img_path, mask_path, swc_path, vis_dir, pct_val, gsdt_thr, error_log_path, soma_json_path, soma_mask_path, meta_df, result_dir = args
    else:
        img_path, mask_path, swc_path, vis_dir, pct_val, gsdt_thr, error_log_path, soma_json_path, soma_mask_path, meta_df = args
        result_dir = None
    try:
        if result_dir is not None:
            selected_path = selected_swc_path(result_dir, img_path)
            if swc_has_nodes(selected_path):
                return img_path.name, (0, []), f"跳过: selected_for_pruning 已存在 {selected_path}"

        neuron_id = int(extract_id(img_path.name))
        
        # 直接从传入的 DataFrame 获取数据
        curr_meta = meta_df.loc[neuron_id]
        
        soma_x_raw = float(curr_meta['soma_x'])
        soma_y_raw = float(curr_meta['soma_y'])
        soma_z_raw = float(curr_meta['soma_z'])
        xy_res = float(curr_meta['xy_resolution'])
        z_res = float(curr_meta['z_resolution'])
        
        center_z = int(round(soma_z_raw * (z_res / 1000)))
        center_y = int(round(soma_y_raw * (xy_res / 1000)))
        center_x = int(round(soma_x_raw * (xy_res / 1000)))
        csv_target_coord = (center_z, center_y, center_x) 

        res, log_msg = process_image_to_gcut(
            img_path, mask_path, swc_path, vis_dir, 
            csv_target_coord, pct_val, gsdt_thr,
            soma_json_path, soma_mask_path
        )
        if result_dir is not None and res is not None:
            selected_path = prepare_selected_swc(result_dir, img_path, vis_dir)
            if selected_path is None:
                log_msg = f"{log_msg}; 失败: 未找到可用于 pruning 的 G-Cut SWC"
                res = None
            else:
                log_msg = f"{log_msg}; selected_for_pruning: {selected_path}"
        
        if "跳过" in log_msg or "失败" in log_msg or "清理了" in log_msg:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{img_path.name}] {log_msg}\n")
                f.flush() 
                
        return img_path.name, res, log_msg
    
    except Exception as e:
        error_str = f"崩溃: 在获取坐标或处理时出错 -> {str(e)}"
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{img_path.name}] {error_str}\n")
            f.flush()
        return img_path.name, None, error_str


# =========================================================================
# 6. 多进程调度与执行主程序
# =========================================================================
if __name__ == "__main__":
    base_dir = BASE_DIR
    # base_dir = "/data/disk2/B4.5"
    print(PATHS["gcut"])
    img_dir = PATHS["image_1um_dir"]
    mask_dir = PATHS["merged_mask_dir"]
    swc_dir = PATHS["trace_swc_dir"] 
    vis_dir = PATHS["gcut_output_dir"]
    result_dir = PATHS["gcut_selected_swc_dir"] 
    
    soma_img_dir = PATHS["soma_crop_dir"]
    soma_mask_dir = PATHS["soma_seg_dir"]
    
    vis_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    error_log_path = PATHS["gcut_error_log"]
    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write("=== G-Cut Processing Exception Log ===\n")
        f.flush()
        
    # 读取 Meta 文件
    meta_file_path = PATHS["meta_file"]
    print("Loading Metadata...")
    meta_df = pd.read_csv(meta_file_path, index_col='cell_id', low_memory=False)
    
    TARGET_PERCENTILE = GCUT["target_percentile"] 
    GSDT_THRESHOLD_X = GCUT["gsdt_threshold_x"]
    
    specific_task_ids = list()
    
    files_to_process = []
    if specific_task_ids and len(specific_task_ids) > 0:
        print(f"检测到特定任务列表，准备处理 {len(specific_task_ids)} 个指定的 Neuron ID...")
        for nid in specific_task_ids:
            img_file = img_dir / f"image_{nid}_0000.tif"
            if img_file.exists():
                files_to_process.append(img_file)
            else:
                print(f"警告: 指定的任务文件不存在 -> {img_file}")
    else:
        print("未指定特定任务，准备扫描整个 img 文件夹...")
        files_to_process = list(img_dir.glob("image_*.tif"))
        
    tasks = []
    print("正在构建任务队列并检查已有结果，请稍候...")
    
    for original_img_file in files_to_process:
        temp_stem = original_img_file.stem.replace("_0000", "")
        selected_path = result_dir / f"{temp_stem}.swc"
        if swc_has_nodes(selected_path):
            print(f"{original_img_file} 已经处理: {selected_path}")
            continue
        prepared_selected = prepare_selected_swc(result_dir, original_img_file, vis_dir)
        if prepared_selected is not None:
            print(f"{original_img_file} 已有 G-Cut 结果，已补齐 selected_for_pruning: {prepared_selected}")
            continue
            
        mask_file = mask_dir / original_img_file.name.replace("_0000.tif", ".tif")
        swc_file = swc_dir / original_img_file.name.replace("_0000.tif", ".swc")
        
        # 构造 Crop 文件的路径
        soma_json_path = soma_img_dir / f"{temp_stem}.json"
        soma_mask_path = soma_mask_dir / f"{temp_stem}.tif"
        
        if not mask_file.exists():
            continue

        tasks.append((
            original_img_file, mask_file, swc_file, vis_dir, 
            TARGET_PERCENTILE, GSDT_THRESHOLD_X, error_log_path,
            soma_json_path, soma_mask_path, meta_df, result_dir
        ))

    results = {}
    MAX_WORKERS = 8 
    print(f"队列构建完毕！需实际计算的任务共 {len(tasks)} 个。启用 {MAX_WORKERS} 个进程开始处理...\n")

    if len(tasks) > 0:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(worker_task, task): task for task in tasks}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="处理进度"):
                    img_name, res, log_msg = future.result()
                    
                    if res is not None:
                        threshold_val, extracted_coords = res
                        results[img_name] = {
                            "intensity_threshold": threshold_val,
                            "soma_coords": extracted_coords
                        }
                    else:
                        results[img_name] = {"intensity_threshold": 0, "soma_coords": []}
                        
                    if "跳过" in log_msg or "失败" in log_msg or "崩溃" in log_msg or "清理了" in log_msg:
                        tqdm.write(f"[{img_name}] {log_msg}")
                    
                    del future
                    
            print(f"\n所有图片处理完成！异常记录已安全保存至: {error_log_path}")
            
        except KeyboardInterrupt:
            print(f"\n\n[警告] 检测到手动中断 (Ctrl+C) 或强制终止！")
            print(f"程序正在安全退出。由于启用了实时落盘机制，中断前产生的所有日志已安全保存在: {error_log_path}")
            
    else:
        print("\n没有需要处理的新任务。")