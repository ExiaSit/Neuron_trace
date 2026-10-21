import os
import glob
from collections import defaultdict
import numpy as np
import tifffile
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Optional dependency used only when recall diagnostic images are enabled.
try:
    from image_enhancer import do_gamma
except ImportError:
    do_gamma = None

# ==========================================
# 1. SWC 极速解析与重采样模块 (复用)
# ==========================================
def resample_swc_customized_core(swc_lines, step_length=1.0):
    nodes = {}
    children = defaultdict(list)
    comments = []

    for line in swc_lines:
        line_s = line.strip()
        if not line_s: continue
        if line_s.startswith('#'):
            comments.append(line_s + '\n')
            continue

        parts = line_s.split()
        if len(parts) >= 7:
            n, t, x, y, z, r, p = parts
            nodes[n] = [int(n), int(t), float(x), float(y), float(z), float(r), p]
            if p != '-1':
                children[p].append(n)

    key_nodes = set(['-1'])
    roots = []

    for n, data in nodes.items():
        p = data[6]
        c_list = children[n]
        if p == '-1':
            roots.append(n)
            key_nodes.add(n)
        elif len(c_list) != 1:
            key_nodes.add(n)

    branches = []
    queue = roots.copy()

    while queue:
        start_node = queue.pop(0)
        for child in children[start_node]:
            branch_path = [start_node]
            curr = child
            while curr not in key_nodes:
                branch_path.append(curr)
                if not children[curr]:
                    break
                curr = children[curr][0]
            branch_path.append(curr)
            branches.append(branch_path)
            if len(children[curr]) > 0:
                queue.append(curr)

    new_nodes = []
    new_id_counter = 1
    key_node_map = {}
    
    for branch in branches:
        coords = np.array([[nodes[n][2], nodes[n][3], nodes[n][4], nodes[n][5]] for n in branch])
        types = [nodes[n][1] for n in branch]

        diffs = np.diff(coords[:, :3], axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cum_length = np.insert(np.cumsum(segment_lengths), 0, 0)
        total_length = cum_length[-1]

        if total_length <= step_length:
            new_cum = np.array([0, total_length])
        else:
            new_cum = np.arange(0, total_length, step_length)
            if total_length - new_cum[-1] > 1e-5:
                new_cum = np.append(new_cum, total_length)

        new_x = np.interp(new_cum, cum_length, coords[:, 0])
        new_y = np.interp(new_cum, cum_length, coords[:, 1])
        new_z = np.interp(new_cum, cum_length, coords[:, 2])
        new_r = np.interp(new_cum, cum_length, coords[:, 3])
        
        parent_node_original = branch[0]
        child_node_original = branch[-1]

        if parent_node_original in key_node_map:
            current_parent_id = key_node_map[parent_node_original]
        else:
            current_parent_id = new_id_counter
            key_node_map[parent_node_original] = current_parent_id
            new_nodes.append(f"{current_parent_id} {types[0]} {new_x[0]:.3f} {new_y[0]:.3f} {new_z[0]:.3f} {new_r[0]:.3f} -1")
            new_id_counter += 1

        for i in range(1, len(new_cum)):
            this_id = new_id_counter
            new_id_counter += 1
            t = types[-1] if i == len(new_cum) - 1 else types[0]
            new_nodes.append(f"{this_id} {t} {new_x[i]:.3f} {new_y[i]:.3f} {new_z[i]:.3f} {new_r[i]:.3f} {current_parent_id}")
            current_parent_id = this_id

        key_node_map[child_node_original] = current_parent_id

    return new_nodes

def read_and_resample_swc(swc_path):
    with open(swc_path, 'r') as f:
        lines = f.readlines()
    resampled_lines = resample_swc_customized_core(lines, step_length=1.0)
    
    nodes = {}
    for line in resampled_lines:
        parts = line.split()
        nid, t, x, y, z, r, pid = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), int(parts[6])
        nodes[nid] = {'x': x, 'y': y, 'z': z, 'r': r, 'pid': pid}
        
    coords, parent_indices, radii = [], [], []
    id_to_idx = {nid: i for i, nid in enumerate(nodes.keys())}
    
    for nid, data in nodes.items():
        coords.append([data['x'], data['y'], data['z']])
        # 保存半径 r (如果有重采样，最好在重采样时也插值 r，这里假设 nodes 已经包含了准确的 r)
        radii.append(data.get('r', 1.0)) 
        p_idx = id_to_idx.get(data['pid'], id_to_idx[nid])
        parent_indices.append(p_idx)
        
    return np.array(coords), np.array(parent_indices), np.array(radii)
    

# ==========================================
# 2. Precision 引擎：SNR 与 局部线性度
# ==========================================
def compute_robust_snr(raw_img, coords_zyx, parent_indices, bg_radius=4.0, num_ring_points=8):
    """计算抗串线干扰的鲁棒局部信噪比"""
    N = len(coords_zyx)
    if N < 2: return np.zeros(N), np.zeros(N), np.zeros(N)

    tangents = coords_zyx - coords_zyx[parent_indices]
    zero_mask = np.linalg.norm(tangents, axis=1) == 0
    tangents[zero_mask] = np.array([1.0, 0.0, 0.0]) 
    tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]

    UP = np.zeros_like(tangents)
    UP[:, 0] = 1.0
    collinear_mask = np.abs(tangents[:, 0]) > 0.99
    UP[collinear_mask, 0] = 0.0
    UP[collinear_mask, 1] = 1.0

    U = np.cross(tangents, UP)
    U = U / np.linalg.norm(U, axis=1)[:, np.newaxis]
    V = np.cross(tangents, U) 

    angles = np.linspace(0, 2 * np.pi, num_ring_points, endpoint=False)
    ring_coords = np.zeros((num_ring_points, N, 3))
    for i, theta in enumerate(angles):
        ring_coords[i] = coords_zyx + bg_radius * (np.cos(theta) * U + np.sin(theta) * V)

    center_signal = map_coordinates(raw_img, coords_zyx.T, order=1, mode='nearest')
    ring_signals = np.zeros((num_ring_points, N))
    for i in range(num_ring_points):
        ring_signals[i] = map_coordinates(raw_img, ring_coords[i].T, order=1, mode='nearest')

    # 【关键升级】：取第 20 百分位数作为鲁棒背景，避开路过的其他神经元
    bg_robust_baseline = np.percentile(ring_signals, 20, axis=0)
    bg_median = np.median(ring_signals, axis=0)
    bg_robust_std = np.clip((bg_median - bg_robust_baseline) / 0.6745, 1e-3, None)

    snr = (center_signal - bg_robust_baseline) / bg_robust_std
    return center_signal, bg_robust_baseline, bg_robust_std, np.clip(snr, 0, 50)

def compute_local_linearity(coords_zyx, parent_indices, window_steps=10):
    """计算基于拓扑树的局部线性度 (Euclidean / Path Length)"""
    N = len(coords_zyx)
    linearity = np.ones(N)
    
    for i in range(N):
        curr = i
        path_len = 0.0
        # 沿拓扑树向上回溯 window_steps 步
        for _ in range(window_steps):
            p = parent_indices[curr]
            if p == curr: break
            path_len += np.linalg.norm(coords_zyx[curr] - coords_zyx[p])
            curr = p
            
        if path_len > 0:
            euclidean = np.linalg.norm(coords_zyx[i] - coords_zyx[curr])
            linearity[i] = euclidean / path_len
            
    return linearity

# ==========================================
# 3. Recall 引擎：射线连通性与暗区探测
# ==========================================
def detect_missed_branches(raw_img, coords_zyx, parent_indices, radii, bgs, stds, r_search=5.0):
    """提取了诊断坐标的漏检探测器"""
    N = len(coords_zyx)
    children_map = {i: [] for i in range(N)}
    for i, p in enumerate(parent_indices):
        if i != p: children_map[p].append(i)
        
    soma_idx = np.where(parent_indices == np.arange(N))[0]
    if len(soma_idx) > 0:
        soma_coord, soma_r = coords_zyx[soma_idx[0]], radii[soma_idx[0]]
    else:
        soma_coord, soma_r = coords_zyx[0], 5.0

    raw_anchors = np.where(np.array([len(children_map[i]) for i in range(N)]) == 0)[0]
    
    anchors = [a for a in raw_anchors if np.linalg.norm(coords_zyx[a] - soma_coord) > (soma_r + 3.0)]
    
    samples = 100
    phi = np.pi * (3. - np.sqrt(5.))
    theta = phi * np.arange(samples)
    y = 1 - (np.arange(samples) / float(samples - 1)) * 2
    x = np.cos(theta) * np.sqrt(1 - y * y)
    z = np.sin(theta) * np.sqrt(1 - y * y)
    sphere_dirs = np.vstack((z, y, x)).T 
    
    tree = KDTree(coords_zyx)
    
    missed_branch_count = 0
    total_known_branches = 0
    
    # ==========================================
    # 【新增】：用于可视化的诊断记录列表
    # 格式: [(起点Z,Y,X, 终点Z,Y,X), ...]
    # ==========================================
    diag_passed = []       # 真正漏追的
    diag_failed_angle = [] # 因为角度冲突失败的 (光晕)
    diag_failed_gap = []   # 因为暗区断开失败的 (路过)

    for anchor in anchors:
        center = coords_zyx[anchor]
        bg, std = bgs[anchor], stds[anchor]
        
        known_dirs = []
        curr = anchor
        for _ in range(5):
            if parent_indices[curr] == curr: break
            curr = parent_indices[curr]
        if curr != anchor:
            v = coords_zyx[curr] - center
            known_dirs.append(v / (np.linalg.norm(v) + 1e-5))
            
        queue = [(c, 1) for c in children_map[anchor]]
        while queue:
            curr, depth = queue.pop(0)
            if depth == 5 or not children_map[curr]:
                v = coords_zyx[curr] - center
                known_dirs.append(v / (np.linalg.norm(v) + 1e-5))
            else:
                for c in children_map[curr]: queue.append((c, depth + 1))
                
        total_known_branches += len(known_dirs)

        shell_coords = center + r_search * sphere_dirs
        dists, nearest_idx = tree.query(shell_coords)
        local_radii = radii[nearest_idx]
        
        dynamic_valid_mask = dists > (local_radii + 3.0) 
        soma_valid_mask = np.linalg.norm(shell_coords - soma_coord, axis=1) > (soma_r + 2.0)
        
        valid_mask = dynamic_valid_mask & soma_valid_mask
        if not np.any(valid_mask): continue
        valid_shell_coords = shell_coords[valid_mask]

        # 1. 将 center (形状 3,) 变形为 (1, 3)，并拼接到 valid_shell_coords (形状 N, 3) 的最前面
        combined_coords = np.vstack([center, valid_shell_coords])

        # 2. 一次性调用 map_coordinates 进行插值
        combined_I = map_coordinates(raw_img, combined_coords.T, order=1, mode='nearest')

        # 3. 将结果拆包解构：第 0 个元素是中心点亮度，剩下的都是球壳候选点亮度
        center_I = combined_I[0]
        shell_I = combined_I[1:]
        
        candidate_mask = shell_I > max(bg + 3.0 * std, center_I * 0.4)
        #print(bg, std)
        #print(bg + 3.0*std, center_I)
        #print(shell_I)
        #print('\n')
        candidates = valid_shell_coords[candidate_mask]
        if len(candidates) == 0: continue
            
        if len(candidates) > 1:
            cand_tree = KDTree(candidates)
            pairs = cand_tree.query_pairs(r=2.5)
            keep = np.ones(len(candidates), dtype=bool)
            for i, j in pairs:
                if keep[i] and keep[j]: keep[j] = False
            candidates = candidates[keep]
            
        for cand in candidates:
            cand_vec = cand - center
            cand_dir = cand_vec / (np.linalg.norm(cand_vec) + 1e-5)
            
            is_redundant = False
            for kd in known_dirs:
                if np.dot(cand_dir, kd) > 0.8: 
                    is_redundant = True
                    break
            
            if is_redundant: 
                # 【记录】：因为角度冲突被淘汰
                diag_failed_angle.append((center, cand))
                continue 

            ray_steps = np.linspace(0.4, 1.0, num=6)
            ray_coords = center + ray_steps[:, None] * (cand - center)
            ray_I = map_coordinates(raw_img, ray_coords.T, order=1, mode='nearest')
            
            if np.min(ray_I) > (bg + 1.5 * std): 
                # 【记录】：完全通过，确实漏追
                missed_branch_count += 1
                diag_passed.append((center, cand))
            else:
                # 【记录】：因为射线中间断崖（有暗区）被淘汰
                diag_failed_gap.append((center, cand))
                
    # 返回诊断列表
    return missed_branch_count, total_known_branches, diag_passed, diag_failed_angle, diag_failed_gap

def plot_recall_diagnostics(raw_img, coords_zyx, parent_indices, 
                            diag_passed, diag_failed_angle, diag_failed_gap, 
                            base_name, output_dir):
    """绘制 Recall 漏检诊断的 XY 叠加图"""
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    
    # 1. 绘制图像底图 (XY MIP)
    mip_xy = np.max(raw_img, axis=0) # 维度 (Y, X)
    vmin, vmax = np.percentile(mip_xy, 0.5), np.percentile(mip_xy, 99.8)
    ax.imshow(mip_xy, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    
    # 2. 绘制已有的 SWC 骨架 (半透明红色)
    X, Y = coords_zyx[:, 2], coords_zyx[:, 1]
    for i, p in enumerate(parent_indices):
        if i == p: continue
        ax.plot([X[p], X[i]], [Y[p], Y[i]], color='red', linewidth=0.8, alpha=0.4)
        
    # 3. 辅助函数：绘制诊断射线
    def plot_rays(rays, color, linestyle, linewidth, marker='o'):
        for (center, cand) in rays:
            # 坐标转换：ZYX -> (X, Y) 即 (index 2, index 1)
            cx, cy = center[2], center[1]
            cand_x, cand_y = cand[2], cand[1]
            ax.plot([cx, cand_x], [cy, cand_y], color=color, 
                    linestyle=linestyle, linewidth=linewidth, alpha=0.9)
            ax.scatter(cand_x, cand_y, color=color, s=15, marker=marker, zorder=5)

    # 绘制被淘汰的射线：角度冲突 (紫色虚线，大概率是光晕)
    plot_rays(diag_failed_angle, color='magenta', linestyle=':', linewidth=1.5, marker='x')
    
    # 绘制被淘汰的射线：暗区断裂 (橙色虚线，大概率是路过)
    plot_rays(diag_failed_gap, color='orange', linestyle='--', linewidth=1.5, marker='s')
    
    # 绘制成功的射线：真正的漏检 (绿色实线)
    plot_rays(diag_passed, color='lime', linestyle='-', linewidth=2.5, marker='*')
    
    # 4. 图例设置
    red_patch = mpatches.Patch(color='red', alpha=0.4, label='Existing SWC')
    green_line = plt.Line2D([0], [0], color='lime', lw=2.5, marker='*', label=f'Missed (Passed) - {len(diag_passed)}')
    magenta_line = plt.Line2D([0], [0], color='magenta', lw=1.5, ls=':', marker='x', label=f'Rejected (Angle) - {len(diag_failed_angle)}')
    orange_line = plt.Line2D([0], [0], color='orange', lw=1.5, ls='--', marker='s', label=f'Rejected (Gap) - {len(diag_failed_gap)}')
    
    ax.legend(handles=[red_patch, green_line, magenta_line, orange_line], loc='upper right', framealpha=0.8)
    
    ax.set_title(f"Recall Diagnostics (XY MIP) - {base_name}", fontsize=14)
    ax.axis('off')
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_name}_recall_diagnostics.png")
    plt.tight_layout()
    plt.savefig(out_path, facecolor='black')
    plt.close()

def estimate_prec_rec_for_all(img_files, swc_dir, perturbation=False):
    results = []

    for img_path in img_files:
        filename = os.path.basename(img_path)
        base_name = filename[:-9]
        swc_path = os.path.join(swc_dir, f"{base_name}.swc")
        
        if not os.path.exists(swc_path): continue
            
        print(f"Evaluating: {base_name}")
        raw_img = tifffile.imread(img_path).astype(np.float32)
        coords_xyz, parent_indices, radii = read_and_resample_swc(swc_path)
        if len(coords_xyz) < 10: continue
            
        coords_zyx = coords_xyz[:, [2, 1, 0]]
        if perturbation :
            coords_zyx += np.random.uniform(-10, 10, size=3)
        
        # 预检：防止坐标系完全错位
        mean_intensity = np.mean(map_coordinates(raw_img, coords_zyx.T, order=1))
        if mean_intensity < np.mean(raw_img) * 1.2:
            print("  -> [警告] 坐标错位或彻底脱靶，跳过。")
            continue

        # 1. 计算局部特征
        _, bgs, stds, snrs = compute_robust_snr(raw_img, coords_zyx, parent_indices)
        linearities = compute_local_linearity(coords_zyx, parent_indices, window_steps=10)
        
        # ==========================================
        # 2. 统计 Precision (保真度) - 【新增：胞体排除逻辑】
        # ==========================================
        # 找到 Soma 坐标和半径
        soma_idx = np.where(parent_indices == np.arange(len(coords_zyx)))[0]
        if len(soma_idx) > 0:
            soma_coord = coords_zyx[soma_idx[0]]
            soma_r = radii[soma_idx[0]]
        else:
            soma_coord = coords_zyx[0]
            soma_r = 5.0
            
        # 计算所有节点到 Soma 中心的距离
        dists_to_soma = np.linalg.norm(coords_zyx - soma_coord, axis=1)
        
        # 建立 Soma 排除掩膜 (Buffer 依然设定为半径外加 2.0um)
        outside_soma_mask = dists_to_soma > (soma_r + 2.0)
        
        # 满足高质量条件的节点 (SNR 强且未出现极端弯折)
        valid_nodes = (snrs > 2.0) & (linearities > 0.6)
        
        # 最终 Precision Score：只在胞体外部的有效树突区域进行统计
        valid_eval_count = np.sum(outside_soma_mask)
        if valid_eval_count > 0:
            # 统计：(既是好节点 AND 又在胞体外) 的数量 / 胞体外的总节点数
            precision_score = np.sum(valid_nodes & outside_soma_mask) / valid_eval_count
        else:
            precision_score = 0.0 # 极端情况：整个 SWC 全在胞体里
        # ==========================================
        
        # 3. 统计 Recall (防漏检) 并获取诊断数据
        missed_branches, total_known_branches, diag_passed, diag_failed_angle, diag_failed_gap = detect_missed_branches(
            raw_img, coords_zyx, parent_indices, radii, bgs, stds
        )
        
        # 计算 Recall 分数
        if total_known_branches + missed_branches == 0:
            recall_score = 1.0 
        else:
            recall_score = total_known_branches / (total_known_branches + missed_branches)

        # ==========================================
        # 生成诊断可视化图
        # ==========================================
        visual_diagnostic = False
        if visual_diagnostic and not perturbation:
            if do_gamma is None:
                raise RuntimeError(
                    "Recall diagnostics require the optional image_enhancer module."
                )
            #vis_recall_dir = os.path.join('.', "vis_recall_diagnostics")
            vis_recall_dir = '.'
            # enhance weak fibers
            img_enh = do_gamma(raw_img, 0.2, trunc_thresh=0, normalize=True, epsilon=1e-7).astype(np.uint8)
            plot_recall_diagnostics(
                img_enh, coords_zyx, parent_indices, 
                diag_passed, diag_failed_angle, diag_failed_gap, 
                base_name, vis_recall_dir
            )

        results.append({
            "Neuron_ID": base_name,
            "Precision_Score": precision_score,
            "Recall_Score": recall_score,
            "Mean_SNR": np.mean(snrs),
            "Missed_Branches": missed_branches,
            "Total_Nodes": len(coords_zyx)
        })
        
        print(f"  -> Precision: {precision_score*100:.1f}% | Recall: {recall_score*100:.1f}% (漏检: {missed_branches} 处). [#Nodes]={len(coords_zyx)}")

        #break # debug
    
    df_results = pd.DataFrame(results)
    return df_results

# ==========================================
# 4. 主流程
# ==========================================
def main(perturbation=False):
    base_dir = os.path.expanduser("~/Research/human_recon/image_sholl/random_samples")
    raw_img_dir = os.path.join(base_dir, "raw_img")
    # LYF: We must use resampled swc with `step size < 3`
    swc_dir = os.path.join(base_dir, "swc_resampled2um")
    
    img_files = glob.glob(os.path.join(raw_img_dir, "*.tif"))

    df_results = estimate_prec_rec_for_all(img_files, swc_dir, perturbation=perturbation)

    if perturbation:
        out_swc = "zero_prior_evaluation_perturbated.csv"
    else:
        out_swc = "zero_prior_evaluation.csv"
    
    df_results.to_csv(os.path.join('.', out_swc), index=False)
    print("全部评估完成！")

if __name__ == "__main__":
    perturbation = True
    main(perturbation=perturbation)
