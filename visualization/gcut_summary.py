from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_comprehensive_summary(img, binary_img, dist_transform, gsdt_mask, global_soma_mask, pre_nms_coords, post_nms_coords,
                               app2_tree, split_trees, target_soma_id,
                               save_path, nms_radius=20.0, title_prefix="", individual_output_dir=None,
                               distance_map_name="EDT Distance Map (um)"):
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
    ax[2].set_title(f"3. {distance_map_name}", fontsize=16)
    ax[2].axis('off')

    # 图 4：NMS 筛选前的所有候选点与抑制范围
    ax[3].imshow(mask_show, cmap='gray')
    soma_overlay = np.ma.masked_where(soma_mask_show == 0, soma_mask_show)
    ax[3].imshow(soma_overlay, cmap='cool', alpha=0.6)
    ax[3].set_title(f"4. Evidence-filtered / Pre-NMS (N={len(pre_nms_coords)})", fontsize=16)
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
    ax[7].set_title(f"5. After NMS + 3D Signal Merge (N={len(post_nms_coords)})", fontsize=16)
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
                ax[6].plot([node[2], p_node[2]], [node[3], p_node[3]], color='cyan', linewidth=1.5, alpha=0.6)
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
                    ax[5].plot([node[2], p_node[2]], [node[3], p_node[3]], color=color, linewidth=1.5, alpha=0.8)
            soma_node = node_dict.get(sid)
            if soma_node:
                ax[5].plot(soma_node[2], soma_node[3], marker='*', color=color, markersize=15, markeredgecolor='white')
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
                ax[4].plot([node[2], p_node[2]], [node[3], p_node[3]], color=color, linewidth=1.5, alpha=0.8)
        soma_node = node_dict.get(target_soma_id)
        if soma_node:
            ax[4].plot(soma_node[2], soma_node[3], marker='*', color=color, markersize=15, markeredgecolor='white')
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
