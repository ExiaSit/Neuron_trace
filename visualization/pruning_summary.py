import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_pruning_summary(show_steps, viz_buffer, raw_image, morph, sphere_zradius, pruning_reasons, prefix, mip_dir):
    if not show_steps or not viz_buffer: return
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei",
        "Arial Unicode MS", "DejaVu Sans"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(30, 35), dpi=110)
    grid = fig.add_gridspec(
        4, 6, height_ratios=[1.0, 1.0, 1.0, 1.45],
        hspace=0.10, wspace=0.04
    )
    axes = [
        fig.add_subplot(grid[row, col * 2:(col + 1) * 2])
        for row in range(3)
        for col in range(3)
    ]

    step_order = [
        ("step01_original_input", "Step 1: Original Input"), 
        ("step02_sphere_cuts", "Step 2: Mask Cuts"),
        ("step03_graph_traversed", "Step 3: Graph Traversed"), 
        ("step04_binary_mask", "Step 4: Binary Mask"),
        ("step05_soma_mask", "Step 5: Soma Mask"),
        ("step06_retained_roots", "Step 6: Retained Roots"),
        ("step07_new_connections", "Step 7: New Connections"),
        ("step08_geometry_cleanup", "Step 8: Merge / Zigzag"),
        ("step09_final_result", "Step 9: Final Result"),
    ]

    from visualization.swc_view import crop_xy_axes
    view_points = [n[2:5] for data in viz_buffer.values() for n in (data.get('tree') or [])]
    type_colors = {1: 'white', 2: 'magenta', 3: 'cyan', 4: 'purple', 0: 'dodgerblue'}
    for i, (key, title) in enumerate(step_order):
        ax = axes[i]
        if key not in viz_buffer:
            ax.text(0.5, 0.5, 'Skipped', ha='center', va='center', color='white')
            ax.set_title(title)
            ax.axis('off'); ax.set_facecolor('black')
            continue
        data = viz_buffer[key]
        bg_img = data['img'] if data['img'] is not None else raw_image 
        bg_img_2d = np.max(bg_img, axis=0) if bg_img.ndim == 3 else bg_img
        # np.percentile cannot interpolate boolean arrays. Step 4's binary
        # mask is bool, so convert all panel backgrounds to a numeric dtype
        # before calculating the display range.
        bg_img_2d = np.asarray(bg_img_2d, dtype=np.float32)
        vmax_val = np.percentile(bg_img_2d, 99.5)
        if vmax_val <= 0:  
            vmax_val = np.max(bg_img_2d)
            if vmax_val <= 0: vmax_val = 1 
        ax.imshow(bg_img_2d, cmap='gray', vmax=vmax_val)

        swc_list = data['tree']
        if swc_list and key != "step05_soma_mask":
            node_map = {n[0]: (n[2], n[3]) for n in swc_list}
            for node in swc_list:
                nid, ntype, x, y, z, r, pid = node
                color = type_colors.get(ntype, 'dodgerblue')
                if pid != -1 and pid in node_map:
                    px, py = node_map[pid]
                    ax.plot([x, px], [y, py], color=color, linewidth=0.8, alpha=0.7)
                if pid == -1: ax.scatter(x, y, s=30, c='red', linewidths=1.2, marker='o', zorder=10)

        if key == "step02_sphere_cuts":
            soma_node = morph.pos_dict.get(list(morph.pos_dict.keys())[0]) 
            for pid, info in morph.pos_dict.items():
                if info[1] == 1:
                    soma_node = info; break
            raw_soma_x, raw_soma_y = soma_node[2], soma_node[3]
            soma_circle = plt.Circle((raw_soma_x, raw_soma_y), sphere_zradius, color='white', fill=False, linestyle='--', linewidth=1.5, alpha=0.9)
            ax.add_patch(soma_circle)
            ax.scatter(raw_soma_x, raw_soma_y, c='red', s=40, marker='x', zorder=10)

        # 🌟 统一渲染各种标注
        if 'annotations' in data and data['annotations']:
            for ann in data['annotations']:
                if 'text_x' in ann: 
                    ax.plot([ann['rx'], ann['text_x']], [ann['ry'], ann['text_y']], color='black', linewidth=1.5, zorder=5)
                    ax.text(ann['text_x'], ann['text_y'], ann['text'], color=ann['color'], fontsize=9, 
                            ha='center', va='center', fontweight='bold', zorder=9,
                            bbox=dict(facecolor='black', alpha=0.8, edgecolor=ann['color'], pad=1.5))
                    if 'vsc_end_x' in ann:
                        ax.annotate('', xy=(ann['vsc_end_x'], ann['vsc_end_y']), xytext=(ann['rx'], ann['ry']),
                                    arrowprops=dict(arrowstyle="->", color='yellow', lw=2, alpha=0.9), zorder=8)
                    if 'vc_end_x' in ann:
                        ax.annotate('', xy=(ann['vc_end_x'], ann['vc_end_y']), xytext=(ann['rx'], ann['ry']),
                                    arrowprops=dict(arrowstyle="->", color='green', lw=2, alpha=0.9), zorder=8)

        ax.set_title(title, color='black', fontsize=12)
        ax.axis('off')

    comparison_input_ax = fig.add_subplot(grid[3, :3])
    comparison_output_ax = fig.add_subplot(grid[3, 3:])
    input_tree = viz_buffer.get("step01_original_input", {}).get("tree") or []
    output_tree = viz_buffer.get("step09_final_result", {}).get("tree") or []
    output_ids = {int(node[0]) for node in output_tree}
    removed_ids = {
        int(node[0]) for node in input_tree if int(node[0]) not in output_ids
    }

    def draw_comparison(ax, tree_data, title, show_removed_overlay):
        bg_img_2d = np.max(raw_image, axis=0) if raw_image.ndim == 3 else raw_image
        vmax_val = np.percentile(bg_img_2d, 99.5)
        if vmax_val <= 0:
            vmax_val = max(float(np.max(bg_img_2d)), 1.0)
        ax.imshow(bg_img_2d, cmap="gray", vmax=vmax_val)
        node_map = {
            int(node[0]): (float(node[2]), float(node[3]))
            for node in tree_data
        }
        for node in tree_data:
            node_id, parent_id = int(node[0]), int(node[6])
            if parent_id in node_map:
                x, y = node_map[node_id]
                px, py = node_map[parent_id]
                ax.plot(
                    [px, x], [py, y], color="#38d9ff", linewidth=0.75,
                    alpha=0.75, zorder=3
                )

        if show_removed_overlay:
            original_xy = {
                int(node[0]): (float(node[2]), float(node[3]))
                for node in input_tree
            }
            for node in input_tree:
                node_id, parent_id = int(node[0]), int(node[6])
                if node_id not in removed_ids or parent_id not in original_xy:
                    continue
                x, y = original_xy[node_id]
                px, py = original_xy[parent_id]
                ax.plot(
                    [px, x], [py, y], color="red", linewidth=1.35,
                    linestyle="--", alpha=0.9, zorder=5
                )
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.axis("off")

    draw_comparison(
        comparison_input_ax, input_tree,
        "剪枝前：红色虚线为最终被删除的原始分支", True
    )
    draw_comparison(
        comparison_output_ax, output_tree,
        "剪枝后：青色为保留结构，红色虚线为被删除分支", True
    )

    removed_roots = []
    input_parent = {int(node[0]): int(node[6]) for node in input_tree}
    for node_id in sorted(removed_ids):
        if input_parent.get(node_id, -1) not in removed_ids:
            removed_roots.append(node_id)

    input_xy = {
        int(node[0]): (float(node[2]), float(node[3])) for node in input_tree
    }
    reason_categories = {}
    root_entries = []
    for node_id in removed_roots:
        info = pruning_reasons.get(
            node_id,
            {
                "reason": "前序筛选后未进入最终树",
                "code": "process_neuron_morphology",
            },
        )
        category_key = (info["reason"], info["code"])
        if category_key not in reason_categories:
            reason_categories[category_key] = len(reason_categories) + 1
        root_entries.append((node_id, info, reason_categories[category_key]))

    category_counts = {
        category_number: sum(
            entry_category == category_number
            for _, _, entry_category in root_entries
        )
        for category_number in reason_categories.values()
    }

    for node_id, info, category_number in root_entries:
        x, y = input_xy.get(node_id, (None, None))
        if x is None:
            continue
        comparison_output_ax.scatter(
            [x], [y], c="yellow", edgecolors="red", s=32, zorder=8
        )
        comparison_output_ax.text(
            x, y, f"R{category_number}",
            fontsize=7.0, color="yellow", fontweight="bold",
            ha="left", va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.15", facecolor="black",
                edgecolor="red", alpha=0.78
            ),
            zorder=9,
        )

    legend_lines = ["剪枝原因编号："]
    for (reason, code), category_number in reason_categories.items():
        legend_lines.append(
            f"R{category_number}（{category_counts[category_number]}处）：{reason}"
        )
        legend_lines.append(f"代码：{code}")
    legend_text = "\n".join(legend_lines)
    comparison_output_ax.text(
        0.01, 0.99, legend_text,
        transform=comparison_output_ax.transAxes,
        fontsize=7.5,
        color="white",
        ha="left",
        va="top",
        linespacing=1.25,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="black",
            edgecolor="red", alpha=0.78
        ),
        zorder=10,
    )

    for ax in [*axes, comparison_input_ax, comparison_output_ax]:
        crop_xy_axes(ax, raw_image.shape[-2:], view_points)

    fig.suptitle(
        f"{prefix} 剪枝流程与原因对照", fontsize=19, fontweight="bold", y=0.995
    )
    plt.savefig(
        os.path.join(mip_dir, f'{prefix}_combined_process.png'),
        bbox_inches="tight",
    )

    plt.close('all')
