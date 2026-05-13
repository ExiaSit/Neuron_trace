import os
import json
import numpy as np
import tifffile
import cv2
from collections import defaultdict
from tqdm import tqdm

# ================= 图像处理辅助函数 =================

def normalize_to_uint8(img_data):
    """
    将图像数据(可能是uint16或其他)归一化到 0-255 的 uint8 格式，以便可视化。
    """
    img_data = img_data.astype(np.float32)
    
    min_val = np.min(img_data)
    max_val = np.max(img_data)
    
    if max_val == min_val:
        return np.zeros(img_data.shape, dtype=np.uint8)
        
    norm_img = (img_data - min_val) / (max_val - min_val) * 255.0
    return norm_img.astype(np.uint8)

def create_overlay_composite(orig_vol, mask_vol, output_path):
    """
    生成原图和叠加图的2x3组合视图。
    Row 1: Original MIPs (XY, XZ, YZ)
    Row 2: Overlay MIPs (Red mask on grayscale original)
    """
    orig_u8 = normalize_to_uint8(orig_vol)
    mask_bin = (mask_vol > 0).astype(np.uint8) # 确保是 0/1

    orig_xy = np.max(orig_u8, axis=0)
    mask_xy = np.max(mask_bin, axis=0)
    orig_xz = np.max(orig_u8, axis=1)
    mask_xz = np.max(mask_bin, axis=1)
    orig_yz = np.max(orig_u8, axis=2)
    mask_yz = np.max(mask_bin, axis=2)

    def make_overlay(gray_mip, mask_mip):
        bgr_base = cv2.cvtColor(gray_mip, cv2.COLOR_GRAY2BGR)
        red_layer = np.zeros_like(bgr_base)
        red_layer[:] = [0, 0, 255] 
        alpha = 0.7
        mask_bool = mask_mip > 0
        bgr_base[mask_bool] = cv2.addWeighted(bgr_base[mask_bool], alpha, red_layer[mask_bool], 1-alpha, 0).squeeze()
        return bgr_base

    ov_xy = make_overlay(orig_xy, mask_xy)
    ov_xz = make_overlay(orig_xz, mask_xz)
    ov_yz = make_overlay(orig_yz, mask_yz)
    orig_xy_bgr = cv2.cvtColor(orig_xy, cv2.COLOR_GRAY2BGR)
    orig_xz_bgr = cv2.cvtColor(orig_xz, cv2.COLOR_GRAY2BGR)
    orig_yz_bgr = cv2.cvtColor(orig_yz, cv2.COLOR_GRAY2BGR)
    
    col1 = np.vstack([orig_xy_bgr, ov_xy])
    col2 = np.vstack([orig_xz_bgr, ov_xz])
    col3 = np.vstack([orig_yz_bgr, ov_yz])

    target_height = col1.shape[0]

    col2_resized = cv2.resize(col2, (col2.shape[1], target_height), interpolation=cv2.INTER_LINEAR)
    col3_resized = cv2.resize(col3, (col3.shape[1], target_height), interpolation=cv2.INTER_LINEAR)

    spacer = np.zeros((target_height, 10, 3), dtype=np.uint8)
    final_composite = np.hstack([col1, spacer, col2_resized, spacer, col3_resized])

    cv2.imwrite(output_path, final_composite)


def restore_crops_to_original(
    original_dir,
    json_dir,
    seg_dir,
    output_dir,
    parse_parent_name_func,
    save_preview=True 
):
    os.makedirs(output_dir, exist_ok=True)
    preview_dir = os.path.join(output_dir, "mips")
    if save_preview:
        os.makedirs(preview_dir, exist_ok=True)

    tasks = defaultdict(list)
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    print(f"正在扫描 {len(json_files)} 个 crop 任务...")
    for json_file in json_files:
        basename = os.path.splitext(json_file)[0]
        seg_filename = basename + ".tif" 
        seg_path = os.path.join(seg_dir, seg_filename)
        json_path = os.path.join(json_dir, json_file)
        if not os.path.exists(seg_path): continue
        if os.path.exists(os.path.join(output_dir, seg_filename)): continue
        parent_image_name = parse_parent_name_func(json_file)
        tasks[parent_image_name].append((json_path, seg_path))
    
    for parent_name, crop_list in tqdm(tasks.items()):
        if os.path.exists(os.path.join(output_dir, parent_name)):
            continue
        original_path = os.path.join(original_dir, parent_name)
        if not os.path.exists(original_path):
            print(f"\nError: 找不到原图 {parent_name}，跳过。")
            continue
            
        with tifffile.TiffFile(original_path) as tif:
            original_vol = tif.asarray()
            original_shape = original_vol.shape
        try:
            soma_mask = np.zeros(original_shape, dtype=np.uint8)
            for json_path, seg_path in crop_list:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    b = data['bounds_1um_zxy'] 
                    z_start, z_end, y_start, y_end, x_start, x_end = b[0], b[1], b[2], b[3], b[4], b[5]

                seg_crop = tifffile.imread(seg_path)
                
                # 简单的尺寸安全校验
                d_z, d_y, d_x = seg_crop.shape
                # print(d_z, d_y, d_x)
                # print(f"    - 读取 crop: {os.path.basename(seg_path)} 尺寸 {seg_crop.shape}, 目标位置 Z:{z_start}-{z_end}, Y:{y_start}-{y_end}, X:{x_start}-{x_end}")
                t_z, t_y, t_x = (z_end-z_start), (y_end-y_start), (x_end-x_start)
                # print(f"    - 目标区域尺寸: {(t_z, t_y, t_x)}")
                real_z, real_y, real_x = min(d_z, t_z), min(d_y, t_y), min(d_x, t_x)
                # print(real_z,real_y,real_x)
                # print(full_mask.shape)
                
                soma_mask[z_start:z_end, y_start:y_end, x_start:x_end] = seg_crop[:real_z, :real_y, :real_x]
        

        # --- 4. 验证环节 (新增) ---
            if save_preview:
                preview_prefix = os.path.join(preview_dir, f"Preview_{parent_name}")
                create_overlay_composite(original_vol,soma_mask, preview_prefix)
        except Exception as e:
            print(f"  - 生成预览图时出错: {e},跳过{parent_name}")
            continue
        # --- 5. 保存最终大图 ---
        merged_mask = np.maximum(original_vol, soma_mask)
        full_mask = (merged_mask > 0).astype(np.uint8) * 255
        output_filename = f"{parent_name}"
        save_path = os.path.join(output_dir, output_filename)
        # print(f"  - 保存大图到: {save_path}") 
        tifffile.imwrite(save_path, full_mask, compression='zlib') 
        
def name_parser(crop_filename):
    base_id = crop_filename.split('.')[0]
    return base_id + ".tif"

if __name__ == "__main__":
    base_dir = "/data/disk/C6.0/app_test"
    Neurite_seg_DIR = os.path.join(base_dir,"1um_neurite_seg_0424")
    JSON_DIR = os.path.join(base_dir,"soma_img")
    Soma_SEG_DIR = os.path.join(base_dir,"soma_seg")
    OUTPUT_DIR = os.path.join(base_dir,"mask_new_version_0424")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    SAVE_PREVIEW_IMAGES = False 

    restore_crops_to_original(
        Neurite_seg_DIR, 
        JSON_DIR, 
        Soma_SEG_DIR, 
        OUTPUT_DIR,
        name_parser,
        save_preview=SAVE_PREVIEW_IMAGES
    )
