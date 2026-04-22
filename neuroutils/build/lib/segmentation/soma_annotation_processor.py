import os

import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from tqdm import tqdm
from skimage.transform import resize
from enum import Enum
from typing import Tuple, List, Optional, Union

# 假设外部依赖函数
from neuroutils.image.io import rotate_img_to_mip
from neuroutils.image.preprocessor import generate_2d_mask_from_polygonal, generate_3d_mask_from_2d_mip_mask
from neuroutils.meta.neuron import get_neuron_meta, get_soma_pos, get_tile_id, get_xy_z_resolution
from neuroutils.meta.tile import get_tile_resolution, get_mapped_somas_in_neuron_block, get_somas_in_tile
from neuroutils.visualization.plot import IMG_SEG_QCVisualStyle, IMG_SEG_QCGenerator

# ==========================================
# 1. 枚举与配置 (Configuration)
# ==========================================

class DataSourceType(Enum):
    SINGLE_NEURON = "CELL_BLOCK"  # 单个神经元数据
    TILE_SCAN     = "PTRSB_DB"    # 扫描块数据

class PipelineMode(Enum):
    PREPARE_LABELS = "GenerateMIPtolabelme"  # 阶段一：生成投影供标注
    RECONSTRUCT_3D = "GenerateMask"          # 阶段二：从标注生成Mask

class ProcessingConfig:
    def __init__(self,
                 source_volume_path: Union[str, Path],
                 workspace_dir: Union[str, Path],
                 mode: PipelineMode,
                 data_type: DataSourceType,
                 qc_style: IMG_SEG_QCVisualStyle = IMG_SEG_QCVisualStyle.OVERLAY,
                 crop_size_um: Tuple[int, int, int] = (50, 50, 50),
                 rotate_times: int = 12):
        """
        :param source_volume_path: 原始 3D 图像路径 (.tif)
        :param workspace_dir: 项目根目录，将自动创建 projections, masks 等子目录
        :param mode: 运行模式
        :param data_type: 数据源类型
        :param crop_size_um: 裁剪块的物理尺寸 (z, y, x) 单位 um
        """
        self.source_path = Path(source_volume_path)
        self.workspace = Path(workspace_dir)
        self.mode = mode
        self.data_type = data_type
        self.crop_size_um = crop_size_um
        self.rotate_times = rotate_times
        self.rotation_range_deg = 180  # 默认旋转范围
        self.qc_style = qc_style


# ==========================================
# 2. 路径管理器 (Path Manager) - 核心优化点
# ==========================================

class PathManager:
    """专门负责管理文件路径结构，将文件管理逻辑从业务逻辑中剥离"""

    def __init__(self, config: ProcessingConfig, sample_id: str):
        self.cfg = config
        self.sample_id = sample_id  # 比如 neuron_id 或 tile_flag

        # 定义子目录结构
        self.proj_dir = self.cfg.workspace / "projections" / self.sample_id
        self.mask_dir = self.cfg.workspace / "masks"
        self.qc_dir   = self.cfg.workspace / "qc"

        # 自动创建目录
        for p in [self.proj_dir, self.mask_dir, self.qc_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def _index_to_degree(self, idx: int) -> int:
        """
        将循环索引转换为实际角度。
        例如：rotate_angles=4, range=180
        idx 0 -> 0度
        idx 1 -> 45度
        idx 2 -> 90度
        idx 3 -> 135度
        """
        if self.cfg.rotate_times <= 1:
            return 0
        step = self.cfg.rotation_range_deg / self.cfg.rotate_times
        return int(idx * step)

    def _get_filename_candidates(self, center: Tuple[int, int, int], angle_idx: int, extension: str) -> List[str]:
        z, y, x = center

        # 计算实际角度
        degree = self._index_to_degree(angle_idx)

        # 1. 老版本命名 (Legacy): 使用实际角度 (Degree)
        # 格式：..._mip_axis_0_angle_135.json
        legacy_name = f"{self.sample_id}_{x}_{y}_{z}_mip_axis_0_angle_{degree}{extension}"

        # 2. 新版本命名 (Standard): 建议使用索引，或者保持清晰的命名
        # 这里演示使用更短的索引命名，方便管理
        standard_name = f"{self.sample_id}_{x}_{y}_{z}_mip_angle{angle_idx}{extension}"

        return [legacy_name, standard_name]

    def get_target_path(self, center: Tuple[int, int, int], angle_idx: int, file_type: str = "json") -> Path:
        ext = f".{file_type}"
        candidates = self._get_filename_candidates(center, angle_idx, ext)

        # 读取优先：尝试寻找老格式文件
        for fname in candidates:
            p = self.proj_dir / fname
            if p.exists():
                return p

        # 都不存在（写入模式）：使用新标准
        return self.proj_dir / candidates[-1]

    # def get_mip_path(self, center: Tuple[int, int, int], angle_idx: int) -> Path:
    #     """获取特定角度的 MIP 图像路径"""
    #     z, y, x = center
    #     filename = f"{self.sample_id}_{x}_{y}_{z}_mip_angle{angle_idx}.tif"
    #     return self.proj_dir / filename
    #
    # def get_json_path(self, center: Tuple[int, int, int], angle_idx: int) -> Path:
    #     """获取对应的 LabelMe JSON 路径"""
    #     # 对应 MIP 文件的 .json
    #     mip_path = self.get_mip_path(center, angle_idx)
    #     return mip_path.with_suffix(".json")
    #
    # def get_final_mask_path(self) -> Path:
    #     """获取最终合并后的全图 Mask 路径"""
    #     return self.mask_dir / f"{self.sample_id}_mask.tif"
    #
    # def get_qc_image_path(self) -> Path:
    #     """获取质检对比图路径"""
    #     return self.qc_dir / f"{self.sample_id}_qc.png"

    def get_mip_path(self, center, angle_idx):
        return self.get_target_path(center, angle_idx, "tif")

    def get_json_path(self, center, angle_idx):
        return self.get_target_path(center, angle_idx, "json")

    def get_final_mask_path(self) -> Path:
        return self.mask_dir / f"{self.sample_id}_mask.tif"

    def get_qc_image_path(self) -> Path:
        return self.qc_dir / f"{self.sample_id}_qc.png"

# ==========================================
# 3. 业务处理器 (Processor)
# ==========================================

class SomaAnnotationProcessor:
    def __init__(self, config: ProcessingConfig):
        self.cfg = config
        self.volume_shape = None
        self.resolutions = (1.0, 1.0) # (xy, z)

    def run(self):
        # 1. 解析元数据
        sample_id, self.resolutions, soma_locations = self._parse_metadata()

        # 2. 初始化路径管理器
        paths = PathManager(self.cfg, sample_id)

        # 3. 读取数据 (Lazy loading or full load)
        print(f"[{self.cfg.mode.name}] Loading volume: {self.cfg.source_path.name}...")
        volume = tifffile.imread(self.cfg.source_path)
        self.volume_shape = volume.shape

        # 4. 准备全图 Mask (仅在重建模式下)
        full_mask_accumulator = None
        if self.cfg.mode == PipelineMode.RECONSTRUCT_3D:
            full_mask_accumulator = np.zeros(self.volume_shape, dtype=np.uint8)

        # 5. 遍历每个胞体坐标
        # 将 DataFrame 转为 list dict 方便处理
        soma_iter = tqdm(soma_locations.iterrows(), total=len(soma_locations), desc=f"Processing {sample_id}")

        for _, marker in soma_iter:
            # 坐标转换与裁剪
            center_px = (int(marker['z']), int(marker['y']), int(marker['x'])) # ZYX
            crop_data = self._crop_and_rescale(volume, center_px)

            if crop_data is None:
                continue # 越界跳过

            block_1um, original_shape, crop_bounds = crop_data

            # 分发任务
            if self.cfg.mode == PipelineMode.PREPARE_LABELS:
                self._generate_projection_files(block_1um, center_px, paths, rotate_times=self.cfg.rotate_times)

            elif self.cfg.mode == PipelineMode.RECONSTRUCT_3D:
                local_mask = self._reconstruct_local_mask(center_px, block_1um.shape, original_shape, paths)

                if local_mask is not None:
                    # 将局部 Mask 累加到全图 Mask
                    z_s, z_e, y_s, y_e, x_s, x_e = crop_bounds
                    # 使用 maximum 避免重叠区域数值错误
                    current_roi = full_mask_accumulator[z_s:z_e, y_s:y_e, x_s:x_e]
                    full_mask_accumulator[z_s:z_e, y_s:y_e, x_s:x_e] = np.maximum(current_roi, local_mask)

        # 6. 收尾：保存结果
        if self.cfg.mode == PipelineMode.RECONSTRUCT_3D and full_mask_accumulator is not None and np.sum(full_mask_accumulator) > 0:
            self._save_results(full_mask_accumulator, volume, paths)

    # -------------------------------------------------
    # 核心逻辑方法
    # -------------------------------------------------

    def _generate_projection_files(self, block_1um: np.ndarray, center: Tuple[int,int,int], paths: PathManager,
                                   rotate_times: int = 12, axes_rot: Tuple[int,int]=(1,2), mip_axis: int =1):
        """生成多角度 MIP 并保存到 projections 目录"""
        # 检查是否已存在（避免重复计算）
        first_mip = paths.get_mip_path(center, 0)
        if first_mip.exists():
            return

        # 旋转并投影 (External Function logic)
        # rotated_blocks = [rotate(block_1um, angle) for angle in ...]
        # mips = [np.max(blk, axis=0) for blk in rotated_blocks]
        # 这里模拟获取了 MIP 列表
        mips_list = rotate_img_to_mip(block_1um, rotate_times=rotate_times, axes_rot=axes_rot, mip_axis=mip_axis) # 假设返回 list of 2d arrays

        for i, mip in enumerate(mips_list):
            if i >= self.cfg.rotate_times: break

            save_path = paths.get_mip_path(center, i)

            # 归一化并保存
            mip_norm = self._normalize_to_uint8(mip)
            tifffile.imwrite(save_path, mip_norm)

    # def _reconstruct_local_mask(self, center: Tuple[int,int,int], shape_1um, shape_orig, paths: PathManager) -> Optional[np.ndarray]:
    #     """读取 JSON -> 生成 2D Mask -> 重建 3D Mask -> Resize 回原分辨率"""
    #
    #     # 1. 收集所有角度的 JSON 路径
    #     json_paths = [paths.get_json_path(center, i) for i in range(self.cfg.rotate_times)]
    #
    #     # 检查文件完整性
    #     if not all(p.exists() for p in json_paths):
    #         print(f"Warning: Incomplete annotations for soma at {center}")
    #
    #         return None
    #
    #     # 2. 解析 JSON 生成 2D Masks
    #     masks_2d = []
    #     for json_p in json_paths:
    #         # 外部函数：读取 labelme json 转为 binary mask
    #         mask = generate_2d_mask_from_polygonal(str(json_p))
    #         print(np.sum(mask > 0), "pixels labeled in 2D mask.")
    #         masks_2d.append(mask)
    #
    #     # 3. 视觉壳重建 (Visual Hull) - 得到 1um 分辨率的 3D Mask
    #     mask_3d_1um = generate_3d_mask_from_2d_mip_mask(
    #         masks_2d, shape_1um, rotate_times=self.cfg.rotate_times
    #     )
    #
    #     # 4. Resize 回原始数据的像素分辨率
    #     # 注意：Mask 的 resize 必须用 order=0 (最近邻插值)
    #     mask_3d_orig = resize(mask_3d_1um, shape_orig, order=0, mode='reflect', anti_aliasing=False)
    #
    #     return mask_3d_orig.astype(np.uint8)
    def _reconstruct_local_mask(self, center: Tuple[int,int,int], shape_1um, shape_orig, paths: PathManager) -> Optional[np.ndarray]:
        """
        核心重建逻辑：利用 PathManager 自动兼容老版本 JSON
        """

        # 1. 获取 JSON 路径 (PathManager 会自动处理是 'angle_0' 还是 'angle0')
        json_paths = []
        for i in range(self.cfg.rotate_times):
            p = paths.get_json_path(center, i)
            json_paths.append(p)

        # 2. 检查完整性
        # 如果文件不存在，get_json_path 会返回新版路径，这里检测其是否存在
        missing_files = [p.name for p in json_paths if not p.exists()]
        if missing_files:
            # 可以选择打印日志调试
            # print(f"Skipping soma at {center}: Missing {len(missing_files)} annotations.")
            return None

        # 3. 解析 JSON -> 2D Mask
        masks_2d = []
        try:
            for json_p in json_paths:
                # 调用外部函数解析
                mask = generate_2d_mask_from_polygonal(str(json_p))
                masks_2d.append(mask)
        except Exception as e:
            print(f"Error parsing json for {center}: {e}")
            return None

        # 4. 3D 重建 (Visual Hull)
        try:
            mask_3d_1um = generate_3d_mask_from_2d_mip_mask(
                masks_2d, shape_1um, rotate_times=self.cfg.rotate_times
            )
        except Exception as e:
            print(f"Reconstruction failed for {center}: {e}")
            return None

        # 5. Resize 回原图分辨率
        mask_3d_orig = resize(mask_3d_1um, shape_orig, order=0, mode='reflect', anti_aliasing=False)

        return mask_3d_orig.astype(np.uint8)

    # def _save_results(self, full_mask: np.ndarray, full_volume: np.ndarray, paths: PathManager):
    #     """保存最终 Mask 和 QC 图"""
    #
    #     # 1. 二值化并保存全图 Mask
    #     final_mask = np.where(full_mask > 0, 255, 0).astype(np.uint8)
    #     # print(np.sum(final_mask > 0), "voxels labeled as soma.")
    #     print(f"Saving final mask to: {paths.get_final_mask_path()}")
    #     tifffile.imwrite(paths.get_final_mask_path(), final_mask, compression=5)
    #
    #     # 2. 生成 QC 对比图 (MIP)
    #     print("Generating QC visualization...")
    #     # 简单做 Z 轴投影
    #     vol_mip = np.max(full_volume, axis=0)
    #     mask_mip = np.max(final_mask, axis=0)
    #
    #     qc_img = np.concatenate([
    #         self._normalize_to_uint8(vol_mip),
    #         self._normalize_to_uint8(mask_mip)
    #     ], axis=1)
    #
    #     tifffile.imwrite(paths.get_qc_image_path(), qc_img)

    def _save_results(self, full_mask: np.ndarray, full_volume: np.ndarray, paths: PathManager):
        """
        保存最终 Mask 并生成三视图 QC 对比图 (XY, XZ, YZ)
        """
        # 1. 二值化并保存 3D Mask (保持 uint8 0-255)
        final_mask = np.where(full_mask > 0, 255, 0).astype(np.uint8)
        print(f"Saving final mask to: {paths.get_final_mask_path()}")
        tifffile.imwrite(paths.get_final_mask_path(), final_mask, compression=5)

        # 2. 生成三视图可视化 QC 图
        print("Generating 3D Orthogonal QC visualization...")
        # qc_image = self._create_ortho_overlay(full_volume, final_mask)
        qc_image = IMG_SEG_QCGenerator.generate(full_volume, final_mask, style=self.cfg.qc_style)

        # 保存为 PNG
        tifffile.imwrite(paths.get_qc_image_path(), qc_image)

    # -------------------------------------------------
    # 辅助工具方法
    # -------------------------------------------------

    def _crop_and_rescale(self, volume, center_px):
        """裁剪并归一化分辨率"""
        cz, cy, cx = center_px

        # 根据物理尺寸计算像素裁剪框
        # 注意：这里需要外部的 xy_res, z_res
        xy_res, z_res = self.resolutions

        # 计算 Block 的像素大小
        bz_px = int(self.cfg.crop_size_um[0] / (z_res / 1000))
        by_px = int(self.cfg.crop_size_um[1] / (xy_res / 1000))
        bx_px = int(self.cfg.crop_size_um[2] / (xy_res / 1000))

        # 计算边界 (使用 helper 或手动)
        z_s, z_e = self._get_bounds(cz, bz_px, volume.shape[0])
        y_s, y_e = self._get_bounds(cy, by_px, volume.shape[1])
        x_s, x_e = self._get_bounds(cx, bx_px, volume.shape[2])

        # 越界检查
        if (z_e - z_s) <= 0 or (y_e - y_s) <= 0 or (x_e - x_s) <= 0:
            return None

        crop = volume[z_s:z_e, y_s:y_e, x_s:x_e]

        # 计算 1um 目标尺寸
        target_shape = (
            int(crop.shape[0] * z_res / 1000),
            int(crop.shape[1] * xy_res / 1000),
            int(crop.shape[2] * xy_res / 1000)
        )

        # Resize
        rescaled = resize(crop, target_shape, order=0, mode='reflect', anti_aliasing=True)

        return rescaled, crop.shape, (z_s, z_e, y_s, y_e, x_s, x_e)

    def _parse_metadata(self):
        """解析 ID 和分辨率 (此处封装原本混乱的 if/else)"""
        fname = self.cfg.source_path.name

        if self.cfg.data_type == DataSourceType.SINGLE_NEURON:
            # 逻辑：从文件名提取 ID
            neuron_id = int(fname.split('.')[0].split("_")[0])
            xy_res, z_res = get_xy_z_resolution(neuron_id) # External

            markers = get_mapped_somas_in_neuron_block(neuron_id) # External
            # 如果没有 marker，造一个 dummy 的
            if markers is None:
                pos = get_soma_pos(neuron_id) # External
                markers = pd.DataFrame([{'x': pos[0], 'y': pos[1], 'z': pos[2]}])

            return str(neuron_id), (xy_res, z_res), markers

        elif self.cfg.data_type == DataSourceType.TILE_SCAN:
            flag = fname.replace(".tif", "").replace("_0000", "")
            xy_res, z_res = get_tile_resolution(flag) # External
            markers = get_somas_in_tile(flag) # External
            return flag, (xy_res, z_res), markers

        else:
            raise ValueError("Unknown Data Type")

    @staticmethod
    def _get_bounds(center, size, limit):
        start = max(0, center - size // 2)
        end = min(limit, center + size // 2)
        return int(start), int(end)

    @staticmethod
    def _normalize_to_uint8(arr):
        v_min, v_max = arr.min(), arr.max()
        if v_max == v_min: return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - v_min) / (v_max - v_min) * 255).astype(np.uint8)

if __name__ == "__main__":
    # 示例配置
    in_dir = "/data2/kfchen/tracing_ws/soma_seg/tif_image"
    files = os.listdir(in_dir)
    for file in files:
        idx = file.split(".")[0]
        if(not idx.isdigit() or int(idx) >= 15000):
            continue

        config = ProcessingConfig(source_volume_path=os.path.join(in_dir, file),
                                  workspace_dir="/data2/kfchen/tracing_ws/soma_seg",
                                  # mode=PipelineMode.PREPARE_LABELS,
                                  mode=PipelineMode.RECONSTRUCT_3D,
                                  data_type=DataSourceType.SINGLE_NEURON,
                                  crop_size_um=(50, 50, 50), rotate_times=12)

        processor = SomaAnnotationProcessor(config)
        try:
            processor.run()
        except Exception as e:
            print(f"Processing failed for {file}: {e}")