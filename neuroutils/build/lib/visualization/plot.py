import numpy as np
import cv2
import matplotlib.pyplot as plt
from neuroutils.swc.io import load_swc
from neuroutils.marker.io import load_marker
from neuroutils.image.io import load_image
from neuroutils.meta.mapping import extract_neuron_id
from neuroutils.meta.neuron import get_neuron_meta
from neuroutils.image.preprocessor import rescale_image
from neuroutils.config.settings import DEFAULT_RESOLUTION_UNIT
from neuroutils.swc.parser import rescale_swc
from enum import Enum
from typing import List, Optional

# dpi = 300
plt.rcParams['figure.dpi'] = 300

def encoding_projection_direction(projection_direction):
    if projection_direction == 'xy':
        projection_axes = 0
    elif projection_direction == 'xz':
        projection_axes = 1
    elif projection_direction == 'yz':
        projection_axes = 2
    return projection_axes

def get_background_img(img_shape, projection_direction='xy'):
    projection_axes = encoding_projection_direction(projection_direction)
    if projection_axes == 0:
        background_img = np.ones((img_shape[1], img_shape[2])).astype(np.uint8) * 255
    elif projection_axes == 1:
        background_img = np.ones((img_shape[0], img_shape[2])).astype(np.uint8) * 255
    elif projection_axes == 2:
        background_img = np.ones((img_shape[0], img_shape[1])).astype(np.uint8) * 255

    background_img = cv2.cvtColor(background_img, cv2.COLOR_GRAY2RGB)
    return background_img

def plot_img_on_fig(fig, gray_img, projection_direction='xy', alpha=0):
    projection_axes = encoding_projection_direction(projection_direction)
    gray_img = np.max(gray_img, axis=projection_axes)
    # resize
    gray_img = cv2.resize(gray_img, (fig.shape[1], fig.shape[0]))
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # print(gray_img.shape, fig.shape)

    fig = cv2.addWeighted(fig, alpha, gray_img, 1 - alpha, 0)
    return fig

def plot_swc_on_fig(fig, swc_points, plot_mode="sphere", projection_direction='xy', line_color=(255, 0, 0), line_thickness=1, soma_color=(0, 0, 255), soma_thickness=3):
    projection_axes = encoding_projection_direction(projection_direction)

    for swc_point_id in range(len(swc_points)):
        if(plot_mode == 'sphere'):
            if (projection_axes == 0):
                # cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].y)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
                cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].y)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
            elif (projection_axes == 1):
                cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].z)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
            elif (projection_axes == 2):
                cv2.circle(fig, (int(swc_points.iloc[swc_point_id].y), int(swc_points.iloc[swc_point_id].z)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
        # elif(plot_mode=="line"):
        #     if (projection_axes == 0):
        #         cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].y)), 1, line_color, -1)
        #     elif (projection_axes == 1):
        #         cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].z)), 1, line_color, -1)
        #     elif (projection_axes == 2):
        #         cv2.circle(fig, (int(swc_points.iloc[swc_point_id].y), int(swc_points.iloc[swc_point_id].z)), 1, line_color, -1)

        if swc_points.iloc[swc_point_id].parent == -1:
            continue
        swc_point = swc_points.iloc[swc_point_id]
        parent_point = swc_points[swc_points['n'] == swc_point['parent']].iloc[0]
        # print(swc_point, parent_point)

        nx, ny, nz = swc_point.x, swc_point.y, swc_point.z
        px, py, pz = parent_point.x, parent_point.y, parent_point.z

        if (projection_axes == 0):
            cv2.line(fig, (int(nx), int(ny)), (int(px), int(py),), line_color, line_thickness)
        elif (projection_axes == 1):
            cv2.line(fig, (int(nx), int(nz)), (int(px), int(pz),), line_color, line_thickness)
        elif (projection_axes == 2):
            cv2.line(fig, (int(ny), int(nz)), (int(py), int(pz),), line_color, line_thickness)

    if (projection_axes == 0):
        cv2.circle(fig, (int(swc_points.iloc[0].x), int(swc_points.iloc[0].y)), soma_thickness, soma_color, -1)
    elif (projection_axes == 1):
        cv2.circle(fig, (int(swc_points.iloc[0].x), int(swc_points.iloc[0].z)), soma_thickness, soma_color, -1)
    elif (projection_axes == 2):
        cv2.circle(fig, (int(swc_points.iloc[0].y), int(swc_points.iloc[0].z)), soma_thickness, soma_color, -1)

    return fig

def plot_markers_on_fig(fig, markers, projection_direction='xy', markers_color=(0, 255, 0), thickness=3, marker_type='rectangle'):
    projection_axes = encoding_projection_direction(projection_direction)

    for marker_id in range(len(markers)):
        marker = markers.iloc[marker_id]
        if marker_type == 'circle':
            if (projection_axes == 0):
                cv2.circle(fig, (int(marker.x), int(marker.y)), int(marker.radius), (marker.color_b, marker.color_g, marker.color_r), thickness)
            elif (projection_axes == 1):
                cv2.circle(fig, (int(marker.x), int(marker.z)), int(marker.radius), (marker.color_b, marker.color_g, marker.color_r), thickness)
            elif (projection_axes == 2):
                cv2.circle(fig, (int(marker.y), int(marker.z)), int(marker.radius), (marker.color_b, marker.color_g, marker.color_r), thickness)
        elif marker_type == 'rectangle':
            if (projection_axes == 0):
                cv2.rectangle(fig, (int(marker.x - marker.radius), int(marker.y - marker.radius)), (int(marker.x + marker.radius), int(marker.y + marker.radius)), markers_color, thickness)
            elif (projection_axes == 1):
                cv2.rectangle(fig, (int(marker.x - marker.radius), int(marker.z - marker.radius)), (int(marker.x + marker.radius), int(marker.z + marker.radius)), markers_color, thickness)
            elif (projection_axes == 2):
                cv2.rectangle(fig, (int(marker.y - marker.radius), int(marker.z - marker.radius)), (int(marker.y + marker.radius), int(marker.z + marker.radius)), markers_color, thickness)

    return fig

def plot_img_and_swc(img_file=None, img=None, swc_file=None, swc_points=None, save_file=None,
                     projection_direction='xy',
                     rescale_img_flag=False, rescale_swc_flag=False,
                     flip_y=False, down_sample_flag=False,
                     return_result=False
                     ):
    assert (img_file is not None or img is not None), "Either img_file or img must be provided."
    assert (swc_file is not None or swc_points is not None), "Either swc_file or swc_points must be provided."
    if img is None:
        img = load_image(img_file)
    if swc_points is None:
        swc_points = load_swc(swc_file)

    if(rescale_img_flag or rescale_swc_flag):
        neuron_id = extract_neuron_id(swc_file)
        meta = get_neuron_meta(neuron_id)
        xy_resolution = float(meta['xy_resolution'].values[0]) / DEFAULT_RESOLUTION_UNIT
        z_resolution = float(meta['z_resolution'].values[0]) / DEFAULT_RESOLUTION_UNIT
        if rescale_img_flag:
            img = rescale_image(img, xy_resolution=xy_resolution, z_resolution=z_resolution)
        if rescale_swc_flag:
            swc_points = rescale_swc(swc_points, xy_resolution=xy_resolution, z_resolution=z_resolution)
    if(flip_y):
        img = np.flip(img, axis=1)
    if(down_sample_flag):
        img = img[::2, ::2, ::2]

    background = get_background_img(img.shape, projection_direction=projection_direction)
    background = plot_img_on_fig(background, img)
    background = plot_swc_on_fig(background, swc_points, line_color=(255, 0, 0), plot_mode='line')

    if(save_file is not None):
        plt.imshow(background)
        plt.savefig(save_file)
        plt.close()
    if(return_result):
        return background

def test(img_file, swc_file, marker_file, save_file, projection_direction='xy'):
    img = load_image(img_file)
    swc_points = load_swc(swc_file)
    markers = load_marker(marker_file)

    background = get_background_img(img.shape, projection_direction=projection_direction)

    background = plot_img_on_fig(background, img)
    background = plot_swc_on_fig(background, swc_points, line_color=(255, 0, 0))
    background = plot_markers_on_fig(background, markers)

    plt.imshow(background)
    plt.savefig(save_file)
    plt.close()


class IMG_SEG_QCVisualStyle(Enum):
    OVERLAY = "overlay"  # 风格一：红绿叠加
    GRID    = "grid"     # 风格二：网格对比

class IMG_SEG_QCGenerator:
    """
    专门负责生成质检(QC)图像的静态工具类。
    将 3D Volume 和 3D Mask 转换为可视化的 2D 图片。
    """

    @staticmethod
    def generate(volume: np.ndarray, mask: np.ndarray, style: IMG_SEG_QCVisualStyle) -> np.ndarray:
        """统一入口函数"""
        if style == IMG_SEG_QCVisualStyle.OVERLAY:
            return IMG_SEG_QCGenerator._create_ortho_overlay(volume, mask)
        elif style == IMG_SEG_QCVisualStyle.GRID:
            return IMG_SEG_QCGenerator._create_ortho_grid(volume, mask)
        else:
            raise ValueError(f"Unsupported QC Style: {style}")

    # =========================================================
    #  风格实现 (Internal Implementation)
    # =========================================================

    @staticmethod
    def _create_ortho_overlay(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """生成三视图叠加：Mask显示为红色，原图为灰色背景"""
        views = []
        # Z, Y, X axis
        for axis in [0, 1, 2]:
            vol_mip = np.max(volume, axis=axis)
            mask_mip = np.max(mask, axis=axis)

            vol_norm = IMG_SEG_QCGenerator._normalize_to_uint8(vol_mip)

            # 创建 RGB 底图 (灰度)
            img_rgb = np.stack([vol_norm, vol_norm, vol_norm], axis=-1)

            # 叠加红色 Mask
            # 逻辑：Mask区域 R通道拉满，G/B通道变暗
            is_mask = mask_mip > 0
            img_rgb[is_mask, 0] = 255
            img_rgb[is_mask, 1] = np.clip(img_rgb[is_mask, 1] * 0.5, 0, 255)
            img_rgb[is_mask, 2] = np.clip(img_rgb[is_mask, 2] * 0.5, 0, 255)

            views.append(img_rgb)

        return IMG_SEG_QCGenerator._stack_images_horizontally(views)

    @staticmethod
    def _create_ortho_grid(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """生成 3x2 网格图 (Raw vs Mask)"""
        rows = []
        padding = 10

        for axis in [0, 1, 2]:
            vol_mip = np.max(volume, axis=axis)
            mask_mip = np.max(mask, axis=axis)

            # 归一化并转 RGB
            vol_rgb = np.stack([IMG_SEG_QCGenerator._normalize_to_uint8(vol_mip)] * 3, axis=-1)
            # Mask 转为白色 RGB
            mask_viz = np.where(mask_mip > 0, 255, 0).astype(np.uint8)
            mask_rgb = np.stack([mask_viz] * 3, axis=-1)

            # 拼接当前行: [原图] | [Mask]
            row_img = np.concatenate([vol_rgb, mask_rgb], axis=1)

            # 可选：中间加灰色分隔线
            sep_line = np.zeros((row_img.shape[0], padding, 3), dtype=np.uint8) + 128
            row_img_sep = np.concatenate([vol_rgb, sep_line, mask_rgb], axis=1)

            rows.append(row_img_sep)

        return IMG_SEG_QCGenerator._stack_images_vertically(rows)

    # =========================================================
    #  图像处理工具 (Helpers)
    # =========================================================

    @staticmethod
    def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        """归一化工具"""
        v_min, v_max = arr.min(), arr.max()
        if v_max == v_min:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - v_min) / (v_max - v_min) * 255).astype(np.uint8)

    @staticmethod
    def _stack_images_horizontally(images: List[np.ndarray], padding: int = 10) -> Optional[np.ndarray]:
        """水平拼接，高度居中"""
        if not images: return None
        max_h = max(img.shape[0] for img in images)
        total_w = sum(img.shape[1] for img in images) + padding * (len(images) - 1)

        canvas = np.zeros((max_h, total_w, 3), dtype=np.uint8)
        current_x = 0
        for img in images:
            h, w, _ = img.shape
            y_off = (max_h - h) // 2
            canvas[y_off:y_off+h, current_x:current_x+w, :] = img
            current_x += w + padding
        return canvas

    @staticmethod
    def _stack_images_vertically(images: List[np.ndarray], padding: int = 10) -> Optional[np.ndarray]:
        """垂直拼接，宽度居中"""
        if not images: return None
        max_w = max(img.shape[1] for img in images)
        total_h = sum(img.shape[0] for img in images) + padding * (len(images) - 1)

        canvas = np.zeros((total_h, max_w, 3), dtype=np.uint8)
        current_y = 0
        for img in images:
            h, w, _ = img.shape
            x_off = (max_w - w) // 2
            canvas[current_y:current_y+h, x_off:x_off+w, :] = img
            current_y += h + padding
        return canvas