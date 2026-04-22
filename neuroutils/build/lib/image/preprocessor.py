# Preprocessing functions for cropping, resizing, and denoising images
import numpy as np
from skimage.transform import resize
from neuroutils.meta.mapping import extract_neuron_id
from neuroutils.meta.neuron import get_neuron_meta
from neuroutils.config.settings import TILE_SOMA_MAP_DIR, DEFAULT_RESOLUTION_UNIT
from neuroutils.image.io import load_image, save_image
from scipy.ndimage import rotate
import os
import json
import cv2

def rescale_image(image, xy_resolution, z_resolution, order=0):
    """
    Rescale the image to the specified xy and z resolution.

    Parameters:
    - image: The input image to be rescaled.
    - xy_resolution: The desired xy resolution.
    - z_resolution: The desired z resolution.

    Returns:
    - rescaled_image: The rescaled image.
    """
    # Placeholder for actual rescaling logic
    img_shape = image.shape
    # print("Original image shape:", img_shape)
    # print(z_resolution, xy_resolution)
    new_shape = (
        int(img_shape[0] * z_resolution),
        int(img_shape[1] * xy_resolution),
        int(img_shape[2] * xy_resolution)
    )
    rescaled_image = resize(image, new_shape, order, mode='reflect', anti_aliasing=True)
    # print("Rescaled image shape:", rescaled_image.shape)
    return rescaled_image

def down_sample(image, factor):
    """
    Downsample the image by a given factor.

    Parameters:
    - image: The input image to be downsampled.
    - factor: The downsampling factor.

    Returns:
    - downsampled_image: The downsampled image.
    """
    # Placeholder for actual downsampling logic
    downsampled_image = image[::factor, ::factor, ::factor]
    return downsampled_image

def rescale_image_file(in_file, out_file, order=0, flip=False):
    """
    Rescale an image file to the specified xy and z resolution and save it.

    Parameters:
    - in_file: Path to the input image file.
    - out_file: Path to save the rescaled image.
    - order: Interpolation order for resizing.
    - flip: Whether to flip the image along the z-axis.
    """
    neuron_id = extract_neuron_id(in_file)
    neuron_meta = get_neuron_meta(neuron_id)
    xy_res = float(neuron_meta["xy_resolution"].values[0]) / DEFAULT_RESOLUTION_UNIT
    z_res = float(neuron_meta["z_resolution"].values[0]) / DEFAULT_RESOLUTION_UNIT

    image = load_image(in_file)
    if flip:
        image = np.flip(image, axis=1)
    rescaled_image = rescale_image(image, xy_res, z_res, order)
    save_image(rescaled_image, out_file)

def pad_to_blocksize(image, blocksize):
    """
    Pad the input image to ensure its dimensions are at least as large as the specified blocksize.
    :param image: Input 3D numpy array representing the image.
    :param blocksize: Tuple of three integers specifying the desired blocksize (depth, height, width).
    :return: Padded image as a 3D numpy array.
    """
    # 检查 blocksize 是否为三维
    if len(blocksize) != 3:
        raise ValueError("blocksize must be a tuple of three integers (depth, height, width).")

    # 获取输入图像的尺寸
    input_shape = image.shape

    # 计算每个维度需要填充的大小
    pad_depth = max(blocksize[0] - input_shape[0], 0)
    pad_height = max(blocksize[1] - input_shape[1], 0)
    pad_width = max(blocksize[2] - input_shape[2], 0)

    # 计算每个维度的前后填充量（保持居中）
    pad_depth_before = pad_depth // 2
    pad_depth_after = pad_depth - pad_depth_before

    pad_height_before = pad_height // 2
    pad_height_after = pad_height - pad_height_before

    pad_width_before = pad_width // 2
    pad_width_after = pad_width - pad_width_before

    # 使用 np.pad 进行填充
    padded_image = np.pad(
        image,
        pad_width=((pad_depth_before, pad_depth_after),
                   (pad_height_before, pad_height_after),
                   (pad_width_before, pad_width_after)),
        mode='constant',  # 填充值为常数（默认填充0）
        constant_values=0  # 填充值为0
    )

    return padded_image


def calculate_bounds_3d(img_shape, center_pos, block_size):
    def limit_pos_in_shape(x, max_x, min_x=0):
        return max(min_x, min(max_x, x))  # 合并 max 和 min 操作
    def calculate_bounds(center, block_size, img_shape, axis):
        """
        计算起始和结束位置，并限制在图像范围内。
        :param center: 中心点坐标（如 soma_z, soma_y, soma_x）
        :param block_size: 当前块的尺寸
        :param img_shape: 图像的形状
        :param axis: 图像的轴（0, 1, 2 分别对应 z, y, x）
        :return: 限制后的起始和结束位置
        """
        start = int(center - block_size / 2)
        end = int(center + block_size / 2)
        return limit_pos_in_shape(start, img_shape[axis]), limit_pos_in_shape(end, img_shape[axis])
    soma_z, soma_y, soma_x = center_pos
    z_start, z_end = calculate_bounds(soma_z, block_size[0], img_shape, axis=0)
    y_start, y_end = calculate_bounds(soma_y, block_size[1], img_shape, axis=1)
    x_start, x_end = calculate_bounds(soma_x, block_size[2], img_shape, axis=2)
    assert z_end >= z_start and y_end >= y_start and x_end >= x_start, (
        "End positions must be greater than or equal to start positions. z_start: {}, z_end: {}, "
        "y_start: {}, y_end: {}, x_start: {}, x_end: {}".format(z_start, z_end, y_start, y_end, x_start, x_end))
    return z_start, z_end, y_start, y_end, x_start, x_end


def generate_2d_mask_from_polygonal(labelme_mask_file):
    if(not os.path.exists(labelme_mask_file)):
        return None
    with open(labelme_mask_file, 'r') as f:
        label_info = json.load(f)
    imageHeight, imageWidth = label_info['imageHeight'], label_info['imageWidth']
    mask = np.zeros((imageHeight, imageWidth), dtype=np.uint8)
    # Polygonal to mask
    # mask_point_list = []
    # print(label_info['shapes'])
    for i in range(len(label_info['shapes'])):
        # mask_point_list.append(np.array(mask['shapes'][i]['points'], dtype=np.int32))
        mask_point = np.array(label_info['shapes'][i]['points'], dtype=np.int32)
        mask = cv2.fillPoly(mask, [mask_point], 1)
    return mask

def generate_3d_mask_from_2d_mip_mask(mip_list, img_shape, rotate_times=12, axes_rot=(1, 2), mip_axis=1):
    total_mask = np.ones((img_shape[0], img_shape[1], img_shape[2]), dtype=np.uint8)
    rotate_step = int(180 / rotate_times)
    for i in range(rotate_times):
        mip = mip_list[i]
        if(mip is None):
            continue
        # print(mip.shape, img_shape)
        mask = np.expand_dims(mip, axis=mip_axis)
        mask = np.repeat(mask, img_shape[mip_axis], axis=mip_axis)

        angle = i * rotate_step
        mask = rotate(mask, -angle, axes=axes_rot, reshape=False)
        mask = np.where(mask > 0, 1, 0)
        total_mask = total_mask * mask

    if(total_mask.sum() == total_mask.size):
        total_mask = np.zeros((img_shape[0], img_shape[1], img_shape[2]), dtype=np.uint8)
    return total_mask