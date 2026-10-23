"""Validated 1 µm resampling and crop bounds, extracted from Neuron_trace."""
import numpy as np
from skimage.transform import resize

def validate_resolution(xy_resolution, z_resolution):
    """Validate and convert metadata resolutions to finite positive floats."""
    try:
        xy_resolution = float(xy_resolution)
        z_resolution = float(z_resolution)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "invalid resolution in metadata: "
            f"xy_resolution={xy_resolution!r}, "
            f"z_resolution={z_resolution!r}"
        ) from exc

    if not np.isfinite(xy_resolution) or not np.isfinite(z_resolution):
        raise ValueError(
            "non-finite resolution in metadata: "
            f"xy_resolution={xy_resolution!r}, "
            f"z_resolution={z_resolution!r}"
        )

    if xy_resolution <= 0 or z_resolution <= 0:
        raise ValueError(
            "non-positive resolution in metadata: "
            f"xy_resolution={xy_resolution!r}, "
            f"z_resolution={z_resolution!r}"
        )

    return xy_resolution, z_resolution


def get_target_shape(img_shape, xy_resolution, z_resolution):
    """
    Compute nearest integer image dimensions for 1 µm isotropic spacing.

    img_shape is assumed to be (Z, Y, X).
    Metadata unit is assumed to be 10^-3 µm / pixel, matching your
    original /1000 conversion.
    """
    scale = np.array(
        [z_resolution, xy_resolution, xy_resolution],
        dtype=np.float64,
    ) / 1000.0

    target_shape = np.rint(
        np.asarray(img_shape, dtype=np.float64) * scale
    ).astype(np.int64)

    target_shape = np.maximum(target_shape, 1)

    return tuple(int(x) for x in target_shape)


def to_uint8_after_resize(image):
    """
    Convert resized grayscale image to uint8 without another min-max stretch.

    Input image has already been normalized to approximately 0~255 by
    load_image(normalize=True). Resize may introduce float values such as
    43.17 through interpolation, so we:

      1. verify finite values;
      2. clip to [0, 255];
      3. round to nearest integer;
      4. cast to uint8.

    This intentionally does NOT re-normalize according to the resized image's
    new min/max. Therefore, resize-induced smoothing does not cause each image
    to be re-stretched to full contrast again.
    """
    image = np.asarray(image)

    img_min = float(np.min(image))
    img_max = float(np.max(image))

    if not np.isfinite(img_min) or not np.isfinite(img_max):
        raise ValueError(
            f"invalid resized image range: min={img_min}, max={img_max}"
        )

    # resize() should return a writable float array. This fallback keeps the
    # function robust if that assumption changes.
    if (
        not np.issubdtype(image.dtype, np.floating)
        or not image.flags.writeable
    ):
        image = image.astype(np.float32, copy=True)

    np.clip(image, 0.0, 255.0, out=image)

    # rint is preferred for newly generated data:
    # 127.6 -> 128, rather than astype(uint8) truncating it to 127.
    np.rint(image, out=image)

    return image.astype(np.uint8, copy=False)
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

def resample_1um(image, xy_resolution, z_resolution):
    xy_resolution,z_resolution=validate_resolution(xy_resolution,z_resolution)
    target=get_target_shape(image.shape,xy_resolution,z_resolution)
    resized=resize(image,target,order=1,mode='reflect',anti_aliasing=True,preserve_range=True)
    return to_uint8_after_resize(resized)
