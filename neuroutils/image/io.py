# Functions for reading and writing image files (TIFF, PNG, etc.)
import tifffile
import numpy as np
from v3dpy.loaders import Raw, PBD
from scipy.ndimage import rotate

def load_image(filename, normalize=True, out_dtype="uint8"):
    """
    Load an image from a file.

    Parameters
    ----------
    filename : str
        Path to .tif / .tiff / .v3dpbd / .v3draw image.
    normalize : bool
        Whether to normalize each image to uint8 [0, 255].
    out_dtype : str or np.dtype
        Output dtype when normalize=True.

    Notes
    -----
    Most tested PBD images are returned by PBD.load() as float32 arrays,
    but their actual voxel values are exact integers in [0, 255].

    For those standard images:
        float32 values 0~255 -> uint8
    is exactly equivalent to the previous min-max normalization, while
    avoiding expensive full-volume arithmetic.

    For unusual ranges such as cell 46646:
        min=2, max=360
    this function falls back to per-image min-max normalization.
    """

    if filename.endswith((".tif", ".tiff")):
        image = tifffile.imread(filename)

    elif filename.endswith(".v3dpbd"):
        pbd = PBD()
        image = pbd.load(filename)

        # Preserve your original "sum channels" behavior for 4D data.
        # dtype=np.float32 avoids an unnecessary full float32 copy before sum.
        if image.ndim == 4:
        # 常见情况：shape=(1, Z, Y, X)，直接去掉 channel 维，
        # 不做 sum，也不把 uint8 扩成 float32。
            if image.shape[0] == 1:
                image = image[0]
            else:
                # 真正多通道时，才按原逻辑求和。
                image = np.sum(image, axis=0, dtype=np.float32)

    elif filename.endswith(".v3draw"):
        raw = Raw()
        image = raw.load(filename)[0]

    else:
        raise ValueError(
            "Unsupported file format. Supported formats are "
            ".tif, .tiff, .v3dpbd, and .v3draw."
        )

    if not normalize:
        return image

    image = np.asarray(image)
    out_dtype = np.dtype(out_dtype)

    img_min = float(np.min(image))
    img_max = float(np.max(image))

    if not np.isfinite(img_min) or not np.isfinite(img_max):
        raise ValueError(
            f"Invalid image range: min={img_min}, max={img_max}"
        )

    # Avoid division by zero for constant images.
    if img_max <= img_min:
        return np.zeros(image.shape, dtype=out_dtype)

    # ------------------------------------------------------------
    # Fast path:
    #
    # Standard PBD images are float32 in storage but contain exact
    # integer values from 0 to 255. For these images, directly converting
    # to uint8 is exactly the same as the previous min-max normalization.
    # ------------------------------------------------------------
    if (
        out_dtype == np.dtype(np.uint8)
        and img_min == 0.0
        and img_max == 255.0
    ):
        return image.astype(np.uint8, copy=False)


    # ------------------------------------------------------------
    # Fallback path:
    #
    # Preserve old semantics for images whose range is not exactly 0~255,
    # e.g. 46646: 2 -> 0, 360 -> 255.
    #
    # In-place operations reduce large temporary float32 arrays.
    # ------------------------------------------------------------
    image = image.astype(np.float32, copy=False)

    image -= img_min
    image /= (img_max - img_min)
    image *= 255.0

    np.clip(image, 0.0, 255.0, out=image)

    # Keep the old behavior: astype(uint8) truncates fractional values.
    return image.astype(out_dtype, copy=False)

def save_image(image, filename, normalize=True, out_dtype="uint8"):
    """
    Save an image to a file.

    Args:
        image (np.ndarray): Image to save.
        filename (str): Path to the output file.
    """
    # Normalize the image if required
    if normalize:
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = (image * 255).astype(out_dtype)

    # Use tifffile to write the image
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        tifffile.imwrite(filename, image)
    elif filename.endswith('.v3dpbd'):
        pbd = PBD()
        pbd.save(filename, image)
    elif filename.endswith('.v3draw'):
        raw = Raw()
        raw.save(filename, image)
    else:
        raise ValueError("Unsupported file format. Supported formats are .tif, .tiff, .v3dpbd, and .v3draw.")

def rotate_img_to_mip(image, rotate_times=12, axes_rot=(1, 2), mip_axis=1):
    rotate_step = int(180 / rotate_times)
    mip_list = []
    for i in range(rotate_times):
        angle = i * rotate_step
        rotated_image = rotate(image, angle, axes=axes_rot, reshape=False)
        mip = np.max(rotated_image, axis=mip_axis)
        # print(mip.shape)
        mip_list.append(mip)

    return mip_list