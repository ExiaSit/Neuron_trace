# Functions for reading and writing image files (TIFF, PNG, etc.)
import tifffile
import numpy as np
from v3dpy.loaders import Raw, PBD
from scipy.ndimage import rotate

def load_image(filename, normalize=True, out_dtype="uint8"):
    """
    Load an image from a file.

    Args:
        filename (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image as a NumPy array.
    """
    # Use tifffile to read the image
    if( filename.endswith('.tif') or filename.endswith('.tiff')):
        image = tifffile.imread(filename)
    elif(filename.endswith('.v3dpbd')):
        pbd = PBD()
        image = pbd.load(filename)
        tifffile.imwrite('/data/disk/C6.0/app_test/mip.tif', image)  # Save MIP for quick check
        if image.ndim == 4:
            image = np.sum(image.astype(np.float32), axis=0)
    elif(filename.endswith('.v3draw')):
        raw = Raw()
        image = raw.load(filename)[0]

    else:
        raise ValueError("Unsupported file format. Supported formats are .tif, .tiff, .v3dpbd, and .v3draw.")

    # Normalize the image if required
    if normalize:
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = (image * 255).astype(out_dtype)

    return image

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