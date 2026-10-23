# core/io/image_parser.py

"""
image_parser.py

Utilities for loading, inspecting, and writing microscopy image volumes
(e.g. TIFF stacks) for neuron tracing and reconstruction workflows.
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tifffile


class ImageParser:
    """
    Parses 3D microscopy images (multi-page TIFFs)
    and exposes image data + metadata in a consistent API.
    """
    # TODO: support other format
    def __init__(self, filepath: Union[str, os.PathLike]):
        """
        Parameters
        ----------
        filepath
            Path to a TIFF file or a folder containing a sequence of TIFFs.
            Or existed image array.
        """
        self.filepath = str(filepath)
        self._metadata: Dict = {}
        self._shape: Optional[Tuple[int, ...]] = None
        self._dtype: Optional[np.dtype] = None

        self._read_header()
        # print(self._dtype, self._shape)

    def _read_header(self) -> None:
        """
        Read image dimensions, dtype, and basic metadata from the first file.
        """
        with tifffile.TiffFile(self.filepath) as tif:
            series = tif.series[0]
            self._shape = series.shape
            self._dtype = series.dtype

        # TODO: Load metadata from json 
        self._metadata = {

        }

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the full 3D shape: (Z, Y, X)."""
        assert self._shape is not None
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of the image array."""
        assert self._dtype is not None
        return self._dtype

    @property
    def metadata(self) -> Dict:
        """metadata loaded from json file or extracted from the image."""
        return self._metadata

    def load(self) -> np.ndarray:
        """
        Load the entire image volume into memory.

        Returns
        -------
        img : np.ndarray 3D (Z,Y,X) array.
        """
        img = tifffile.imread(self.filepath)

        return img

    @staticmethod
    def save(img: np.ndarray, 
             out_path: Union[str, os.PathLike], 
             overwrite: bool = False
            ) -> None:
        """
        Save an image array.

        Parameters
        ----------
        img : np.ndarray
          3D array.
        out_path
        overwrite : bool
        """
        if os.path.exists(out_path) and not overwrite:
            raise OSError(f"File exists: {out_path}")
        tifffile.imwrite(out_path, img)


if __name__ == '__main__':
    parser = ImageParser(r'D:\science\pyNeuTube\15257_6631_12759_1147.tif')
    volume = parser.load()
