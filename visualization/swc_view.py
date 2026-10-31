"""SWC-centered display: XY bounding box + 15 um, full original Z.

The reconstruction pipeline uses 1 um image coordinates. Keep image extents in
those original coordinates: this is a display crop, never an SWC transform.
"""
import numpy as np


class SwcMipView:
    def __init__(self, shape, points, margin_um=15., spacing_xy=(1., 1.)):
        self.shape = tuple(shape) if len(shape) == 3 else (1, *shape)
        z, h, w = self.shape
        points = np.asarray(list(points), dtype=float)
        self.x0, self.x1, self.y0, self.y1 = 0, w, 0, h
        if points.size:
            xy = points.reshape(-1, points.shape[-1])[:, :2]
            xy = xy[np.isfinite(xy).all(axis=1)]
            spacing = np.asarray(spacing_xy, float)
            if spacing.shape != (2,) or np.any(spacing <= 0):
                raise ValueError('XY spacing must be two positive um/voxel values')
            if len(xy):
                lo = np.maximum(0, np.floor(xy.min(0)-margin_um/spacing)).astype(int)
                hi = np.minimum([w, h], np.ceil(xy.max(0)+margin_um/spacing)+1).astype(int)
                if np.all(hi > lo):
                    self.x0, self.y0 = map(int, lo)
                    self.x1, self.y1 = map(int, hi)

    @property
    def slices(self):
        return (slice(None), slice(self.y0, self.y1), slice(self.x0, self.x1))

    def extent(self, axis=0):
        x, y, z = (self.x0-.5, self.x1-.5), (self.y0-.5, self.y1-.5), (-.5, self.shape[0]-.5)
        horizontal, vertical = [(x, y), (x, z), (y, z)][axis]
        return (*horizontal, vertical[1], vertical[0])

    def project(self, volume, axis=0):
        volume = np.asarray(volume)
        crop = volume[self.slices] if volume.ndim == 3 else volume[self.slices[1:]]
        return crop.max(axis=axis) if volume.ndim == 3 else crop

    def apply(self, ax, axis=0):
        left, right, bottom, top = self.extent(axis)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)


def crop_xy_axes(ax, shape_yx, points):
    """Crop already rendered XY backgrounds/overlays without moving SWC lines."""
    view = SwcMipView(shape_yx, points)
    for artist in ax.images:
        data = artist.get_array()
        if tuple(data.shape[:2]) != tuple(shape_yx):
            continue
        artist.set_data(data[view.y0:view.y1, view.x0:view.x1])
        extent = view.extent()
        if artist.origin == 'lower':
            extent = (extent[0], extent[1], extent[3], extent[2])
        artist.set_extent(extent)
    view.apply(ax)
    if ax.images and ax.images[0].origin == 'lower':
        ax.set_ylim(view.y0-.5, view.y1-.5)
    return view
