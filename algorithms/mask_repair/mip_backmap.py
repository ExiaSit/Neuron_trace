"""Depth-preserving XY MIP seeds and locally supported secondary Z peaks."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils import bootstrap as _bootstrap

import numpy as np
from utils.gpu_ndimage import ndi

def legacy_backmap(raw, contrast):
    zmax=raw.argmax(axis=0)
    mip=raw.max(axis=0)
    primary=np.zeros(raw.shape,bool)
    yy,xx=np.indices(raw.shape[1:]);primary[zmax,yy,xx]=True
    # A maximum in every column exists even in noise; require positive local evidence.
    primary &= contrast>2
    smooth=ndi.gaussian_filter1d(raw.astype('float32'),.6,axis=0)
    peaks=(smooth==ndi.maximum_filter1d(smooth,5,axis=0))&(contrast>2)
    # Count different neighboring XY columns supporting this depth (+/- 2 slices).
    nearby=ndi.maximum_filter1d(peaks.astype('uint8'),5,axis=0)
    votes=ndi.convolve(nearby,np.ones((1,3,3),np.uint8),mode='constant')-nearby
    zz=np.arange(raw.shape[0])[:,None,None]
    secondary=peaks&(abs(zz-zmax[None])>=3)&(votes>=3)
    primary &= votes>=2
    seeds=primary|secondary
    seed_values=np.where(seeds,np.maximum(contrast,0),0).astype('float32')
    # Recover only a one-voxel neighborhood with measured weak raw contrast.
    # Never copy an XY maximum down the whole Z column.
    local_peak=ndi.maximum_filter(seed_values,size=3)
    near=ndi.maximum_filter(seeds.astype('uint8'),size=3)>0
    measured=(contrast>.75)
    recovered=near&measured&~seeds
    guide=contrast.copy()
    # Conservative bounded reinforcement, without replacing the raw image.
    boost=np.minimum(local_peak*.5,np.maximum(contrast,0)+3)
    guide[recovered]=np.maximum(guide[recovered],boost[recovered])
    classes=np.zeros(raw.shape,np.uint8);classes[recovered]=3;classes[secondary]=2;classes[primary]=1
    stats=dict(primary_peak_voxels=int(primary.sum()),secondary_peak_voxels=int(secondary.sum()),reinforced_voxels=int(np.sum(guide>contrast)),multi_depth_xy=int(np.sum(secondary.any(0)&primary.any(0))),secondary_neighbor_columns=3,neighbor_depth_tolerance=2,reinforcement_radius=1,min_measured_contrast=.75,max_contrast_boost=3.)
    return guide,classes,mip,zmax,stats


def backmap(raw, contrast, config=None):
    from .multiscale_mip import configuration, reinforce
    cfg = configuration(config)
    if raw.ndim != 3 or raw.shape != contrast.shape or any(n == 0 for n in raw.shape):
        raise ValueError('MIP guidance requires matching nonempty 3-D volumes')
    if not np.isfinite(contrast).all():
        raise ValueError('Non-finite MIP contrast')
    guide, classes, mip, zmax, stats = legacy_backmap(raw, contrast)
    stats['mode'] = cfg['mode']
    if cfg['mode'] == 'multiscale':
        guide, classes, report = reinforce(raw, contrast, guide, classes, cfg)
        stats['multiscale'] = report
        stats['reinforced_voxels'] = int(np.count_nonzero(guide > contrast))
    return guide, classes, mip, zmax, stats
