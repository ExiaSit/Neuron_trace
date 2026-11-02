"""Exact slab argmax/vote computation with one GPU upload per volume.

The intensity comparison, first-index tie rule, overlap windows and two-axis
consensus match the NumPy implementation. No interpolation or reduced precision.
"""
import time
import numpy as np
from utils.gpu_ndimage import ndi


def seeds_cuda(raw,contrast,cfg):
    with ndi.gpu_slot():
        ndi.initialize();started=time.perf_counter()
        # Keep CuPy arrays inside a nested scope so they are released before
        # gpu_slot returns its memory pool to the next sample.
        result=_compute(ndi.cp,raw,contrast,cfg)
        key='multiscale_mip_seeds'
        ndi.stats['gpu_calls'][key]=ndi.stats['gpu_calls'].get(key,0)+1
        times=ndi.stats['gpu_seconds_including_transfers'];times[key]=times.get(key,0)+time.perf_counter()-started
        return result


def _compute(cp,raw,contrast,cfg):
    from .multiscale_mip import window_ranges
    raw=cp.asarray(raw);contrast=cp.asarray(contrast)
    support=cp.zeros(raw.shape,cp.uint8);records=[]
    vote=cp.ElementwiseKernel('raw int64 depths, raw bool valid, int32 height, int32 width','bool supported',r'''
        const int row=i/width, col=i%width;
        int votes=0;
        for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx){
            if(dy==0 && dx==0) continue;
            int y=row+dy,x=col+dx;
            if(y<0 || y>=height || x<0 || x>=width) continue;
            int j=y*width+x;
            long long delta=depths[i]-depths[j];
            if(valid[j] && delta>=-2 && delta<=2) ++votes;
        }
        supported=valid[i] && votes>=2;
    ''','neuron_slab_neighbor_votes')
    for axis,plane in enumerate(('XY','XZ','YZ')):
        volume=cp.moveaxis(raw,axis,0);local_contrast=cp.moveaxis(contrast,axis,0)
        widths=sorted({min(w,len(volume)) for w in cfg['thicknesses']}|{len(volume)})
        counts=cp.zeros(raw.shape,cp.uint8);scales=[]
        h,w=volume.shape[1:];row=cp.arange(h)[:,None];col=cp.arange(w)[None,:]
        for width in widths:
            flags=cp.zeros(raw.shape,cp.bool_);local_flags=cp.moveaxis(flags,axis,0)
            windows=window_ranges(len(volume),width,cfg['overlap'])
            for start,stop in windows:
                depth=volume[start:stop].argmax(axis=0).astype(cp.int64,copy=False)
                valid=local_contrast[depth+start,row,col]>2
                ok=vote(depth,valid,h,w,size=h*w).reshape(h,w)
                # One unique location per column; OR preserves earlier windows.
                local_flags[depth+start,row,col]=local_flags[depth+start,row,col]|ok
            counts+=flags
            scales.append(dict(thickness=width,stride=max(1,int(np.floor(width*(1-cfg['overlap'])))),
                               windows=[[a,b] for a,b in windows],seed_voxels=int(flags.sum())))
        support+=counts>0
        records.append(dict(axis_zyx=axis,plane=plane,scales=scales,
                            seed_voxels=int(cp.count_nonzero(counts)),
                            multiple_scale_voxels=int(cp.count_nonzero(counts>=2))))
    seeds=support>=cfg['min_axes']
    report=dict(config=cfg,axes=records,consensus_seed_voxels=int(seeds.sum()),
                axis_support_counts={str(i):int(cp.count_nonzero(support==i)) for i in range(4)})
    return cp.asnumpy(seeds),report
