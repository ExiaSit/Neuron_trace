from pathlib import Path
import json
import numpy as np
import tifffile
from utils import bootstrap
from utils.mask_batch import atomic_tiff
from utils.case_worker import save_json

def run(spec):
    from algorithms.preprocessing import calculate_bounds_3d
    image=tifffile.imread(spec['raw']);m=spec['metadata'];cfg=spec['config']
    xyz=cfg.get('soma_overrides_image_xyz',{}).get(spec['id'])
    if xyz is None:xyz=[float(m[k])*float(m[r])/1000 for k,r in [('soma_x','xy_resolution'),('soma_y','xy_resolution'),('soma_z','z_resolution')]]
    center=tuple(int(round(v)) for v in xyz[::-1])
    if any(v<0 or v>=n for v,n in zip(center,image.shape)):raise ValueError(f'Soma outside image: {xyz}')
    shape=tuple(cfg.get('soma_crop_shape',[128,128,128]))
    b=calculate_bounds_3d(image.shape,center,shape)
    crop=image[b[0]:b[1],b[2]:b[3],b[4]:b[5]]
    if not all(crop.shape):raise ValueError('Empty soma crop')
    crop=np.pad(crop,tuple((0,n-s) for n,s in zip(shape,crop.shape)),'constant')
    if crop.max()==crop.min():crop=np.zeros(crop.shape,np.uint8)
    else:crop=((crop-crop.min())/(crop.max()-crop.min())*255).astype(np.uint8)
    atomic_tiff(Path(spec['crop']),crop)
    save_json(Path(spec['soma_json']),dict(original_1um_shape=list(crop.shape),bounds_1um_zxy=list(b),
                                        spacing=[1.,1.,1.],source_file=spec['raw']))

if __name__=="__main__":
    from utils.preparation import cli
    cli(3)
