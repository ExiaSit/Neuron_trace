"""Numbered v0.83 reconstruction stage; run_pipeline.py is the batch entry."""
from pathlib import Path
import json, shutil, sys, time, re
import numpy as np
import tifffile
from utils import bootstrap
from utils.case_worker import setup_config, load_module, check_swc, save_json, comparison
from utils import mask_batch as mask_run

def run(spec):
    p=setup_config(spec);cfg=spec['config'];r=p['root'];cid=spec['id']
    input_mode=cfg.get('app2_input_mode','mask')
    if input_mode not in ('mask','skeleton_soma'):raise ValueError('app2_input_mode must be mask or skeleton_soma')
    expected_cnn=cfg.get('app2_expected_cnn_type')
    if expected_cnn is not None and (type(expected_cnn) is not int or expected_cnn not in (1,2,3)):
        raise ValueError('app2_expected_cnn_type must be 1, 2 or 3')
    import pandas as pd
    meta=pd.DataFrame([spec['metadata']], index=[int(cid)])
    mod=load_module('app2')
    from utils.worker_launcher import local_app2
    native_app2=mod.run_app2_locally
    mod.run_app2_locally=lambda *a,**kw:local_app2(native_app2,*a,temp_root=cfg.get('app2_temp_dir','/tmp/neuron-app2-v083'),**kw)
    marker_dir=r/'trace_app2/markers';marker_dir.mkdir(parents=True,exist_ok=True)
    m=spec['metadata'];xyz=[int(round(float(m[k])*float(m[res])/1000)) for k,res in [('soma_x','xy_resolution'),('soma_y','xy_resolution'),('soma_z','z_resolution')]]
    with tifffile.TiffFile(p['mask']) as t:shape=t.series[0].shape
    if any(x<0 or x>=bound for x,bound in zip(xyz,shape[::-1])):raise ValueError(f'Metadata soma {xyz} outside image {shape}')
    marker_xyz=mod.image_to_app2_marker(xyz,shape)
    mask=tifffile.imread(p['mask']);seed_value=int(mask[tuple(xyz[::-1])]);del mask
    save_json(r/'logs'/cid/'app2-seed.json',dict(image_xyz=xyz,marker_xyz=marker_xyz,shape_zyx=list(shape),mask_value=seed_value,coordinate_version=2))
    if seed_value<=10 and input_mode=='mask':raise ValueError(f'Target soma {xyz} is outside mask foreground; refusing to seed another fragment')
    trace_input=p['mask']
    if input_mode=='skeleton_soma':
        from algorithms.tracing.skeleton_input import prepare_input
        trace_input=prepare_input(spec,p,xyz)
    marker=marker_dir/f'{cid}.marker';marker.write_text('##x,y,z,radius,shape,name,comment,color_r,color_g,color_b\n'+','.join(map(str,marker_xyz))+',1,1,soma,,255,0,0\n')
    native=r/'trace_app2/native_swcs'/f'{cid}.swc';native.parent.mkdir(parents=True,exist_ok=True)
    if native.exists():native.rename(native.with_name(f'{cid}_retry_{time.time_ns()}.swc'))
    mod.run_app2_locally(str(trace_input),str(marker),str(native),300,native_log=r/'logs'/cid/'app2-native.log')
    if expected_cnn is not None:
        actual=re.findall(r'^#cnn_type = (\d+)\s*$',(r/'logs'/cid/'app2-native.log').read_text(),re.M)
        if actual!=[str(expected_cnn)]:raise RuntimeError(f'APP2 neighborhood mismatch: expected {expected_cnn}, log reported {actual}')
    check_swc(native);shutil.copy2(native,p['app2']);mod.flip_swc_y_to_image_coordinates(str(p['app2']),shape[-2]);check_swc(p['app2'])
    intermediate=r/'trace_app2/intermediates'/cid;intermediate.mkdir(parents=True,exist_ok=True)
    for f in trace_input.parent.glob(trace_input.name+'*.swc'):shutil.move(str(f),intermediate/f.name)

if __name__=="__main__":
    from utils.stage_cli import run_spec
    run_spec(run)
