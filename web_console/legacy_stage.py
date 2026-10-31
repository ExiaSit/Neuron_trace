"""Run one web stage against a selected live v0.81/v0.82/v0.82a source tree.

This is a process boundary: no v0.83 algorithm modules enter the native process.
Native image_<id> outputs stay in this experiment; canonical exports are copies.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HOST_ROOT=Path(__file__).resolve().parents[1]
VERSIONS={'v0.81':'NEURON_V081_CONFIG','v0.82':'NEURON_V082_CONFIG','v0.82a':'NEURON_V082A_CONFIG'}


def source_root(version):
    if version not in VERSIONS:raise ValueError('Unsupported native version')
    return HOST_ROOT.parent/version


def code_digest(version):
    root=source_root(version)
    names=['run_pipeline.py','pipeline_config.py','1_rescale_cpu.py','2_soma_crop_seg.py',
           '2_neurite_seg.py','3.concat_neurite_soma.py','4_pre_trace_app2.py',
           '5.gcut_pipeline.py','5.5_resample_app2_swc.py','6_gcut_pruning_no_radius_jump_no_long_edge_copy.py']
    files=[root/name for name in names]
    for folder in ['algorithms','utils','neuroutils','vendor']:
        for suffix in ['*.py','*.json']:
            files.extend(p for p in (root/folder).rglob(suffix) if '__pycache__' not in p.parts)
    digest=hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.writing.json');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n');tmp.replace(path)


def copy(source,dest):
    source,dest=Path(source),Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.native-copy.tmp')
    shutil.copy2(source,tmp);tmp.replace(dest)


def bind(source,dest):
    """Read-only inputs may be linked; never repoint an existing native output."""
    source,dest=Path(source).resolve(),Path(dest)
    if not source.is_file():return
    if dest.exists() or dest.is_symlink():
        if dest.resolve()!=source:
            if dest.is_symlink() or not dest.is_file() or dest.stat().st_size!=source.stat().st_size or dest.read_bytes()!=source.read_bytes():
                raise ValueError(f'Native input conflict: {dest}')
        return
    dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(source)


def crop_for_native(source,dest):
    data=json.loads(Path(source).read_text())
    order=data.get('bounds_order','ZYX')
    if order.startswith('ZXY'):
        data['bounds_1um_zxy']=[data['bounds_1um_zxy'][i] for i in (0,1,4,5,2,3)]
    elif order!='ZYX':raise ValueError(f'Unknown crop bounds order: {order}')
    data['bounds_order']='ZYX'
    save(dest,data)


def initialize(spec):
    cfg=spec['config'];version=cfg['method_version'];root=source_root(version)
    native=Path(cfg['output_root'])/'native';native.mkdir(parents=True,exist_ok=True)
    cid=spec['id'];stem='image_'+cid
    # Keep a small per-case metadata table; native lookups use the original fields.
    import pandas as pd
    metadata=native/'metadata'/f'{cid}.csv';metadata.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([spec['metadata']],index=[int(cid)]).to_csv(metadata,index_label='cell_id')
    config=dict(neuron_ids=[int(cid)],output_root=str(native),metadata_file=str(metadata),
        v3d_root=cfg.get('v3d_root') or '/mnt/hndb/Cell_Image',
        vaa3d_app2=cfg['vaa3d_app2'],vaa3d_resample=cfg['vaa3d_resample'],
        pipeline=dict(mode='neuron',stage_max_tasks=1,gpu_max_tasks=1,infer_batch_size=1,isolate_pruning=True),
        nnunet=dict(soma_results=cfg['models']['soma']['results'],neurite_results=cfg['models']['neurite']['results'],
                    gpu_id=cfg.get('gpu_id',0),device='cuda'),
        gcut=dict(target_percentile=55.,gsdt_threshold_x=5.))
    if version!='v0.81':
        config['gcut'].update(candidate_method='app2_gsdt',candidate_backend='cuda',candidate_evidence={'enabled':True},
                             gpu_lock_path=str(native/'.gpu-stage.lock'))
    path=native/'configs'/f'{cid}.json';save(path,config)
    os.environ[VERSIONS[version]]=str(path)
    # Each native stage is launched in a fresh interpreter. Prepend only this tree.
    sys.path[:]=[str(root)]+[p for p in sys.path if Path(p or '.').resolve() not in (HOST_ROOT,HOST_ROOT/'web_console')]
    import pipeline_config
    # Native modules require directories/log parents to exist before import.
    for key,value in pipeline_config.PATHS.items():
        p=Path(value)
        if key.endswith('_dir'):p.mkdir(parents=True,exist_ok=True)
        elif key.endswith('_log') or p.suffix=='.log':p.parent.mkdir(parents=True,exist_ok=True)
    module=importlib.util.spec_from_file_location('native_pipeline',root/'run_pipeline.py')
    pipeline=importlib.util.module_from_spec(module);module.loader.exec_module(pipeline)
    return root,native,stem,pipeline


def run(spec,stage):
    import numpy as np
    import tifffile
    root,native,stem,pipeline=initialize(spec)
    cid=spec['id'];out=Path(spec['config']['output_root']);version=spec['config']['method_version']
    def load(filename):return pipeline.load_stage_module('web_native_'+str(stage),filename)
    def native_path(key):return Path(pipeline.PATHS[key])
    if stage==1:
        if version=='v0.82a':
            from neuroutils.image.paper_preprocess import load_paper_image
            image=load_paper_image(spec['source'])
        else:
            from neuroutils.image.io import load_image
            image=load_image(spec['source'],normalize=True)
        Path(spec['normalized']).parent.mkdir(parents=True,exist_ok=True)
        tifffile.imwrite(spec['normalized'],image,photometric='minisblack')
    elif stage==2:
        module=load('1_rescale_cpu.py');image=tifffile.imread(spec['normalized']);m=spec['metadata']
        shape=module.get_target_shape(image.shape,m['xy_resolution'],m['z_resolution'])
        if version=='v0.82a':
            result=module.preprocess_cuda(image,shape,gpu_id=spec['config'].get('gpu_id',0))
        else:
            result=module.to_uint8_after_resize(module.resize(image,shape,order=module.RESIZE_ORDER,
                mode=module.RESIZE_MODE,anti_aliasing=module.RESIZE_ANTI_ALIASING,preserve_range=True))
        Path(spec['raw']).parent.mkdir(parents=True,exist_ok=True)
        module.write_tiff_atomic(spec['raw'],result)
    elif stage in (3,4,5,6):
        bind(spec['raw'],native_path('image_1um_dir')/(stem+'_0000.tif'))
        if stage==3:
            pipeline.run_soma_crop(cid)
            copy(native_path('soma_crop_dir')/(stem+'_0000.tif'),spec['crop'])
            copy(native_path('soma_crop_dir')/(stem+'.json'),spec['soma_json'])
        elif stage==4:
            bind(spec['crop'],native_path('soma_crop_dir')/(stem+'_0000.tif'))
            pipeline.run_soma_infer(cid)
            copy(native_path('soma_seg_dir')/(stem+'.tif'),spec['soma_mask'])
        elif stage==5:
            pipeline.run_neurite_infer(cid)
            copy(native_path('neurite_seg_dir')/(stem+'.tif'),spec['neurite_mask'])
        else:
            bind(spec['neurite_mask'],native_path('neurite_seg_dir')/(stem+'.tif'))
            bind(spec['soma_mask'],native_path('soma_seg_dir')/(stem+'.tif'))
            crop_for_native(spec['soma_json'],native_path('soma_crop_dir')/(stem+'.json'))
            pipeline.run_merge(cid)
            copy(native_path('merged_mask_dir')/(stem+'.tif'),spec['mask'])
    elif stage in (8,9,10,11):
        if stage!=10:
            bind(spec['raw'],native_path('image_1um_dir')/(stem+'_0000.tif'))
            # Use a copied mask because some native utilities may write their inputs.
            mask=native_path('merged_mask_dir')/(stem+'.tif')
            if not mask.exists():copy(spec['repaired'],mask)
            crop_for_native(spec['soma_json'],native_path('soma_crop_dir')/(stem+'.json'))
            bind(spec['soma_mask'],native_path('soma_seg_dir')/(stem+'.tif'))
        import pandas as pd
        meta=pd.DataFrame([spec['metadata']],index=[int(cid)])
        if stage==8:
            pipeline.run_app2_trace(cid)
            copy(native_path('trace_swc_dir')/(stem+'.swc'),spec['app2'])
        elif stage==9:
            app2=native_path('trace_swc_dir')/(stem+'.swc')
            if not app2.exists():copy(spec['app2'],app2)
            if pipeline.run_gcut(cid,meta) is False:raise RuntimeError('Native G-Cut produced no target')
            copy(pipeline.gcut_selected_swc_path(cid),spec['selected'])
            figures=list(native_path('gcut_output_dir').rglob(stem+'*_comprehensive_vis.png'))
            if figures:copy(figures[0],out/'gcut_mips'/f'{cid}.png')
            # Export native G-Cut diagnostics without applying any v0.83 rule.
            folder=native_path('gcut_output_dir')/cid
            for name in ['nms-audit.json','candidate-evidence.json','soma-merge-audit.json']:
                if (folder/name).is_file():copy(folder/name,out/'logs'/cid/name)
        elif stage==10:
            selected=pipeline.gcut_selected_swc_path(cid)
            if not selected.exists():copy(spec['selected'],selected)
            pipeline.run_gcut_resample(cid)
            copy(pipeline.gcut_resampled_swc_path(cid),spec['resampled'])
        else:
            resampled=pipeline.gcut_resampled_swc_path(cid)
            if not resampled.exists():copy(spec['resampled'],resampled)
            pipeline.run_prune(cid,meta)
            copy(pipeline.prune_output_swc_path(cid),spec['final'])
            for path in native_path('prune_mip_dir').glob(stem+'*'):
                if path.is_file() and path.suffix in ('.png','.json'):
                    copy(path,out/'pruning_mips'/path.name.replace(stem,cid,1))
    else:raise ValueError(f'Native stage {stage} must be handled by the console adapter')
    save(out/'logs'/cid/f'native-stage-{stage:02d}.json',dict(method_version=version,source_root=str(root),
         source_code_digest=spec['config'].get('native_code_digest'),stage=stage,mask_repair=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--stage',type=int,required=True)
    args=ap.parse_args();run(json.loads(args.spec.read_text()),args.stage)
