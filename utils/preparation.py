"""Input preparation, checkpointing and nnUNet inference for numbered steps 01–06."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, hashlib, importlib.util, json, os, re, subprocess, sys, tempfile
import numpy as np
import pandas as pd
import tifffile
from utils.bootstrap import ROOT
from utils.configuration import load_config
from utils.case_worker import save_json
from utils.mask_batch import atomic_tiff

STEPS={1:'01_v3d_to_tif.py',2:'02_resample_1um.py',3:'03_crop_soma.py',
       4:'04_segment_soma.py',5:'05_segment_neurite.py',6:'06_merge_masks.py'}

def read_metadata(cfg):
    for encoding in ['utf-8-sig','gb18030']:
        try:
            table=pd.read_csv(cfg['metadata_file'],encoding=encoding,low_memory=False)
            break
        except UnicodeDecodeError:continue
    else:raise ValueError('Cannot decode metadata CSV')
    return table.set_index('cell_id')

def requested_ids(cfg,text=None):
    values=text.split(',') if text else cfg.get('neuron_ids')
    if values:return [str(int(x)) for x in values]
    if cfg.get('v3d_dir'):
        return sorted(index_v3d(cfg['v3d_dir']),key=int)
    raise ValueError('请通过 --ids 15257,17254 或 config.json 的 neuron_ids 指定 ID；也可设置 v3d_dir 自动扫描。')

def index_v3d(directory):
    found={}
    for pattern in ['*.v3dpbd','*.v3draw']:
        for p in Path(directory).rglob(pattern):
            m=re.match(r'^(?:image_)?(\d+)(?:_|$)',p.stem)
            if m:
                cid=str(int(m[1]))
                if cid in found:raise ValueError(f'Duplicate V3D ID: {cid}')
                found[cid]=p
    return found

def source_path(cfg,cid,row,index=None):
    if cfg.get('v3d_dir'):
        if index is None:index=index_v3d(cfg['v3d_dir'])
        if cid not in index:raise FileNotFoundError(f'{cid}: V3D source missing')
        return index[cid]
    value=str(row['v3dpbd_file']).replace('\\','/')
    if 'Cell_Image/' in value:value=value.split('Cell_Image/',1)[1]
    p=Path(value)
    return p if p.is_absolute() else Path(cfg['v3d_root'])/p

def case(cfg,cid,row,index=None):
    root=Path(cfg['prepared_root'])
    return dict(id=cid,config=cfg,metadata=row,source=str(source_path(cfg,cid,row,index)),
                normalized=str(root/'source_tif'/f'{cid}.tif'),raw=str(root/'img'/f'{cid}_0000.tif'),
                crop=str(root/'soma_img'/f'{cid}_0000.tif'),soma_json=str(root/'soma_img'/f'{cid}.json'),
                soma_mask=str(root/'soma_seg'/f'{cid}.tif'),neurite_mask=str(root/'neurite_seg'/f'{cid}.tif'),
                mask=str(root/'initial_mask'/f'{cid}.tif'))

def output_files(spec,step):
    keys={1:['normalized'],2:['raw'],3:['crop','soma_json'],4:['soma_mask'],5:['neurite_mask'],6:['mask']}[step]
    return [Path(spec[k]) for k in keys]

def stamp(p):
    p=Path(p);s=p.stat()
    return [str(p.resolve()),s.st_size,s.st_mtime_ns]

def fingerprint(spec,step):
    keys={1:['source'],2:['normalized'],3:['raw'],4:['crop','soma_json'],5:['raw'],6:['neurite_mask','soma_mask','soma_json']}[step]
    code=[ROOT/STEPS[step],ROOT/'utils/preparation.py',ROOT/'algorithms/preprocessing.py',ROOT/'utils/v3d_io.py']
    options=dict(metadata=spec['metadata'],crop_shape=spec['config'].get('soma_crop_shape',[128]*3),
                 override=spec['config'].get('soma_overrides_image_xyz',{}).get(spec['id']))
    if step in (4,5):
        model=spec['config']['models']['soma' if step==4 else 'neurite']
        options['model']=model
        folder=Path(model['results'])/model['dataset_name']/f"{model.get('trainer','nnUNetTrainer')}__nnUNetPlans__3d_fullres"
        options['model_files']=[stamp(folder/f'fold_{model.get("fold",0)}'/model.get('checkpoint','checkpoint_final.pth')),
                                stamp(folder/'plans.json'),stamp(folder/'dataset.json')]
    value=dict(inputs=[stamp(spec[k]) for k in keys],options=options,
               code=hashlib.sha256(b''.join(p.read_bytes() for p in code)).hexdigest())
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def checkpoint(spec,step):
    return Path(spec['config']['prepared_root'])/'status'/f"{spec['id']}_{step:02d}.json"

def is_complete(spec,step,key):
    p=checkpoint(spec,step)
    if not p.exists():return False
    record=json.loads(p.read_text())
    if record['fingerprint']!=key:
        raise ValueError(f"{spec['id']}: step {step:02d} inputs/config changed; use a fresh prepared_root")
    return all(p.is_file() and p.stat().st_size>0 for p in output_files(spec,step))

def mark(spec,step,key):
    for p in output_files(spec,step):
        if not p.is_file() or not p.stat().st_size:raise RuntimeError(f'Missing output: {p}')
    save_json(checkpoint(spec,step),dict(status='complete',fingerprint=key,outputs=list(map(str,output_files(spec,step)))))

def load_step(step):
    item=importlib.util.spec_from_file_location(f'preparation_{step}',ROOT/STEPS[step])
    module=importlib.util.module_from_spec(item);item.loader.exec_module(module)
    return module

def run_one(item):
    spec,step=item
    try:
        key=fingerprint(spec,step)
        if is_complete(spec,step,key):return spec,None
        load_step(step).run(spec)
        mark(spec,step,key)
        print(f"[{spec['id']}] {STEPS[step]} 完成",flush=True)
        return spec,None
    except Exception as exc:
        return spec,f'{type(exc).__name__}: {exc}'

def inference(specs,kind):
    if not specs:return
    step=4 if kind=='soma' else 5
    cfg=specs[0]['config'];model=cfg['models'][kind]
    pending=[]
    for spec in specs:
        key=fingerprint(spec,step)
        if not is_complete(spec,step,key):pending.append((spec,key))
    size=int(cfg.get('inference_batch_size',4))
    if size<1:raise ValueError('inference_batch_size must be positive')
    for start in range(0,len(pending),size):
        group=pending[start:start+size]
        # Scratch links are removed when inference ends; they never enter final outputs.
        scratch=Path(cfg.get('app2_temp_dir','/tmp/neuron-app2-v083'))
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f'nnunet-{kind}-',dir=scratch) as temp:
            temp=Path(temp);inp=temp/'input';out=temp/'output';inp.mkdir();out.mkdir()
            for spec,_ in group:
                src=Path(spec['crop' if kind=='soma' else 'raw'])
                (inp/f"{spec['id']}_0000.tif").symlink_to(src)
                # Tiff3DIO reads case spacing, independently of the crop-bounds sidecar.
                (inp/f"{spec['id']}.json").write_text(json.dumps({'spacing':[1.,1.,1.]}))
            command=[cfg['nnunet_executable'],'-i',str(inp),'-o',str(out),
                     '-d',str(model['dataset_id']),'-c','3d_fullres','-f',str(model.get('fold',0)),
                     '-tr',model.get('trainer','nnUNetTrainer'),'-chk',model.get('checkpoint','checkpoint_final.pth'),
                     '-device',cfg.get('inference_device','cuda'),'-npp','1','-nps','1']
            env=dict(os.environ,nnUNet_raw='Wait_No_Need',nnUNet_preprocessed='Wait_No_Need',
                     nnUNet_results=model['results'],CUDA_VISIBLE_DEVICES=str(cfg.get('gpu_id',0)),
                     OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            log=Path(cfg['prepared_root'])/'logs'/f"{kind}_{group[0][0]['id']}.log"
            log.parent.mkdir(parents=True,exist_ok=True)
            print(f"nnUNet {kind}: {[s['id'] for s,_ in group]}；日志 {log}",flush=True)
            with log.open('w') as stream:subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            for spec,key in group:
                result=tifffile.imread(out/f"{spec['id']}.tif")
                with tifffile.TiffFile(spec['crop' if kind=='soma' else 'raw']) as tif:
                    if result.shape!=tif.series[0].shape:raise ValueError(f"{spec['id']}: nnUNet output shape mismatch")
                dest=Path(spec['soma_mask' if kind=='soma' else 'neurite_mask'])
                atomic_tiff(dest,result)
                mark(spec,step,key)

def run_preparation(cfg,ids,first=1,last=6,check=False):
    table=read_metadata(cfg);specs=[];skipped=[]
    index=index_v3d(cfg['v3d_dir']) if cfg.get('v3d_dir') else None
    for cid in ids:
        try:
            row=table.loc[int(cid)]
            if isinstance(row,pd.DataFrame):raise ValueError('duplicate metadata')
            row=row.where(pd.notnull(row),None).to_dict()
            spec=case(cfg,cid,row,index)
            if first==1 and not Path(spec['source']).is_file():raise FileNotFoundError(spec['source'])
            specs.append(spec)
        except (KeyError,ValueError,FileNotFoundError) as exc:
            skipped.append(dict(id=cid,reason=str(exc)));print(f'[{cid}] 跳过：{exc}',flush=True)
    if check:
        print(f'V3D 输入检查：就绪 {len(specs)}，跳过 {len(skipped)}',flush=True)
        return [s['id'] for s in specs]
    root=Path(cfg['prepared_root']);root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'.preparation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save_json(root/'skipped-inputs.json',{'records':skipped})
    failures=[]
    for step in range(first,last+1):
        if not specs:break
        if step in (4,5):
            inference(specs,'soma' if step==4 else 'neurite')
        else:
            with ProcessPoolExecutor(max_workers=int(cfg.get('preprocess_workers',1))) as pool:
                results=list(pool.map(run_one,[(s,step) for s in specs]))
            specs=[]
            for spec,error in results:
                if error:
                    failures.append(dict(id=spec['id'],step=step,error=error))
                    print(f"[{spec['id']}] 失败：{error}",flush=True)
                else:specs.append(spec)
    save_json(root/'preparation-summary.json',dict(completed_ids=[s['id'] for s in specs],failed=failures,skipped=skipped))
    if failures:print(f'前处理失败 {len(failures)} 个；详情见 {root/"preparation-summary.json"}',flush=True)
    return [s['id'] for s in specs]

def cli(step):
    parser=argparse.ArgumentParser(description=STEPS[step])
    parser.add_argument('--config',default=str(ROOT/'config.json'));parser.add_argument('--ids');parser.add_argument('--check',action='store_true')
    args=parser.parse_args();cfg=load_config(args.config)
    run_preparation(cfg,requested_ids(cfg,args.ids),step,step,args.check)
