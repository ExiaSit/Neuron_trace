"""Validated stage-range execution, isolated outputs and resumable checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import bootstrap
from utils.case_worker import save_json, check_swc
from utils.preparation import read_metadata, source_path, index_v3d
from utils.reconstruction_batch import index_files, file_stamp, validate_tiffs
from algorithms.preprocessing import validate_resolution

STAGES = ['V3D 解码', '1 µm 重采样', '胞体裁剪', '胞体分割', '神经突分割',
          '合并 mask', 'Mask 修复', 'APP2 追踪', 'G-Cut 分离', 'SWC 重采样',
          '信号剪枝', '历史回归保护', '结果比较']
SCRIPTS = ['01_v3d_to_tif.py', '02_resample_1um.py', '03_crop_soma.py',
           '04_segment_soma.py', '05_segment_neurite.py', '06_merge_masks.py',
           '07_reconnect_mask.py', '08_trace_app2.py', '09_separate_gcut.py',
           '10_resample_swc.py', '11_prune_swc.py', '12_guard_regression.py',
           '13_compare_results.py']
OUTPUT_KEYS = {1:['normalized'], 2:['raw'], 3:['crop','soma_json'], 4:['soma_mask'],
               5:['neurite_mask'], 6:['mask'], 7:['repaired'], 8:['app2'],
               9:['selected'], 10:['resampled'], 11:['final'], 12:['guard'], 13:['comparison']}
COMMON = {'raw','mask','soma_json','soma_mask'}
DEPS = {1:{'source'},2:{'normalized'},3:{'raw'},4:{'crop','soma_json'},5:{'raw'},
        6:{'neurite_mask','soma_mask','soma_json'},7:COMMON,
        8:COMMON|{'repaired'},9:COMMON|{'repaired','app2'},
        10:{'selected'},11:COMMON|{'repaired','resampled'},
        12:COMMON|{'repaired','final'},13:COMMON|{'repaired','app2','final'}}
INPUT_FIELDS = {'normalized':('source_tif_dir',{'.tif','.tiff'}),
                'raw':('raw_dir',{'.tif','.tiff'}), 'crop':('soma_crop_dir',{'.tif','.tiff'}),
                'soma_json':('soma_crop_dir',{'.json'}), 'soma_mask':('soma_seg_dir',{'.tif','.tiff'}),
                'neurite_mask':('neurite_seg_dir',{'.tif','.tiff'}), 'mask':('mask_dir',{'.tif','.tiff'})}
UPSTREAM = {'repaired':'mask/{id}.tif','app2':'trace_app2/down_sampled_swcs_app2/{id}.swc',
            'selected':'gcut_output/selected_for_pruning/{id}.swc',
            'resampled':'gcut_output/selected_resampled_step1/{id}.swc','final':'pruning_swcs/{id}.swc'}


def code_digest():
    paths = sorted(p for p in ROOT.rglob('*.py') if '.git' not in p.parts and
                   not any(x in p.parts for x in ('outputs','results','validation','__pycache__')))
    return hashlib.sha256(b''.join(str(p.relative_to(ROOT)).encode()+p.read_bytes() for p in paths)).hexdigest()


def parse_request(body):
    if not isinstance(body,dict): raise ValueError('请求必须是 JSON 对象')
    req = {k:body.get(k,'') for k in ['name','ids','metadata_file','v3d_root','v3d_dir',
           'previous_app2_dir','source_tif_dir','raw_dir','mask_dir','soma_crop_dir','soma_seg_dir','neurite_seg_dir','upstream_root']}
    for k,v in req.items():
        if not isinstance(v,str) or len(v)>4096: raise ValueError(f'{k} 格式不正确')
        req[k]=v.strip()
    tokens = re.split(r'[,，\s]+',req['ids'])
    if not tokens or any(not re.fullmatch(r'\d{1,12}',x) for x in tokens):
        raise ValueError('请输入数字 ID，以逗号或空格分隔')
    req['ids']=list(dict.fromkeys(str(int(x)) for x in tokens))
    if len(req['ids'])>200: raise ValueError('单次最多 200 个 ID')
    for k,default in [('start',7),('stop',13),('gpu_id',0)]:
        value=body.get(k,default)
        if isinstance(value,bool) or not isinstance(value,int): raise ValueError(f'{k} 必须是整数')
        req[k]=value
    if not 1<=req['start']<=req['stop']<=13: raise ValueError('阶段范围必须满足 1 ≤ 开始 ≤ 结束 ≤ 13')
    if not 0<=req['gpu_id']<=15: raise ValueError('GPU 编号须为 0–15')
    from web_console.versions import DEFAULT_METHOD, method_root
    req['method_version']=body.get('method_version',DEFAULT_METHOD)
    method_root(req['method_version'])
    # Experimental console deliberately exposes candidate results, without historical substitution.
    req['name']=req['name'][:80] or req['method_version']+' 实验'
    return req


def required_keys(first,last):
    produced=set(); needed=set()
    for stage in range(first,last+1):
        needed |= DEPS[stage]-produced
        produced.update(OUTPUT_KEYS[stage])
    return needed


def base_config(req, job):
    cfg=json.loads((ROOT/'config.example.json').read_text())
    for key in ['python_executable','vaa3d_app2','vaa3d_resample','nnunet_executable']:
        cfg[key]=str(Path(cfg[key]).resolve())
    cfg.update(input_mode='prepared',metadata_file=req['metadata_file'],
               prepared_root=str(job/'prepared'), output_root=str(job/'reconstruction'),
               output_mask_dir=str(job/'reconstruction/mask'), output_mip_dir=str(job/'reconstruction/mask_mips'),
               comparison_dir=str(job/'reconstruction/comparisons'),direct_inputs=True,
               workers=1,preprocess_workers=1,inference_batch_size=1,gpu_id=req['gpu_id'],
               baseline_root=None,validated_baseline_registry=None,previous_swc_dirs=[],
               soma_overrides_image_xyz={})
    cfg['method_version']=req.get('method_version',cfg['method_version'])
    from web_console.versions import METHODS
    if METHODS[cfg['method_version']].get('engine')=='legacy':
        cfg['method_revision']=cfg['method_version']+'-native-console'
        cfg.pop('mask_mip',None)
    if req.get('previous_app2_dir'):
        cfg['previous_app2_dirs']=[str(local_path(req['previous_app2_dir'],'旧版 APP2 目录'))]
    return cfg


def local_path(value, label):
    if not value or not Path(value).is_absolute(): raise ValueError(f'{label} 请填写服务器上的绝对路径')
    return Path(value).resolve()


def preflight(req):
    import numpy as np
    import tifffile
    cfg=base_config(req,Path('/unused'))
    from web_console.versions import METHODS
    legacy=METHODS[req.get('method_version','v0.83a')].get('engine')=='legacy'
    cfg['metadata_file']=str(local_path(req['metadata_file'],'metadata_file'))
    try:
        table=read_metadata(cfg)
    except OSError as exc:
        raise ValueError(f"读取元数据 CSV 失败：{cfg['metadata_file']}。"
                         f"系统错误：{exc}。请检查该文件及所在磁盘是否可正常读取，"
                         "或选择一份已核实的完整元数据文件；尚未启动重建。") from exc
    needed=required_keys(req['start'],req['stop'])
    indexes={}
    for key in sorted(needed & INPUT_FIELDS.keys()):
        field,suffixes=INPUT_FIELDS[key];folder=local_path(req[field],field)
        if not folder.is_dir():raise ValueError(f'目录不存在：{folder}')
        indexes[key]=index_files(folder,suffixes)
    v3d_index=None
    if 'source' in needed:
        if req['v3d_dir']:v3d_index=index_v3d(local_path(req['v3d_dir'],'v3d_dir'))
        else:local_path(req['v3d_root'],'v3d_root')
    for stage,key in [(4,'nnunet_executable'),(5,'nnunet_executable'),(8,'vaa3d_app2'),(10,'vaa3d_resample')]:
        if req['start']<=stage<=req['stop'] and not Path(cfg[key]).is_file():raise ValueError(f'程序不存在：{cfg[key]}')
    for stage,kind in [(4,'soma'),(5,'neurite')]:
        if req['start']<=stage<=req['stop']:
            m=cfg['models'][kind];folder=Path(m['results'])/m['dataset_name']/f"{m['trainer']}__nnUNetPlans__3d_fullres"
            for rel in ['plans.json','dataset.json',f"fold_{m['fold']}/{m['checkpoint']}"]:
                if not (folder/rel).is_file():raise ValueError(f'模型文件缺失：{folder/rel}')
    cases=[]
    for cid in req['ids']:
        if int(cid) not in table.index:raise ValueError(f'{cid}：元数据缺失')
        row=table.loc[int(cid)]
        if getattr(row,'ndim',1)!=1:raise ValueError(f'{cid}：元数据重复')
        meta=row.where(row.notnull(),None).to_dict()
        for key in ['soma_x','soma_y','soma_z','xy_resolution','z_resolution']:
            if meta.get(key) is None or not math.isfinite(float(meta[key])):raise ValueError(f'{cid}：{key} 无效')
        validate_resolution(meta['xy_resolution'],meta['z_resolution'])
        inputs={}
        for key in sorted(needed):
            if key=='source': p=source_path(req,cid,meta,v3d_index)
            elif key in UPSTREAM:
                upstream=local_path(req['upstream_root'],'upstream_root')
                p=upstream/UPSTREAM[key].format(id=cid)
                if not p.is_file():
                    prefixed=p.with_name('image_'+p.name)
                    if prefixed.is_file():p=prefixed
                    elif key=='final':
                        native_final=upstream/'gcut_pruned/pruning_swcs'/f'image_{cid}.swc'
                        if native_final.is_file():p=native_final
            else:
                found,duplicates=indexes[key]
                if cid in duplicates:raise ValueError(f'{cid}：{INPUT_FIELDS[key][0]} 存在重复文件')
                if cid not in found:raise ValueError(f'{cid}：{INPUT_FIELDS[key][0]} 缺少匹配文件')
                p=found[cid]
            if not p.is_file() or p.stat().st_size==0:raise ValueError(f'{cid}：文件不存在或为空：{p}')
            inputs[key]=str(p.resolve())
            if p.suffix=='.swc':check_swc(p)
            elif p.suffix=='.json':json.loads(p.read_text())
            elif p.suffix.lower() in ('.tif','.tiff'):
                with tifffile.TiffFile(p) as tif:
                    if len(tif.series[0].shape)!=3:raise ValueError(f'{p} 不是三维 TIFF')
                    if key in ('raw','normalized') and tif.series[0].dtype!=np.uint8:raise ValueError(f'{p} 需要 uint8 灰度图')
        if COMMON<=inputs.keys(): validate_tiffs(inputs['raw'],inputs['mask'],inputs['soma_mask'])
        if 'raw' in inputs:
            with tifffile.TiffFile(inputs['raw']) as tif:shape=tif.series[0].shape
            xyz=[float(meta[k])*float(meta[r])/1000 for k,r in [('soma_x','xy_resolution'),('soma_y','xy_resolution'),('soma_z','z_resolution')]]
            if any(round(v)<0 or round(v)>=n for v,n in zip(xyz,shape[::-1])):raise ValueError(f'{cid}：元数据胞体坐标超出 1 µm 图像')
            if 'repaired' in inputs:
                with tifffile.TiffFile(inputs['repaired']) as tif:
                    if tif.series[0].shape!=shape:raise ValueError(f'{cid}：修复 mask 与原图尺寸不一致')
        cases.append(dict(id=cid,metadata=meta,inputs=inputs,stamps={k:file_stamp(p) for k,p in inputs.items()}))
    native={}
    if legacy:
        from web_console.legacy_stage import code_digest as native_digest
        native['native_code_digest']=native_digest(req['method_version'])
    return dict(**native,cases=cases,required=sorted(needed),code_digest=code_digest(),
                metadata_stamp=file_stamp(cfg['metadata_file']),checked_at=time.time())


def case_paths(job,cid):
    prep=job/'prepared';out=job/'reconstruction'
    return dict(source='',normalized=str(prep/'source_tif'/f'{cid}.tif'),raw=str(prep/'img'/f'{cid}_0000.tif'),
                crop=str(prep/'soma_img'/f'{cid}_0000.tif'),soma_json=str(prep/'soma_img'/f'{cid}.json'),
                soma_mask=str(prep/'soma_seg'/f'{cid}.tif'),neurite_mask=str(prep/'neurite_seg'/f'{cid}.tif'),
                mask=str(prep/'initial_mask'/f'{cid}.tif'),repaired=str(out/'mask'/f'{cid}.tif'),
                app2=str(out/'trace_app2/down_sampled_swcs_app2'/f'{cid}.swc'),
                selected=str(out/'gcut_output/selected_for_pruning'/f'{cid}.swc'),
                resampled=str(out/'gcut_output/selected_resampled_step1'/f'{cid}.swc'),
                final=str(out/'pruning_swcs'/f'{cid}.swc'),guard=str(out/'regression_guard'/f'{cid}.json'),
                comparison=str(out/'comparisons'/f'{cid}.png'))


def run_legacy_stage(spec_path,spec,stage):
    cfg=spec['config'];out=Path(cfg['output_root']);cid=spec['id']
    if stage==7:
        # This slot is a passthrough for versions with no mask completion.
        copy=Path(spec['repaired']);copy.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(spec['mask'],copy)
        save_json(out/'logs'/cid/'continuity.json',dict(method_version=cfg['method_version'],
            mask_repair=False,decision='original_merged_mask',root_geodesic_enabled=False))
        from utils import mask_batch
        mask_batch.make_mip((Path(spec['raw']),Path(spec['mask']),copy),out/'mask_mips'/f'{cid}.png')
    elif stage==12:
        save_json(Path(spec['guard']),dict(method_version=cfg['method_version'],
            decision='native_version_no_publication_guard',historical_replacement=False))
    elif stage==13:
        from visualization.comparison import comparison
        comparison(dict(spec,selection_reason=cfg['method_version']+' native pipeline'))
    else:
        from web_console.versions import method_root
        subprocess.run([sys.executable,str(ROOT/'web_console/legacy_stage.py'),'--spec',str(spec_path),
                        '--stage',str(stage)],cwd=method_root(cfg['method_version']),check=True)


def run_stage(spec_path,stage):
    spec=json.loads(spec_path.read_text())
    from web_console.versions import METHODS
    legacy=METHODS[spec['config'].get('method_version','v0.83a')].get('engine')=='legacy'
    if not legacy and stage>=7 and Path(spec['mask']).is_file():
        from utils.mask_storage import preserve_original_mask
        preserve_original_mask(spec)
    if legacy:
        run_legacy_stage(spec_path,spec,stage)
    elif stage<=6:
        from utils.preparation import load_step,inference
        if stage in (4,5):inference([spec],'soma' if stage==4 else 'neurite')
        else:load_step(stage).run(spec)
    else:
        import importlib.util
        loader=importlib.util.spec_from_file_location('range_stage',ROOT/SCRIPTS[stage-1])
        mod=importlib.util.module_from_spec(loader);loader.loader.exec_module(mod)
        if stage==7:
            from utils.case_worker import run_worker
            run_worker(spec_path,mask_only=True)
        else:mod.run(spec)
    if stage<=6:
        if not legacy and stage==6:
            from utils.mask_storage import preserve_original_mask
            preserve_original_mask(spec)
        import numpy as np
        import tifffile
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        key=OUTPUT_KEYS[stage][0]
        volume=tifffile.imread(spec[key]);projection=volume.max(axis=0);del volume
        out=Path(spec['config']['output_root'])/'preparation_mips'/f"{spec['id']}_{stage:02d}.png"
        out.parent.mkdir(parents=True,exist_ok=True)
        fig,ax=plt.subplots(figsize=(8,6));ax.imshow(projection,cmap='gray',vmin=0,vmax=max(1,float(np.percentile(projection,99.5))))
        ax.set_title(f"{spec['id']} | stage {stage:02d} | XY MIP");ax.axis('off')
        fig.tight_layout();fig.savefig(out,dpi=120);plt.close(fig)
    elif stage==8:
        from visualization.app2_mip import generate_xy_mip_visualization
        out=Path(spec['config']['output_root'])/'app2_mips'/f"{spec['id']}.png"
        result=generate_xy_mip_visualization(spec['raw'],spec['repaired'],spec['app2'],str(out),spec['id'])
        if result!='OK':raise RuntimeError(f'APP2 可视化失败：{result}')


def run_job(job):
    import fcntl
    with (job/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run_locked_job(job)


def run_locked_job(job):
    record=json.loads((job/'job.json').read_text());req=record['request'];plan=record['plan']
    progress_path=job/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'completed':{},'failed':{},'status':'running'}
    progress['status']='running';progress.pop('error',None);save_json(progress_path,progress)
    try:
        if plan['code_digest']!=code_digest():raise ValueError('代码已变更，请创建新实验；旧结果已保留')
        if file_stamp(req['metadata_file'])!=plan['metadata_stamp']:raise ValueError('元数据已变化，请创建新实验')
        for case in plan['cases']:
            for k,p in case['inputs'].items():
                if file_stamp(p)!=case['stamps'][k]:raise ValueError(f'输入已变化：{p}；请创建新实验')
        cfg=base_config(req,job)
        if 'native_code_digest' in plan:cfg['native_code_digest']=plan['native_code_digest']
        save_json(job/'config.json',cfg)
        for case in plan['cases']:
            cid=case['id'];p=case_paths(job,cid)
            (job/'reconstruction/logs'/cid).mkdir(parents=True,exist_ok=True)
            for key,src in case['inputs'].items():
                if key in UPSTREAM:
                    dst=Path(p[key]);dst.parent.mkdir(parents=True,exist_ok=True)
                    # Import once into this job. Never link writable intermediates to old results.
                    if not dst.exists():shutil.copy2(src,dst)
                else:p[key]=src
            spec=dict(p,id=cid,config=cfg,metadata=case['metadata'],old_swcs=[],
                      selection_reason='web_console_candidate_without_baseline',fingerprint=plan['code_digest']+job.name)
            manifest=job/'reconstruction/manifests'/f'{cid}.json';save_json(manifest,spec)
            from web_console.versions import stage_names
            names=stage_names(req.get('method_version','v0.83a'),STAGES)
            for stage in range(req['start'],req['stop']+1):
                if 'native_code_digest' in plan:
                    from web_console.legacy_stage import code_digest as native_digest
                    if native_digest(req['method_version'])!=plan['native_code_digest']:
                        raise ValueError('所选版本代码已变化，请创建新实验')
                if (job/'pause.request').exists():
                    progress.update(status='paused',current=None);save_json(progress_path,progress);return
                key=f'{cid}:{stage}'
                outputs=[Path(p[k]) for k in OUTPUT_KEYS[stage]]
                previous=progress['completed'].get(key)
                if previous and all(x.is_file() and x.stat().st_size for x in outputs):
                    # Stage 12 can deliberately modify stage 11's output.
                    if stage!=11 or f'{cid}:12' not in progress['completed']:
                        if [file_stamp(x) for x in outputs]!=previous['outputs']:raise ValueError(f'{key} 输出已变化，请新建实验')
                    continue
                progress.update(current=dict(id=cid,stage=stage,name=names[stage-1]),status='running')
                save_json(progress_path,progress)
                print(f'[{cid}] {stage:02d} {names[stage-1]} 开始',flush=True)
                log=job/'logs'/f'{cid}-{stage:02d}.log';log.parent.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
                started=time.time()
                with log.open('a') as stream:
                    result=subprocess.run([cfg['python_executable'],str(Path(__file__).resolve()),'--spec',str(manifest),'--stage',str(stage)],
                                          cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(f'{cid} 第 {stage} 步失败，查看 logs/{log.name}')
                if file_stamp(req['metadata_file'])!=plan['metadata_stamp']:raise ValueError('运行期间元数据发生变化，请创建新实验')
                for input_key,input_path in case['inputs'].items():
                    if file_stamp(input_path)!=case['stamps'][input_key]:raise ValueError(f'运行期间输入发生变化：{input_path}')
                if not all(x.is_file() and x.stat().st_size for x in outputs):raise RuntimeError(f'{cid} 第 {stage} 步未生成所需输出')
                for x in outputs:
                    if x.suffix=='.swc':check_swc(x)
                progress['completed'][key]=dict(seconds=round(time.time()-started,2),outputs=[file_stamp(x) for x in outputs])
                progress['failed'].pop(key,None);save_json(progress_path,progress)
                print(f'[{cid}] {stage:02d} 完成',flush=True)
        progress.update(status='complete',current=None)
    except Exception as exc:
        progress.update(status='failed',error=str(exc))
        if progress.get('current'):
            c=progress['current'];progress['failed'][f"{c['id']}:{c['stage']}"]=str(exc)
        save_json(progress_path,progress)
        raise
    save_json(progress_path,progress)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path);ap.add_argument('--spec',type=Path);ap.add_argument('--stage',type=int)
    args=ap.parse_args()
    if args.job:run_job(args.job.resolve())
    else:run_stage(args.spec,args.stage)
