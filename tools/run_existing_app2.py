"""Prepare historical APP2 inputs, then run the live v0.82 downstream pipeline."""
from pathlib import Path
import argparse,csv,fcntl,hashlib,json,os,subprocess,sys,time,math
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]

def save(path,value):
    path=Path(path);temporary=path.with_suffix('.writing.json')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_csv(path,rows,fields):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def check_swc_content(path):
    """Reject empty/header-only SWCs before preparation; conversion checks all rows."""
    try:
        with Path(path).open(encoding='utf-8',errors='strict') as stream:
            for number,line in enumerate(stream,1):
                line=line.strip()
                if not line:continue
                if line.startswith('#'):
                    if 'app2_native_to_image' in line or 'y_axis_flipped_to_image' in line:
                        return 'already_converted_swc','输入已转换过 Y 坐标，不能再次翻转'
                    continue
                parts=line.split()
                if len(parts)<7:return 'invalid_swc',f'第 {number} 行不足 7 列'
                try:
                    values=[float(x) for x in parts[:7]]
                except ValueError:return 'invalid_swc',f'第 {number} 行包含非数字字段'
                if not all(math.isfinite(x) for x in values):return 'invalid_swc',f'第 {number} 行包含非有限数值'
                if any(not values[i].is_integer() for i in (0,1,6)):
                    return 'invalid_swc',f'第 {number} 行节点编号、类型或父节点不是整数'
                return None,''
        return 'empty_swc','SWC 没有有效节点（空文件或只有注释）'
    except (OSError,UnicodeError) as exc:return 'unreadable_swc',str(exc)

def link(source,destination):
    source=Path(source).resolve();destination=Path(destination)
    if destination.is_symlink() and destination.resolve()==source:return
    if destination.exists() or destination.is_symlink():raise ValueError(f'路径冲突：{destination}')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.symlink_to(source,target_is_directory=source.is_dir())

def prepared_inputs_unchanged(cid,sources,out):
    """Reuse previously committed inputs only while every source remains older."""
    imported=out/'trace_app2/down_sampled_swcs_app2'/f'image_{cid}.swc'
    crop=out/'soma_img'/f'image_{cid}.json'
    try:
        target_stats=[p.stat() for p in (imported,crop)]
        if any(s.st_size==0 for s in target_stats):return False
        prepared_at=min(s.st_mtime_ns for s in target_stats)
        for key,name in [('app2_dir',f'image_{cid}.swc'),('raw_dir',f'image_{cid}_0000.tif'),
                         ('mask_dir',f'image_{cid}.tif'),('crop_dir',f'image_{cid}.json'),
                         ('soma_dir',f'image_{cid}.tif')]:
            source=(sources[key]/name).stat()
            if source.st_size==0 or max(source.st_mtime_ns,source.st_ctime_ns)>prepared_at:return False
        return True
    except OSError:return False

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.existing_app2.local.json')
    parser.add_argument('--ids',nargs='+',type=int,help='仅处理这些编号；不指定时使用配置中的 neuron_ids，空列表代表全部历史 APP2')
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--check',action='store_true',help='只检查文件和元数据，不转换输入或运行算法')
    actions.add_argument('--prepare-only',action='store_true',help='准备输入，不运行 G-Cut')
    args=parser.parse_args()
    config_path=args.config.expanduser().resolve()
    cfg=json.loads(config_path.read_text())
    def configured_path(key,default=None):
        path=Path(cfg.get(key,default)).expanduser()
        return path.resolve() if path.is_absolute() else (config_path.parent/path).resolve()
    out=configured_path('output_root');out.mkdir(parents=True,exist_ok=True)
    code=configured_path('code_root',str(ROOT))
    # Reuse current import/crop adapters. Algorithm code is never copied.
    sys.path.insert(0,str(code/'tools'))
    from compare_reconstructions import normalize_crop,pipeline_hashes
    from reconstruction_comparison.archived_app2 import convert
    import numpy as np
    import pandas as pd
    import tifffile
    sources={k:configured_path(k) for k in ['app2_dir','raw_dir','mask_dir','crop_dir','soma_dir','metadata_file']}
    names={k:{p.name for p in sources[k].iterdir()} for k in ['app2_dir','raw_dir','mask_dir','crop_dir','soma_dir']}
    available={int(name[6:-4]) for name in names['app2_dir'] if name.startswith('image_') and name.endswith('.swc') and name[6:-4].isdigit()}
    ids=sorted(set(args.ids if args.ids is not None else (cfg.get('neuron_ids') or available)))
    if not ids:raise ValueError('没有待处理编号')
    columns=['cell_id','soma_x','soma_y','soma_z','xy_resolution','z_resolution']
    for encoding in ['utf-8-sig','gb18030','gbk']:
        try:meta=pd.read_csv(sources['metadata_file'],encoding=encoding,usecols=columns,low_memory=False);break
        except UnicodeDecodeError:continue
    else:raise ValueError('无法解码元数据 CSV')
    meta=meta.apply(pd.to_numeric,errors='coerce')
    meta=meta[meta['cell_id'].isin(ids)]
    duplicate=set(meta.loc[meta.cell_id.duplicated(False),'cell_id'].astype(int))
    meta=meta.drop_duplicates('cell_id').set_index('cell_id')
    ready=[];missing=[]
    for cid in ids:
        required=dict(app2_dir=f'image_{cid}.swc',raw_dir=f'image_{cid}_0000.tif',mask_dir=f'image_{cid}.tif',crop_dir=f'image_{cid}.json',soma_dir=f'image_{cid}.tif')
        absent=[key for key,name in required.items() if name not in names[key]]
        if cid not in meta.index:absent.append('metadata')
        elif cid in duplicate:absent.append('duplicate_metadata')
        elif not np.isfinite(meta.loc[cid].to_numpy(dtype=float)).all() or (meta.loc[cid,['xy_resolution','z_resolution']]<=0).any():absent.append('invalid_metadata')
        if absent:missing.append(dict(id=cid,reason=','.join(absent)))
        else:ready.append(cid)
    # Existing files can still be header-only APP2 failures. Cache only this
    # inexpensive first-node check; convert() still validates every input row.
    cache_path=out/'app2_content_cache.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    invalid=[];checked=[];cache_hits=0
    def inspect(cid):
        path=sources['app2_dir']/f'image_{cid}.swc'
        try:
            stat=path.stat();stamp=[str(path),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
        except OSError as exc:return cid,None,'unreadable_swc',str(exc),False
        previous=cache.get(str(cid),{})
        if previous.get('stamp')==stamp:
            return cid,stamp,previous['category'],previous['reason'],True
        category,reason=check_swc_content(path)
        return cid,stamp,category,reason,False
    print(f'检查 APP2 内容：{len(ready)} 份文件',flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(cid,stamp,category,reason,cached) in enumerate(pool.map(inspect,ready),1):
            cache_hits+=int(cached)
            if stamp is not None:cache[str(cid)]=dict(stamp=stamp,category=category,reason=reason)
            if category:invalid.append(dict(id=cid,category=category,reason=reason,path=str(sources['app2_dir']/f'image_{cid}.swc')))
            else:checked.append(cid)
            if i%5000==0:print(f'APP2 内容检查 {i}/{len(ready)}，无效 {len(invalid)} 例',flush=True)
    save(cache_path,cache);ready=checked
    write_csv(out/'invalid_app2.csv',invalid,['id','category','reason','path'])
    preflight=dict(requested=len(ids),ready=len(ready),missing=len(missing),invalid_app2=len(invalid),content_cache_hits=cache_hits,metadata_encoding=encoding,ids=ids)
    save(out/'preflight.json',preflight)
    with (out/'missing_inputs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','reason']);writer.writeheader();writer.writerows(missing)
    print(f"输入检查：请求 {len(ids)} 例，通过预检查 {len(ready)} 例，无效 APP2 {len(invalid)} 例，缺失或元数据异常 {len(missing)} 例。",flush=True)
    if invalid:print('无效 APP2 已记录到 invalid_app2.csv；运行时将跳过这些样本，其余继续。',flush=True)
    if missing:print('缺失输入已记录到 missing_inputs.csv；运行时将跳过这些样本，其余继续。',flush=True)
    if args.check:return 1 if missing or invalid else 0
    if not ready:raise ValueError('没有通过输入预检查的样本，详见输入检查 CSV')
    with (out/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gc=dict(target_percentile=55.0,gsdt_threshold_x=5.0,candidate_method='app2_gsdt',candidate_backend=cfg.get('candidate_backend','cuda'),gpu_lock_path=str(configured_path('gpu_lock_path',str(out/'.gpu-stage.lock'))),candidate_evidence={'enabled':True})
        identity=dict(code=pipeline_hashes(),sources={k:str(v) for k,v in sources.items()},metadata_sha256=sha(sources['metadata_file']),gcut=gc)
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        manifest_path=out/'input_run_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text())['fingerprint']!=fingerprint:
            raise ValueError('代码或输入配置发生变化，已有结果不能混用；请为新运行设置新的 output_root')
        current_manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        current_manifest.update(fingerprint=fingerprint,identity=identity,code_root=str(code),policy='live v0.82; archived APP2 Y reflection only; original masks; no mask completion')
        save(manifest_path,current_manifest)
        for key,folder in [('raw_dir','img'),('mask_dir','mask'),('soma_dir','soma_seg')]:link(sources[key],out/folder)
        (out/'soma_img').mkdir(exist_ok=True)
        previous_preparation=out/'input_preparation_summary.json'
        prior=set(json.loads(previous_preparation.read_text()).get('prepared_ids',[])) if previous_preparation.exists() else set()
        started=time.monotonic();prepared=[];preparation_failures=[];reused=0
        fields=['id','category','reason','path']
        write_csv(out/'input_preparation_failures.csv',[],fields)
        for i,cid in enumerate(ready,1):
            source=sources['app2_dir']/f'image_{cid}.swc'
            if cid in prior and prepared_inputs_unchanged(cid,sources,out):
                prepared.append(cid);reused+=1
                if i==1 or i%1000==0 or i==len(ready):print(f'[断点续跑] 输入 {i}/{len(ready)}，复用未变化输入 {reused} 例，累计 {time.monotonic()-started:.1f} 秒',flush=True)
                continue
            try:
                with tifffile.TiffFile(sources['raw_dir']/f'image_{cid}_0000.tif') as f:shape=f.series[0].shape
                with tifffile.TiffFile(sources['mask_dir']/f'image_{cid}.tif') as f:mask_shape=f.series[0].shape
                with tifffile.TiffFile(sources['soma_dir']/f'image_{cid}.tif') as f:soma_shape=f.series[0].shape
                if len(shape)!=3 or tuple(mask_shape)!=tuple(shape):raise ValueError(f'{cid}: 原图必须为三维且与 mask 形状一致')
                crop,_=normalize_crop(json.loads((sources['crop_dir']/f'image_{cid}.json').read_text()),soma_shape)
                convert(source,out/'trace_app2/down_sampled_swcs_app2'/f'image_{cid}.swc',int(shape[1]))
                save(out/'soma_img'/f'image_{cid}.json',crop)
            except (ValueError,KeyError,IndexError,FileNotFoundError,UnicodeError,tifffile.TiffFileError) as exc:
                # An output conflict indicates incompatible cached results, not
                # a bad sample. Never hide it by skipping that sample.
                if 'Conflicting imported APP2 output' in str(exc) or 'Archived input must never be overwritten' in str(exc):raise
                failure=dict(id=cid,category=type(exc).__name__,reason=str(exc),path=str(source))
                preparation_failures.append(failure)
                with (out/'input_preparation_failures.csv').open('a',newline='') as f:
                    csv.DictWriter(f,fieldnames=fields).writerow(failure)
                print(f'[跳过输入] {cid}: {type(exc).__name__}: {exc}',flush=True)
            else:prepared.append(cid)
            if i==1 or i%100==0 or i==len(ready):print(f'准备输入 {i}/{len(ready)}，成功 {len(prepared)}，失败 {len(preparation_failures)}，当前 {cid}，累计 {time.monotonic()-started:.1f} 秒',flush=True)
        save(out/'input_preparation_summary.json',dict(requested=len(ids),preflight_ready=len(ready),prepared=len(prepared),reused_inputs=reused,invalid_app2=len(invalid),missing_inputs=len(missing),preparation_failed=len(preparation_failures),prepared_ids=prepared,preparation_failed_ids=[r['id'] for r in preparation_failures]))
        if not prepared:raise ValueError('没有成功准备输入的样本，详见 input_preparation_failures.csv')
        meta.loc[prepared].to_csv(out/'cohort_metadata.csv',index_label='cell_id',encoding='utf-8')
        pipeline=dict(neuron_ids=prepared,output_root=str(out),metadata_file=str(out/'cohort_metadata.csv'),vaa3d_resample=str(configured_path('vaa3d_resample')),pipeline={'mode':'stage-batch','stage_max_tasks':cfg.get('workers',4),'gpu_max_tasks':1,'infer_batch_size':1},gcut=gc,
                      enabled_stages={key:key in ['gcut','gcut_resample','prune'] for key in ['rescale','soma_crop','soma_infer','neurite_infer','merge','app2_trace','gcut','gcut_resample','prune']})
        save(out/'pipeline_config.json',pipeline)
        if args.prepare_only:return 0
        command=[sys.executable,'-u',str(code/'run_pipeline.py'),'--config',str(out/'pipeline_config.json'),'--mode','stage-batch','--stage-max-tasks',str(cfg.get('workers',4))]
        print('开始运行：'+' '.join(command),flush=True)
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
        result=subprocess.run(command,cwd=code,env=env)
        final=out/'gcut_pruned/pruning_swcs'
        completed=[cid for cid in prepared if (final/f'image_{cid}.swc').is_file() and (final/f'image_{cid}.swc').stat().st_size>0]
        failed=sorted(set(prepared)-set(completed))
        skipped=sorted(set(ids)-set(prepared))
        save(out/'run_summary.json',dict(requested=len(ids),runnable=len(prepared),skipped_inputs=len(skipped),skipped_input_ids=skipped,completed=len(completed),missing_final=len(failed),missing_final_ids=failed,pipeline_returncode=result.returncode))
        print(f'最终 SWC：{len(completed)}/{len(prepared)}；输入跳过 {len(skipped)} 例；算法未生成最终结果 {len(failed)} 例，见 run_summary.json。',flush=True)
        if pipeline_hashes()!=identity['code']:raise RuntimeError('运行期间代码发生变化，结果需要重新核对')
        return result.returncode or (1 if failed else 0)

if __name__=='__main__':sys.exit(main())
