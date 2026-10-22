#!/usr/bin/env python3
"""Rebuild and compare a fixed SWC cohort using the live checkout.

--check: resolve inputs; --run: rebuild APP2/G-Cut/resample/prune; --report:
compute all versions' metrics and MIPs. With no action, run all three phases.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('app2_trace', 'gcut', 'gcut_resample', 'prune')


def stages(cfg):
    mode=cfg.get('app2_input_mode','retrace')
    if mode=='retrace':return STAGES
    if mode=='archived_native':return ('import_app2', *STAGES[1:])
    raise ValueError('Unknown APP2 input mode: '+mode)


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def configuration(path):
    path = Path(path).resolve();cfg = json.loads(path.read_text())
    for key in ('cohort_csv','reference_root','output_root','pipeline_config','input_alias_root'):
        p = Path(cfg[key]).expanduser();cfg[key] = str(p if p.is_absolute() else (path.parent/p).resolve())
    cfg['config_path'] = str(path)
    if cfg.get('reuse_from'):
        p=Path(cfg['reuse_from']).expanduser();cfg['reuse_from']=str(p if p.is_absolute() else (path.parent/p).resolve())
    stages(cfg)  # Validate before preparing or writing outputs.
    for version in cfg.get('versions',[]):
        if 'swc_root' in version:
            p=Path(version['swc_root']).expanduser()
            version['swc_root']=str(p if p.is_absolute() else (path.parent/p).resolve())
    if not cfg.get('versions') or len({v['key'] for v in cfg['versions']}) != len(cfg['versions']):
        raise ValueError('Version keys must be nonempty and unique')
    return cfg


def pipeline_hashes():
    paths = list(ROOT.glob('*.py'))
    for folder in ['algorithms','utils','visualization','neuroutils']:
        paths.extend((ROOT/folder).rglob('*.py'))
    paths.extend((ROOT/'vendor/app2_gsdt').glob('*.h'))
    paths.append(ROOT/'vendor/app2_gsdt/wrapper.cpp')
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def resolve_input(reference, alias, cid, key, original):
    specs = dict(raw=(f'img/{cid}_0000.tif','1um_img',f'image_{cid}_0000.tif'),
                 mask=(f'original_masks/{cid}.tif','mask',f'image_{cid}.tif'),
                 crop=(f'soma_img/{cid}.json','soma_img',f'image_{cid}.json'),
                 soma=(f'soma_seg/{cid}.tif','soma_seg',f'image_{cid}.tif'))
    relative, kind, name = specs[key]
    candidates = [reference/relative]
    if key == 'mask':candidates.append(reference/'input_masks'/f'{cid}.tif')
    candidates.extend([Path(original.get({'raw':'img_path','mask':'mask_path'}.get(key,''),'/nonexistent')),
                       alias/kind/name])
    if key == 'raw':candidates.append(alias/'1um_img/img'/name)
    for candidate in candidates:
        if candidate.is_file():return candidate.resolve()
    raise FileNotFoundError(f'{cid}: {key}: '+', '.join(map(str,candidates)))


def normalize_crop(data, soma_shape):
    """Produce the original v0.81 writer's ZYX bounds without editing its source."""
    result = dict(data);bounds = list(data['bounds_1um_zxy']);order=data.get('bounds_order','')
    if len(bounds)!=6:raise ValueError('Crop must have six bounds')
    if order.startswith('ZYX'):source_order='ZYX'
    elif order.startswith('ZXY'):source_order='ZXY'
    elif order:raise ValueError(f'Unknown crop order: {order}')
    else:
        zxy=(bounds[1]-bounds[0],bounds[5]-bounds[4],bounds[3]-bounds[2])
        zyx=(bounds[1]-bounds[0],bounds[3]-bounds[2],bounds[5]-bounds[4])
        if tuple(soma_shape)==zyx and tuple(soma_shape)!=zxy:source_order='ZYX'
        else:source_order='ZXY'  # Historical bounds_1um_zxy writer's contract.
    result['bounds_1um_zxy']=[bounds[i] for i in (0,1,4,5,2,3)] if source_order=='ZXY' else bounds
    result['bounds_order']='ZYX'
    # Old nnUNet exports can be padded or one voxel smaller than the bounds.
    # The tracing readers place the actual array at its recorded origin and
    # clip it to the image. Preserve that behavior, without resizing the mask.
    result['comparison_source_bounds_order']=source_order
    return result,source_order


def link(source, destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_symlink() and destination.resolve()==source.resolve():return
    if destination.exists() or destination.is_symlink():raise ValueError(f'Conflicting input: {destination}')
    destination.symlink_to(source)


def prepare(cfg, lightweight=False):
    import tifffile
    out=Path(cfg['output_root']);reference=Path(cfg['reference_root']);alias=Path(cfg['input_alias_root'])
    with Path(cfg['cohort_csv']).open(newline='') as stream:cohort=list(csv.DictReader(stream))
    ids=[str(int(row['cell_id'])) for row in cohort]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Cohort IDs must be nonempty and unique')
    records=[];errors=[]
    for row,cid in zip(cohort,ids):
        try:
            inputs={k:resolve_input(reference,alias,cid,k,row) for k in ('raw','mask','crop','soma')}
            spec=json.loads((reference/'manifests'/f'{cid}.json').read_text())
            with tifffile.TiffFile(inputs['raw']) as f:shape=f.series[0].shape;dtype=str(f.series[0].dtype)
            with tifffile.TiffFile(inputs['mask']) as f:mask_shape=f.series[0].shape
            with tifffile.TiffFile(inputs['soma']) as f:soma_shape=f.series[0].shape
            if len(shape)!=3 or tuple(mask_shape)!=tuple(shape) or dtype!='uint8':raise ValueError('Expected matching uint8 3D image and mask')
            crop,order=normalize_crop(json.loads(inputs['crop'].read_text()),soma_shape)
            rec=dict(id=cid,inputs={k:str(p) for k,p in inputs.items()},metadata=spec['metadata'],
                     shape=list(shape),source_crop_order=order,crop_for_v081=crop)
            if cfg.get('app2_input_mode')=='archived_native':
                from reconstruction_comparison.metrics import read_swc
                from reconstruction_comparison.report import validate_topology
                source=Path(row['archived_app2_swc_path']).resolve()
                validate_topology(read_swc(source))
                inputs['archived_app2']=source;rec['inputs']['archived_app2']=str(source)
            if not lightweight:
                rec['input_sha256']={k:sha(p) for k,p in inputs.items()}
                rec['input_stat']={k:dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for k,p in inputs.items()}
                if row.get('archived_app2_sha256') and rec['input_sha256'].get('archived_app2')!=row['archived_app2_sha256']:
                    raise ValueError('Archived APP2 does not match the registered cohort hash')
            records.append(rec)
        except Exception as exc:errors.append(dict(id=cid,error=str(exc)))
    preflight=dict(requested=len(ids),ready=len(records),errors=errors,ids=ids)
    print(json.dumps({k:v for k,v in preflight.items() if k!='ids'},ensure_ascii=False),flush=True)
    if errors:raise RuntimeError(f'Input preflight failed for {len(errors)} IDs; no sample replacement')
    if lightweight:return preflight
    out.mkdir(parents=True,exist_ok=True)
    reconstruction=out/'reconstruction'
    cfg_pipeline=json.loads(Path(cfg['pipeline_config']).read_text())
    cfg_pipeline.update(output_root=str(reconstruction),neuron_ids=list(map(int,ids)),metadata_file=str(out/'cohort_metadata.csv'))
    cfg_pipeline['enabled_stages']={s:s in stages(cfg) for s in ('rescale','soma_crop','soma_infer','neurite_infer','merge',*STAGES)}
    cfg_pipeline.setdefault('gcut',{}).update(candidate_method='app2_gsdt')
    code=pipeline_hashes();cohort_sha=sha(cfg['cohort_csv'])
    if cfg.get('app2_input_mode')=='archived_native':
        name='tools/reconstruction_comparison/archived_app2.py';code[name]=sha(ROOT/name)
    input_identity={r['id']:dict(hashes=r['input_sha256'],metadata=r['metadata']) for r in records}
    identity=dict(code=code,config=cfg_pipeline,cohort=cohort_sha,inputs=input_identity)
    if cfg.get('app2_input_mode')=='archived_native':identity['stages']=stages(cfg)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    existing=out/'run_manifest.json'
    if existing.exists() and json.loads(existing.read_text())['fingerprint']!=fingerprint:
        raise ValueError('This output directory belongs to different code/configuration; choose a fresh output_root')
    for rec in records:
        cid=rec['id'];inputs={k:Path(p) for k,p in rec['inputs'].items()}
        for key,folder,name in [('raw','img',f'image_{cid}_0000.tif'),('mask','mask',f'image_{cid}.tif'),('soma','soma_seg',f'image_{cid}.tif')]:
            link(inputs[key],reconstruction/folder/name)
        save(reconstruction/'soma_img'/f'image_{cid}.json',rec['crop_for_v081'])
        save(out/'manifests'/f'{cid}.json',rec)
    with (out/'cohort_metadata.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['cell_id','soma_x','soma_y','soma_z','xy_resolution','z_resolution']);writer.writeheader()
        for rec in records:writer.writerow(dict(cell_id=rec['id'],**{k:rec['metadata'][k] for k in writer.fieldnames[1:]}))
    save(out/'pipeline_config.json',cfg_pipeline)
    save(out/'preflight.json',preflight)
    save(existing,dict(fingerprint=fingerprint,created_at=datetime.now().astimezone().isoformat(),
         project=str(ROOT),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         code=code,cohort_sha256=cohort_sha,reference_thread=cfg.get('reference_thread'),config=cfg,
         stages=stages(cfg),policy='live v0.82; original masks; no mask completion; '+
         ('archived APP2 imported with Y=H-1-Y only; no retracing or seed correction; ' if cfg.get('app2_input_mode')=='archived_native' else 'APP2 recomputed; ')+
         'G-Cut/resample/prune recomputed; fixed cohort, no replacement'))
    return records


def stage_worker(cfg,cid,stage):
    if stage not in stages(cfg):raise ValueError(f'{stage} is disabled for this input mode')
    out=Path(cfg['output_root']);rec=json.loads((out/'manifests'/f'{cid}.json').read_text())
    if stage=='import_app2':
        from reconstruction_comparison.archived_app2 import convert
        source=Path(rec['inputs']['archived_app2'])
        if sha(source)!=rec['input_sha256']['archived_app2']:raise ValueError('Archived APP2 changed since preflight')
        convert(source,out/'reconstruction/trace_app2/down_sampled_swcs_app2'/f'image_{cid}.swc',int(rec['shape'][1]))
        return
    os.environ['NEURON_V082_CONFIG']=str(out/'pipeline_config.json')
    sys.path.insert(0,str(ROOT))
    import pandas as pd
    import run_pipeline as p
    p.ensure_dirs()
    meta=pd.DataFrame([rec['metadata']],index=[int(cid)])
    functions=dict(app2_trace=lambda:p.run_app2_trace(cid),gcut=lambda:p.run_gcut(cid,meta),
                   gcut_resample=lambda:p.run_gcut_resample(cid),prune=lambda:p.run_prune(cid,meta))
    if functions[stage]() is False:raise RuntimeError(f'{cid}: {stage} returned failure')


def reconstruct(cfg,records):
    out=Path(cfg['output_root']);fingerprint=json.loads((out/'run_manifest.json').read_text())['fingerprint']
    gate=threading.Semaphore(int(cfg.get('gcut_workers',2)))
    def case(rec):
        cid=rec['id'];path=out/'status'/f'{cid}.json'
        previous=json.loads(path.read_text()) if path.exists() else {}
        if previous.get('fingerprint') not in (None,fingerprint):raise ValueError(f'{cid}: checkpoint fingerprint mismatch')
        if previous.get('state')=='complete' or (previous.get('state')=='failed' and previous.get('reused_terminal')):return previous
        state=dict(id=cid,fingerprint=fingerprint,state='running',completed_stages=previous.get('completed_stages',[]),seconds=previous.get('seconds',{}))
        started=time.monotonic()
        try:
            for stage in stages(cfg):
                if stage in state['completed_stages']:continue
                state['current_stage']=stage;save(path,state)
                stage_start=time.monotonic();log=out/'logs'/cid/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMPY_MADVISE_HUGEPAGE='0',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
                if stage=='gcut':gate.acquire()
                try:
                    with log.open('w') as f:
                        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--config',cfg['config_path'],
                                        '--case',cid,'--stage',stage],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,
                                       timeout=int(cfg.get('stage_timeout_seconds',1800)),check=True)
                finally:
                    if stage=='gcut':gate.release()
                state['seconds'][stage]=round(time.monotonic()-stage_start,3)
                state['completed_stages'].append(stage);save(path,state)
            state.update(state='complete',elapsed_seconds=round(time.monotonic()-started,3))
        except Exception:
            state.update(state='failed',error=traceback.format_exc(),elapsed_seconds=round(time.monotonic()-started,3))
        save(path,state);print(f"{cid}: {state['state']} ({state['elapsed_seconds']:.1f}s)",flush=True)
        return state
    results=[];started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(cfg.get('workers',6))) as pool:
        futures=[pool.submit(case,r) for r in records]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result());save(out/'reconstruction_summary.json',dict(planned=len(records),finished=len(results),
                  succeeded=sum(r['state']=='complete' for r in results),failed=sum(r['state']=='failed' for r in results),seconds=round(time.monotonic()-started,3),records=results))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'comparison/original200.json')
    parser.add_argument('--check',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--report',action='store_true')
    parser.add_argument('--render-only',action='store_true',help='Redraw figures from completed metrics without recomputing PR')
    parser.add_argument('--refresh-structure',action='store_true',help='Refresh structural metrics from unchanged SWCs; reuse cached PR and existing MIPs')
    parser.add_argument('--verify',action='store_true',help='Verify saved metrics, cohort, coordinates, figures and historical compatibility')
    parser.add_argument('--case');parser.add_argument('--stage',choices=(*STAGES,'import_app2'))
    args=parser.parse_args();cfg=configuration(args.config)
    if args.case:
        if not args.stage:parser.error('--case requires --stage')
        stage_worker(cfg,args.case,args.stage);return
    if args.check:prepare(cfg,lightweight=True);return
    if args.refresh_structure:
        from reconstruction_comparison.report import refresh_structure
        from reconstruction_comparison.audit import verify
        refresh_structure(cfg);verify(cfg);return
    if args.verify:
        from reconstruction_comparison.audit import verify
        verify(cfg);return
    if args.render_only:
        from reconstruction_comparison.report import render_existing
        render_existing(cfg);return
    default=not(args.run or args.report)
    if args.run or default:
        import fcntl
        out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True)
        with (out/'.run.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            records=prepare(cfg)
            from reconstruction_comparison.reuse import adopt
            adopt(cfg,records)
            reconstruct(cfg,records)
    if args.report or default:
        from reconstruction_comparison.report import build
        build(cfg)
        from reconstruction_comparison.audit import verify
        verify(cfg)

if __name__=='__main__':main()
