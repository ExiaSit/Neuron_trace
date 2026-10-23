"""Reuse only finished upstream checkpoints whose original inputs still match."""
from pathlib import Path
import json,hashlib,os,shutil
from functools import lru_cache

@lru_cache(maxsize=8)
def baseline_code(root):
    r=Path(root);runtime=json.loads((r/'runtime.json').read_text());source=Path(runtime['source'])
    files=[source/'batch_v083.py',source/'incremental_worker.py',source/'lean_outputs.py',source/'run.py',*sorted((source/'algorithm').rglob('*.py')),*sorted((source/'trace_algorithm').rglob('*.py'))]
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    if digest!=runtime['sha256']:raise ValueError('Baseline source no longer matches recorded runtime')
    return digest

def copy_output(source,dest,link=False):
    source=Path(source);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name('.'+dest.name+'.migration')
    if tmp.exists():tmp.unlink()
    method='copy'
    try:
        if link:
            try:os.link(source,tmp);method='hardlink'
            except OSError:shutil.copy2(source,tmp)
        else:shutil.copy2(source,tmp)
        os.replace(tmp,dest)
    finally:
        if tmp.exists():tmp.unlink()
    return method

def seed(spec,stamps,state_path,save_json,check_swc):
    cfg=spec['config'];baseline=cfg.get('baseline_root')
    if not baseline:return
    base=Path(baseline);root=Path(cfg['output_root']);cid=spec['id'];path=base/'status'/f'{cid}.json';manifest=base/'manifests'/f'{cid}.json'
    if not path.is_file() or not manifest.is_file():return
    state=json.loads(path.read_text());old=json.loads(manifest.read_text());code=baseline_code(str(base))
    expected=hashlib.sha256(json.dumps(dict(inputs=stamps,metadata=spec['metadata'],code=code,tools={k:cfg[k] for k in ['vaa3d_app2','vaa3d_resample','python_executable']}),sort_keys=True).encode()).hexdigest()
    if state.get('fingerprint')!=expected or old.get('fingerprint')!=expected:
        save_json(root/'migration'/f'{cid}.json',dict(id=cid,reused=False,reason='baseline_inputs_or_runtime_differ'));return
    if state.get('status')=='complete' and (base/'pruning_swcs'/f'{cid}.swc').is_file():spec['guard_baseline_root']=str(base)
    if Path(state_path).exists():return
    stages=state.get('completed_stages',[]);mask=base/'mask'/f'{cid}.tif';continuity=base/'logs'/cid/'continuity.json'
    if 'continuity' not in stages or not mask.is_file() or not continuity.is_file():return
    reused=['mask_'+x for x in 'abcdef']+['continuity'];files=[]
    for source,dest,link in [(mask,root/'mask'/f'{cid}.tif',True),(continuity,root/'logs'/cid/'continuity.json',False)]:
        files.append(dict(source=str(source),destination=str(dest),mode=copy_output(source,dest,link)))
    app2=base/'trace_app2/down_sampled_swcs_app2'/f'{cid}.swc'
    if 'app2' in stages and app2.is_file():
        check_swc(app2);dest=root/'trace_app2/down_sampled_swcs_app2'/f'{cid}.swc';files.append(dict(source=str(app2),destination=str(dest),mode=copy_output(app2,dest,True)));reused.append('app2')
    save_json(root/'migration'/f'{cid}.json',dict(id=cid,reused=True,baseline_fingerprint=expected,baseline_root=str(base),completed_stages=reused,files=files))
    save_json(Path(state_path),dict(id=cid,status='queued',fingerprint=spec['fingerprint'],mask_saved=True,completed_stages=reused,seconds={},method_revision=cfg['method_revision']))
