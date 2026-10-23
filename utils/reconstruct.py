"""Full soma-core revision with v0.81 comparison and validated upstream reuse."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parents[1]
from utils import reconstruction_batch as engine
from utils import case_worker as worker


def previous_index(folders):
    """Prefer the first dataset containing an ID; use recovery as fallback."""
    import re
    index = {}
    for folder in folders:
        local = {}
        for p in sorted(Path(folder).rglob('*.swc')):
            match = re.match(r'^(?:image_)?(\d+)(?:_|$)', p.stem)
            if match:
                local.setdefault(str(int(match[1])), []).append(str(p))
        for cid, files in local.items():
            index.setdefault(cid, files)
    return index


def comparison_key(spec):
    p=worker.paths(spec)
    from visualization.previous_app2 import resolve_previous_app2
    previous_app2=resolve_previous_app2(spec)
    return hashlib.sha256(json.dumps(dict(
        style='mask-history-previous-app2-2',
        previous_app2=engine.file_stamp(previous_app2) if previous_app2 else None,
        previous_app2_coordinates=spec['config'].get('previous_app2_coordinates','native'),
        app2=engine.file_stamp(p['app2']),old=[engine.file_stamp(x) for x in spec['old_swcs']],
        inputs=[engine.file_stamp(x) for x in [spec['raw'],spec['mask'],p['mask'],p['final']]],
        metadata=spec['metadata']),sort_keys=True).encode()).hexdigest()


def render(spec_path, sidecar, key):
    spec=json.loads(Path(spec_path).read_text()); output=worker.paths(spec)['comparison']
    audit=Path(spec['config']['output_root'])/'regression_guard'/f"{spec['id']}.json"
    if audit.is_file():
        decision=json.loads(audit.read_text()).get('decision')
        spec['selection_reason']='soma-core revision | '+str(decision)
    # Validate before replacing an existing image; malformed references remain errors.
    for old in spec['old_swcs']:
        worker.check_swc(Path(old))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    original_title=Axes.set_title
    original_text=Axes.text
    def title(self,label,*args,**kwargs):
        return original_title(self,'v0.81 SWC' if label=='Previous SWC' else label,*args,**kwargs)
    def text(self,x,y,s,*args,**kwargs):
        return original_text(self,x,y,'No matching v0.81 SWC' if s=='Previous SWC not supplied' else s,*args,**kwargs)
    Axes.set_title=title;Axes.text=text
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.comparison-',dir=output.parent) as temp:
        spec['config']['comparison_dir']=temp
        worker.comparison(spec)
        os.replace(Path(temp)/output.name,output)
    worker.save_json(Path(sidecar),dict(key=key,version='v0.81',old_swcs=spec['old_swcs'],status='present' if spec['old_swcs'] else 'missing'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--refresh-only',action='store_true')
    args,remaining=ap.parse_known_args()
    cfg=json.loads(Path(args.config).read_text());root=Path(cfg['output_root'])
    # Preserve input identity across a directory-only move. Size and mtime
    # remain part of the frozen fingerprint, so changed inputs still fail closed.
    aliases=cfg.get('input_path_aliases', {})
    original_stamp=engine.file_stamp
    def stamp(path):
        value=original_stamp(path)
        for current,previous in aliases.items():
            try:
                relative=Path(value[0]).relative_to(Path(current).resolve())
            except ValueError:
                continue
            value[0]=str(Path(previous)/relative)
            break
        return value
    engine.file_stamp=stamp
    index=previous_index(cfg['previous_swc_dirs'])
    registry_path=cfg.get('validated_baseline_registry')
    registry=json.loads(Path(registry_path).read_text()) if registry_path else {}
    def validated_reference(spec):
        record=registry.get(spec['id'])
        if not record or record['metadata']!=spec['metadata']:return None
        for key,digest in record['input_sha256'].items():
            h=hashlib.sha256()
            with Path(spec[key]).open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            if h.hexdigest()!=digest:return None
        reference=cfg.get('baseline_path_overrides',{}).get(record['root'],record['root'])
        if not (Path(reference)/'pruning_swcs'/f"{spec['id']}.swc").is_file():
            raise FileNotFoundError(f"Validated baseline missing: {reference}; restore it or configure baseline_path_overrides")
        return reference
    original_save=engine.save_json
    def save(path,data):
        if Path(path).parent==root/'manifests':
            data['old_swcs']=index.get(data['id'],[])
            reference=validated_reference(data)
            if reference:data['guard_baseline_root']=reference
        return original_save(path,data)
    engine.save_json=save
    def refresh(cid):
        manifest=root/'manifests'/f'{cid}.json'
        spec=json.loads(manifest.read_text());spec['old_swcs']=index.get(cid,[])
        for field in ['raw','mask','soma_json','soma_mask']:
            if field not in spec or Path(spec[field]).exists():
                continue
            for current,previous in aliases.items():
                try:
                    relocated=Path(current)/Path(spec[field]).relative_to(previous)
                except ValueError:
                    continue
                if relocated.is_file():
                    spec[field]=str(relocated)
                    break
        original_save(manifest,spec)
        sidecar=root/'comparison_sources'/f'{cid}.json'
        key=comparison_key(spec)
        if sidecar.exists() and json.loads(sidecar.read_text()).get('key')==key and worker.paths(spec)['comparison'].is_file():
            return
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
        log=root/'logs'/f'{cid}-comparison-v081.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as stream:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--render',str(manifest),str(sidecar),key],
                           env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(f'[{cid}] 比较图已加载 {len(spec["old_swcs"])} 个 v0.81 SWC'+('（该 ID 无旧结果）' if not spec['old_swcs'] else ''),flush=True)
    print(f'已索引 {len(index)} 个 ID 的 v0.81 结果；全量采用 ring-guard-20261016；重新计算 mask，禁用旧延伸结果复用；历史已验证样本启用内容核验回归保护。',flush=True)
    if args.refresh_only:
        import fcntl
        lock=(root/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ids=[p.stem for p in (root/'status').glob('*.json') if json.loads(p.read_text()).get('status')=='complete']
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(refresh,ids))
        original_save(root/'comparison-refresh.json',dict(completed=len(ids),ids=ids,source_dirs=cfg['previous_swc_dirs']))
        return 0
    original_group=engine.batch.process_group
    def group(*a,**kw):
        original_process=engine.batch.process
        def process(pair,config,stop):
            result=original_process(pair,config,stop)
            cid=pair[2].stem;state=root/'status'/f'{cid}.json'
            if state.exists() and json.loads(state.read_text()).get('status')=='complete':
                refresh(cid)
            return result
        engine.batch.process=process
        return original_group(*a,**kw)
    engine.batch.process_group=group
    sys.argv=[sys.argv[0],'--config',args.config,*remaining]
    return engine.main()


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--render':
        render(*sys.argv[2:])
    else:
        sys.exit(main())
