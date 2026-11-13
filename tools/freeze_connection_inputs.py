#!/usr/bin/env python3
"""Copy a v0.83b cell's experiment inputs and evidence without overwriting data."""
from pathlib import Path
import argparse,hashlib,json,re,shutil

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def freeze(cid,source,data):
    manifest_path=source/'manifests'/f'{cid}.json'
    manifest=json.loads(manifest_path.read_text())
    if str(manifest['id'])!=cid:raise ValueError('Source manifest cell mismatch')
    status=json.loads((source/'status'/f'{cid}.json').read_text())
    if status.get('status')!='complete':raise ValueError('Source cell run is not complete')
    pairs=[]
    for key,folder,suffix in [('raw','raw','tif'),('mask','initial_mask','tif'),
                              ('soma_json','soma_img','json'),('soma_mask','soma_seg','tif')]:
        pairs.append((Path(manifest[key]),Path('inputs')/folder/f'{cid}.{suffix}'))
    required=['mask/'+cid+'.tif','gcut_output/selected_resampled_step1/'+cid+'.swc',
              'pruning_swcs/'+cid+'.swc']
    for rel in required:
        if not (source/rel).is_file():raise FileNotFoundError(source/rel)
    folders=['mask','mask_original','trace_app2/down_sampled_swcs_app2',
             'gcut_output/selected_for_pruning','gcut_output/selected_resampled_step1',
             'pruning_swcs','app2_input','comparisons','pruning_mips','gcut_mips',
             'mask_mips','status','manifests','regression_guard']
    pattern=re.compile(re.escape(cid)+r'(?:[_.-]|$)')
    for folder in folders:
        directory=source/folder
        if not directory.is_dir():continue
        for p in sorted(directory.iterdir()):
            if pattern.match(p.name) and p.is_file():pairs.append((p,Path('source_v083b')/p.relative_to(source)))
    logs=source/'logs'/cid
    if logs.exists():
        pairs.extend((p,Path('source_v083b')/p.relative_to(source)) for p in sorted(logs.rglob('*')) if p.is_file())
    pairs.append((source/f'pruning_swcs/{cid}.swc',Path(f'references/v0.83b/{cid}.swc')))
    old=Path(f'/data/disk/v0.81/swc/{cid}.swc')
    if old.is_file():pairs.append((old,Path(f'references/v0.81/{cid}.swc')))
    pairs.append((source/'run-config.json',Path(f'provenance/{cid}/run-config.json')))
    records=[]
    for src,relative in pairs:
        before=src.stat();sha=digest(src);dst=data/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if digest(dst)!=sha:raise FileExistsError(f'Existing frozen file differs: {dst}')
        else:shutil.copy2(src,dst)
        after=src.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError(f'Source changed during copy: {src}')
        if digest(dst)!=sha:raise RuntimeError(f'Checksum mismatch: {dst}')
        records.append(dict(source=str(src),resolved_source=str(src.resolve()),destination=str(relative),bytes=before.st_size,sha256=sha))
    meta=data/'metadata'/f'{cid}.json';meta.parent.mkdir(parents=True,exist_ok=True)
    if meta.exists() and json.loads(meta.read_text())!=manifest['metadata']:raise FileExistsError(meta)
    meta.write_text(json.dumps(manifest['metadata'],indent=2)+'\n')
    stamp_matches={k:list(v[1:])==[Path(manifest[k]).stat().st_size,Path(manifest[k]).stat().st_mtime_ns]
                   for k,v in manifest.get('input_stamps',{}).items()}
    result=dict(cell=cid,source_root=str(source),source_manifest=str(manifest_path),
                source_fingerprint=manifest['fingerprint'],source_input_stamps_match=stamp_matches,
                metadata_file=str(meta.relative_to(data)),metadata_sha256=digest(meta),files=records,
                count=len(records),total_bytes=sum(r['bytes'] for r in records))
    target=data/f'copy-manifest-{cid}.json'
    if target.exists() and json.loads(target.read_text())!=result:raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('cells',nargs='+');args=ap.parse_args()
    for cid in args.cells:
        if not cid.isdigit():ap.error('Cell IDs must be numeric')
        freeze(cid,Path('/data/disk/C6.0/v0.83b'),ROOT.parent/'test/0.84/data')
