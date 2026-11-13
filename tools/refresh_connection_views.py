#!/usr/bin/env python3
"""Regenerate display-only artifacts from saved connection diagnostics."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import bootstrap
import tifffile
from visualization.projection_viewer import write_site_viewer,write_viewer_index,render_projection_comparison
from algorithms.tracing.connection_diagnostics import save_json


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-run');ap.add_argument('--name');args=ap.parse_args()
    experiment=ROOT.parent/'test/0.84'
    source=Path(args.source_run or (experiment/'latest-result.txt').read_text().strip()).resolve()
    run=json.loads((source/'run.json').read_text());cid=run['cell']
    src=source/f'mips/{cid}_connections';report=json.loads((src/'diagnosis.json').read_text())
    watched=[src/'diagnosis.json',src/'cuts.json',source/f'swc/{cid}.swc']
    hashes={str(p):digest(p) for p in watched}
    raw_path=Path(run['inputs']['raw']);raw=tifffile.imread(raw_path)
    name=args.name or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+f'_{cid}_projection_viewer'
    if Path(name).name!=name:ap.error('name must be a single directory name')
    out=experiment/'runs'/name;out.mkdir(parents=True,exist_ok=False)
    for file in ['diagnosis.json','cuts.json','neighbor_soma_cuts.json']:
        if (src/file).exists():shutil.copy2(src/file,out/file)
    selected=[sid for sid in ['004','011','031','071'] if any(s['site_id']==sid and s.get('image') for s in report['sites'])]
    for n,site in enumerate(report['sites'],1):
        if not site.get('image'):continue
        box=site['bbox_xyz'];lo=[b[0] for b in box];hi=[b[1] for b in box]
        crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
        shutil.copy2(src/site['image'],out/site['image'])
        write_site_viewer(crop,lo,site,out/f"site_{site['site_id']}.html")
        if site['site_id'] in selected:
            render_projection_comparison(crop,lo,site,out/f"display_comparison_{site['site_id']}.png")
        if n%25==0:print(f'{n}/{len(report["sites"])} sites exported',flush=True)
    write_viewer_index(report,out,selected);shutil.copy2(out/'interactive.html',out/'index.html')
    snapshots={}
    for rel in ['visualization/projection_viewer.py','visualization/assets/projection_viewer.html',
                'visualization/connection_diagnostics.py','tools/refresh_connection_views.py']:
        target=out/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
        snapshots[rel]=digest(ROOT/rel)
    assert all(digest(p)==sha for p,sha in hashes.items()),'Source diagnosis/SWC changed during display export'
    save_json(out/'display-manifest.json',dict(cell=cid,source_run=str(source),raw_path=str(raw_path),
        raw_sha256=digest(raw_path),source_hashes=hashes,source_unchanged=True,source_sha256=snapshots,
        sites=len(report['sites']),mode='display_only',default_mode='5-um slab MIP',
        available_thickness_um=[3,5,9],voxel_spacing_um=[1,1,1],
        window='5th/99.5th raw-crop percentiles, shared across views and fixed while scrolling; gamma .85',
        overlay='SWC segments clipped to selected slab; off by default'))
    (experiment/'latest-visualization.txt').write_text(str(out)+'\n')
    print(json.dumps(dict(output=str(out),sites=len(report['sites']),source_unchanged=True),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
