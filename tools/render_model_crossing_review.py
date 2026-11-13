"""Add a slice explorer to completed model diagnoses, without changing decisions."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visualization.projection_viewer import write_site_viewer,write_viewer_index

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args();out=Path(args.run)
    report=json.loads((out/'diagnosis.json').read_text());inputs=json.loads((out/'input.json').read_text())
    raw=tifffile.imread(inputs['raw']);tree=np.loadtxt(inputs['swc']);nodes={int(n[0]):n for n in tree}
    folder=out/'review';folder.mkdir(exist_ok=True);sites=[]
    for s in report['sites']:
        if s['status'] in ('pending_model','protected_soma'):continue
        accepted=s.get('transfer_gate',s.get('gate',{})).get('accepted',False);state='suspected_switch' if accepted else 'uncertain'
        lo,hi=np.array(s['bbox_xyz'])
        site=dict(site_id=s['site_id'],kind='gemma_xy_crossing',status=s['status'],center_xyz=s['center_xyz'],
                  bbox_xyz=np.column_stack([lo,hi]).tolist(),
                  context_xyz=[nodes[i][2:5].tolist() for i in s['incoming_ids']],image=f'../sites/{s["site_id"]}/xy.png',
                  arms=[dict(xyz=[nodes[i][2:5].tolist() for i in s['outgoing_ids']],diagnosis=dict(status=state,cut_xyz=s['center_xyz'] if accepted else None))])
        crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
        write_site_viewer(crop,lo,site,folder/f'site_{s["site_id"]}.html');sites.append(site)
    write_viewer_index(dict(sites=sites),folder)
    (folder/'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    index=out/'index.html'
    if index.exists() and 'review/interactive.html' not in index.read_text():
        text=index.read_text();text=text.replace('<h1>', '<p><a href="review/interactive.html">逐层 / 薄层三视图交互复核</a></p><h1>',1);index.write_text(text)
    print(f'Wrote {len(sites)} interactive review volumes')

if __name__=='__main__':main()
