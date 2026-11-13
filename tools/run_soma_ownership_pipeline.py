#!/usr/bin/env python3
"""Run candidate/model review, reconcile ownership conflicts, then validate/export."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-name',default='20261111_97117_global_soma_v3_checked');ap.add_argument('--endpoint',default='http://127.0.0.1:18484');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--limit',type=int);ap.add_argument('--finalize-only',action='store_true');args=ap.parse_args()
    if Path(args.run_name).name!=args.run_name:ap.error('Invalid run name')
    base=ROOT.parent/'test/0.84';out=base/'runs'/args.run_name
    if (out/'diagnosis.json').exists() and json.loads((out/'diagnosis.json').read_text()).get('consistency'):
        args.finalize_only=True
    if not args.finalize_only:
        cmd=[sys.executable,str(ROOT/'tools/run_soma_ownership_experiment.py'),'--run-name',args.run_name,'--endpoint',args.endpoint]
        if args.prepare_only:cmd.append('--prepare-only')
        if args.limit:cmd.extend(['--limit',str(args.limit)])
        subprocess.run(cmd,cwd=ROOT,check=True)
        if args.prepare_only or args.limit:return
    if not (out/'summary.json').exists():raise ValueError('Complete model review first')
    report=json.loads((out/'diagnosis.json').read_text())
    if not report.get('consistency'):
        subprocess.run([sys.executable,str(ROOT/'tools/reconcile_soma_ownership.py'),'--run',str(out)],cwd=ROOT,check=True)
    if not (out/'validation.json').exists() or not (out/'review/interactive.html').exists() or 'id="effect"' not in (out/'index.html').read_text():
        subprocess.run([sys.executable,str(ROOT/'tools/render_soma_ownership_effect.py'),'--run',str(out)],cwd=ROOT,check=True)
    (base/'latest-model-result-97117.txt').write_text(str(out)+'\n')
    print('Final reviewed result: '+str(out/'index.html'))

if __name__=='__main__':
    main()
