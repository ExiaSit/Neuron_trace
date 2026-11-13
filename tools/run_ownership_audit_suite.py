#!/usr/bin/env python3
"""Execute staged experiments sequentially on the already deployed Gemma server."""
from pathlib import Path
import argparse,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'test/0.84'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--wait-pid',type=int);args=ap.parse_args();config=json.loads(args.config.read_text());logdir=Path(config['log_directory']);logdir.mkdir(parents=True,exist_ok=True)
    if args.wait_pid:
        process=Path('/proc')/str(args.wait_pid)
        while process.exists():time.sleep(5)
    stages=[]
    def execute(label,argv):
        log=logdir/(label+'.log');print(json.dumps(dict(stage=label,status='running',command=argv)),flush=True)
        start=time.time()
        with log.open('a') as out:subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT,check=True,cwd=ROOT.parent)
        stages.append(dict(stage=label,command=argv,seconds=time.time()-start,log=str(log)));(logdir/'suite_stages.json').write_text(json.dumps(stages,indent=2));print(json.dumps(dict(stage=label,status='complete')),flush=True)
    for item in config['experiments']:
        cell=str(item['cell']);parent=BASE/'runs'/item['parent_run'];final=BASE/'runs'/item['final_run'];python=config['python']
        if not (parent/'index.html').exists():
            if item.get('require_completed_parent'):raise ValueError('Existing main experiment did not complete: '+str(parent))
            command=[python,str(ROOT/'tools/run_ownership_audit.py'),'--cell',cell,'--swc',item['swc'],'--run-name',item['parent_run'],'--workers','2']
            if item.get('history'):command+=['--history',item['history']]
            if item.get('representative'):command+=['--representative',str(item['representative'])]
            execute(cell+'_main',command)
        if (final/'export_repair_identity.json').exists():
            # An output-only repair retains the original inference fingerprint.
            # This helper rejects any changed inference AST, prompt, image or runtime.
            execute(cell+'_export_checked',[python,str(ROOT/'tools/finish_ownership_export.py'),'--run',str(final)])
        else:
            execute(cell+'_segment_joint',[python,str(ROOT/'tools/refine_ownership_conflicts.py'),'--parent-run',str(parent),'--run-name',item['final_run']])
        execute(cell+'_report',[python,str(ROOT/'tools/report_ownership_joint.py'),'--run',str(final),'--server-log',config['server_log']])
        execute(cell+'_delivery',[python,str(ROOT/'tools/build_audit_delivery.py'),'--run',str(final),'--browser-check'])
    print('SUITE_COMPLETE',flush=True)
if __name__=='__main__':main()
