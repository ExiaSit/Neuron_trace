#!/usr/bin/env python3
"""v0.83a: V3D → TIFF → 1 µm → segmentation → mask repair → reconstruction."""
from pathlib import Path
import argparse, json, os, subprocess, sys
from utils import bootstrap
from utils.configuration import load_config
from utils.case_worker import save_json

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'config.json'))
    parser.add_argument('--ids',help='数字 ID，以逗号分隔')
    parser.add_argument('--check',action='store_true',help='检查输入及依赖，不执行计算')
    parser.add_argument('--start',type=int,default=1,choices=range(1,8),help='起始前处理步骤；7 表示已有 img/mask，直接重建')
    parser.add_argument('--stop',type=int,default=13,choices=[1,2,3,4,5,6,13],help='完成指定前处理步骤后停止；13 表示完成全部重建')
    args=parser.parse_args();cfg=load_config(args.config)
    if args.stop<args.start:
        raise ValueError('--stop 必须不小于 --start')
    if Path(sys.executable).resolve()!=Path(cfg['python_executable']).resolve():
        raise ValueError('请使用 config.json 指定的 hb_seg Python')
    root=Path(cfg['output_root'])
    entry_lock=None
    if not args.check:
        import fcntl
        root.mkdir(parents=True,exist_ok=True)
        entry_lock=(root/'.entry.lock').open('a')
        fcntl.flock(entry_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ids=args.ids or (','.join(map(str,cfg['neuron_ids'])) if cfg.get('neuron_ids') else None)
    preparation_failed=False
    if cfg.get('input_mode','v3d')=='v3d' and args.start<=6:
        from utils.preparation import requested_ids,run_preparation
        wanted=requested_ids(cfg,args.ids)
        if args.check:
            from utils.check_environment import check
            check(cfg)
        ready=run_preparation(cfg,wanted,args.start,min(6,args.stop),args.check)
        if args.check:return 0
        preparation_failed=bool(json.loads((Path(cfg['prepared_root'])/'preparation-summary.json').read_text())['failed'])
        if args.stop<=6:return int(preparation_failed)
        if not ready:
            print('无已准备好的样本，本轮结束。',flush=True);return 1
        ids=','.join(ready)
        # Our own completed files have atomic checkpoints; no transfer wait is needed.
        cfg['minimum_stable_seconds']=0
    elif args.stop<=6:
        raise ValueError('准备阶段 --stop 必须不小于 --start，且 input_mode 必须为 v3d')
    if args.check:
        from utils.check_environment import check
        check(cfg)
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    config_path=root/'resolved-config.json'
    save_json(config_path,cfg)
    command=[sys.executable,str(ROOT/'utils/reconstruct.py'),'--config',str(config_path)]
    if ids:command+=['--ids',ids]
    if args.check:command+=['--check']
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
    result=subprocess.call(command,env=env)
    return result or int(preparation_failed)
if __name__=='__main__':
    sys.exit(main())
