"""Restored v0.84 base pruning; soma seed experiments do not invoke this stage."""
import json
import numpy as np
from utils import bootstrap
from utils.case_worker import setup_config, load_module, check_swc, save_json

def run(spec):
    p=setup_config(spec);cfg=spec['config'];r=p['root'];cid=spec['id']
    import pandas as pd
    meta=pd.DataFrame([spec['metadata']], index=[int(cid)])
    mod=load_module('prune');mip_dir=r/'pruning_mips';mip_dir.mkdir(parents=True,exist_ok=True)
    settings=dict(selected_swc_dir=str(p['resampled'].parent),mask_dir=str(p['mask'].parent),raw_image_dir=str(p['raw'].parent),out_swc_dir=str(p['final'].parent),mip_dir=str(mip_dir),meta=meta,sphere_zradius=10,downsample_scale=np.array([1,1,1]),binary_input=True,pre_traced=True,debug=True,verbose=True,skip_existing_outputs=False,soma_img_dir=str(r/'soma_img'),soma_mask_dir=str(r/'soma_seg'))
    settings.update(raw_image_path=str(p['raw']),soma_json_path=str(p['soma_json']),soma_mask_path=str(p['soma_mask']))
    result=mod.worker_task(str(p['resampled']),settings)
    if result['status']!='success':raise RuntimeError(json.dumps(result,ensure_ascii=False))
    check_swc(p['final'])
    audit=mip_dir/f'{cid}_pruning_reasons.json'
    if audit.exists():
        from collections import Counter
        details=json.loads(audit.read_text())
        save_json(r/'logs'/cid/'pruning-audit.json',dict(pruning_policy=details.get('pruning_policy'),active_rules=details.get('active_rules',[]),signal_gap_decisions=details.get('signal_gap_decisions',[]),crossing_decisions=details.get('crossing_decisions',[]),sheet_signal_decisions=details.get('sheet_signal_decisions',[]),
            initial_short_leaf_audit=details.get('initial_short_leaf_audit'),connection_audit=details.get('connection_audit'),neighbor_soma_audit=details.get('neighbor_soma_audit'),repair_evidence_audit=details.get('repair_evidence_audit'),soma_terminal_audit=details.get('soma_terminal_audit'),final_short_leaf_audit=details.get('final_short_leaf_audit'),removal_counts=dict(Counter(v['code'] for v in details.get('removed_nodes',{}).values()))))

def run_branchwalk_experiment(spec):
    from pathlib import Path
    import shutil
    from tools.run_soma_branchwalk import run as branchwalk
    p=setup_config(spec);cid=str(spec['id']);cfg=spec['config']
    name=cfg.get('gemma_run_name') or (cid+'_soma_branchwalk_'+spec['fingerprint'][:12])
    experiment=Path(__file__).resolve().parent.parent/'test/0.84/runs'/name
    summary=branchwalk(cid,p['resampled'],p['raw'],spec['metadata'],experiment,cfg.get('gemma_endpoint','http://127.0.0.1:18484'))
    if summary.get('incomplete'):raise RuntimeError('Incomplete Gemma traversal; no final output')
    shutil.copy2(experiment/'swc'/f'{cid}.swc',p['final'])
    check_swc(p['final'])
    save_json(p['root']/'logs'/cid/'pruning-audit.json',dict(policy=summary['policy'],legacy_pruning_steps=[],experiment=str(experiment),summary=summary))

if __name__=="__main__":
    from utils.stage_cli import run_spec
    run_spec(run)
