"""Numbered v0.83 reconstruction stage; run_pipeline.py is the batch entry."""
from pathlib import Path
import json, shutil, sys, time
import numpy as np
import tifffile
from utils import bootstrap
from utils.case_worker import setup_config, load_module, check_swc, save_json, comparison
from utils import mask_batch as mask_run

def run(spec):
    p=setup_config(spec);cfg=spec['config'];r=p['root'];cid=spec['id']
    import pandas as pd
    meta=pd.DataFrame([spec['metadata']], index=[int(cid)])
    mod=load_module('prune');mip_dir=r/'pruning_mips';mip_dir.mkdir(parents=True,exist_ok=True)
    settings=dict(selected_swc_dir=str(p['resampled'].parent),mask_dir=str(p['mask'].parent),raw_image_dir=str(p['raw'].parent),out_swc_dir=str(p['final'].parent),mip_dir=str(mip_dir),meta=meta,sphere_zradius=10,downsample_scale=np.array([1,1,1]),binary_input=True,pre_traced=True,debug=True,verbose=True,skip_existing_outputs=False,soma_img_dir=str(r/'soma_img'),soma_mask_dir=str(r/'soma_seg'))
    from algorithms.tracing.soma_signal_groups import auxiliary_groups,exclude_confirmed_neighbors
    m=spec['metadata'];target_zyx=np.array([float(m[k])*float(m[res])/1000 for k,res in [('soma_z','z_resolution'),('soma_y','xy_resolution'),('soma_x','xy_resolution')]])
    aux_labels,aux_origin,aux_same,aux_audit=auxiliary_groups(tifffile.imread(p['raw']),target_zyx,p['soma_json'],p['soma_mask'])
    from algorithms.tracing.soma_signal_groups import confirmed_mask
    local_target=np.rint(target_zyx-aux_origin).astype(int)
    own_aux=int(aux_labels[tuple(local_target)]) if np.all(local_target>=0) and np.all(local_target<aux_labels.shape) else 0
    protected_aux=aux_same+([own_aux] if own_aux else [])
    mod.CONFIRMED_SOMA_MASK=confirmed_mask(tifffile.imread(p['raw']).shape,aux_labels,aux_origin,protected_aux,aux_audit)
    from algorithms.tracing.soma_signal_groups import target_core_region
    foreign=confirmed_mask(tifffile.imread(p['raw']).shape,aux_labels,aux_origin,[int(x) for x in np.unique(aux_labels) if x and x not in protected_aux],dict(same_auxiliary_components=[]))
    core_sl,core_region,core_audit=target_core_region(tifffile.imread(p['raw']),target_zyx,foreign)
    mod.CONFIRMED_SOMA_MASK[core_sl] |= core_region
    from algorithms.tracing.sheet_signal import completed_soma_protection
    mod.SHEET_PROTECTED_SOMA_MASK,sheet_protection_source=completed_soma_protection(
        r/'mask_steps'/cid/'f/completed-somas.npz',mod.CONFIRMED_SOMA_MASK.shape,mod.CONFIRMED_SOMA_MASK)
    save_json(r/'logs'/cid/'sheet-soma-protection.json',dict(source=sheet_protection_source))
    save_json(r/'logs'/cid/'target-core-protection.json',core_audit)
    # Preserve foreign-soma entry evidence before old pruning clips those children.
    from algorithms.tracing.repair_evidence import foreign_entry_ancestors
    neighbor_points = []
    if not aux_audit['same_auxiliary_components']:
        for label in np.unique(aux_labels):
            if not label:
                continue
            points = np.argwhere(aux_labels == label) + aux_origin
            if len(points) >= 20 and np.linalg.norm(points-target_zyx, axis=1).min() > 5.:
                neighbor_points.extend(points.tolist())
    original_path = p['original_mask']
    measured_soma_path = r/'app2_input'/f'{cid}_soma.tif'
    mod.REPAIR_EVIDENCE_CONTEXT = None
    if original_path.exists() and measured_soma_path.exists():
        pre_tree = np.loadtxt(p['resampled'], ndmin=2).tolist()
        mod.REPAIR_EVIDENCE_CONTEXT = dict(
            original_mask_path=str(original_path), measured_soma_path=str(measured_soma_path),
            neighbor_points=neighbor_points,
            foreign_witness_ids=foreign_entry_ancestors(pre_tree, neighbor_points))
    del foreign
    confirmed=[x['center_zyx'] for x in aux_audit['same_auxiliary_components']]
    save_json(r/'logs'/cid/'prune-soma-groups.json',aux_audit)
    original_inside=mod.get_soma_mask_inside_ids
    def inside(*args,**kwargs):
        ids,radius,context=original_inside(*args,**kwargs)
        return ids,radius,exclude_confirmed_neighbors(context,confirmed)
    mod.get_soma_mask_inside_ids=inside
    settings.update(raw_image_path=str(p['raw']),soma_json_path=str(p['soma_json']),soma_mask_path=str(p['soma_mask']))
    result=mod.worker_task(str(p['resampled']),settings)
    if result['status']!='success':raise RuntimeError(json.dumps(result,ensure_ascii=False))
    check_swc(p['final'])
    audit=mip_dir/f'{cid}_pruning_reasons.json'
    if audit.exists():
        from collections import Counter
        details=json.loads(audit.read_text())
        save_json(r/'logs'/cid/'pruning-audit.json',dict(signal_gap_decisions=details.get('signal_gap_decisions',[]),crossing_decisions=details.get('crossing_decisions',[]),sheet_signal_decisions=details.get('sheet_signal_decisions',[]),
            repair_evidence_audit=details.get('repair_evidence_audit'),soma_terminal_audit=details.get('soma_terminal_audit'),final_short_leaf_audit=details.get('final_short_leaf_audit'),removal_counts=dict(Counter(v['code'] for v in details.get('removed_nodes',{}).values()))))

if __name__=="__main__":
    from utils.stage_cli import run_spec
    run_spec(run)
