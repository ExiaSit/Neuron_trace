"""Apply the same publication guard used in the validated refinement runner."""
from pathlib import Path
import json,numpy as np,tifffile
from algorithms.tracing.regression_guard import choose,foreign_context
from utils.baseline_io import copy_output

def run(spec,paths,save_json,check_swc):
    cid=spec['id'];root=paths['root'];base=spec.get('guard_baseline_root');audit_path=root/'regression_guard'/f'{cid}.json'
    if not base:
        save_json(audit_path,dict(id=cid,decision='new_case_no_completed_baseline',method_revision=spec['config']['method_revision']));return
    base=Path(base);raw=tifffile.imread(spec['raw']);old=np.loadtxt(base/'pruning_swcs'/f'{cid}.swc',comments='#',ndmin=2);new=np.loadtxt(paths['final'],comments='#',ndmin=2)
    m=spec['metadata'];target=[float(m[k])*float(m[res])/1000 for k,res in [('soma_z','z_resolution'),('soma_y','xy_resolution'),('soma_x','xy_resolution')]]
    labels,origin=foreign_context(raw,target,spec['soma_json'],spec['soma_mask']);result,audit=choose(old,new,raw,foreign_labels=labels,foreign_origin_zyx=origin)
    if audit['decision']=='baseline':
        for sub,ext in [('mask','.tif'),('trace_app2/down_sampled_swcs_app2','.swc'),('gcut_output/selected_for_pruning','.swc'),('gcut_output/selected_resampled_step1','.swc'),('pruning_swcs','.swc'),('mask_mips','.png'),('gcut_mips','.png'),('pruning_mips','_combined_process.png')]:copy_output(base/sub/(cid+ext),root/sub/(cid+ext))
    elif audit['decision']=='candidate_with_supported_baseline_branches':
        tmp=paths['final'].with_suffix('.guard.swc');np.savetxt(tmp,result,fmt=['%d','%d','%.6f','%.6f','%.6f','%.6f','%d']);tmp.replace(paths['final'])
    check_swc(paths['final']);audit.update(id=cid,baseline_root=str(base),method_revision=spec['config']['method_revision']);save_json(audit_path,audit)
