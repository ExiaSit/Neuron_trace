#!/usr/bin/env python3
"""Complete an output-only repair without repeating or changing model decisions.

The frozen inference source must match exactly except the export function. Both
source versions remain saved; every request/image/runtime identity is rechecked.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,copy,json,shutil,sys
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import refine_ownership_conflicts as stage
from algorithms.tracing.connection_diagnostics import save_json

def inference_ast(path):
    tree=ast.parse(Path(path).read_text())
    tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='export']
    return ast.dump(tree,include_attributes=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();out=args.run.resolve();identity=stage.read(out/'stage_identity.json');parent=Path(identity['parent']);stage.check_parent(parent)
    assert stage.sha(parent/'diagnosis.json')==identity['parent_diagnosis_sha256']
    for relative,digest in identity['code'].items():
        frozen=out/'source'/relative;assert stage.sha(frozen)==digest
        if relative=='tools/refine_ownership_conflicts.py':assert inference_ast(frozen)==inference_ast(ROOT/relative),'Only exporter changes may reuse these model decisions'
        else:assert stage.sha(ROOT/relative)==digest,'Changed inference implementation or prompt: '+relative
    runtime=stage.runner.runtime_identity('http://127.0.0.1:18484');assert runtime==identity['runtime']
    report=stage.read(out/'diagnosis.json');assert 'joint_settlement' in report
    assert {t['candidate_id'] for t in report['segment_tasks']}=={u['candidate_id'] for u in report['segment_updates']}
    for call in stage.all_calls(report):
        req=stage.read(call['request']);assert req['image_count']==len(req['images'])
        assert all(stage.sha(i['path'])==i['sha256'] for i in req['images'])
        assert stage.oa.fingerprint({k:req[k] for k in ['final_text','images','parameters','run_fingerprint']})==req['fingerprint']
        assert Path(call['request'].replace('_request.json','_prompt.txt')).read_text()==req['final_text']
    repair=dict(reason='Protected/unsent cases have no reviews: fix summary-only optional field access',inference_ast_unchanged=True,old_source_sha256=identity['code']['tools/refine_ownership_conflicts.py'],export_source_sha256=stage.sha(ROOT/'tools/refine_ownership_conflicts.py'),helper_sha256=stage.sha(__file__),model_calls_repeated=0,model_decisions_changed=False)
    save_json(out/'export_repair_identity.json',repair)
    for file in [ROOT/'tools/refine_ownership_conflicts.py',Path(__file__)]:
        dest=out/'export_source/tools'/file.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
    tree=np.loadtxt(report['input']['swc'],ndmin=2).tolist();raw=tifffile.imread(report['input']['raw'])
    stage.export(out,parent,report,tree,raw,SimpleNamespace(cell=report['input']['cell']))
if __name__=='__main__':main()
