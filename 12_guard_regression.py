"""Numbered v0.83 reconstruction stage; run_pipeline.py is the batch entry."""
from utils import bootstrap
from utils.case_worker import setup_config, check_swc, save_json
from algorithms.tracing.soma_branchwalk import tree_index
from algorithms.tracing.connection_diagnostics import POLICY
import numpy as np

def run(spec):
    p=setup_config(spec);cfg=spec['config'];r=p['root'];cid=spec['id']
    # The current pruning result is authoritative: a historical baseline
    # must not restore removed branches or reintroduce older pruning rules.
    check_swc(p['final'])
    tree_index(np.loadtxt(p['final'],ndmin=2).tolist())
    save_json(r/'regression_guard'/f'{cid}.json', dict(
        id=cid, decision='validated_restored_base_pruning_output',
        pruning_policy=POLICY,
        method_revision=cfg.get('method_revision')))

if __name__=="__main__":
    from utils.stage_cli import run_spec
    run_spec(run)
