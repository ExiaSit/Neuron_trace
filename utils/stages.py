from pathlib import Path
import importlib.util
STAGES={'continuity': '07_reconnect_mask.py', 'app2': '08_trace_app2.py', 'gcut': '09_separate_gcut.py', 'resample': '10_resample_swc.py', 'prune': '11_prune_swc.py', 'guard': '12_guard_regression.py', 'comparison': '13_compare_results.py'}
ROOT=Path(__file__).resolve().parents[1]
def load_stage(name):
    spec=importlib.util.spec_from_file_location('v083_stage_'+name,ROOT/STAGES[name])
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
