"""Configuration paths are resolved relative to the configuration file."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load_config(path):
    path=Path(path).resolve()
    cfg=json.loads(path.read_text())
    def absolute(value):
        p=Path(value).expanduser()
        return str(p.resolve() if p.is_absolute() else (path.parent/p).resolve())
    for key in ['output_root','prepared_root','metadata_file','v3d_root','v3d_dir','python_executable','nnunet_executable',
                'vaa3d_app2','vaa3d_resample','app2_temp_dir','validated_baseline_registry',
                'raw_dir','mask_dir','soma_crop_dir','soma_seg_dir']:
        if cfg.get(key):cfg[key]=absolute(cfg[key])
    cfg['previous_app2_dirs']=[absolute(p) for p in cfg.get('previous_app2_dirs',[])]
    cfg['previous_swc_dirs']=[absolute(p) for p in cfg.get('previous_swc_dirs',[])]
    for model in cfg.get('models',{}).values():
        model['results']=absolute(model['results'])
    cfg.setdefault('method_version','v0.83a')
    cfg.setdefault('method_revision','base-pruning-restored-gemma-soma-seeds-20261112')
    cfg.setdefault('gemma_endpoint','http://127.0.0.1:18484')
    from algorithms.mask_repair.multiscale_mip import configuration
    cfg['mask_mip']=configuration(cfg.get('mask_mip', {}))
    cfg['direct_inputs']=True
    cfg.setdefault('baseline_root',None)
    if cfg['baseline_root'] is not None:
        raise ValueError('ring-guard 禁止复用旧 mask/APP2；baseline_root 必须为 null')
    cfg.setdefault('minimum_stable_seconds',120)
    cfg.setdefault('workers',4)
    cfg.setdefault('memory_budget_gb',96)
    cfg.setdefault('memory_reserve_gb',32)
    root=Path(cfg['output_root'])
    cfg.update(output_mask_dir=str(root/'mask'),output_mip_dir=str(root/'mask_mips'),comparison_dir=str(root/'comparisons'))
    if cfg.get('input_mode','v3d')=='v3d':
        prep=Path(cfg['prepared_root'])
        cfg.update(raw_dir=str(prep/'img'),mask_dir=str(prep/'initial_mask'),
                   soma_crop_dir=str(prep/'soma_img'),soma_seg_dir=str(prep/'soma_seg'))
        for other in [root,Path(cfg.get('v3d_dir') or cfg['v3d_root'])]:
            if prep==other or prep in other.parents or other in prep.parents:
                raise ValueError('prepared_root、output_root 与 V3D 输入目录必须相互独立')
    return cfg
