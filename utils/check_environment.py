from pathlib import Path
import importlib
def check(cfg):
    for name in ['numpy','scipy','skimage','tifffile','matplotlib','pandas','cc3d','edt','networkx','cv2','psutil','ortools']:
        importlib.import_module(name)
    keys=['metadata_file','python_executable','vaa3d_app2','vaa3d_resample']
    if cfg.get('input_mode')=='v3d':
        importlib.import_module('v3dpy.loaders')
        keys.append('nnunet_executable')
        for model in cfg['models'].values():
            folder=Path(model['results'])/model['dataset_name']/f"{model.get('trainer','nnUNetTrainer')}__nnUNetPlans__3d_fullres"
            for file in ['plans.json','dataset.json',f"fold_{model.get('fold',0)}/{model.get('checkpoint','checkpoint_final.pth')}"]:
                if not (folder/file).is_file():raise FileNotFoundError(folder/file)
    for key in keys:
        if not Path(cfg[key]).is_file():raise FileNotFoundError(f'{key}: {cfg[key]}')
    print('Python、算法依赖、外部程序和所需模型检查通过。',flush=True)
