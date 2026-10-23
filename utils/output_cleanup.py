"""Remove generated detail products; keep final outputs and small SWC checkpoints."""
from pathlib import Path
import shutil


def cleanup(root, cid):
    root=Path(root)
    if not str(cid).isdigit():raise ValueError('Expected numeric ID')
    directories=[root/'mask_steps'/cid,root/'gcut_output'/cid,root/'gcut_output/resample_intermediates'/cid,
                 root/'trace_app2/intermediates'/cid,root/'pruning_mips/process_data'/cid]
    directories+=list((root/'gcut_output').glob(f'{cid}_retry_*'))
    for path in directories:
        if path.is_dir() and not path.is_symlink():shutil.rmtree(path)
    files=[root/'trace_app2/native_swcs'/f'{cid}.swc',root/'trace_app2/markers'/f'{cid}.marker',root/'trace_app2/vis_mip'/f'{cid}.png']
    for parent in [root/'trace_app2/native_swcs',root/'gcut_output/selected_for_pruning']:
        files+=list(parent.glob(f'{cid}_retry_*.swc'))
    for pattern in [f'{cid}_*.png',f'{cid}_*.json']:
        files+=[p for p in (root/'pruning_mips').glob(pattern) if p.name != f'{cid}_combined_process.png']
    for path in files:
        if path.is_file() or path.is_symlink():path.unlink()
