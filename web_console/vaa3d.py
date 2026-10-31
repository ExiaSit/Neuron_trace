"""Read-only local Vaa3D launch choices from registered sample outputs.

Input paths may live outside the output root, but come only from the saved
sample manifest (never from HTTP path parameters). No files are converted.
"""
import json
import os
from pathlib import Path
import re


def existing(value, extensions):
    if not value or not isinstance(value, (str, Path)):
        return None
    try:
        path = Path(value)
        if (not path.is_absolute() or path.suffix.lower() not in extensions
                or any(ord(c) < 32 or ord(c) == 127 for c in str(path))):
            return None
        path = path.resolve()
        if path.suffix.lower() in extensions and path.is_file() and os.access(path, os.R_OK):
            return str(path)
    except (OSError, RuntimeError, ValueError):
        pass
    return None


def launch_bundle(root, cid, items, prepared=None):
    if not re.fullmatch(r'\d{1,12}', str(cid)):
        raise ValueError('ID 无效')
    root = Path(root).resolve()
    spec = {}
    try:
        saved = json.loads((root / 'manifests' / f'{cid}.json').read_text())
        if isinstance(saved, dict) and str(saved.get('id')) == str(cid):
            spec = saved
    except (OSError, ValueError, TypeError):
        pass
    tif = {'.tif', '.tiff'}
    raw_candidates = [spec.get('raw')]
    # These directories contain full-size 1 µm images, never soma crops or MIPs.
    for base in ([Path(prepared)] if prepared else []) + [root]:
        raw_candidates.extend(base / 'img' / name for name in
                              (f'{cid}_0000.tif', f'image_{cid}_0000.tif', f'{cid}.tif'))
    image = next((p for v in raw_candidates if (p := existing(v, tif | {'.v3dpbd'}))), None)
    choices, seen = [], {image} if image else set()

    def add(value, role, label, priority, default=False):
        path = existing(value, tif if role == 'mask' else {'.swc', '.eswc'})
        if not path or path in seen:
            return
        seen.add(path)
        choices.append(dict(path=path, role=role, label=label, priority=priority, default=default))

    labels = {'pruning_swcs': ('最终剪枝 SWC', 0),
              'gcut_output/selected_resampled_step1': ('重采样 SWC', 30),
              'gcut_output/selected_for_pruning': ('G-Cut 目标 SWC', 40),
              'trace_app2/down_sampled_swcs_app2': ('APP2 SWC', 50)}
    # Own final output wins over a duplicate registered comparison alias.
    for a in sorted(items, key=lambda a: bool(a.get('scope'))):
        group, path = a.get('group'), a.get('storage_path')
        if group in ('mask', 'mask_original'):
            add(path, 'mask', '补全后 / 当前 Mask' if group == 'mask' else '原始 Mask',
                10 if group == 'mask' else 20, group == 'mask')
        elif group == 'app2_input':
            name=Path(path).stem if path else ''
            label=('骨架 + 胞体（APP2 实际输入）' if name.endswith('_skeleton_soma') else
                   '三维骨架（诊断）' if name.endswith('_skeleton') else '保留的目标胞体（诊断）')
            add(path, 'mask', label, 70)
        elif group in labels:
            label, priority = labels[group]
            add(path, 'swc', label, priority, group == 'pruning_swcs')
        elif path and Path(path).suffix.lower() in ('.swc', '.eswc'):
            add(path, 'swc', ('版本对照' if a.get('scope') else '参考 / 诊断 SWC') + ' · ' + a.get('name', Path(path).name), 60)
    if not any(c['label'] == '原始 Mask' for c in choices):
        add(spec.get('mask'), 'mask', '原始 Mask', 20)
    choices.sort(key=lambda c: (c['priority'], c['label'], c['path']))
    for role in ('swc', 'mask'):
        members = [c for c in choices if c['role'] == role]
        # Only choose a known reconstruction stage by default, not diagnostic points.
        if members and not any(c['default'] for c in members) and members[0]['priority'] < 60:
            members[0]['default'] = True
    return dict(image=image, choices=choices,
                unavailable=None if image else '未找到本样本的原始三维图像，暂不能组合打开。')
