"""Resolve an exact neuron ID, then normalize a reference copy in memory only."""
from pathlib import Path
import hashlib
import numpy as np


def resolve_previous_app2(spec):
    cid = str(int(spec['id']))
    for folder in spec['config'].get('previous_app2_dirs', []):
        for name in [f'{cid}.swc', f'image_{cid}.swc']:
            path = Path(folder)/name
            if path.is_file():
                return path
    return None


def reference_nodes(spec, height):
    from utils.case_worker import check_swc
    path = resolve_previous_app2(spec)
    if path is None:
        return {}, dict(status='missing', searched_dirs=spec['config'].get('previous_app2_dirs', []))
    nodes = check_swc(path)
    header = path.read_text()
    frame = spec['config'].get('previous_app2_coordinates', 'native')
    if any(marker in header for marker in ['# app2_native_to_image_zero_based_v2',
                                         '# archived_app2_native_to_image_zero_based_v1']):
        frame = 'image'
    elif '# y_axis_flipped_to_image_coordinates' in header:
        raise ValueError(f'Legacy coordinate marker requires explicit conversion from native source: {path}')
    if frame not in ('image', 'native'):
        raise ValueError('previous_app2_coordinates must be native or image')
    rows = {nid: np.array([n.x, n.y, n.z], dtype=float) for nid, n in nodes.items()}
    if frame == 'native':
        for xyz in rows.values():
            xyz[1] = height - 1 - xyz[1]
    return {nid: (rows[nid], n.parent) for nid, n in nodes.items()}, dict(
        status='present', path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        input_coordinates=frame, transform='y = height - 1 - y' if frame == 'native' else 'none',
        image_height=height, output_coordinates='image zero-based', nodes=len(nodes))
