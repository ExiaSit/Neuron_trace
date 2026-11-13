#!/usr/bin/env python3
"""Export accepted soma coordinates as enlarged Vaa3D display landmarks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import bootstrap
from v3dpy.loaders import Raw


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--radius', type=float, default=8.0, help='Display radius in um, not a measured soma radius')
    args = parser.parse_args()
    if not np.isfinite(args.radius) or args.radius <= 0:
        parser.error('radius must be positive and finite')
    run = args.run.resolve()
    identity = json.loads((run / 'input.json').read_text())
    bundle = json.loads((run / 'candidates.json').read_text())
    catalogue_path = run / 'normalized/soma_catalogue.json'
    catalogue = json.loads(catalogue_path.read_text())
    if bundle['voxel_size_um_xyz'] != [1, 1, 1]:
        raise ValueError('This exporter requires the experiment 1 um isotropic image')
    raw_path = Path(identity['raw'])
    if digest(raw_path) != identity['raw_sha256']:
        raise ValueError('Raw image differs from the inference input')
    raw = tifffile.imread(raw_path)
    points = catalogue['accepted']
    xyz = np.array([p['xyz_um'] for p in points], dtype=float)
    if not np.all(np.isfinite(xyz) & (xyz >= 0) & (xyz < np.array(raw.shape[::-1]))):
        raise ValueError('Invalid point coordinates')
    if not np.array_equal(xyz, np.array([p['zyx'][::-1] for p in points])):
        raise ValueError('XYZ/ZYX mismatch')
    diameter = f'{2 * args.radius:g}'
    cell = identity['cell']
    out = run / f'vaa3d_soma_overlay_d{diameter}'
    out.mkdir(exist_ok=False)
    prefix = f'{cell}_somas_d{diameter}'
    image_path = out / f'{cell}_image_axes.v3draw'
    Raw().save(str(image_path), raw[None])
    # Raw stacks preserve array axes; TIFF readers in the calibrated Vaa3D
    # installation reverse Y. Do not use the APP2 plugin's special +2 offset.
    for suffix, tiff in [('', False), ('_for_tiff', True)]:
        coordinates = xyz.copy()
        if tiff:
            coordinates[:, 1] = raw.shape[1] - 1 - coordinates[:, 1]
        coordinates += 1  # Vaa3D landmark UI/file coordinates are one based.
        with (out / f'{prefix}{suffix}.marker').open('w') as stream:
            stream.write('##x,y,z,radius,shape,name,comment,color_r,color_g,color_b\n')
            writer = csv.writer(stream)
            for p, pos in zip(points, coordinates):
                color = [255, 64, 64] if p['candidate_id'] == 'T' else [0, 255, 96]
                writer.writerow([*pos, args.radius, 1, p['candidate_id'],
                                 'display size only; known target' if p['candidate_id'] == 'T' else 'display size only; Gemma accepted', *color])
        with (out / f'{prefix}{suffix}.marker').open() as stream:
            loaded = list(csv.reader(line for line in stream if not line.startswith('#')))
        assert len(loaded) == len(points)
        assert np.array_equal(np.array([r[:3] for r in loaded], float), coordinates)
        assert all(float(r[3]) == args.radius for r in loaded)
    # Independent roots are sphere landmarks, not reconstructed neuron branches.
    swc = out / f'{prefix}_spheres.swc'
    with swc.open('w') as stream:
        stream.write('# DISPLAY LANDMARKS ONLY: independent roots; radius is not anatomical.\n')
        for i, (p, pos) in enumerate(zip(points, xyz), 1):
            stream.write(f'# {i} = {p["candidate_id"]}\n')
            stream.write(f'{i} {1 if p["candidate_id"] == "T" else 3} {pos[0]:g} {pos[1]:g} {pos[2]:g} {args.radius:g} -1\n')
    (out / f'{prefix}.ano').write_text(f'RAWIMG={image_path.name}\nSWCFILE={swc.name}\n')
    reread = Raw().load(str(image_path))
    assert np.array_equal(reread, raw[None])
    rows = np.loadtxt(swc)
    assert np.array_equal(rows[:, 2:5], xyz)
    assert np.all(rows[:, 5] == args.radius) and np.all(rows[:, 6] == -1)
    audit = dict(accepted_count=len(points), candidate_ids=[p['candidate_id'] for p in points],
                 display_radius_um=args.radius, display_diameter_um=2*args.radius,
                 anatomical_radii_modified=False, image_voxels_unchanged=True,
                 catalogue_sha256=digest(catalogue_path), raw_sha256=digest(raw_path),
                 v3draw_marker_transform='XYZ+1; no Y flip',
                 tiff_marker_transform='X+1, height-Y, Z+1; calibrated Vaa3D TIFF loader',
                 swc_transform='zero-based XYZ, no Y flip; pair with supplied v3draw only',
                 exports_read_back_and_checked=True, vaa3d_gui_verified=False,
                 excluded_review_ids=catalogue['review'], source_sha256=digest(__file__))
    (out / 'export_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    (out / 'README.txt').write_text(f'''95918 胞体点显示叠加
共 {len(points)} 点：已知目标 T + 15 个 Gemma ACCEPT；不包含重复、拒绝或待复核点。
显示半径 {args.radius:g} µm，直径 {2*args.radius:g} µm，仅为便于观察，不是胞体真实大小。

推荐：在 Vaa3D 打开 {image_path.name}，然后载入/拖入 {prefix}.marker。
红色 T 为目标胞体，绿色 Cxxx 为模型选中胞体。点名与审核页面一致。
配套 v3draw 与实验原图逐体素相同，避免不同 TIFF 读取器的 Y 翻转差异。

需要用 SWC 球形点叠加时，打开 {prefix}.ano，或在配套 v3draw 的 3D 窗口载入 {swc.name}。
SWC 每点独立、半径 {args.radius:g}，没有连线；它是显示标记文件，不是剪枝结果。
SWC 的颜色由 Vaa3D 显示设置决定。部分 Vaa3D 版本的 marker 显示还受全局 Marker size 控件影响。

若已在本机 Vaa3D 中打开原 95918.tif，可使用 {prefix}_for_tiff.marker（按本项目已校准的 TIFF Y 翻转生成）。
不要把 for_tiff 版本叠加到配套 v3draw，也不要把本标记用于原始 0.28 µm XY 分辨率的大图。
推荐使用配套 v3draw + marker，以保持明确的坐标约定。未做 Vaa3D GUI 加载测试；文件已读回核对坐标、大小及图像像素。
''')
    print(json.dumps(dict(output=str(out), count=len(points), diameter_um=2*args.radius), ensure_ascii=False))


if __name__ == '__main__':
    main()
