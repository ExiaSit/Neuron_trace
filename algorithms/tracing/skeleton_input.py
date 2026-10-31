"""Optional APP2-only Lee skeleton + measured target soma input.

Adapted from SEU-ALLEN-codebase/neuron_seg_human/tracer.py (skel_seg,
skel_with_soma). This is not an exact reproduction of its soma finder or APP2
settings. The repaired mask used by G-Cut/pruning is never overwritten.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import ball, skeletonize


def opened_somas(soma, seed, max_radius):
    """Offer bounded, anti-extensive soma openings, strongest first.

    Repeating the same opening is idempotent. Try increasing ball sizes on
    the measured body instead; never manufacture an ellipsoid outside it.
    The current pipeline uses isotropic 1 um voxels.
    """
    if type(max_radius) is not int or not 0 <= max_radius <= 3:
        raise ValueError('app2_soma_opening_radius_um must be an integer from 0 to 3')
    points = np.argwhere(soma)
    lo = np.maximum(0, points.min(0)-max_radius-1)
    hi = np.minimum(soma.shape, points.max(0)+max_radius+2)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    patch = soma[sl]
    center = tuple(np.asarray(seed)-lo)
    trials = []
    for radius in range(max_radius, 0, -1):
        local = ndi.binary_opening(patch, structure=ball(radius)) & patch
        labels, _ = ndi.label(local, np.ones((3, 3, 3), bool))
        cid = labels[center]
        local = labels == cid if cid else np.zeros_like(local)
        retained = float(local.sum()/patch.sum())
        rec = dict(radius_um=radius, retained_fraction=retained,
                   seed_preserved=bool(cid))
        trials.append(rec)
        if not cid or retained < .8:
            rec['status'] = 'rejected_seed_or_volume'
            continue
        candidate = np.zeros_like(soma)
        candidate[sl] = local
        yield candidate, radius, trials, rec
    yield soma, 0, trials, None


def skeleton_with_soma(mask, soma, seed_xyz, soma_opening_radius=3,
                       raw=None, soma_shape='signal_surface'):
    mask = np.asarray(mask) > 0
    soma = np.asarray(soma) > 0
    if mask.ndim != 3 or soma.shape != mask.shape:
        raise ValueError('Mask and soma must be matching ZYX volumes')
    seed = tuple(np.rint(seed_xyz).astype(int)[::-1])
    if any(q < 0 or q >= n for q, n in zip(seed, mask.shape)):
        raise ValueError('Soma seed is outside image')
    soma = soma & mask
    if not soma[seed]:
        raise ValueError('Measured soma does not contain APP2 seed; refusing to move seed')
    if soma_shape not in ('opening', 'signal_surface'):
        raise ValueError('app2_soma_shape must be opening or signal_surface')
    start = time.monotonic()
    skeleton = skeletonize(mask, method='lee').astype(bool)
    structure = np.ones((3, 3, 3), bool)
    measured = soma
    baseline_labels, _ = ndi.label(skeleton | measured, structure)
    required = skeleton & (baseline_labels == baseline_labels[seed])
    del baseline_labels
    for candidate, radius, trials, trial in opened_somas(measured, seed, soma_opening_radius):
        merged = skeleton | candidate
        labels, after = ndi.label(merged, structure)
        reached = labels == labels[seed]
        lost = int((required & ~reached).sum())
        if trial is not None:
            trial.update(disconnected_skeleton_voxels=lost,
                         status='rejected_connectivity' if lost else 'accepted')
        if not lost:
            soma = candidate
            break
    surface_report = dict(status='disabled' if soma_shape=='opening' or soma_opening_radius==0 else 'raw_unavailable')
    if soma_shape == 'signal_surface' and raw is not None and soma_opening_radius > 0:
        from algorithms.tracing.soma_shape import signal_surface
        candidate, surface_report = signal_surface(raw, mask, measured, soma, seed)
        if candidate is not None:
            candidate_merged = skeleton | candidate
            candidate_labels, candidate_count = ndi.label(candidate_merged, structure)
            candidate_reached = candidate_labels == candidate_labels[seed]
            lost = int((required & ~candidate_reached).sum())
            # Smoothing may fill a measured concavity, but cannot join a new
            # component that was disconnected from the target before fitting.
            gained = int((skeleton & candidate_reached & ~required).sum())
            surface_report.update(disconnected_skeleton_voxels=lost,
                                  newly_connected_skeleton_voxels=gained)
            if not lost and not gained:
                soma, merged, reached, after = candidate, candidate_merged, candidate_reached, candidate_count
                surface_report['status'] = 'accepted'
            else:
                surface_report.update(status='rejected', reason='connectivity_guard')
    labels, before = ndi.label(mask, structure)
    target = labels == labels[seed]
    # Thinning can erase a small component: record this rather than claiming
    # unconditional topology preservation from the algorithm's name.
    audit = dict(method='3d_lee_skeleton_union_regularized_target_soma',
                 mask_voxels=int(mask.sum()), soma_voxels=int(soma.sum()),
                 skeleton_voxels=int(skeleton.sum()), input_voxels=int(merged.sum()),
                 components_before=int(before), components_after=int(after),
                 target_skeleton_voxels=int((target & skeleton).sum()),
                 target_skeleton_unreachable=int((target & skeleton & ~reached).sum()),
                 seconds=round(time.monotonic()-start, 4), seed_xyz=list(seed_xyz),
                 outside_repaired_mask=int((merged & ~mask).sum()))
    audit['soma_regularization'] = dict(method=('signal_surface' if surface_report['status']=='accepted' else 'bounded_ball_opening'),
        requested_radius_um=soma_opening_radius, selected_radius_um=radius,
        measured_voxels=int(measured.sum()), processed_voxels=int(soma.sum()),
        removed_voxels=int((measured & ~soma).sum()), min_retained_fraction=.8,
        skeleton_unchanged=True, disconnected_previously_reachable_skeleton=0,
        trials=trials, surface_fit=surface_report)
    return skeleton, soma, merged, audit


def load_segmented_soma(shape, seed_xyz, json_path, mask_path):
    """Place only the seed's segmented component in 1-um image ZYX space.

    Crop metadata is ZXY, whereas TIFF arrays are ZYX. Use the recorded origin
    and actual TIFF dimensions (historical crop end bounds differ by a voxel).
    Never infer a replacement soma from intensity or snap to a nearby cell.
    """
    bounds = json.loads(Path(json_path).read_text())['bounds_1um_zxy']
    if len(bounds) != 6 or not np.isfinite(bounds).all() or not np.equal(bounds, np.rint(bounds)).all():
        raise ValueError('Soma crop bounds must be six integer 1-um coordinates')
    origin = np.array([bounds[0], bounds[4], bounds[2]], dtype=int)
    crop = tifffile.imread(mask_path) > 0
    if crop.ndim != 3:
        raise ValueError('Soma segmentation must be a 3-D ZYX TIFF')
    seed = np.rint(seed_xyz).astype(int)[::-1]
    local = seed-origin
    if (np.any(seed < 0) or np.any(seed >= shape) or
            np.any(local < 0) or np.any(local >= crop.shape)):
        raise ValueError('Target seed is outside the soma segmentation crop')
    labels, count = ndi.label(crop, np.ones((3, 3, 3), bool))
    selected = int(labels[tuple(local)])
    if selected == 0:
        raise ValueError('Target seed is not inside segmented soma; no estimated-soma fallback')
    component = labels == selected
    lo = np.maximum(origin, 0)
    hi = np.minimum(origin+crop.shape, shape)
    src = tuple(slice(int(a), int(b)) for a, b in zip(lo-origin, hi-origin))
    dst = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    soma = np.zeros(shape, bool)
    soma[dst] = component[src]
    report = dict(source='segmentation', source_path=str(mask_path),
                  crop_json=str(json_path), origin_zyx=origin.tolist(),
                  crop_shape_zyx=list(crop.shape), component_id=selected,
                  component_count=int(count), voxels=int(soma.sum()),
                  clipped_outside_image_voxels=int(component.sum()-soma.sum()),
                  segmentation_sha256=hashlib.sha256(Path(mask_path).read_bytes()).hexdigest(),
                  crop_json_sha256=hashlib.sha256(Path(json_path).read_bytes()).hexdigest())
    return soma, report


def skeleton_with_segmented_soma(mask, soma, seed_xyz):
    """Exact union: repaired-mask skeleton OR unmodified segmented target soma."""
    mask, soma = np.asarray(mask) > 0, np.asarray(soma) > 0
    if mask.ndim != 3 or soma.shape != mask.shape:
        raise ValueError('Mask and segmented soma must have matching ZYX shapes')
    seed = tuple(np.rint(seed_xyz).astype(int)[::-1])
    if any(q < 0 or q >= n for q, n in zip(seed, mask.shape)) or not soma[seed]:
        raise ValueError('Target seed is not inside segmented soma')
    start = time.monotonic()
    structure = np.ones((3, 3, 3), bool)
    skeleton = skeletonize(mask, method='lee').astype(bool)
    merged = skeleton | soma
    labels, after = ndi.label(merged, structure)
    reached = labels == labels[seed]
    labels, before = ndi.label(mask, structure)
    target_ids = np.unique(labels[soma]); target_ids = target_ids[target_ids > 0]
    target = np.isin(labels, target_ids)
    audit = dict(method='3d_lee_skeleton_union_segmented_target_soma',
                 mask_voxels=int(mask.sum()), soma_voxels=int(soma.sum()),
                 skeleton_voxels=int(skeleton.sum()), input_voxels=int(merged.sum()),
                 components_before=int(before), components_after=int(after),
                 target_skeleton_voxels=int((target & skeleton).sum()),
                 target_skeleton_unreachable=int((target & skeleton & ~reached).sum()),
                 seconds=round(time.monotonic()-start, 4), seed_xyz=list(seed_xyz),
                 outside_repaired_mask=int((merged & ~mask).sum()),
                 soma_regularization=dict(method='none_exact_segmentation',
                     measured_voxels=int(soma.sum()), processed_voxels=int(soma.sum()),
                     removed_voxels=0, skeleton_unchanged=True))
    return skeleton, soma, merged, audit


def prepare_input(spec, paths, seed_xyz):
    from utils.case_worker import save_json
    mask = tifffile.imread(paths['mask'])
    source = spec['config'].get('app2_soma_source', 'segmentation')
    if source == 'segmentation':
        measured, soma_report = load_segmented_soma(mask.shape, seed_xyz,
            paths['soma_json'], paths['soma_mask'])
        skeleton, soma, merged, audit = skeleton_with_segmented_soma(mask, measured, seed_xyz)
    elif source == 'measured':
        from algorithms.mask_repair.guarded_continuation import target_body
        raw = tifffile.imread(paths['raw'])
        sl, body, _, soma_report = target_body(raw, seed_xyz, paths['soma_json'], paths['soma_mask'])
        measured = np.zeros(mask.shape, bool)
        measured[sl] = body
        measured &= mask > 0
        skeleton, soma, merged, audit = skeleton_with_soma(mask, measured, seed_xyz,
            soma_opening_radius=spec['config'].get('app2_soma_opening_radius_um', 3),
            raw=raw, soma_shape=spec['config'].get('app2_soma_shape', 'signal_surface'))
    else:
        raise ValueError('app2_soma_source must be segmentation or measured')
    folder = paths['root'] / 'app2_input'
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in [('skeleton', skeleton), ('soma_measured', measured),
                       ('soma', soma), ('skeleton_soma', merged)]:
        tifffile.imwrite(folder / f"{spec['id']}_{name}.tif", data.astype('uint8')*255,
                         compression='zlib', metadata={'axes': 'ZYX'})
    dest = folder / f"{spec['id']}_skeleton_soma.tif"
    audit.update(soma=soma_report, soma_source=source, source_mask=str(paths['mask']), input=str(dest),
                 source_mask_sha256=hashlib.sha256(Path(paths['mask']).read_bytes()).hexdigest(),
                 spacing_zyx_um=[1, 1, 1],
                 reference='https://github.com/SEU-ALLEN-codebase/neuron_seg_human/blob/main/tracer.py')
    save_json(paths['root']/'logs'/spec['id']/'app2-input.json', audit)
    return dest
