#!/usr/bin/env python3
"""Read-only engineering audit of saved soma seed inference and image inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(run, partial=False):
    run = Path(run)
    identity = read(run / 'input.json')
    runtime = read(run / 'runtime.json')
    cmdline = runtime['cmdline']
    options = {}
    for name in ['--ctx-size', '--batch-size', '--ubatch-size', '--n-gpu-layers', '--reasoning-budget', '--image-min-tokens', '--image-max-tokens']:
        occurrences = [i for i, value in enumerate(cmdline) if value == name]
        if occurrences:
            options[name] = int(cmdline[occurrences[-1] + 1])
    bundle = read(run / 'candidates.json')
    results = read(run / 'decisions.json')
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    for key in ['raw', 'mask']:
        check(sha(identity[key]) == identity[key + '_sha256'], key + ' changed')
    for name, digest in identity['code'].items():
        check(sha(run / 'source' / name) == digest, 'snapshot changed: ' + name)
    expected = {c['candidate_id'] for c in bundle['candidates']}
    check(set(results) <= expected, 'unexpected candidate')
    if not partial:
        check(set(results) == expected, 'incomplete inference')
    kept = {'T'}
    images = 0
    prompt_tokens = []
    finishes = Counter()
    planes = {'XY': (0, [0, 1]), 'XZ': (1, [0, 2]), 'YZ': (2, [1, 2])}
    for candidate in bundle['candidates']:
        cid = candidate['candidate_id']
        if cid not in results:
            continue
        result = results[cid]
        folder = run / 'cases' / cid
        request = read(folder / 'review_request.json')
        response = read(folder / 'review_response.json')
        views = read(folder / 'views.json')
        check(request['final_text'] == (folder / 'review_prompt.txt').read_text(), cid + ' prompt mismatch')
        metadata = json.loads(request['final_text'].split('CASE AND VIEW INVENTORY:\n', 1)[1])
        check(metadata['candidate']['candidate_id'] == cid, cid + ' request ID mismatch')
        check({p['candidate_id'] for p in metadata['previously_retained']} == kept, cid + ' retained context mismatch')
        check(set(metadata['allowed_duplicate_ids']) == set(result['reference_ids']), cid + ' reference mismatch')
        check(len(request['images']) == request['image_count'] == result['call']['image_count'] == 3, cid + ' image count')
        image_paths = {item['path'] for item in request['images']}
        check(image_paths == set(result['images']), cid + ' image inventory mismatch')
        for item in request['images']:
            images += 1
            check(sha(item['path']) == item['sha256'], cid + ' image hash mismatch')
        check(set(metadata['views']) == set(views), cid + ' view inventory mismatch')
        for name, view in views.items():
            axis, xy = planes[view['plane']]
            lo, hi = np.asarray(view['xyz_bounds'])
            check(view['array_projection_axis'] == axis and view['world_axes'] == xy, cid + '/' + name + ' axes')
            check(np.all(lo >= 0) and np.all(hi < np.asarray(bundle['shape_zyx'][::-1])) and np.all(lo <= hi), cid + '/' + name + ' bounds')
            check(view['image'] in image_paths, cid + '/' + name + ' unsent image')
            check(all(metadata['views'][name][k] == view[k] for k in metadata['views'][name]), cid + '/' + name + ' metadata mismatch')
            if name.startswith('slice_'):
                check(lo[2-axis] == hi[2-axis], cid + '/' + name + ' slice thickness')
            matrix = np.asarray(view['xyz_to_png_uv'])
            check(matrix.shape == (2, 4) and np.isfinite(matrix).all(), cid + '/' + name + ' mapping')
            for mark in view['landmarks'].values():
                xyz = np.asarray(mark['xyz_um'])
                check(np.array_equal(xyz[xy], mark['uv_um']), cid + '/' + name + ' landmark axes')
                check(mark['in_depth'] == bool(lo[2-axis]-.5 <= xyz[2-axis] <= hi[2-axis]+.5), cid + '/' + name + ' depth flag')
                uv = matrix @ np.r_[xyz, 1]
                check(np.all(uv >= 0) and np.all(uv < view['image_size_wh']), cid + '/' + name + ' landmark off image')
        choice = response['choices'][0]
        finishes[choice['finish_reason']] += 1
        check(choice['finish_reason'] == 'stop', cid + ' output truncated')
        check(response.get('__verbose', {}).get('truncated') is False, cid + ' input truncation not ruled out')
        check(response['model'] == request['parameters']['model'], cid + ' model mismatch')
        text = choice['message'].get('content', '').strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0]
        try:
            parsed = json.loads(text)
        except ValueError:
            check(bool(result['call'].get('error')) and result['decision']['decision'] == 'REVIEW', cid + ' malformed output not held for review')
        else:
            check(parsed == result['call'].get('judgment'), cid + ' saved model judgment differs from raw response')
        check(response['usage']['prompt_tokens'] + request['parameters']['max_tokens'] <= options['--ctx-size'], cid + ' insufficient context budget')
        prompt_tokens.append(response['usage']['prompt_tokens'])
        decision = result['decision']
        if decision['decision'] == 'DUPLICATE':
            check(decision['duplicate_of'] in kept, cid + ' duplicate ref not retained')
        if decision['decision'] == 'ACCEPT':
            kept.add(cid)
    log = (run / 'server.log').read_text(errors='replace')
    visual_tokens = [int(v) for v in re.findall(r'image_tokens->nx = (\d+)', log)]
    check(len(visual_tokens) >= images, 'fewer encoded images than completed requests')
    if not partial:
        check(len(visual_tokens) == images, 'server image count differs from request manifests')
    check(not re.search(r'truncated = 1\b', log), 'server reports truncated input')
    check(not visual_tokens or max(visual_tokens) <= options['--ubatch-size'], 'image exceeds configured microbatch')
    no_swc = not list(run.rglob('*.swc'))
    check(no_swc, 'unexpected SWC output')
    report = dict(passed=not failures, failures=failures, completed=len(results), total=len(expected),
                  input_and_snapshot_hashes_checked=True, request_images_checked=images,
                  server_encoded_images=len(visual_tokens), visual_tokens_histogram=dict(Counter(visual_tokens)),
                  effective_runtime_options=options,
                  finish_reasons=dict(finishes), prompt_token_range=[min(prompt_tokens), max(prompt_tokens)] if prompt_tokens else [],
                  retained_including_metadata_target=len(kept), no_swc_output=no_swc,
                  partial=partial, biological_accuracy_evaluated=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    report = audit(args.run, args.partial)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
