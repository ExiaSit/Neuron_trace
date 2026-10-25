#!/usr/bin/env python3
"""Read-only signal-supported branch-retention audit of completed MIP pairs.

Reuses v0.83's existing 3-D signal and 3-voxel coverage definition. This is a
heuristic regression metric, not accuracy against manual neuronal ownership.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import bootstrap
from algorithms.tracing.regression_guard import coverage
from tools.compare_mip_repair import sha, save, valid_reconstruction_pair


def audit(rec, output):
    cid = rec['id']
    sources = [output/side/'pruning_swcs'/f'{cid}.swc' for side in ('baseline','candidate')]
    identity = dict(swc_sha256=[sha(p) for p in sources],
                    raw_sha256=rec['input_identity']['raw']['sha256'],
                    metric_sha256=sha(ROOT/'algorithms/tracing/regression_guard.py'))
    path = output/'branch_audits'/f'{cid}.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('identity') == identity:
            return old
    raw = tifffile.imread(rec['inputs']['raw'])
    baseline, candidate = [np.loadtxt(p,comments='#',ndmin=2) for p in sources]
    retained = coverage(baseline,candidate,raw)
    reverse = coverage(candidate,baseline,raw)
    data = dict(id=cid,identity=identity,baseline_supported_coverage_by_candidate=retained,
                candidate_supported_coverage_by_baseline=reverse,
                below_98_percent_retention=retained < .98,
                note='3-D raw-signal heuristic; 3 voxel tolerance; no baseline substitution.')
    save(path,data)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    args = ap.parse_args()
    cohort = json.loads(args.cohort.read_text())['cases']
    ready = []
    for r in cohort:
        path = args.output/'cases'/f'{r["id"]}.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if valid_reconstruction_pair(state):
            ready.append(r)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(audit,r,args.output):r['id'] for r in ready}
        for future in as_completed(jobs):
            records.append(future.result())
            values = [x['baseline_supported_coverage_by_candidate'] for x in records]
            summary = dict(pairs=len(records),median_baseline_supported_retention=float(np.median(values)),
                           min_baseline_supported_retention=float(min(values)),
                           below_98_percent_cases=[x['id'] for x in records if x['below_98_percent_retention']],
                           records=sorted(records,key=lambda x:int(x['id'])))
            save(args.output/'branch_retention.json',summary)
            print(f'{len(records)}/{len(ready)} branch audits | {jobs[future]}',flush=True)


if __name__ == '__main__':
    main()
