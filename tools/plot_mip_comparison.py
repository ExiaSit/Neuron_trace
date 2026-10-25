#!/usr/bin/env python3
"""Export paired statistics and a scientific overview from saved MIP results."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.compare_mip_repair import saved_records, save, valid_reconstruction_pair


def export(output):
    records = saved_records(output)
    masks = [r for r in records if 'mask' in r]
    pairs = [r for r in records if valid_reconstruction_pair(r)]
    summary = dict(mask_pairs=len(masks), reconstruction_pairs=len(pairs),
                   failures=[r['id'] for r in records if r['status']=='failed'],
                   note='Paired input comparison; per-neuron means; no manual ground truth.',stages={})
    for stage in ('app2','gcut','resample','final'):
        metrics = {}
        for metric in ('nodes','length_um','branches','mean_branch_length_um'):
            a = np.array([r['baseline_swc'][stage][metric] for r in pairs],float)
            b = np.array([r['candidate_swc'][stage][metric] for r in pairs],float)
            d = b-a
            metrics[metric] = dict(baseline_mean=float(a.mean()),candidate_mean=float(b.mean()),
                                   mean_delta=float(d.mean()),median_delta=float(np.median(d)),
                                   increased=int(np.sum(d>1e-6)),decreased=int(np.sum(d < -1e-6)),
                                   equal=int(np.sum(abs(d)<=1e-6)))
        for side in ('baseline','candidate'):
            length = sum(r[side+'_swc'][stage]['length_um'] for r in pairs)
            branches = sum(r[side+'_swc'][stage]['branches'] for r in pairs)
            metrics[side+'_pooled_mean_branch_length_um'] = length/max(1,branches)
        summary['stages'][stage] = metrics
    save(output/'aggregate_metrics.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(12,10))
    retention = {}
    for r in pairs:
        path = output/'branch_audits'/f'{r["id"]}.json'
        if path.exists():
            retention[r['id']] = json.loads(path.read_text())['baseline_supported_coverage_by_candidate']
    def scatter(ax,a,b,title,unit,colors=None,symlog=False):
        maximum = max(max(a,default=1),max(b,default=1),1)*1.05
        ax.scatter(a,b,s=15,c=colors if colors is not None else '#2c7fb8',alpha=.7)
        ax.plot([0,maximum],[0,maximum],color='#777',linestyle='--',linewidth=1)
        if symlog:
            ax.set_xscale('symlog',linthresh=10);ax.set_yscale('symlog',linthresh=10)
        ax.set_xlim(0,maximum);ax.set_ylim(0,maximum)
        ax.set_title(title);ax.set_xlabel('v0.83 '+unit);ax.set_ylabel('v0.83a '+unit)
        ax.grid(alpha=.15)
    scatter(axes[0,0],[r['mask']['baseline']['added_voxels'] for r in masks],
            [r['mask']['candidate']['added_voxels'] for r in masks],
            'Added mask voxels | all valid mask pairs','(voxels)',symlog=True)
    colors = ['#e68613' if retention.get(r['id'],1)<.98 else '#2c7fb8' for r in pairs]
    for ax,field,title in [(axes[0,1],'length_um','Final SWC total length'),
                           (axes[1,0],'mean_branch_length_um','Final SWC mean branch length')]:
        scatter(ax,[r['baseline_swc']['final'][field] for r in pairs],
                [r['candidate_swc']['final'][field] for r in pairs],title,'(um)',colors)
    _,edges,bars=axes[1,1].hist(list(retention.values()),
                               bins=np.unique(np.r_[np.linspace(0,1,21),.98]),color='#2c7fb8')
    for edge,bar in zip(edges[1:],bars):
        if edge<=.98:bar.set_facecolor('#e68613')
    axes[1,1].axvline(.98,color='#d7301f',linestyle='--',label='98% review threshold')
    axes[1,1].set(xlim=(0,1.01),xlabel='Baseline supported branch coverage',ylabel='Cases',
                  title=f'3-D signal retention audit | {len(retention)} pairs')
    axes[1,1].legend()
    axes[1,1].text(.04,.92,f'Below 98%: {sum(v<.98 for v in retention.values())} / {len(retention)}',
                   transform=axes[1,1].transAxes)
    fig.suptitle(f'v0.83 vs v0.83a | {len(pairs)} reconstruction pairs | {len(summary["failures"])} failed cases')
    fig.text(.02,.01,'Orange: supported coverage <98%. Signal heuristics are not manual ownership truth. No historical substitution.',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.96));fig.savefig(output/'summary_metrics.png',dpi=150);plt.close(fig)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();print(json.dumps(export(args.output),indent=2))
