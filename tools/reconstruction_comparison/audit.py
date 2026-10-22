"""Independent integrity checks for the saved comparison, including failures."""
import csv
from collections import Counter
import json
import math
from pathlib import Path
import re
import numpy as np
from PIL import Image
from .metrics import read_swc,swc_total_length
from .report import METRICS,save,sha,summarize,validate_topology,write_csv,write_text_report


def verify(cfg):
    out=Path(cfg['output_root']);run=json.loads((out/'run_manifest.json').read_text())
    pre=json.loads((out/'preflight.json').read_text());summary=json.loads((out/'comparison_summary.json').read_text())
    with Path(cfg['cohort_csv']).open(newline='') as f:ids=[str(int(r['cell_id'])) for r in csv.DictReader(f)]
    assert ids==pre['ids'] and len(ids)==len(set(ids))
    assert sha(cfg['cohort_csv'])==run['cohort_sha256']
    rows=list(csv.DictReader((out/'per_cell_metrics.csv').open()))
    keys={(r['id'],r['version_key']) for r in rows}
    assert len(keys)==len(rows)==len(ids)*len(cfg['versions'])
    reused_count=0;reused_metrics=0;reused_files=0
    reuse_path=out/'reuse_manifest.json'
    if reuse_path.exists():
        reuse=json.loads(reuse_path.read_text());source=Path(reuse['source'])
        previous=json.loads((source/'run_manifest.json').read_text())
        assert previous['fingerprint']==reuse['source_fingerprint'] and previous['code']==run['code']
        indexed={(r['id'],r['version_key']):r for r in rows}
        for case in reuse['cases']:
            cid=case['id'];reused_count+=1
            before=json.loads((source/'manifests'/f'{cid}.json').read_text())
            after=json.loads((out/'manifests'/f'{cid}.json').read_text())
            assert all(before[k]==after[k] for k in ('input_sha256','metadata','shape','crop_for_v081'))
            for artifact in case['files']:
                assert sha(out/'reconstruction'/artifact['relative'])==artifact['sha256']
                assert sha(source/'reconstruction'/artifact['relative'])==artifact['sha256']
                reused_files+=1
            for old in json.loads((source/'metrics'/f'{cid}.json').read_text()):
                new=indexed[cid,old['version_key']]
                assert bool(old['error'])==bool(new['error'])
                if old['error']:continue
                assert new['swc_sha256']==old['swc_sha256']
                assert all(math.isclose(float(new[k]),float(old[k]),rel_tol=1e-12,abs_tol=1e-9) for k in METRICS)
                reused_metrics+=1
        assert reused_count==reuse['reused_count']
    checked_swcs=0;new_coordinates=0;source_count=0;failed=[];stat_verified=0;rehash_verified=0
    archived=cfg.get('app2_input_mode')=='archived_native';imported_verified=0
    for cid in ids:
        manifest=json.loads((out/'manifests'/f'{cid}.json').read_text())
        state=json.loads((out/'status'/f'{cid}.json').read_text())
        assert state['state'] in ('complete','failed')
        if state['state']=='failed':
            log=out/'logs'/cid/(state['current_stage']+'.log')
            lines=[s for s in log.read_text().splitlines() if s.strip()]
            failed.append(dict(id=cid,stage=state['current_stage'],reason=lines[-1],log_path=str(log)))
        for key,path in manifest['inputs'].items():
            path=Path(path);stat=path.stat();original=manifest['input_stat'][key]
            if stat.st_size==original['size'] and stat.st_mtime_ns==original['mtime_ns']:stat_verified+=1
            else:
                assert sha(path)==manifest['input_sha256'][key],(cid,key,'input changed');rehash_verified+=1
            source_count+=1
        if archived:
            source=Path(manifest['inputs']['archived_app2'])
            assert sha(source)==manifest['input_sha256']['archived_app2']
            native=read_swc(source)
            imported_path=out/'reconstruction/trace_app2/down_sampled_swcs_app2'/f'image_{cid}.swc'
            imported=read_swc(imported_path)
            source_rows=np.loadtxt(source,ndmin=2);imported_rows=np.loadtxt(imported_path,ndmin=2)
            assert source_rows.shape==imported_rows.shape
            kept_columns=[i for i in range(source_rows.shape[1]) if i!=3]
            assert np.array_equal(source_rows[:,kept_columns],imported_rows[:,kept_columns])
            assert np.allclose(imported_rows[:,3],manifest['shape'][1]-1-source_rows[:,3],rtol=0,atol=1e-12)
            assert native.parents==imported.parents and native.nodes.keys()==imported.nodes.keys()
            for node,a in native.nodes.items():
                b=imported.nodes[node]
                assert (a[0],a[2],a[3])==(b[0],b[2],b[3])
                assert math.isclose(b[1],manifest['shape'][1]-1-a[1],abs_tol=1e-12)
            assert math.isclose(swc_total_length(native)[0],swc_total_length(imported)[0],rel_tol=1e-12,abs_tol=1e-7)
            imported_verified+=1
        with Image.open(out/'comparisons'/f'{cid}.png') as im:im.verify()
    for row in rows:
        if row['error']:continue
        path=Path(row['swc_path']);assert sha(path)==row['swc_sha256']
        swc=read_swc(path);validate_topology(swc)
        assert all(math.isfinite(value) for node in swc.nodes.values() for value in node)
        # Independently contract non-root degree-two nodes to count segments.
        children=Counter(swc.parents.values())
        edges=sum(p!=-1 for p in swc.parents.values())
        segments=edges-sum(p!=-1 and children[node]==1 for node,p in swc.parents.items())
        length=swc_total_length(swc)[0]
        assert int(row['branch_segments'])==segments,(row['id'],'branch segments')
        assert math.isclose(float(row['mean_branch_length']),length/segments if segments else 0.,rel_tol=1e-12,abs_tol=1e-9)
        checked_swcs+=1
        if row['version_key'] in ('v082_gsdt','app2_current'):
            if not (archived and row['version_key']=='app2_current'):
                assert sum(p==-1 for p in swc.parents.values())==1
            assert int(row['out_of_image_nodes'])==0,(row['id'],row['version_key'])
            new_coordinates+=1
            if row['version_key']=='app2_current':
                expected='# archived_app2_native_to_image_zero_based_v1' if archived else '# app2_native_to_image_zero_based_v2'
                assert expected in path.read_text().splitlines()
    numeric=[]
    for row in rows:
        rec=dict(row)
        for k in METRICS:
            if row.get(k):rec[k]=float(row[k])
        numeric.append(rec)
    recalculated,common=summarize(numeric,cfg['versions'])
    assert common==summary['paired_ids'] and len(common)==summary['paired_count']
    for a,b in zip(recalculated,summary['summary']):
        assert a['count']==b['count']
        for key in ('mean_length','total_length','mean_branches','mean_branch_length','mean_nodes','mean_precision','mean_recall','mean_snr'):
            assert a[key] is None and b[key] is None or math.isclose(a[key],b[key],rel_tol=1e-12,abs_tol=1e-6)
    old_csv=Path(cfg['reference_root'])/'precision_recall_with_app2.csv'
    old={(r['id'],r['version']):r for r in csv.DictReader(old_csv.open())}
    legacy_labels={v['key']:v['legacy_label'] for v in cfg['versions'] if 'legacy_label' in v}
    maxima={k:0. for k in ('length','nodes','branches','precision','recall','mean_snr')};legacy_count=0
    for row in rows:
        if row['version_key'] not in legacy_labels or row['error']:continue
        baseline=old[row['id'],legacy_labels[row['version_key']]]
        for key in maxima:
            difference=abs(float(row[key])-float(baseline[key]));maxima[key]=max(maxima[key],difference)
            assert difference<=1e-6,(row['id'],key,difference)
        legacy_count+=1
    write_csv(out/'reconstruction_failures.csv',failed,fields=['id','stage','reason','log_path'])
    links=re.findall(r'(?:src|href)="([^"]+)"',(out/'index.html').read_text())
    assert all((out/p).is_file() for p in links)
    with Image.open(out/'summary_metrics.png') as im:im.verify()
    for relative,digest in run['code'].items():
        assert sha(Path(run['project'])/relative)==digest,relative
    method_dir=Path(__file__).parent
    report=dict(validation_passed=True,cohort_count=len(ids),analysed_count=summary['analyzed'],
        reconstruction_succeeded=len(ids)-len(failed),reconstruction_failed=len(failed),failures=failed,
        paired_count=len(common),legacy_records_reproduced=legacy_count,legacy_metric_max_abs_difference=maxima,
        valid_metric_swc_files=checked_swcs,branch_length_records_verified=checked_swcs,new_image_coordinate_swcs=new_coordinates,
        archived_app2_imports_verified=imported_verified,
        reused_cases_verified=reused_count,reused_stage_files_verified=reused_files,
        reused_metric_records_verified=reused_metrics,newly_processed_cases=len(ids)-reused_count,
        input_files_verified=source_count,input_stat_unchanged=stat_verified,input_rehashed=rehash_verified,
        valid_comparison_pngs=len(ids),valid_summary_pngs=1,valid_html_links=len(links),
        pipeline_source_unchanged=True,comparison_code_sha256={p.name:sha(p) for p in method_dir.glob('*.py')},
        note='Integrity validation passed. This does not mean all reconstructions succeeded or establish accuracy against manual truth.')
    save(out/'validation.json',report)
    write_text_report(cfg,summary,report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report
