"""Extend a cohort by reusing checked terminal cases from an earlier run."""
from pathlib import Path
import json
import shutil

from .report import save,sha


def processing_config(config):
    """Only output placement and the batch's ID list may differ."""
    return {k:v for k,v in config.items() if k not in ('output_root','neuron_ids','metadata_file')}


def adopt(cfg,records):
    if not cfg.get('reuse_from'):return
    if cfg.get('app2_input_mode')!='archived_native':
        raise ValueError('reuse_from currently requires archived_native APP2 input mode')
    out=Path(cfg['output_root']);source=Path(cfg['reuse_from']).resolve()
    if source==out.resolve():raise ValueError('Reuse source must be a different completed run')
    previous=json.loads((source/'run_manifest.json').read_text())
    current=json.loads((out/'run_manifest.json').read_text())
    validation=json.loads((source/'validation.json').read_text())
    if not validation['validation_passed'] or previous['code']!=current['code'] or previous.get('stages')!=current.get('stages'):
        raise ValueError('Reuse requires verified results from identical pipeline code and stages')
    a=json.loads((source/'pipeline_config.json').read_text());b=json.loads((out/'pipeline_config.json').read_text())
    if processing_config(a)!=processing_config(b):raise ValueError('Processing parameters differ; refusing reuse')
    saved=out/'reuse_manifest.json'
    prior=json.loads(saved.read_text()) if saved.exists() else {}
    if prior and (prior['source']!=str(source) or prior['source_fingerprint']!=previous['fingerprint']):
        raise ValueError('Existing reuse manifest refers to a different source run')
    adopted=list(prior.get('cases',[]))
    for rec in records:
        cid=rec['id'];old_manifest=source/'manifests'/f'{cid}.json'
        if not old_manifest.exists():continue
        if (out/'status'/f'{cid}.json').exists():continue
        old=json.loads(old_manifest.read_text())
        for key in ('input_sha256','metadata','shape','crop_for_v081'):
            if old[key]!=rec[key]:raise ValueError(f'{cid}: {key} differs; refusing reuse')
        state=json.loads((source/'status'/f'{cid}.json').read_text())
        if state['fingerprint']!=previous['fingerprint'] or state['state'] not in ('complete','failed'):
            raise ValueError(f'{cid}: source case is not a matching terminal result')
        # Ensure source APP2/final files still match the evaluated source run.
        metrics=json.loads((source/'metrics'/f'{cid}.json').read_text())
        for row in metrics:
            if row['version_key'] in ('app2_current','v082_gsdt') and not row['error']:
                if sha(row['swc_path'])!=row['swc_sha256']:raise ValueError(f'{cid}: source output changed')
        products=[('import_app2',f'trace_app2/down_sampled_swcs_app2/image_{cid}.swc'),
                  ('gcut',f'gcut_output/selected_for_pruning/image_{cid}.swc'),
                  ('gcut_resample',f'gcut_output/selected_resampled_step1/image_{cid}.swc'),
                  ('prune',f'gcut_pruned/pruning_swcs/image_{cid}.swc')]
        files=[]
        for stage,relative in products:
            if stage not in state['completed_stages']:continue
            src=source/'reconstruction'/relative;dst=out/'reconstruction'/relative
            digest=sha(src);dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
            if sha(dst)!=digest:raise ValueError('Copy integrity error: '+str(dst))
            files.append(dict(relative=relative,sha256=digest))
        # Keep diagnostic products available without duplicating large images.
        diagnostic=source/'reconstruction/gcut_output'/cid
        link=out/'reconstruction/gcut_output'/cid
        if diagnostic.is_dir() and not link.exists():link.symlink_to(diagnostic,target_is_directory=True)
        shutil.copytree(source/'logs'/cid,out/'logs'/cid,dirs_exist_ok=True)
        state.update(fingerprint=current['fingerprint'],reused_from=str(source),reused_terminal=True)
        save(out/'status'/f'{cid}.json',state)
        adopted.append(dict(id=cid,state=state['state'],files=files))
    save(out/'reuse_manifest.json',dict(source=str(source),source_fingerprint=previous['fingerprint'],
        reused_count=len(adopted),cases=adopted,policy='Identical processing code/parameters and per-case inputs; terminal failures retained without retry'))
    print(f'Reused {len(adopted)} checked terminal cases from {source}',flush=True)
