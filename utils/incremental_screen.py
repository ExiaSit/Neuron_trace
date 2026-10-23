"""Optional one-sided SWC screening helpers; full v0.83 batches process all matched IDs."""
from pathlib import Path
from types import SimpleNamespace
import re
from algorithms.tracing import one_sided
HERE=Path(__file__).resolve().parents[1]

def cell_id(name):
    match=re.fullmatch(r'(?:image_)?(\d+)(?:_0000)?',Path(name).stem)
    if not match:raise ValueError(f'Cannot identify ID: {name}')
    return str(int(match[1]))

def screen(mask, index, cfg):
    cid=cell_id(mask.name);old=index.get(cid,[]);audits=[]
    for path in old:
        result,_,_=one_sided.analyze_swc(path,SimpleNamespace(**cfg['one_sided']))
        audits.append(dict(path=str(path),suspicious=result.suspicious,reason=result.reason,error=result.error,
            major_fraction=result.balance.major_fraction if result.balance else None,
            minor_fraction=result.balance.minor_fraction if result.balance else None))
    if not old:return dict(id=cid,selected=True,reason='missing_swc',swcs=[],audit=[])
    if any(a['error'] for a in audits):return dict(id=cid,selected=False,reason='swc_analysis_error',swcs=[str(x) for x in old],audit=audits)
    selected=any(a['suspicious'] for a in audits)
    return dict(id=cid,selected=selected,reason='one_sided_xy' if selected else 'not_one_sided_xy',swcs=[str(x) for x in old],audit=audits)
